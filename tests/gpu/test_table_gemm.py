"""Tuned TMA GEMM bodies inside the table-driven launch.

Large graphs use the table launch, whose GEMM phases build their TMA
descriptors in the kernel prologue from the device address table. These tests
force that launch on a two-layer model so it compiles in seconds; the task 31
tests cover the full 30-layer model.
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.fixtures import SmolLMArchitecture, StatefulSmolLMArchitecture


def _sm90_available() -> bool:
    if os.environ.get("MEGABAKE_RUN_GPU_TESTS") != "1":
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)


pytestmark = pytest.mark.skipif(
    not _sm90_available(),
    reason="table GEMM tests require an SM90 CUDA device",
)

LAYERS = 2
# Q, K, V, O, gate and up as one paired GEMM, and down per layer, plus the
# vocabulary projection.
PROJECTIONS = 6 * LAYERS + 1


@pytest.fixture
def table_launch(monkeypatch):
    import megabake.compiler as compiler

    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)


def _assert_tuned_table_projections(backend) -> None:
    compiled = backend.megabake_last_compiled
    launch = compiled.generated.launch
    assert launch.launch_policy == "megabake_cooperative_table"
    gemm_phases = [
        phase for phase in compiled.plan.phases if phase.body.provider == "g0_range"
    ]
    assert len(gemm_phases) == PROJECTIONS
    assert not any(
        phase.body.name == "generic_contraction" for phase in compiled.plan.phases
    )
    assert len(launch.gemm_table) == PROJECTIONS
    # O and down add the residual in their epilogues; gate and up run as one
    # paired GEMM whose epilogue applies SiLU and the product.
    names = [phase.body.name for phase in gemm_phases]
    assert names.count("g0_range_swapab_16x64_epilogue") == 2 * LAYERS
    assert names.count("g0_range_swapab_16x64_paired_epilogue") == LAYERS
    # Equal shapes share one variant whatever their epilogues: Q/O, K/V,
    # gate/up, down, and vocabulary.
    assert len(launch.gemm_template_phases) == 5
    assert compiled.plan.launch.grid_size == compiled.resources.max_cooperative_workers
    assert compiled.resources.local_memory_bytes_per_thread <= 64
    trace = backend.megabake_last_trace[-1]
    assert trace.recurring_gpu_operations == (("workload_kernel", 1),)


def test_table_launch_runs_tuned_gemms_in_prefill(tmp_path, table_launch):
    from megabake import make_backend

    torch.manual_seed(7)
    model = SmolLMArchitecture(layers=LAYERS, device="cuda", dtype=torch.bfloat16)
    backend = make_backend(
        mode="prefill", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    calls = (
        (torch.tensor([[1, 2, 3, 4]], device="cuda"), torch.ones(1, 4, dtype=torch.int64, device="cuda")),
        (torch.tensor([[9, 8, 7, 6]], device="cuda"), torch.tensor([[1, 1, 1, 0]], device="cuda")),
    )
    with torch.no_grad():
        for input_ids, mask in calls:
            actual = optimized(input_ids, mask).logits
            expected = model(input_ids, mask).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    _assert_tuned_table_projections(backend)


def _run_stateful_decode(tmp_path):
    """Four decode calls of the two-layer model, bit-exact against eager."""
    from megabake import make_backend

    torch.manual_seed(7)
    model = StatefulSmolLMArchitecture(layers=LAYERS, device="cuda", dtype=torch.bfloat16)

    def state():
        return (
            *(torch.zeros(1, 4, 3, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2 * LAYERS)),
            *(torch.zeros(1, device="cuda", dtype=torch.int32) for _ in range(LAYERS)),
        )

    actual_state, reference_state = state(), state()
    backend = make_backend(
        mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for position, token in enumerate((5, 6, 7, 8)):
            token_ids = torch.tensor([[token]], device="cuda")
            position_ids = torch.tensor([[position]], device="cuda")
            actual = optimized(token_ids, position_ids, *actual_state)
            expected = model(token_ids, position_ids, *reference_state)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_tensor, reference_tensor in zip(actual_state, reference_state, strict=True):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)
    return backend


def test_table_launch_runs_tuned_gemms_in_stateful_decode(tmp_path, table_launch):
    backend = _run_stateful_decode(tmp_path)

    _assert_tuned_table_projections(backend)
    # Attention computes the rotated query and new key itself.
    attention_regions = [
        phase.region_id
        for phase in backend.megabake_last_compiled.plan.phases
        if phase.body.provider == "attention"
    ]
    assert len(attention_regions) == LAYERS
    assert all(region.endswith(":attention_prologue") for region in attention_regions)
    assert not any(
        phase.region_id.endswith(":rope") for phase in backend.megabake_last_compiled.plan.phases
    )


def test_split_context_decode_matches_eager_at_long_contexts(tmp_path, table_launch):
    """Decode at contexts that use one, several and a full wave of key chunks
    per row, with one query head per row and, at the longest context, the
    whole KV group per row: logits and the appended entries are as close to
    fp32 as eager bf16 is (bit-exact with one chunk), other entries are
    untouched and lengths advance by one."""
    import copy

    from megabake import make_backend
    from megabake.bodies.cuda.sm90.attention import (
        DECODE_PACKED_KEYS_PER_HEAD,
        decode_packed_heads,
        decode_split_steps,
    )

    torch.manual_seed(11)
    model = StatefulSmolLMArchitecture(layers=LAYERS, device="cuda", dtype=torch.bfloat16)
    exact_model = copy.deepcopy(model).float()
    capacity = 1100
    caches = [
        torch.randn(1, capacity, 3, 64, device="cuda", dtype=torch.bfloat16)
        for _ in range(2 * LAYERS)
    ]
    backend = make_backend(
        mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)

    def relative(actual, expected):
        return (actual.float() - expected.float()).norm().item() / expected.float().norm().item()

    with torch.no_grad():
        for context in (0, 63, 200, 1000):
            lengths = [
                torch.full((1,), context, device="cuda", dtype=torch.int32) for _ in range(LAYERS)
            ]
            actual_state = [tensor.clone() for tensor in (*caches, *lengths)]
            reference_state = [tensor.clone() for tensor in (*caches, *lengths)]
            exact_state = [*(cache.float() for cache in caches), *(length.clone() for length in lengths)]
            token_ids = torch.tensor([[5]], device="cuda")
            position_ids = torch.tensor([[context]], device="cuda")
            actual = optimized(token_ids, position_ids, *actual_state)
            expected = model(token_ids, position_ids, *reference_state)
            exact = exact_model(token_ids, position_ids, *exact_state)
            if context < 64:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert relative(actual, exact) <= 1.25 * relative(expected, exact) + 1e-3
            for cache, actual_cache, reference_cache, exact_cache in zip(
                caches, actual_state, reference_state, exact_state
            ):
                assert torch.equal(actual_cache[:, :context], cache[:, :context])
                assert torch.equal(actual_cache[:, context + 1 :], cache[:, context + 1 :])
                row = slice(context, context + 1)
                assert relative(actual_cache[:, row], exact_cache[:, row]) <= (
                    1.25 * relative(reference_cache[:, row], exact_cache[:, row]) + 1e-3
                )
            assert all(int(length.item()) == context + 1 for length in actual_state[2 * LAYERS :])
    # The longest context splits each row over several chunks, and the
    # contexts run both the one-head and the packed rows.
    workers = backend.megabake_last_compiled.plan.launch.grid_size
    steps = decode_split_steps(3, workers, 3, 64, capacity)
    assert any(first <= 1001 and count > 1 for first, count in steps)
    packed = decode_packed_heads(3, 64)
    assert packed == 3 and 200 < DECODE_PACKED_KEYS_PER_HEAD * packed <= 1000


def test_table_launch_streams_weights_in_stateful_decode(tmp_path, table_launch, monkeypatch):
    import megabake.codegen as codegen

    # Streaming is off by default (ROADMAP step 3); it must stay exact.
    monkeypatch.setattr(codegen, "WEIGHT_STREAM_STAGES", 64)
    backend = _run_stateful_decode(tmp_path)

    _assert_tuned_table_projections(backend)
    generated = backend.megabake_last_compiled.generated
    # Every barrier before a GEMM interval streams, and every variant streams
    # somewhere: Q/O, K/V, gate/up, down and the vocabulary projection.
    stream_rows = generated.launch.stream_table[1:]
    assert len(stream_rows) == 4 * LAYERS
    assert {row[3 * cta + 1] for row in stream_rows for cta in range(len(row) // 3) if row[3 * cta]} == set(
        range(len(generated.launch.gemm_template_phases))
    )
    assert generated.source.count(".preload_table(") == len(generated.launch.gemm_template_phases)


def test_graphed_stateful_decode_matches_eager(tmp_path, table_launch):
    """Replays of a CUDA-graphed decode step advance the KV state bit-exactly."""
    from megabake import graphed, make_backend

    torch.manual_seed(7)
    model = StatefulSmolLMArchitecture(layers=LAYERS, device="cuda", dtype=torch.bfloat16)

    def state():
        return (
            *(torch.zeros(1, 6, 3, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2 * LAYERS)),
            *(torch.zeros(1, device="cuda", dtype=torch.int32) for _ in range(LAYERS)),
        )

    actual_state, reference_state = state(), state()
    backend = make_backend(
        mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        step = graphed(
            optimized,
            torch.tensor([[1]], device="cuda"),
            torch.tensor([[0]], device="cuda"),
            *actual_state,
        )
        # Warm-up calls must leave the state as it was.
        for tensor in actual_state:
            assert not tensor.any()
        for position, token in enumerate((5, 6, 7, 8)):
            token_ids = torch.tensor([[token]], device="cuda")
            position_ids = torch.tensor([[position]], device="cuda")
            actual = step(token_ids, position_ids, *actual_state)
            expected = model(token_ids, position_ids, *reference_state)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Eager calls on the same compiled step still line up with the device counter.
        token_ids = torch.tensor([[9]], device="cuda")
        position_ids = torch.tensor([[4]], device="cuda")
        actual = optimized(token_ids, position_ids, *actual_state)
        expected = model(token_ids, position_ids, *reference_state)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_tensor, reference_tensor in zip(actual_state, reference_state, strict=True):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)
