from __future__ import annotations

import copy
import os

import pytest
import torch
from torch.autograd import DeviceType

from tests.fixtures import SmolLMArchitecture, StatefulSmolLMArchitecture


def _sm90_available() -> bool:
    if os.environ.get("MEGABAKE_RUN_GPU_TESTS") != "1":
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)


pytestmark = pytest.mark.skipif(
    not _sm90_available(),
    reason="task 31 requires an SM90 CUDA device",
)


def _assert_tuned_projections(compiled, layers: int = 30) -> None:
    # Every Q/K/V/O, gate/up/down, and vocabulary projection runs the tuned
    # TMA GEMM inside the table launch, on every cooperative worker; gate and
    # up run as one paired GEMM.
    assert compiled.generated.launch.launch_policy == "megabake_cooperative_table"
    assert (
        sum(phase.body.provider == "g0_range" for phase in compiled.plan.phases) == 6 * layers + 1
    )
    assert compiled.plan.launch.grid_size == compiled.resources.max_cooperative_workers


def test_task31_compiles_complete_smollm_architecture(tmp_path):
    from megabake import make_backend, verify_plan

    torch.manual_seed(31)
    model = SmolLMArchitecture(layers=30, device="cuda", dtype=torch.bfloat16)
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    second_input_ids = torch.tensor([[5, 6, 7, 8]], device="cuda")
    attention_mask = torch.ones(1, 4, dtype=torch.int64, device="cuda")
    second_attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.int64, device="cuda")

    backend = make_backend(
        mode="prefill",
        compile_kernel=True,
        cache=False,
        dump_dir=tmp_path,
        scheduler="phase",
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    inputs = (input_ids, attention_mask)
    second_inputs = (second_input_ids, second_attention_mask)

    with torch.no_grad():
        actual = optimized(*inputs)
        expected = model(*inputs)
        second_actual = optimized(*second_inputs)
        second_expected = model(*second_inputs)

    for result, reference in (
        (actual.logits, expected.logits),
        (second_actual.logits, second_expected.logits),
    ):
        error = (result.float() - reference.float()).norm().item()
        scale = reference.float().norm().item()
        assert error / scale < 0.02

    compiled = backend.megabake_last_compiled
    context = backend.megabake_contexts[-1]
    verify_plan(compiled.plan)
    assert context.requirements["execution"] == "megakernel"
    # Dynamo names the positional input_ids and attention_mask arguments.
    assert {"l_args_0_", "l_args_1_"} <= set(context.input_order)
    assert len(compiled.plan.graph.ops) > 3_000
    assert len(compiled.plan.phases) > 100
    assert compiled.plan.storage.state == ()
    assert compiled.resources.workspace_bytes == compiled.plan.storage.workspace_bytes
    assert compiled.resources.body_variant_count < len(compiled.plan.phases)
    assert compiled.resources.compiled_code_bytes > 0
    assert compiled.generated.source_path.exists()
    _assert_tuned_projections(compiled)

    trace = backend.megabake_last_trace[-1]
    assert trace.workload_kernel_launches == 1
    assert trace.recurring_gpu_operations == (("workload_kernel", 1),)

    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler,
        torch.no_grad(),
    ):
        optimized(*second_inputs)
        torch.cuda.synchronize()
    assert len([event for event in profiler.events() if event.device_type == DeviceType.CUDA]) == 1


def test_task31_compiles_complete_stateful_smollm_decode(tmp_path):
    from megabake import make_backend, verify_plan

    torch.manual_seed(31)
    model = StatefulSmolLMArchitecture(layers=30, device="cuda", dtype=torch.bfloat16)
    key_caches = tuple(
        torch.zeros(1, 4, 3, 64, device="cuda", dtype=torch.bfloat16) for _ in range(30)
    )
    value_caches = tuple(
        torch.zeros(1, 4, 3, 64, device="cuda", dtype=torch.bfloat16) for _ in range(30)
    )
    valid_lengths = tuple(torch.zeros(1, device="cuda", dtype=torch.int32) for _ in range(30))
    reference_key_caches = tuple(cache.clone() for cache in key_caches)
    reference_value_caches = tuple(cache.clone() for cache in value_caches)
    reference_lengths = tuple(length.clone() for length in valid_lengths)
    # An fp32 copy gives the accuracy baseline that eager bf16 is held to.
    exact_model = copy.deepcopy(model).float()
    exact_key_caches = tuple(cache.float() for cache in key_caches)
    exact_value_caches = tuple(cache.float() for cache in value_caches)
    exact_lengths = tuple(length.clone() for length in valid_lengths)

    def advance_exact(token, position_ids):
        exact_model(token, position_ids, *exact_key_caches, *exact_value_caches, *exact_lengths)

    backend = make_backend(
        mode="decode",
        compile_kernel=True,
        cache=False,
        dump_dir=tmp_path,
        scheduler="phase",
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    tokens = (
        torch.tensor([[1]], device="cuda"),
        torch.tensor([[2]], device="cuda"),
        torch.tensor([[3]], device="cuda"),
        torch.tensor([[4]], device="cuda"),
    )

    with torch.no_grad():
        for position, token in enumerate(tokens[:3]):
            position_ids = torch.tensor([[position]], device="cuda")
            actual = optimized(token, position_ids, *key_caches, *value_caches, *valid_lengths)
            expected = model(
                token,
                position_ids,
                *reference_key_caches,
                *reference_value_caches,
                *reference_lengths,
            )
            advance_exact(token, position_ids)
            error = (actual.float() - expected.float()).norm().item()
            scale = expected.float().norm().item()
            assert error / scale < 0.02

        position_ids = torch.tensor([[3]], device="cuda")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            actual = optimized(tokens[3], position_ids, *key_caches, *value_caches, *valid_lengths)
            torch.cuda.synchronize()
        expected = model(
            tokens[3],
            position_ids,
            *reference_key_caches,
            *reference_value_caches,
            *reference_lengths,
        )
        advance_exact(tokens[3], position_ids)
        error = (actual.float() - expected.float()).norm().item()
        scale = expected.float().norm().item()
        assert error / scale < 0.02

    # Cached K/V carry bf16 rounding accumulated over earlier layers, so the
    # compiled caches must be as close to fp32 as eager bf16 is; lengths are exact.
    def relative(actual, expected):
        return (actual.float() - expected).norm().item() / expected.norm().item()

    for actual_cache, reference_cache, exact_cache in (
        *zip(key_caches, reference_key_caches, exact_key_caches, strict=True),
        *zip(value_caches, reference_value_caches, exact_value_caches, strict=True),
    ):
        assert relative(actual_cache, exact_cache) <= (
            1.25 * relative(reference_cache, exact_cache) + 1e-3
        )
    for actual_length, reference_length in zip(valid_lengths, reference_lengths, strict=True):
        torch.testing.assert_close(actual_length, reference_length, rtol=0, atol=0)
        assert int(actual_length.item()) == 4

    compiled = backend.megabake_last_compiled
    verify_plan(compiled.plan)
    assert len(compiled.plan.storage.state) == 30
    assert all(state.capacity == 4 for state in compiled.plan.storage.state)
    _assert_tuned_projections(compiled)
    assert compiled.resources.workspace_bytes == compiled.plan.storage.workspace_bytes
    assert compiled.resources.compiled_code_bytes > 0
    trace = backend.megabake_last_trace[-1]
    assert trace.workload_kernel_launches == 1
    assert trace.recurring_gpu_operations == (("workload_kernel", 1),)
    assert len([event for event in profiler.events() if event.device_type == DeviceType.CUDA]) == 1


def _pinned_smollm():
    transformers = pytest.importorskip("transformers")
    from tests.fixtures import SMOLLM_MODEL_NAME, SMOLLM_REVISION

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            SMOLLM_MODEL_NAME,
            revision=SMOLLM_REVISION,
            attn_implementation="eager",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
    except OSError:
        pytest.skip("pinned SmolLM checkpoint is not in the local Hugging Face cache")
    model.config.use_cache = False
    return model.to("cuda").eval()


def test_task31_compiles_pinned_smollm_checkpoint(tmp_path):
    from megabake import make_backend, verify_plan

    model = _pinned_smollm()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    backend = make_backend(
        mode="prefill",
        compile_kernel=True,
        cache=False,
        dump_dir=tmp_path,
        scheduler="phase",
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    attention_mask = torch.ones(1, 4, dtype=torch.int64, device="cuda")
    with torch.no_grad():
        for input_ids in (
            torch.tensor([[1, 504, 3458, 314]], device="cuda"),
            torch.tensor([[1, 22, 333, 4444]], device="cuda"),
        ):
            actual = optimized(input_ids, attention_mask=attention_mask).logits
            expected = model(input_ids, attention_mask=attention_mask).logits
            error = (actual.float() - expected.float()).norm().item()
            assert error / expected.float().norm().item() < 0.02
            assert torch.equal(actual.argmax(-1), expected.argmax(-1))

    compiled = backend.megabake_last_compiled
    verify_plan(compiled.plan)
    assert backend.megabake_last_trace[-1].workload_kernel_launches == 1
    _assert_tuned_projections(compiled)
