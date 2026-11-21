from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch.autograd import DeviceType

from megabake import make_backend, verify_plan
from tests.fixtures import StatefulTransformerBlock, TransformerBlock

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="task 29 requires an SM90 CUDA device",
)


def _compile(model, mode, output):
    backend = make_backend(
        mode=mode,
        compile_kernel=True,
        cache=False,
        dump_dir=output,
        scheduler="phase",
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    return backend, optimized


def _assert_block_plan(backend):
    compiled = backend.megabake_last_compiled
    verify_plan(compiled.plan)
    assert backend.megabake_contexts[-1].requirements["execution"] == "megakernel"
    assert compiled.generated.source_path.exists()

    providers = Counter(phase.body.provider for phase in compiled.plan.phases)
    assert providers["g0_range"] == 7
    assert providers["attention"] == 1
    assert providers["generic"] >= 2
    assert all(phase.work_items for phase in compiled.plan.phases)
    assert all(phase.workers for phase in compiled.plan.phases)
    assert any(phase.barrier is not None for phase in compiled.plan.phases)
    assert compiled.resources.active_blocks_per_sm >= 1
    assert compiled.resources.max_cooperative_workers >= compiled.plan.launch.grid_size
    assert compiled.resources.workspace_bytes == compiled.plan.storage.workspace_bytes
    assert compiled.resources.compiled_code_bytes > 0
    assert compiled.resources.body_variant_count < len(compiled.plan.phases)


def _inference_call(item):
    # Grad mode is thread-local, so worker threads must disable it themselves.
    function, arguments = item
    with torch.no_grad():
        return function(*arguments)


def _assert_one_kernel(function):
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler,
        torch.no_grad(),
    ):
        result = function()
        torch.cuda.synchronize()
    assert len([event for event in profiler.events() if event.device_type == DeviceType.CUDA]) == 1
    return result


def test_task29_compiles_complete_prefill_block(tmp_path):
    torch.manual_seed(29)
    model = TransformerBlock(
        hidden_size=576,
        heads=9,
        key_value_heads=3,
        intermediate_size=1536,
    ).eval()
    source = torch.randn(2, 5, 576, device="cuda", dtype=torch.bfloat16) / 24
    residual = torch.randn_like(source)
    query_cosine = torch.randn(2, 5, 9, 64, device="cuda", dtype=torch.bfloat16)
    query_sine = torch.randn_like(query_cosine)
    key_cosine = torch.randn(2, 5, 3, 64, device="cuda", dtype=torch.bfloat16)
    key_sine = torch.randn_like(key_cosine)
    mask = torch.randn(2, 1, 5, 5, device="cuda", dtype=torch.bfloat16)
    second_source = torch.randn_like(source)
    second_mask = torch.randn_like(mask)
    inputs = (
        source,
        residual,
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        mask,
    )
    second_inputs = (
        second_source,
        residual,
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        second_mask,
    )

    backend, optimized = _compile(model, "prefill", tmp_path)
    with torch.no_grad():
        actual = optimized(*inputs)
        expected = model(*inputs)
        second_actual = optimized(*second_inputs)
        second_expected = model(*second_inputs)
        backend.megabake_last_callable.megabake_runtime_state.poison_workspace()
        poisoned_actual = optimized(*inputs)
        with ThreadPoolExecutor(max_workers=2) as executor:
            concurrent_results = list(
                executor.map(
                    _inference_call,
                    ((optimized, inputs), (optimized, second_inputs)),
                )
            )

    for result, reference in (
        (actual, expected),
        (second_actual, second_expected),
        (poisoned_actual, expected),
        (concurrent_results[0], expected),
        (concurrent_results[1], second_expected),
    ):
        assert all(
            (result_value.float() - reference_value.float()).norm().item()
            / reference_value.float().norm().item()
            < 0.01
            for result_value, reference_value in zip(result, reference, strict=True)
        )
    _assert_block_plan(backend)
    _assert_one_kernel(lambda: optimized(*second_inputs))


def test_task29_compiles_complete_stateful_decode_block(tmp_path):
    torch.manual_seed(29)
    model = StatefulTransformerBlock(
        hidden_size=576,
        heads=9,
        key_value_heads=3,
        intermediate_size=1536,
    ).eval()
    source = torch.randn(2, 1, 576, device="cuda", dtype=torch.bfloat16) / 24
    residual = torch.randn_like(source)
    query_cosine = torch.randn(2, 1, 9, 64, device="cuda", dtype=torch.bfloat16)
    query_sine = torch.randn_like(query_cosine)
    key_cosine = torch.randn(2, 1, 3, 64, device="cuda", dtype=torch.bfloat16)
    key_sine = torch.randn_like(key_cosine)
    key_cache = torch.randn(2, 4, 3, 64, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    valid_length = torch.tensor([1], device="cuda", dtype=torch.int32)
    inputs = (
        source,
        residual,
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        key_cache,
        value_cache,
        valid_length,
    )

    expected_key_cache = key_cache.clone()
    expected_value_cache = value_cache.clone()
    expected_length = valid_length.clone()
    expected_inputs = (
        source,
        residual,
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        expected_key_cache,
        expected_value_cache,
        expected_length,
    )

    backend, optimized = _compile(model, "decode", tmp_path)
    with torch.no_grad():
        first = optimized(*inputs)
        first_expected = model(*expected_inputs)
        second_source = source * 0.5
        second = optimized(second_source, *inputs[1:])
        second_expected = model(second_source, *expected_inputs[1:])
        third = _assert_one_kernel(lambda: optimized(second_source, *inputs[1:]))
        third_expected = model(second_source, *expected_inputs[1:])

    for result, reference in (
        (first, first_expected),
        (second, second_expected),
        (third, third_expected),
    ):
        assert all(
            (result_value.float() - reference_value.float()).norm().item()
            / reference_value.float().norm().item()
            < 0.01
            for result_value, reference_value in zip(result, reference, strict=True)
        )
    torch.testing.assert_close(key_cache, expected_key_cache, rtol=0, atol=0)
    torch.testing.assert_close(value_cache, expected_value_cache, rtol=0, atol=0)
    torch.testing.assert_close(valid_length, expected_length, rtol=0, atol=0)
    assert int(valid_length.item()) == 4

    _assert_block_plan(backend)
    assert backend.megabake_last_compiled.plan.storage.state[0].capacity == 4
    assert backend.megabake_last_compiled.plan.storage.state[0].lifetime == ("boundary_state")
