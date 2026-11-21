from __future__ import annotations

from collections import Counter

import pytest
import torch

from megabake import make_backend
from megabake.fx_handler.transformers import StatefulAttention
from tests.fixtures import SmolLMArchitecture
from tests.gpu.test_task26_attention import Attention, compile_and_compare

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 0),
    reason="SM100 CUDA device required",
)


@pytest.fixture
def precise_bf16_reference():
    # PyTorch's reduced-precision BF16 GEMM can be inaccurate near cancellation
    # on SM100. Compare against its FP32-accumulating BF16 path instead.
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


@pytest.mark.parametrize("query_length", [1, 5])
def test_sm100_prefill_attention_and_rebinding(tmp_path, precise_bf16_reference, query_length):
    torch.manual_seed(26)
    source = torch.randn(2, query_length, 4, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.randn(2, 1, query_length, 7, device="cuda", dtype=torch.bfloat16)

    model = Attention().eval()
    backend, actual, expected = compile_and_compare(model, tmp_path, (source, key, value, mask))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert backend.megabake_last_compiled.plan.launch.architecture == "sm100"
    assert any(
        p.body.provider == "sm100_attention" for p in backend.megabake_last_compiled.plan.phases
    )
    assert backend.megabake_contexts[-1].requirements["execution"] == "megakernel"

    new_mask = torch.randn_like(mask)
    with torch.no_grad():
        actual = backend.megabake_optimized(source, key, value, new_mask)
        expected = model(source, key, value, new_mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sm100_fully_masked_attention_row(tmp_path, precise_bf16_reference):
    torch.manual_seed(27)
    source = torch.randn(2, 5, 4, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.randn(2, 1, 5, 7, device="cuda", dtype=torch.bfloat16)
    mask[0, 0, 0].fill_(float("-inf"))

    _, actual, expected = compile_and_compare(
        Attention().eval(), tmp_path, (source, key, value, mask)
    )
    masked = torch.isnan(expected)
    assert masked.any()
    assert torch.equal(actual[masked], torch.zeros_like(actual[masked]))
    torch.testing.assert_close(actual[~masked], expected[~masked], rtol=0, atol=0)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("value_block", ["4", "8"])
def test_sm100_prefill_value_loads_preserve_results(
    monkeypatch, tmp_path, precise_bf16_reference, value_block
):
    monkeypatch.setenv("MEGABAKE_PREFILL_VALUE_BLOCK", value_block)
    torch.manual_seed(792)
    source = torch.randn(2, 5, 4, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 13, 2, 8, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.randn(2, 1, 5, 13, device="cuda", dtype=torch.bfloat16)
    _, actual, expected = compile_and_compare(
        Attention().eval(), tmp_path, (source, key, value, mask)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sm100_repeated_row_decode_preserves_cache(tmp_path, precise_bf16_reference):
    torch.manual_seed(28)
    query = torch.randn(1, 1, 2, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    keys = torch.randn(1, 3, 2, 8, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    length = torch.tensor([1], device="cuda", dtype=torch.int32)
    expected_keys, expected_values, expected_length = keys.clone(), values.clone(), length.clone()
    model = StatefulAttention().eval()
    backend = make_backend(mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path)
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)

    for scale in (1.0, 0.5):
        with torch.no_grad():
            actual = optimized(query, key * scale, value * scale, keys, values, length)
            expected = model(
                query, key * scale, value * scale, expected_keys, expected_values, expected_length
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
        torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
        torch.testing.assert_close(length, expected_length, rtol=0, atol=0)

    assert length.item() == 3
    assert any(
        p.body.name == "attention_stateful_decode_rows"
        for p in backend.megabake_last_compiled.plan.phases
    )


@pytest.mark.parametrize(
    "capacity,lengths",
    [
        (130, (0, 63, 64, 129)),
        (2050, (511, 512, 1023, 2048)),
    ],
)
def test_sm100_split_decode_preserves_cache(
    tmp_path, monkeypatch, precise_bf16_reference, capacity, lengths
):
    from megabake import compiler

    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)
    torch.manual_seed(28)
    query = torch.randn(1, 1, 4, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    original_keys = torch.randn(1, capacity, 2, 64, device="cuda", dtype=torch.bfloat16)
    original_values = torch.randn_like(original_keys)
    model = StatefulAttention().eval()
    backend = make_backend(
        mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)

    for length in lengths:
        keys, values = original_keys.clone(), original_values.clone()
        expected_keys, expected_values = original_keys.clone(), original_values.clone()
        valid_length = torch.tensor([length], device="cuda", dtype=torch.int32)
        expected_length = valid_length.clone()
        with torch.no_grad():
            actual = optimized(query, key, value, keys, values, valid_length)
            expected = model(query, key, value, expected_keys, expected_values, expected_length)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
        torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
        torch.testing.assert_close(valid_length, expected_length, rtol=0, atol=0)
        assert valid_length.item() == length + 1

    assert (
        backend.megabake_last_compiled.generated.launch.launch_policy
        == "megabake_cooperative_table"
    )
    assert any(
        p.body.provider == "sm100_attention" for p in backend.megabake_last_compiled.plan.phases
    )
    assert backend.megabake_contexts[-1].requirements["execution"] == "megakernel"


def test_sm100_compiles_small_causal_model(tmp_path, precise_bf16_reference):
    torch.manual_seed(31)
    model = SmolLMArchitecture(layers=1, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    mask = torch.ones_like(ids)
    backend = make_backend(mode="prefill", compile_kernel=True, cache=False, dump_dir=tmp_path)
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        actual = optimized(ids, attention_mask=mask).logits.float()
        expected = model(ids, attention_mask=mask).logits.float()
    assert ((actual - expected).norm() / expected.norm()).item() < 0.02
    providers = Counter(p.body.provider for p in backend.megabake_last_compiled.plan.phases)
    assert providers["sm100_attention"] > 0
    assert providers["blackwell_mma"] > 0
    assert backend.megabake_contexts[-1].requirements["execution"] == "megakernel"
    assert backend.megabake_last_trace[-1].workload_kernel_launches == 1
