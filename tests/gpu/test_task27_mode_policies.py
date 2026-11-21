from __future__ import annotations

import pytest
import torch

from megabake import make_backend

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="task 27 requires an SM90 CUDA device",
)


def grouped_attention(source, key, value, mask):
    query = source.permute(0, 2, 1, 3)
    grouped_key = key.permute(0, 2, 1, 3).repeat_interleave(source.shape[2] // key.shape[2], dim=1)
    grouped_value = value.permute(0, 2, 1, 3).repeat_interleave(
        source.shape[2] // value.shape[2], dim=1
    )
    scores = (query @ grouped_key.transpose(-2, -1)) * source.shape[-1] ** -0.5 + mask
    probabilities = torch.softmax(scores.float(), dim=-1).to(source.dtype)
    return probabilities @ grouped_value


class Attention(torch.nn.Module):
    def forward(self, source, key, value, mask):
        return grouped_attention(source, key, value, mask)


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(1536, 576, device="cuda", dtype=torch.bfloat16) / 24
        )

    def forward(self, source):
        return torch.nn.functional.linear(source, self.weight)


def compile_and_compare(model, mode, tmp_path, inputs):
    backend = make_backend(
        mode=mode,
        compile_kernel=True,
        cache=False,
        dump_dir=tmp_path,
        scheduler="phase",
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        actual = optimized(*inputs)
        expected = model(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return backend


def attention_inputs(batch, query_length):
    torch.manual_seed(27)
    return (
        torch.randn(batch, query_length, 4, 8, device="cuda", dtype=torch.bfloat16),
        torch.randn(batch, 7, 2, 8, device="cuda", dtype=torch.bfloat16),
        torch.randn(batch, 7, 2, 8, device="cuda", dtype=torch.bfloat16),
        torch.randn(batch, 1, query_length, 7, device="cuda", dtype=torch.bfloat16),
    )


def test_task27_short_prefill_uses_prefill_attention(tmp_path):
    backend = compile_and_compare(
        Attention().eval(),
        "prefill",
        tmp_path / "prefill",
        attention_inputs(batch=2, query_length=5),
    )
    selected = backend.megabake_last_compiled.plan.selection.selections[0]
    assert selected.body.name == "attention_prefill_rows"


def test_task27_larger_decode_batch_uses_decode_attention(tmp_path):
    backend = compile_and_compare(
        Attention().eval(),
        "decode",
        tmp_path / "decode",
        attention_inputs(batch=8, query_length=1),
    )
    selected = backend.megabake_last_compiled.plan.selection.selections[0]
    assert selected.body.name == "attention_decode_rows"


def test_task27_large_decode_batch_keeps_throughput_gemm(tmp_path):
    torch.manual_seed(27)
    source = torch.randn(128, 576, device="cuda", dtype=torch.bfloat16)
    backend = compile_and_compare(
        Projection().eval(),
        "decode",
        tmp_path / "gemm",
        (source,),
    )
    selected = backend.megabake_last_compiled.plan.selection.selections[0]
    assert selected.body.provider == "g0_range"
