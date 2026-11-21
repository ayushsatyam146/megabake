from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch.autograd import DeviceType

from megabake import make_backend
from megabake.bodies.cuda.sm90.attention import attention_scratch_rows

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="task 26 requires an SM90 CUDA device",
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


def rms_norm(source, weight, residual):
    widened = source.float()
    variance = torch.pow(widened, 2).mean(dim=-1, keepdim=True)
    normalized = torch.rsqrt(variance + 1e-5) * widened
    return (normalized * weight).to(torch.bfloat16) + residual


class AttentionNorm(torch.nn.Module):
    def forward(self, source, key, value, mask, weight, residual):
        return rms_norm(grouped_attention(source, key, value, mask), weight, residual)


class Attention(torch.nn.Module):
    def forward(self, source, key, value, mask):
        return grouped_attention(source, key, value, mask)


class ProjectedAttention(torch.nn.Module):
    def __init__(self, input_width, heads, head_dim):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.weight = torch.nn.Parameter(
            torch.randn(
                heads * head_dim,
                input_width,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / input_width**0.5
        )

    def forward(self, source, key, value, mask):
        projected = torch.nn.functional.linear(source, self.weight)
        projected = projected.view(source.shape[0], source.shape[1], self.heads, self.head_dim)
        return grouped_attention(projected, key, value, mask)


class ProjectedAttentionNorm(ProjectedAttention):
    def forward(self, source, key, value, mask, weight, residual):
        attention = super().forward(source, key, value, mask)
        return rms_norm(attention, weight, residual)


def compile_and_compare(model, tmp_path, inputs):
    backend = make_backend(
        compile_kernel=True,
        cache=False,
        dump_dir=tmp_path,
        scheduler="phase",
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    backend.megabake_optimized = optimized
    with torch.no_grad():
        actual = optimized(*inputs)
        expected = model(*inputs)
    return backend, actual, expected


def test_task26_composes_attention_and_rms_norm(tmp_path):
    torch.manual_seed(26)
    source = torch.randn(2, 5, 4, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(2, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    mask = torch.randn(2, 1, 5, 7, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(8, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(2, 4, 5, 8, device="cuda", dtype=torch.bfloat16)

    backend, actual, expected = compile_and_compare(
        AttentionNorm().eval(),
        tmp_path,
        (source, key, value, mask, weight, residual),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    providers = [phase.body.provider for phase in backend.megabake_last_compiled.plan.phases]
    assert "attention" in providers
    assert "generic" in providers


def test_task26_compiles_attention_with_partial_rows(tmp_path):
    torch.manual_seed(26)
    source = torch.randn(2, 5, 4, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(2, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    mask = torch.randn(2, 1, 5, 7, device="cuda", dtype=torch.bfloat16)

    backend, actual, expected = compile_and_compare(
        Attention().eval(), tmp_path, (source, key, value, mask)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [  # noqa: RUF015  # Keep sequence indexing and its existing empty-result exception.
        (selection.program.source_target, selection.body.name)
        for selection in backend.megabake_last_compiled.plan.selection.selections
    ][0] in {
        ("megabake.attention", "attention_prefill_rows"),
        ("megabake.attention", "attention_decode_rows"),
    }
    assert backend.megabake_contexts[-1].requirements["execution"] == "megakernel"
    selection = backend.megabake_last_compiled.plan.selection.selections[0]
    assert selection.body.pipeline.stage_count == 3
    assert "score-maximum" in selection.body.pipeline.drain_rule
    # One FP32 score row per warp-owned query row.
    key_length = selection.body.tiles.reduction_tile[0]
    assert selection.body.scratch.owner == "cta"
    assert selection.body.scratch.size_bytes == (
        attention_scratch_rows(key_length) * key_length * 4
    )
    resources = backend.megabake_last_compiled.resources
    assert resources.registers_per_thread > 0
    assert resources.active_blocks_per_sm >= 1
    assert resources.local_memory_bytes_per_thread == 0
    assert all(
        value > 0
        for value in backend.megabake_contexts[-1]
        .requirements["generic_coverage"]["performance"]
        .values()
    )


def test_task26_fully_masked_rows_return_zero(tmp_path):
    torch.manual_seed(26)
    source = torch.randn(2, 5, 4, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(2, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    mask = torch.randn(2, 1, 5, 7, device="cuda", dtype=torch.bfloat16)
    mask[0, 0, 0].fill_(float("-inf"))
    _backend, actual, expected = compile_and_compare(
        Attention().eval(), tmp_path, (source, key, value, mask)
    )
    masked = torch.isnan(expected)
    assert torch.equal(actual[masked], torch.zeros_like(actual[masked]))
    torch.testing.assert_close(actual[~masked], expected[~masked], rtol=0, atol=0)
    assert torch.isfinite(actual).all()


def test_task26_repeated_mixed_body_resident_bindings(tmp_path):
    torch.manual_seed(26)
    source = torch.randn(2, 5, 4, 8, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(2, 7, 2, 8, device="cuda", dtype=torch.bfloat16)
    mask = torch.randn(2, 1, 5, 7, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(8, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(2, 4, 5, 8, device="cuda", dtype=torch.bfloat16)
    second_mask = torch.randn_like(mask)

    backend, actual, expected = compile_and_compare(
        AttentionNorm().eval(), tmp_path, (source, key, value, mask, weight, residual)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    with (
        torch.no_grad(),
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler,
    ):
        second_actual = backend.megabake_optimized(
            source, key, value, second_mask, weight, residual
        )
        torch.cuda.synchronize()
    second_expected = AttentionNorm().eval()(source, key, value, second_mask, weight, residual)
    torch.testing.assert_close(second_actual, second_expected, rtol=0, atol=0)

    assert len(backend.megabake_last_trace) == 2
    assert (
        len([event.name for event in profiler.events() if event.device_type == DeviceType.CUDA])
        == 1
    )


def test_task26_rms_norm_accepts_four_dimensional_attention_layout(tmp_path):
    torch.manual_seed(26)
    source = torch.randn(2, 4, 5, 8, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(8, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(2, 4, 5, 8, device="cuda", dtype=torch.bfloat16)

    class Model(torch.nn.Module):
        def forward(self, source, weight, residual):
            return rms_norm(source, weight, residual)

    backend, actual, expected = compile_and_compare(
        Model().eval(), tmp_path, (source, weight, residual)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [
        (selection.program.source_target, selection.body.name)
        for selection in backend.megabake_last_compiled.plan.selection.selections
    ] == [("rms_norm", "generic_reduction")]


def test_task26_composes_gemm_and_attention(tmp_path):
    torch.manual_seed(26)
    model = ProjectedAttention(576, 4, 144).eval()
    source = torch.randn(4, 16, 576, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(4, 31, 2, 144, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(4, 31, 2, 144, device="cuda", dtype=torch.bfloat16)
    mask = torch.randn(4, 1, 16, 31, device="cuda", dtype=torch.bfloat16)

    backend, actual, expected = compile_and_compare(model, tmp_path, (source, key, value, mask))
    relative_error = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative_error < 0.01
    providers = Counter(phase.body.provider for phase in backend.megabake_last_compiled.plan.phases)
    assert providers["g0_range"] >= 1
    assert providers["attention"] == 1


def test_task26_composes_gemm_attention_and_rms_norm(tmp_path):
    torch.manual_seed(26)
    model = ProjectedAttentionNorm(576, 4, 144).eval()
    source = torch.randn(4, 16, 576, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(4, 31, 2, 144, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(4, 31, 2, 144, device="cuda", dtype=torch.bfloat16)
    mask = torch.randn(4, 1, 16, 31, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(144, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(4, 4, 16, 144, device="cuda", dtype=torch.bfloat16)

    backend, actual, expected = compile_and_compare(
        model, tmp_path, (source, key, value, mask, weight, residual)
    )
    relative_error = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative_error < 0.01
    providers = Counter(phase.body.provider for phase in backend.megabake_last_compiled.plan.phases)
    assert providers["g0_range"] >= 1
    assert providers["attention"] == 1
    assert providers["generic"] >= 1
