from __future__ import annotations

import pytest
import torch

from megabake import make_backend

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="requires an SM90 CUDA device",
)


class PointwiseAndViews(torch.nn.Module):
    """Every pointwise operator and view added for Qwen3.5 and Gemma, in one graph."""

    def forward(self, source, flags, other):
        positive = source.abs() + 0.5
        transcendental = (
            torch.log(positive)
            + torch.log1p(positive)
            + torch.log2(positive)
            + torch.exp2(source)
            + torch.expm1(source)
            + torch.erf(source)
            + torch.sqrt(positive)
            + torch.reciprocal(positive)
        )
        rounded = torch.floor(source) + torch.ceil(source) + torch.trunc(source)
        clamped = (
            source.clamp(-0.5, 0.5) + torch.maximum(source, other) + torch.minimum(source, other)
        )
        logical = torch.logical_and(flags, source > 0) | torch.logical_xor(flags, other > 0)
        mixed = torch.where(logical, transcendental, rounded) + clamped
        half = mixed.to(torch.bfloat16)
        unary_half = torch.log1p(half.abs()) + torch.erf(half)  # opmath in float32
        row = unary_half.select(0, 1).unsqueeze(0).squeeze(0)
        return unary_half, row


def test_pointwise_operators_and_views_compile_and_match_eager(tmp_path):
    torch.manual_seed(5)
    model = PointwiseAndViews().eval()
    source = torch.randn(4, 64, device="cuda")
    flags = torch.rand(4, 64, device="cuda") > 0.5
    other = torch.randn(4, 64, device="cuda")
    backend = make_backend(compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase")
    with torch.no_grad():
        actual = torch.compile(model, backend=backend, fullgraph=True)(source, flags, other)
        expected = model(source, flags, other)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got.float(), want.float(), rtol=1e-2, atol=1e-2)
    assert backend.megabake_contexts[-1].requirements["execution"] == "megakernel"


class ScatterPadCopy(torch.nn.Module):
    def forward(self, base, row, block):
        updated = base.clone()
        updated[:, 2:5] = block
        updated[1] = row
        updated[:, 1:8:3] = block  # stepped slice_scatter, as in Qwen3.5's mrope
        padded = torch.nn.functional.pad(updated, (1, 2, 0, 1))
        filled = torch.nn.functional.pad(updated, (2, 0), value=1.5)
        return padded, filled


def test_scatters_and_padding_compile_and_match_eager(tmp_path):
    torch.manual_seed(6)
    model = ScatterPadCopy().eval()
    inputs = (
        torch.randn(3, 8, device="cuda"),
        torch.randn(8, device="cuda"),
        torch.randn(3, 3, device="cuda"),
    )
    backend = make_backend(compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase")
    with torch.no_grad():
        actual = torch.compile(model, backend=backend, fullgraph=True)(*inputs)
        expected = model(*inputs)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want)
    assert backend.megabake_contexts[-1].requirements["execution"] == "megakernel"


class CausalConv1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 1, 4, device="cuda"))

    def forward(self, source):
        length = source.shape[-1]
        mixed = torch.nn.functional.conv1d(source, self.weight, padding=3, groups=8)
        return torch.nn.functional.silu(mixed[..., :length])


def test_depthwise_conv1d_compiles_and_matches_eager(tmp_path):
    torch.manual_seed(7)
    cudnn_was_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False  # an exact float32 reference
    try:
        model = CausalConv1d().eval()
        source = torch.randn(2, 8, 6, device="cuda")
        backend = make_backend(
            compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
        )
        with torch.no_grad():
            actual = torch.compile(model, backend=backend, fullgraph=True)(source)
            expected = model(source)
    finally:
        torch.backends.cudnn.enabled = cudnn_was_enabled
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


class RecurrentStep(torch.nn.Module):
    def forward(self, source, state):
        decayed = state * 0.5
        updated = decayed + source * state.sum(dim=-1, keepdim=True)  # reads state twice
        state.copy_(updated)
        return updated * 2


def test_in_place_state_writes_persist_across_calls(tmp_path):
    torch.manual_seed(8)
    model = RecurrentStep().eval()
    source = torch.randn(4, 64, device="cuda")
    state = torch.randn(4, 64, device="cuda")
    expected_state = state.clone()
    backend = make_backend(
        mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True)
    with torch.no_grad():
        for _ in range(3):
            actual = optimized(source, state)
            expected = model(source, expected_state)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(state, expected_state, rtol=1e-5, atol=1e-5)
    assert backend.megabake_contexts[-1].requirements["execution"] == "megakernel"
