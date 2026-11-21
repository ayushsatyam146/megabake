"""Reduction prologues (ops/prologue.py) compiled into a megakernel: the
elementwise work a reduction reads runs inside the reduction body."""

from __future__ import annotations

import pytest
import torch

import megabake

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _compile(model, *inputs):
    torch._dynamo.reset()
    backend = megabake.make_backend(compile_kernel=True, cache=False)
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        result = compiled(*inputs)
    regions = [selected.region_id for selected in backend.megabake_last_compiled.plan.selection.selections]
    return result, regions


class _DeltaRead(torch.nn.Module):
    """The gated delta rule's read: sum over the key axis of state * key."""

    def forward(self, state, key, decay):
        return (state * decay[..., None, None] * key[..., None]).sum(dim=-2)


class _ScaledMax(torch.nn.Module):
    def forward(self, scores):
        return (scores * 0.125 + 1.0).amax(dim=-1)


@pytest.mark.parametrize("heads,dim", [(4, 32), (16, 128)])
def test_delta_read_fuses_its_products_into_the_sum(heads, dim):
    torch.manual_seed(heads)
    state = torch.randn(1, heads, dim, dim, device="cuda")
    key = torch.randn(1, heads, dim, device="cuda")
    decay = torch.rand(1, heads, device="cuda")
    result, regions = _compile(_DeltaRead(), state, key, decay)
    torch.testing.assert_close(result, _DeltaRead()(state, key, decay), rtol=1e-4, atol=1e-4)
    assert any(region.endswith(":prologue") for region in regions), regions


def test_maximum_reduction_takes_an_affine_prologue():
    torch.manual_seed(7)
    scores = torch.randn(3, 5, 96, device="cuda")
    result, regions = _compile(_ScaledMax(), scores)
    torch.testing.assert_close(result, _ScaledMax()(scores))
    assert any(region.endswith(":prologue") for region in regions), regions


def test_prologue_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("MEGABAKE_REDUCTION_PROLOGUE", "0")
    scores = torch.randn(3, 5, 96, device="cuda")
    result, regions = _compile(_ScaledMax(), scores)
    torch.testing.assert_close(result, _ScaledMax()(scores))
    assert not any(region.endswith(":prologue") for region in regions)
