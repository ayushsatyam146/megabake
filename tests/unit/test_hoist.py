"""Parameter-only work hoisted out of the captured graph (src/megabake/hoist.py). CPU only."""

from __future__ import annotations

import torch

import megabake


class _Norm(torch.nn.Module):
    """Gemma-style: the weight enters as 1 + w, in float32."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8))
        self.linear = torch.nn.Linear(8, 8, bias=False)

    def forward(self, value):
        scaled = value * (1.0 + self.weight.float())
        return self.linear(scaled) - self.weight.exp().sum()


def _inputs(backend) -> int:
    # AOT preparation renames placeholders (arg0_1, ...), so count them.
    return sum(node.op == "placeholder" for node in backend.megabake_last_prepared.graph.graph.nodes)


def _compile(model, monkeypatch, hoist: str):
    monkeypatch.setenv("MEGABAKE_HOIST_PARAMS", hoist)
    torch._dynamo.reset()
    backend = megabake.make_backend(compile_kernel=False, cache=False)
    return backend, torch.compile(model, backend=backend, fullgraph=True)


def test_hoisted_graph_matches_eager_and_moves_parameter_work(monkeypatch):
    torch.manual_seed(3)
    model = _Norm().eval()
    value = torch.randn(2, 8)
    backend, compiled = _compile(model, monkeypatch, "1")
    with torch.no_grad():
        expected = model(value)
        result = compiled(value)
    torch.testing.assert_close(result, expected)
    # Two parameters and the input, plus 1 + w.float() and w.exp().sum().
    assert _inputs(backend) == 5


def test_hoisted_values_follow_in_place_parameter_updates(monkeypatch):
    torch.manual_seed(4)
    model = _Norm().eval()
    value = torch.randn(2, 8)
    _, compiled = _compile(model, monkeypatch, "1")
    with torch.no_grad():
        compiled(value)
        model.weight.add_(0.5)
        torch.testing.assert_close(compiled(value), model(value))


def test_hoisting_can_be_turned_off(monkeypatch):
    torch.manual_seed(5)
    model = _Norm().eval()
    backend, compiled = _compile(model, monkeypatch, "0")
    with torch.no_grad():
        compiled(torch.randn(2, 8))
    assert _inputs(backend) == 3


def test_views_of_parameters_are_not_hoisted(monkeypatch):
    class Viewed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8))

        def forward(self, value):
            return value @ self.weight.t()

    backend, compiled = _compile(Viewed().eval(), monkeypatch, "1")
    with torch.no_grad():
        compiled(torch.randn(2, 8))
    assert _inputs(backend) == 2
