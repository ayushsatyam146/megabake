import pytest
import torch
from torch._inductor import compile_fx as inductor_compile_fx
from torch.fx import GraphModule

from megabake import CompilerError, make_backend, prepare_fx_graph


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, value):
        return torch.nn.functional.gelu(self.linear(value))


_opaque_calls = [0]


@torch.library.custom_op("megabake_task22::opaque", mutates_args=())
def opaque(value: torch.Tensor) -> torch.Tensor:
    _opaque_calls[0] += 1
    return value * 3


@opaque.register_fake
def _(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


class OpaqueModel(torch.nn.Module):
    def forward(self, value):
        return opaque(value) + 1


def test_make_backend_rejects_invalid_mode():
    with pytest.raises(ValueError, match="Unsupported mode"):
        make_backend(mode="training")


def test_make_backend_rejects_invalid_scheduler():
    with pytest.raises(ValueError, match="scheduler must be"):
        make_backend(scheduler="tile-events")


def test_make_backend_rejects_invalid_fallback():
    with pytest.raises(ValueError, match="Unsupported fallback"):
        make_backend(fallback="eager")


def test_make_backend_defaults_to_measured_scheduler_selection():
    backend = make_backend()
    assert backend.megabake_options["scheduler"] == "auto"


def test_make_backend_rejects_training_inputs():
    model = Model().eval()
    value = torch.randn(2, 4, requires_grad=True)
    exported = torch.export.export(model, (value,))
    backend = make_backend()
    with pytest.raises(RuntimeError, match="inference only"):
        backend(exported.graph_module, (value,))


def test_prepare_fx_graph_remains_fx_and_skips_inductor_compilation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Inductor compilation must not be invoked")

    monkeypatch.setattr(inductor_compile_fx, "compile_fx_forward", forbidden)

    class Model(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.gelu(value) * 2

    model = Model()

    value = torch.randn(3, 4)
    exported = torch.export.export(model, (value,))
    prepared = prepare_fx_graph(
        exported.graph_module,
        (value,),
        output_tree_spec=exported.call_spec.out_spec,
    )
    assert isinstance(prepared.graph, GraphModule)
    assert prepared.context.phase == "post_grad"
    with torch.no_grad():
        torch.testing.assert_close(prepared.forward(value), model(value))


def test_prepare_fx_graph_dumps_every_stage(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.gelu(value) * 2

    value = torch.randn(3, 4)
    exported = torch.export.export(Model(), (value,))
    prepared = prepare_fx_graph(
        exported.graph_module,
        (value,),
        dump_dir=tmp_path,
        model_name="test-model",
        output_tree_spec=exported.call_spec.out_spec,
    )
    expected = {
        "phase_0_input_fx.txt",
        "phase_1_normalized.txt",
        "phase_2_pre_grad.txt",
        "phase_3_aot_inference.txt",
        "phase_4_prepared.txt",
        "phase_5_cache_check.txt",
        "phase_6_post_grad.txt",
    }
    dump_directory = tmp_path / "test-model"
    assert {path.name for path in dump_directory.iterdir()} == expected
    assert prepared.context.phase == "post_grad"


def test_torch_compile_returns_fx_reference_and_follows_parameters():
    torch.manual_seed(7)
    model = Model().eval()
    backend = make_backend()
    optimized = torch.compile(model, backend=backend, fullgraph=True)
    first = torch.randn(5, 4)
    second = torch.randn(2, 4)
    with torch.no_grad():
        assert torch.allclose(optimized(first), model(first))
        assert torch.allclose(optimized(second), model(second))
        model.linear.weight.add_(1)
        assert torch.allclose(optimized(second), model(second))
    assert backend.megabake_contexts[-1].phase == "post_grad"


def test_task22_strict_rejection_happens_before_execution():
    value = torch.randn(3, 4)
    exported = torch.export.export(OpaqueModel(), (value,))
    backend = make_backend(fallback="error", compile_kernel=True, cache=False)
    calls_before = _opaque_calls[0]
    with pytest.raises(CompilerError, match="missing lowering rule") as rejection:
        backend(exported.graph_module, (value,))
    assert "File " in str(rejection.value)
    assert "source operations: op:opaque" in str(rejection.value)
    assert _opaque_calls[0] == calls_before


def test_task22_inductor_fallback_handles_changed_inputs():
    value = torch.randn(3, 4)
    backend = make_backend(fallback="inductor", compile_kernel=True, cache=False)
    forward = torch.compile(OpaqueModel(), backend=backend, fullgraph=True)
    calls_before = _opaque_calls[0]
    with torch.no_grad():
        first = forward(value)
        second = forward(value + 1)
    fallback = backend.megabake_last_fallback
    assert torch.allclose(first, value * 3 + 1)
    assert torch.allclose(second, (value + 1) * 3 + 1)
    assert _opaque_calls[0] == calls_before + 2
    assert fallback.megabake_execution == "inductor_fallback"
    assert "missing lowering rule" in fallback.megabake_fallback_reason
    assert fallback.megabake_launch_count == 2
    context = backend.megabake_contexts[-1]
    assert context.requirements["execution"] == "inductor_fallback"
    assert context.requirements["fallback_launch_count"] == 2


def test_splits_become_slices_before_import():
    from megabake.frontend import split_to_slices

    class Model(torch.nn.Module):
        def forward(self, value):
            first, second = torch.split(value, [3, 5], dim=-1)
            halves = torch.split(value, 3, dim=-1)
            return first * 2, second + 1, halves[2]

    value = torch.randn(2, 8)
    exported = torch.export.export(Model(), (value,))
    prepared = prepare_fx_graph(
        exported.graph_module, (value,), output_tree_spec=exported.call_spec.out_spec
    )
    targets = {node.target for node in prepared.graph.graph.nodes}
    assert torch.ops.aten.split.Tensor not in targets
    assert torch.ops.aten.split_with_sizes.default not in targets
    with torch.no_grad():
        for actual, expected in zip(prepared.forward(value), Model()(value), strict=True):
            torch.testing.assert_close(actual, expected)
    split_to_slices(prepared.graph)  # idempotent


def test_eliminate_common_subexpressions_merges_pure_duplicates_only():
    from torch.fx.experimental.proxy_tensor import make_fx

    from megabake.frontend import eliminate_common_subexpressions

    def per_layer(positions, state):
        # Two layers each compute the same cosines; the state write must keep
        # its own copy of the value it mutates.
        first = torch.cos(positions * 2.0)
        second = torch.cos(positions * 2.0)
        scratch = positions * 2.0
        scratch.add_(1.0)
        state.copy_(scratch)
        return first + second, state

    graph = make_fx(per_layer)(torch.randn(4), torch.zeros(4))
    before = [node.target for node in graph.graph.nodes if node.op == "call_function"]
    eliminate_common_subexpressions(graph)
    graph.recompile()
    after = [node.target for node in graph.graph.nodes if node.op == "call_function"]

    assert before.count(torch.ops.aten.cos.default) == 2
    assert after.count(torch.ops.aten.cos.default) == 1
    # One shared positions * 2 for the cosines, one owned by the in-place add.
    assert after.count(torch.ops.aten.mul.Tensor) == 2
    positions = torch.randn(4)
    expected = per_layer(positions, torch.zeros(4))
    actual = graph(positions, torch.zeros(4))
    torch.testing.assert_close(actual, expected)
