from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megabake import import_fx, make_backend, prepare_fx_graph
from megabake.fx_handler.export import capture_exported_program
from megabake.graph import (
    ComputeGraph,
    Op,
    StorageRelation,
    Value,
    verify_compute_graph,
)


def _context(graph, value):
    return SimpleNamespace(
        input_order=[node.name for node in graph.graph.nodes if node.op == "placeholder"],
        input_specs=[
            SimpleNamespace(
                name="x",
                kind="runtime",
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                stride=tuple(value.stride()),
            )
        ],
    )


def test_import_fx_preserves_tuples_reuse_views_and_inventory():
    class Model(torch.nn.Module):
        def forward(self, x):
            viewed = x.view(2, 6)
            return viewed, viewed + 1, [x, viewed]

    value = torch.randn(3, 4)
    exported = torch.export.export(Model(), (value,))
    prepared = prepare_fx_graph(
        exported.graph_module,
        (value,),
        output_tree_spec=exported.call_spec.out_spec,
    )
    graph = import_fx(prepared.graph, prepared.context)

    values_by_name = {item.name: item for item in graph.values.values()}
    input_value = values_by_name["arg0_1"]
    view_value = values_by_name["view"]
    add_value = values_by_name["add"]

    assert graph.inputs == (input_value.id,)
    assert graph.outputs == (
        view_value.id,
        add_value.id,
        input_value.id,
        view_value.id,
    )
    assert graph.output_reconstruction == (
        view_value.id,
        add_value.id,
        [input_value.id, view_value.id],
    )
    assert view_value.storage_relation.storage_id == input_value.storage_relation.storage_id
    assert view_value.storage_relation.aliases == (input_value.id,)
    assert add_value.storage_relation.storage_id == add_value.id
    assert graph.values[add_value.id].users == ()
    assert graph.values[input_value.id].users == ("op:view",)
    assert graph.values[view_value.id].users == ("op:add",)

    source_operation_ids = {
        source_id
        for source_id, node in graph.source_nodes.items()
        if node.op not in {"placeholder", "output"}
    }
    imported_operation_ids = {
        source_id for op in graph.ops.values() for source_id in op.source_fx_ids
    }
    assert imported_operation_ids == source_operation_ids
    assert all(op.tensor_program is None for op in graph.ops.values())
    assert all(op.effects["unknown"] is False for op in graph.ops.values())


def test_import_fx_represents_parameters_as_live_bindings():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.linear(x)

    value = torch.randn(2, 4)
    model = Model().eval()
    exported = torch.export.export(model, (value,))
    capture = capture_exported_program(
        exported,
        args=(value,),
        model=model,
        backend=make_backend(cache=False),
    )
    graph = import_fx(capture.graph, capture.context)
    parameter_values = [
        graph.values[value_id]
        for value_id in graph.inputs
        if graph.values[value_id].kind == "parameter"
    ]
    assert [item.binding_name for item in parameter_values] == [
        capture.context.input_order[index] for index in range(len(parameter_values))
    ]
    assert all(item.storage_relation.storage_id == item.id for item in parameter_values)


def test_import_fx_retains_unknown_target_and_source_node():
    def unknown_operator(value):
        return value * 2

    class Model(torch.nn.Module):
        def forward(self, x):
            return unknown_operator(x)

    value = torch.randn(2, 3)
    graph_module = torch.fx.symbolic_trace(Model())
    unknown_node = next(node for node in graph_module.graph.nodes if node.op == "call_function")
    unknown_node.meta["stack_trace"] = "test_graph.py:unknown_operator"
    graph = import_fx(graph_module, _context(graph_module, value))
    input_value = graph.values[graph.inputs[0]]
    assert input_value.shape == (2, 3)
    assert input_value.dtype == "torch.float32"
    assert input_value.device == "cpu"
    assert input_value.strides == (3, 1)
    assert input_value.binding_name == "x"

    assert len(graph.ops) == 1
    operation = next(iter(graph.ops.values()))
    assert operation.tensor_program is None
    assert operation.numerical_rules == {
        "known": False,
        "reason": "task-03 lowering pending",
    }
    assert operation.effects == {"unknown": True, "may_mutate_inputs": True}
    assert operation.source_location == "test_graph.py:unknown_operator"
    assert graph.source_nodes[operation.source_fx_ids[0]] is unknown_node
    result = graph.values[operation.results[0]]
    assert result.shape is None
    assert result.storage_relation.unknown
    assert result.storage_relation.aliases == graph.inputs


def test_graph_verifier_rejects_missing_operand_invalid_output_and_bad_metadata():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 1

    value = torch.randn(2, 3)
    graph_module = torch.fx.symbolic_trace(Model())
    graph = import_fx(graph_module, _context(graph_module, value))
    operation = next(iter(graph.ops.values()))

    missing_operand = dict(graph.values)
    del missing_operand[operation.operands[0]]
    with pytest.raises(ValueError, match="Missing operand"):
        verify_compute_graph(replace(graph, values=missing_operand))

    missing_user = dict(graph.values)
    operand_id = operation.operands[0]
    missing_user[operand_id] = replace(graph.values[operand_id], users=())
    with pytest.raises(ValueError, match=f"Missing user {operation.id}"):
        verify_compute_graph(replace(graph, values=missing_user))

    with pytest.raises(ValueError, match="Invalid graph output"):
        verify_compute_graph(
            replace(
                graph,
                outputs=("value:missing",),
                output_reconstruction="value:missing",
            )
        )

    inconsistent = dict(graph.values)
    result_id = operation.results[0]
    inconsistent[result_id] = replace(
        graph.values[result_id],
        shape=(2,),
        dtype="torch.float32",
        device="cpu",
        strides=(1, 2),
    )
    with pytest.raises(ValueError, match="Shape and stride rank mismatch"):
        verify_compute_graph(replace(graph, values=inconsistent))


def test_graph_verifier_rejects_cycles():
    def make_value(value_id: str, producer: str, user: str) -> Value:
        return Value(
            id=value_id,
            name=value_id,
            source_fx_id=f"fx:{value_id}",
            shape=None,
            dtype=None,
            device=None,
            strides=None,
            storage_relation=StorageRelation(storage_id=value_id),
            kind="result",
            producer=producer,
            users=(user,),
        )

    def make_op(op_id: str, operand: str, result: str) -> Op:
        return Op(
            id=op_id,
            source_fx_ids=(f"fx:{op_id}",),
            operands=(operand,),
            results=(result,),
            attributes={},
            tensor_program=None,
            effects={"unknown": True},
            numerical_rules={"known": False},
            source_location=None,
            result_reconstruction=result,
        )

    graph = ComputeGraph(
        values={
            "value:a": make_value("value:a", "op:a", "op:b"),
            "value:b": make_value("value:b", "op:b", "op:a"),
        },
        ops={
            "op:a": make_op("op:a", "value:b", "value:a"),
            "op:b": make_op("op:b", "value:a", "value:b"),
        },
        regions=(),
        inputs=(),
        outputs=("value:a",),
        output_reconstruction="value:a",
        output_tree_spec=torch.utils._pytree.tree_flatten("value:a")[1],
        runtime_context=SimpleNamespace(input_order=()),
        source_nodes={
            "fx:op:a": SimpleNamespace(op="call_function"),
            "fx:op:b": SimpleNamespace(op="call_function"),
        },
    )
    with pytest.raises(ValueError, match="contains a cycle"):
        verify_compute_graph(graph)


def test_import_fx_records_view_storage_offsets():
    class Model(torch.nn.Module):
        def forward(self, x):
            doubled = x * 2
            return doubled[..., 4:] + doubled[..., :4]

    value = torch.randn(3, 8)
    exported = torch.export.export(Model(), (value,))
    prepared = prepare_fx_graph(
        exported.graph_module,
        (value,),
        output_tree_spec=exported.call_spec.out_spec,
    )
    graph = import_fx(prepared.graph, prepared.context)

    offsets = {
        operation.attributes["fx_args"][2]: graph.values[operation.results[0]].storage_offset
        for operation in graph.ops.values()
        if operation.attributes.get("target") == "aten.slice.Tensor"
    }
    assert offsets == {0: 0, 4: 4}
