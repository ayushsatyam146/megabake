from __future__ import annotations

import itertools
import operator
from types import SimpleNamespace

import pytest
import torch
from torch.utils._pytree import tree_unflatten

from megabake import prepare_fx_graph
from megabake.bodies import SelectionPolicy, TargetFacts
from megabake.graph import import_fx
from megabake.ir import (
    Accumulator,
    AxisAffine,
    AxisIndex,
    Binary,
    CanonicalProgram,
    Cast,
    Constant,
    DataIndex,
    Load,
    ScalarExpression,
    TensorProgram,
    Unary,
    Where,
)
from megabake.lowering import (
    axis_coordinate,
    canonicalize_semantics,
    dump_lowering_graph,
    lower_semantics,
    semantic_coverage,
)
from megabake.plan import SearchBudget, search_plans
from megabake.regions import form_regions


def _context(graph: torch.fx.GraphModule, *values: torch.Tensor) -> SimpleNamespace:
    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    return SimpleNamespace(
        input_order=[node.name for node in placeholders],
        input_specs=[
            SimpleNamespace(
                name=node.name,
                kind="runtime",
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                stride=tuple(value.stride()),
            )
            for node, value in zip(placeholders, values)
        ],
    )


def _import(model: torch.nn.Module, *values: torch.Tensor):
    exported = torch.export.export(model, values)
    prepared = prepare_fx_graph(
        exported.graph_module,
        values,
        output_tree_spec=exported.call_spec.out_spec,
    )
    return import_fx(prepared.graph, prepared.context)


def _dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name.removeprefix("torch."))
    assert isinstance(dtype, torch.dtype)
    return dtype


def _read_load(program: TensorProgram, load_id: str, coordinates: dict[str, int], values: dict):
    load = next(item for item in program.loads if item.id == load_id)
    offset = 0
    coordinate_values = []
    for axis, stride in zip(load.axes, load.strides):
        if isinstance(axis, DataIndex):
            coordinate = int(_read_load(program, axis.load_id, coordinates, values))
        elif isinstance(axis, AxisAffine):
            coordinate = axis_coordinate(axis, coordinates)
        else:
            coordinate = coordinates[axis] if isinstance(axis, str) else axis
        coordinate_values.append(coordinate)
        offset += coordinate * stride
    if load.source_shape is not None and any(
        not 0 <= coordinate < size for coordinate, size in zip(coordinate_values, load.source_shape)
    ):
        return torch.zeros(1, dtype=_dtype(load.dtype))[0]
    source = values[load.value_id]
    return source.as_strided((1,), (1,), source.storage_offset() + offset)[0]


def _loads(
    program: TensorProgram, coordinates: dict[str, int], values: dict
) -> dict[str, torch.Tensor]:
    return {
        load.id: _read_load(program, load.id, coordinates, values)
        for load in program.loads
        if all(not isinstance(axis, str) or axis in coordinates for axis in load.axes)
    }


def _evaluate_expression(
    expression: ScalarExpression,
    loads: dict[str, torch.Tensor],
    accumulator: torch.Tensor | None,
    coordinates: dict[str, int] | None = None,
):
    if isinstance(expression, Constant):
        return expression.value
    if isinstance(expression, Load):
        return loads[expression.load_id]
    if isinstance(expression, Accumulator):
        assert accumulator is not None
        return accumulator
    if isinstance(expression, Cast):
        return torch.as_tensor(
            _evaluate_expression(expression.operand, loads, accumulator, coordinates)
        ).to(_dtype(expression.dtype))
    if isinstance(expression, Unary):
        value = _evaluate_expression(expression.operand, loads, accumulator, coordinates)
        functions = {
            "relu": torch.relu,
            "neg": torch.neg,
            "abs": torch.abs,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "rsqrt": torch.rsqrt,
            "exp": torch.exp,
            "cos": torch.cos,
            "sin": torch.sin,
            "logical_not": torch.logical_not,
        }
        return functions[expression.operator](value)
    if isinstance(expression, AxisIndex):
        assert coordinates is not None and expression.axis in coordinates
        value = expression.start + coordinates[expression.axis] * expression.step
        return torch.tensor(value, dtype=_dtype(expression.dtype))
    if isinstance(expression, Binary):
        lhs = _evaluate_expression(expression.lhs, loads, accumulator, coordinates)
        rhs = _evaluate_expression(expression.rhs, loads, accumulator, coordinates)
        functions = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": torch.div,
            "pow": torch.pow,
            "maximum": torch.maximum,
            "bitwise_and": torch.bitwise_and,
            "eq": operator.eq,
            "ne": operator.ne,
            "lt": operator.lt,
            "le": operator.le,
            "gt": operator.gt,
            "ge": operator.ge,
        }
        return functions[expression.operator](lhs, rhs)
    if isinstance(expression, Where):
        return torch.where(
            _evaluate_expression(expression.condition, loads, accumulator, coordinates),
            _evaluate_expression(expression.true_value, loads, accumulator, coordinates),
            _evaluate_expression(expression.false_value, loads, accumulator, coordinates),
        )
    raise AssertionError(f"unsupported test expression {expression!r}")


def _evaluate_program(program: TensorProgram, values: dict) -> torch.Tensor:
    reduction_axes = set(program.reduction_axes)
    output_domains = [
        domain for domain in program.iteration_domains if domain.axis not in reduction_axes
    ]
    reduction_domains = [
        domain for domain in program.iteration_domains if domain.axis in reduction_axes
    ]
    maximum_store_offset = 0
    for size, stride in zip(program.store.shape, program.store.strides):
        maximum_store_offset += max(size - 1, 0) * abs(stride)
    storage = torch.empty(maximum_store_offset + 1, dtype=_dtype(program.store.dtype))
    result = storage.as_strided(program.store.shape, program.store.strides, storage_offset=0)
    for output_coordinates in itertools.product(*(range(domain.size) for domain in output_domains)):
        output_map = {
            domain.axis: coordinate
            for domain, coordinate in zip(output_domains, output_coordinates)
        }
        if reduction_domains:
            assert program.init is not None and program.accumulator_dtype is not None
            accumulator = torch.as_tensor(
                program.init.value, dtype=_dtype(program.accumulator_dtype)
            )
            for reduction_coordinates in itertools.product(
                *(range(domain.size) for domain in reduction_domains)
            ):
                coordinates = output_map | {
                    domain.axis: coordinate
                    for domain, coordinate in zip(reduction_domains, reduction_coordinates)
                }
                accumulator = _evaluate_expression(
                    program.combine,
                    _loads(program, coordinates, values),
                    accumulator,
                    coordinates,
                )
            assert program.post is not None
            value = _evaluate_expression(
                program.post,
                _loads(program, output_map, values),
                accumulator,
                output_map,
            )
        else:
            value = _evaluate_expression(
                program.combine,
                _loads(program, output_map, values),
                None,
                output_map,
            )
        result[output_coordinates] = value
    return result


def _evaluate_canonical_program(
    program: CanonicalProgram, values: dict
) -> tuple[torch.Tensor, ...]:
    if program.family == "concat":
        operands = [values[value_id] for value_id in program.semantics["operands"]]
        return (torch.cat(operands, dim=program.semantics["axis"]),)
    if program.family == "gather":
        if "data" in program.semantics:
            data = values[program.semantics["data"]]
            indices = values[program.semantics["indices"]]
            return (data[indices],)
        source = values[program.semantics["source"]]
        indices = tuple(values[value_id] for value_id in program.semantics["indices"])
        return (source[indices],)
    if program.family == "softmax_statistics":
        source = values[program.semantics["input"]]
        axis = program.semantics["axis"]
        maximum = source.amax(dim=axis, keepdim=True)
        sum_exp = (source - maximum).exp().sum(dim=axis, keepdim=True)
        return maximum, sum_exp
    raise AssertionError(f"unsupported canonical program {program.family}")


def _evaluate_graph(graph, values: dict):
    environment = {
        value_id: values[graph.values[value_id].binding_name or graph.values[value_id].name]
        for value_id in graph.inputs
    }
    for operation in graph.ops.values():
        program = operation.tensor_program
        assert program is not None, operation.numerical_rules
        if isinstance(program, TensorProgram):
            assert len(operation.results) == 1
            environment[operation.results[0]] = _evaluate_program(program, environment)
        else:
            results = _evaluate_canonical_program(program, environment)
            assert len(results) == len(operation.results)
            for value_id, value in zip(operation.results, results):
                environment[value_id] = value
    outputs = [environment[value_id] for value_id in graph.outputs]
    return tree_unflatten(outputs, graph.output_tree_spec)


def test_lower_semantics_gemm_elementwise_reduction_fixture():
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.relu(torch.nn.functional.linear(x, weight, bias)).sum(dim=-1)

    torch.manual_seed(7)
    x = torch.randn(2, 3)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)
    model = Model().eval()
    graph = lower_semantics(_import(model, x, weight, bias))

    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    assert [operation.tensor_program.family for operation in graph.ops.values()] == [
        "view",
        "contraction",
        "elementwise",
        "reduction",
    ]
    contraction = graph.ops["op:addmm"].tensor_program
    assert contraction is not None
    assert contraction.reduction_axes == ("r0",)
    assert contraction.accumulator_dtype == "torch.float32"
    assert contraction.post is not None
    assert graph.ops["op:permute"].tensor_program.store.owns_storage is False

    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": x, "weight": weight, "bias": bias}),
        model(x, weight, bias),
    )


def test_dump_lowering_graph_reports_programs_and_unknown_reasons(tmp_path):
    class Known(torch.nn.Module):
        def forward(self, x):
            return x + 1

    x = torch.randn(2, 3)
    graph = lower_semantics(_import(Known(), x))
    path = dump_lowering_graph(graph, tmp_path / "nested" / "lowering_graph.txt")

    text = path.read_text(encoding="utf-8")
    assert path.exists()
    assert "# LoweringGraph" in text
    assert (
        "values=2 operations=1 known=1 unknown=0 canonical=0 generic=1 "
        "unresolved=0 inputs=1 outputs=1" in text
    )
    assert "## Operations" in text
    assert "## Tensor Programs" in text
    assert "### op:add" in text
    assert "source_target=aten.add.Tensor" in text
    assert "## Output Reconstruction" in text


def test_lower_semantics_supports_scalar_attribute_and_lift_fresh_copy():
    mask = torch.tensor([False, True])
    zero = torch.tensor(0.0, dtype=torch.bfloat16)
    fx_graph = torch.fx.Graph()
    with fx_graph.inserting_after():
        mask_node = fx_graph.placeholder("mask")
        mask_node.meta["val"] = mask
        constant_node = fx_graph.get_attr("zero")
        constant_node.meta["val"] = zero
        lifted = fx_graph.call_function(torch.ops.aten.lift_fresh_copy.default, (constant_node,))
        lifted.meta["val"] = zero
        result = fx_graph.call_function(
            torch.ops.aten.where.self, (mask_node, lifted, constant_node)
        )
        result.meta["val"] = torch.where(mask, zero, zero)
        fx_graph.output(result)
    module = torch.fx.GraphModule({"zero": zero}, fx_graph)
    graph = canonicalize_semantics(import_fx(module, _context(module, mask)))

    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    assert graph.ops["op:zero"].tensor_program.combine == Constant(0.0, "torch.bfloat16")
    assert graph.ops["op:lift_fresh_copy_default"].tensor_program.family == "copy"

    torch.testing.assert_close(
        _evaluate_graph(graph, {"mask": mask}), torch.where(mask, zero, zero)
    )


def test_canonicalize_semantics_supports_shared_targets_and_preserves_programs():
    class Model(torch.nn.Module):
        def forward(self, x, y, ids, weights, mask):
            return (
                x.pow(2),
                torch.rsqrt(x),
                torch.exp(x),
                torch.cos(x),
                torch.sin(x),
                x.clone(),
                x.unsqueeze(1),
                x[:1],
                x.expand((2, 2, 3)),
                torch.bmm(x.unsqueeze(0), y.unsqueeze(0)),
                torch.nn.functional.embedding(ids, weights),
                x[ids],
                mask.bitwise_and(mask),
                torch.full((2, 3), 3.0),
                torch.scalar_tensor(4.0),
                torch.arange(4),
            )

    torch.manual_seed(31)
    x = torch.rand(2, 3)
    y = torch.randn(3, 2)
    ids = torch.tensor([[0, 1], [1, 0]])
    weights = torch.randn(2, 4)
    mask = torch.tensor([[True, False], [False, True]])
    model = Model().eval()
    graph = _import(model, x, y, ids, weights, mask)
    lowered = lower_semantics(graph)
    canonical = canonicalize_semantics(graph)

    assert semantic_coverage(canonical) == {
        "canonical": len(canonical.ops) - 2,
        "generic": 2,
        "unresolved": 0,
    }
    assert {
        "aten.pow.Tensor_Scalar",
        "aten.rsqrt.default",
        "aten.exp.default",
        "aten.cos.default",
        "aten.sin.default",
        "aten.clone.default",
        "aten.unsqueeze.default",
        "aten.slice.Tensor",
        "aten.expand.default",
        "aten.bmm.default",
        "aten.embedding.default",
        "aten.index.Tensor",
        "aten.bitwise_and.Tensor",
        "aten.full.default",
        "aten.scalar_tensor.default",
        "prims.iota.default",
    } <= {operation.attributes["target"] for operation in canonical.ops.values()}
    for op_id, operation in lowered.ops.items():
        canonical_operation = canonical.ops[op_id]
        assert canonical_operation.source_fx_ids == operation.source_fx_ids
        assert canonical_operation.tensor_program == operation.tensor_program

    torch.testing.assert_close(
        _evaluate_graph(
            canonical,
            {
                "x": x,
                "y": y,
                "ids": ids,
                "weights": weights,
                "mask": mask,
            },
        ),
        model(x, y, ids, weights, mask),
    )


def test_canonicalize_semantics_supports_multi_result_online_softmax():
    class Model(torch.nn.Module):
        def forward(self, x):
            statistics = torch.ops.prims.prepare_softmax_online(x, -1)
            return statistics[0], statistics[1]

    torch.manual_seed(37)
    x = torch.randn(2, 3, 4)
    model = Model().eval()
    canonical = canonicalize_semantics(_import(model, x))

    statistics_operation = canonical.ops["op:prepare_softmax_online"]
    statistics_program = statistics_operation.tensor_program
    assert isinstance(statistics_program, CanonicalProgram)
    assert statistics_program.family == "softmax_statistics"
    assert statistics_program.result_roles == ("max", "sum_exp")
    assert len(statistics_operation.results) == 2
    for operation in canonical.ops.values():
        if operation.attributes["target"] == str(operator.getitem):
            assert operation.tensor_program is not None
            assert operation.tensor_program.store.owns_storage is False
            assert operation.effects == {
                "unknown": False,
                "mutates_inputs": False,
            }
            result_index = operation.attributes["fx_args"][1]
            source_id = operation.operands[result_index]
            projection = canonical.values[operation.results[0]]
            assert projection.storage_relation.aliases == (source_id,)
            assert projection.storage_relation.unknown is False

    torch.testing.assert_close(_evaluate_graph(canonical, {"x": x}), model(x))


def test_canonicalize_semantics_supports_concat_semantics():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.cat.default((x, x), 0)

    torch.manual_seed(41)
    x = torch.randn(2, 3)
    model = Model().eval()
    graph_module = torch.fx.symbolic_trace(model)
    cat_node = next(node for node in graph_module.graph.nodes if node.op == "call_function")
    cat_node.meta["val"] = model(x)
    canonical = canonicalize_semantics(import_fx(graph_module, _context(graph_module, x)))
    concat_operation = canonical.ops["op:cat_default"]
    program = concat_operation.tensor_program
    assert isinstance(program, TensorProgram)
    assert program.family == "concat"
    assert tuple(load.value_id for load in program.loads) == (
        canonical.inputs[0],
        canonical.inputs[0],
    )
    torch.testing.assert_close(_evaluate_graph(canonical, {"x": x}), model(x))


def test_lower_semantics_supports_broadcast_advanced_indexing():
    class Model(torch.nn.Module):
        def forward(self, source, first, second):
            return source[first, second]

    torch.manual_seed(43)
    source = torch.randn(2, 5)
    first = torch.zeros(1, 1, 1, 1, dtype=torch.int64)
    second = torch.tensor([[[[0, 1, 2, 3]]]])
    model = Model().eval()
    graph = lower_semantics(_import(model, source, first, second))

    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"source": source, "first": first, "second": second}),
        model(source, first, second),
    )


def test_lower_semantics_supports_smollm_mask_and_scalar_operations():
    class Model(torch.nn.Module):
        def forward(self, mask, values):
            negated = torch.logical_not(mask)
            any_masked = negated.any(dim=-1, keepdim=True)
            restored = torch.logical_not(any_masked)
            return restored, values * 0.25

    torch.manual_seed(44)
    mask = torch.tensor([[True, False, False, False], [False, False, False, False]])
    values = torch.randn(2, 4)
    model = Model().eval()
    graph = lower_semantics(_import(model, mask, values))
    targets = {operation.attributes["target"] for operation in graph.ops.values()}

    assert {"aten.logical_not.default", "aten.any.dim"} <= targets
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"mask": mask, "values": values}),
        model(mask, values),
    )


def test_lower_semantics_supports_scalar_multiply():
    class Model(torch.nn.Module):
        def forward(self, values):
            return torch.ops.aten.mul.Scalar(values, 0.25)

    torch.manual_seed(45)
    values = torch.randn(2, 4)
    model = Model().eval()
    graph_module = torch.fx.symbolic_trace(model)
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            node.meta["val"] = model(values)
    graph = lower_semantics(import_fx(graph_module, _context(graph_module, values)))

    assert graph.ops["op:mul_scalar"].attributes["target"] == "aten.mul.Scalar"
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(_evaluate_graph(graph, {"values": values}), model(values))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lower_semantics_preserves_low_precision_gemm_bias_rounding(dtype):
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.nn.functional.linear(x, weight, bias)

    torch.manual_seed(17)
    x = (torch.randn(32, 32) * 8).to(dtype)
    weight = (torch.randn(32, 32) * 8).to(dtype)
    bias = (torch.randn(32) * 8).to(dtype)
    model = Model().eval()
    graph = lower_semantics(_import(model, x, weight, bias))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": x, "weight": weight, "bias": bias}),
        model(x, weight, bias),
        rtol=0,
        atol=0,
    )


def test_lower_semantics_supports_batched_linear_rank_changing_reshapes():
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.nn.functional.linear(x, weight, bias)

    torch.manual_seed(19)
    x = torch.randn(2, 3, 4)
    weight = torch.randn(5, 4)
    bias = torch.randn(5)
    model = Model().eval()
    graph = lower_semantics(_import(model, x, weight, bias))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": x, "weight": weight, "bias": bias}),
        model(x, weight, bias),
    )


def test_lower_semantics_supports_zero_size_and_strided_storage_holes():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    torch.manual_seed(23)
    empty_x = torch.empty(0, 3)
    empty_y = torch.empty(0, 3)
    graph = lower_semantics(_import(Model(), empty_x, empty_y))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": empty_x, "y": empty_y}),
        Model()(empty_x, empty_y),
    )

    strided_x = torch.empty(6).as_strided((2,), (5,))
    strided_y = torch.randn(2)
    graph = lower_semantics(_import(Model(), strided_x, strided_y))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": strided_x, "y": strided_y}),
        Model()(strided_x, strided_y),
    )


def test_lower_semantics_cast_broadcast_transpose_and_reductions():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            casted = (x.view(2, 6).t() + y).to(torch.float16)
            return casted.mean(dim=0), casted.sum(dim=1)

    torch.manual_seed(11)
    x = torch.randn(3, 4)
    y = torch.randn(6, 1)
    model = Model().eval()
    graph = lower_semantics(_import(model, x, y))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    assert {operation.tensor_program.family for operation in graph.ops.values()} == {
        "view",
        "elementwise",
        "reduction",
    }

    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": x, "y": y}),
        model(x, y),
    )


def test_lower_semantics_constants_comparisons_and_selection():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch.where(x > 0.25, x + y, x * -3.0).to(torch.float16)

    torch.manual_seed(13)
    x = torch.randn(2, 3)
    y = torch.randn(1, 3)
    model = Model().eval()
    graph = lower_semantics(_import(model, x, y))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": x, "y": y}),
        model(x, y),
    )


def test_lower_semantics_supports_all_tensor_sum_and_mean():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.sum(), x.mean()

    torch.manual_seed(29)
    x = torch.randn(2, 3)
    model = Model().eval()
    graph = lower_semantics(_import(model, x))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"x": x}),
        model(x),
    )


def test_lower_semantics_leaves_unknown_operation_opaque_and_blocks_effects():
    def unknown_operator(value):
        return value * 2

    class Model(torch.nn.Module):
        def forward(self, x):
            return unknown_operator(x)

    value = torch.randn(2, 3)
    graph_module = torch.fx.symbolic_trace(Model())
    unknown_node = next(node for node in graph_module.graph.nodes if node.op == "call_function")
    unknown_node.meta["val"] = value
    graph = import_fx(graph_module, _context(graph_module, value))
    lowered = lower_semantics(graph)
    operation = next(iter(lowered.ops.values()))
    assert operation.tensor_program is None
    assert operation.numerical_rules["known"] is False
    assert "missing lowering rule" in operation.numerical_rules["reason"]
    assert "unsupported effects" in operation.numerical_rules["reason"]

    canonical = canonicalize_semantics(graph)
    canonical_operation = next(iter(canonical.ops.values()))
    assert semantic_coverage(canonical) == {
        "canonical": 0,
        "generic": 0,
        "unresolved": 1,
    }
    assert canonical_operation.source_fx_ids == operation.source_fx_ids
    assert canonical_operation.numerical_rules["canonical"] is False


def test_lower_semantics_turns_rule_failure_into_an_opaque_reason():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.float16)

    x = torch.randn(2, 3)
    graph = _import(Model(), x)
    cast_node = graph.source_nodes["fx:convert_element_type"]
    cast_node.args = ()

    lowered = lower_semantics(graph)
    operation = next(iter(lowered.ops.values()))
    assert operation.tensor_program is None
    assert operation.numerical_rules["known"] is False
    assert "lowering rule failed" in operation.numerical_rules["reason"]
    assert "IndexError" in operation.numerical_rules["reason"]


def test_lower_semantics_covers_softmax_clamp_and_exact_gelu():
    class Model(torch.nn.Module):
        def forward(self, x):
            return (
                torch.softmax(x, dim=-1),
                torch.clamp(x, -1, 1),
                torch.nn.functional.gelu(x),
            )

    x = torch.randn(2, 3)
    graph = lower_semantics(_import(Model(), x))
    unknown_targets = {
        operation.attributes["target"]
        for operation in graph.ops.values()
        if operation.tensor_program is None
    }
    amax_operation = next(
        operation
        for operation in graph.ops.values()
        if operation.attributes["target"] == "aten.amax.default"
    )
    assert amax_operation.tensor_program is not None
    assert unknown_targets == set()


def test_task25_softmax_max_primitive_has_exact_generic_expansion():
    class Model(torch.nn.Module):
        def forward(self, source):
            return torch.softmax(source, dim=-1)

    torch.manual_seed(25)
    source = torch.randn(3, 5)
    source[1, 2] = float("nan")
    graph = lower_semantics(_import(Model(), source))

    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    torch.testing.assert_close(
        _evaluate_graph(graph, {"source": source}),
        Model()(source),
        equal_nan=True,
    )
    assert any(operation.tensor_program is not None for operation in graph.ops.values())


def test_lower_semantics_reports_higher_order_operator_without_crashing():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.cond(
                x.sum() > 0,
                lambda: x + 1,
                lambda: x * 2,
                [],
            )

    x = torch.randn(2, 3)
    graph = lower_semantics(_import(Model(), x))
    cond_operation = graph.ops["op:cond"]
    assert cond_operation.tensor_program is None
    assert cond_operation.numerical_rules["known"] is False
    assert "missing lowering rule" in cond_operation.numerical_rules["reason"]
    assert "unsupported effects" in cond_operation.numerical_rules["reason"]


def test_import_fx_rejects_symbolic_dimensions_instead_of_specializing_them():
    from torch.export import Dim

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    batch = Dim("batch")
    features = Dim("features")
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    exported = torch.export.export(
        Model(),
        (x, y),
        dynamic_shapes=({0: batch, 1: features}, {0: batch, 1: features}),
    )
    with pytest.raises(ValueError, match="Symbolic tensor shape"):
        prepare_fx_graph(
            exported.graph_module,
            (x, y),
            output_tree_spec=exported.call_spec.out_spec,
        )


def test_task23_convolution_is_an_indexed_grouped_contraction():
    class Model(torch.nn.Module):
        def forward(self, source, weight, bias):
            return torch.nn.functional.conv2d(
                source,
                weight,
                bias,
                stride=(2, 1),
                padding=(1, 2),
                dilation=(2, 1),
                groups=2,
            )

    source = torch.randn(2, 4, 9, 10)
    weight = torch.randn(6, 2, 3, 3)
    bias = torch.randn(6)
    graph = lower_semantics(_import(Model(), source, weight, bias))
    operation = next(iter(graph.ops.values()))
    program = operation.tensor_program

    assert isinstance(program, TensorProgram)
    assert program.family == "contraction"
    assert "groups" in program.decomposition
    assert program.reduction_axes == ("r0",)
    input_channel = program.loads[0].axes[1]
    assert isinstance(input_channel, AxisAffine)
    assert axis_coordinate(input_channel, {"o1": 4, "r0": 9}) == 3


def test_task23_pooling_is_an_indexed_reduction_with_masked_windows():
    class Model(torch.nn.Module):
        def forward(self, source):
            return torch.nn.functional.avg_pool2d(source, (2, 2), stride=(1, 1), padding=(1, 1))

    source = torch.randn(2, 3, 5, 6)
    graph = lower_semantics(_import(Model(), source))
    operation = next(iter(graph.ops.values()))
    program = operation.tensor_program

    assert isinstance(program, TensorProgram)
    assert program.family == "reduction"
    assert program.loads[0].source_shape == (2, 3, 5, 6)
    assert axis_coordinate(program.loads[0].axes[2], {"o2": 0, "r0": 0}) == -1
    assert axis_coordinate(program.loads[0].axes[2], {"o2": 1, "r0": 2}) == 1


def test_task23_unfamiliar_norm_and_fork_join_plan_generically():
    class NormVariant(torch.nn.Module):
        def forward(self, source):
            scale = (source * source).mean(dim=-1, keepdim=True)
            return source * torch.rsqrt(scale + 1e-6)

    class ForkJoin(torch.nn.Module):
        def forward(self, source):
            value = source + 1
            return value * 2, value + 3

    target = TargetFacts(
        architecture="sm90a",
        threads_per_warp=32,
        max_threads_per_cta=1024,
        max_shared_memory_per_cta=232_448,
        max_registers_per_thread=255,
        multiprocessor_count=60,
        supports_tma=True,
        supports_wgmma=True,
    )
    policy = SelectionPolicy(allow_tuned=False)
    for model, value in (
        (NormVariant(), torch.randn(3, 7)),
        (ForkJoin(), torch.randn(3, 7)),
    ):
        graph = form_regions(lower_semantics(_import(model, value)))
        assert semantic_coverage(graph)["unresolved"] == 0
        result = search_plans(graph, target, policy, SearchBudget(evaluator=None))
        assert result.candidates
        assert all(candidate.status == "unmeasured" for candidate in result.candidates)
        assert all("generic/" in candidate.description for candidate in result.candidates)


def test_task23_runtime_gather_uses_validated_conservative_read_domain():
    class Model(torch.nn.Module):
        def forward(self, source, index):
            return torch.gather(source, 1, index)

    source = torch.randn(2, 3, 4)
    index = torch.randint(0, 3, (2, 3, 4))
    graph = form_regions(lower_semantics(_import(Model(), source, index)))
    operation = next(iter(graph.ops.values()))
    program = operation.tensor_program

    assert isinstance(program, TensorProgram)
    assert program.family == "elementwise"
    assert isinstance(program.loads[1].axes[1], DataIndex)
    target = TargetFacts(
        architecture="sm90a",
        threads_per_warp=32,
        max_threads_per_cta=1024,
        max_shared_memory_per_cta=232_448,
        max_registers_per_thread=255,
        multiprocessor_count=60,
        supports_tma=True,
        supports_wgmma=True,
    )
    result = search_plans(
        graph, target, SelectionPolicy(allow_tuned=False), SearchBudget(evaluator=None)
    )
    candidate = result.candidates[0]
    assert candidate.plan is not None
    gather_tile = next(
        tile_map
        for tile_map in candidate.plan.tile_maps
        if operation.id in tile_map.body.semantic.source_op_ids
    )
    source_tile = next(
        tile
        for access in gather_tile.accesses
        for tile in access.inputs
        if tile.value_id == "value:arg0_1"
    )
    assert source_tile.shape[1] == 3


class _ScatterPadCopy(torch.nn.Module):
    def forward(self, base, row, block):
        updated = base.clone()
        updated[:, 2:5] = block  # slice_scatter
        updated[1] = row  # select_scatter
        updated[:, 1:8:3] = block  # stepped slice_scatter, as in Qwen3.5's mrope
        padded = torch.nn.functional.pad(updated, (1, 2, 0, 1))
        filled = torch.nn.functional.pad(updated, (2, 0), value=1.5)
        return padded, filled, updated.to(torch.float64).to(torch.float32)


def test_scatter_pad_and_copy_lower_without_unknown_targets():
    graph = lower_semantics(
        _import(_ScatterPadCopy(), torch.randn(3, 8), torch.randn(8), torch.randn(3, 3))
    )
    targets = {operation.attributes["target"] for operation in graph.ops.values()}
    unknown = {
        operation.attributes["target"]
        for operation in graph.ops.values()
        if operation.tensor_program is None
    }
    assert {"aten.slice_scatter.default", "aten.select_scatter.default"} <= targets
    assert "aten.constant_pad_nd.default" in targets
    assert unknown == set()


class _CausalConv1d(torch.nn.Module):
    """Qwen3.5's depthwise causal convolution over the token axis."""

    def forward(self, source, weight):
        length = source.shape[-1]
        mixed = torch.nn.functional.conv1d(
            source, weight, padding=weight.shape[-1] - 1, groups=source.shape[1]
        )
        return torch.nn.functional.silu(mixed[..., :length])


def test_depthwise_causal_conv1d_lowers_as_a_contraction():
    graph = lower_semantics(_import(_CausalConv1d(), torch.randn(2, 8, 6), torch.randn(8, 1, 4)))
    convolution = next(
        operation
        for operation in graph.ops.values()
        if operation.attributes["target"] == "aten.convolution.default"
    )
    assert convolution.tensor_program.family == "contraction"
    assert all(operation.tensor_program is not None for operation in graph.ops.values())


class _RecurrentStep(torch.nn.Module):
    """A recurrent state updated in place, like a gated delta rule step."""

    def forward(self, source, state):
        updated = state * 0.5 + source
        state.copy_(updated)
        return updated * 2


def test_in_place_state_write_lowers_into_the_input():
    from megabake import canonicalize_semantics, make_backend

    backend = make_backend(mode="decode", compile_kernel=False, cache=False)
    state = torch.zeros(4, 8)
    with torch.no_grad():
        torch.compile(_RecurrentStep(), backend=backend, fullgraph=True)(torch.ones(4, 8), state)
    prepared = backend.megabake_last_prepared
    graph = canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    write = next(
        operation
        for operation in graph.ops.values()
        if operation.attributes["target"] == "aten.copy_.default"
    )
    assert write.tensor_program is not None
    assert write.tensor_program.store.owns_storage is False
    assert write.operands[0] in graph.inputs
