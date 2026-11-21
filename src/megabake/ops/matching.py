"""Graph queries shared by the family recognizers and region formation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import prod

from ..graph import DTYPE_BYTES, ComputeGraph, Op
from ..ir import (
    Accumulator,
    AxisIndex,
    Binary,
    Cast,
    Constant,
    Load,
    ScalarExpression,
    TensorProgram,
    Unary,
    Where,
)


def external_inputs_of(operations: tuple[Op, ...], graph: ComputeGraph) -> tuple[str, ...]:
    produced = {result_id for operation in operations for result_id in operation.results}
    inputs: list[str] = []
    for operation in operations:
        for operand_id in operation.operands:
            if operand_id not in produced and operand_id not in inputs:
                inputs.append(operand_id)
    return tuple(inputs)


def live_outputs_of(operations: tuple[Op, ...], graph: ComputeGraph) -> tuple[str, ...]:
    operation_ids = {operation.id for operation in operations}
    produced = {result_id for operation in operations for result_id in operation.results}
    outputs: list[str] = []
    for result_id in graph.outputs:
        if result_id in produced and result_id not in outputs:
            outputs.append(result_id)
    for operation in operations:
        for result_id in operation.results:
            if result_id in outputs:
                continue
            users = graph.values[result_id].users
            if any(user not in operation_ids for user in users):
                outputs.append(result_id)
    return tuple(outputs)


def producer_of(value_id: str, graph: ComputeGraph) -> Op | None:
    operation_id = graph.values[value_id].producer
    return graph.ops.get(operation_id) if operation_id is not None else None


def sole_user(operation: Op, graph: ComputeGraph) -> Op | None:
    users = graph.values[operation.results[0]].users
    if len(users) != 1:
        return None
    return graph.ops.get(users[0])


def matches_target(operation: Op, target: str) -> bool:
    return operation.attributes.get("target") == target


def operation_target(operation: Op) -> object:
    return operation.attributes.get("target")


def binary_operator(operation: Op | None, operator: str) -> bool:
    if operation is None or not isinstance(operation.tensor_program, TensorProgram):
        return False
    return (
        isinstance(operation.tensor_program.combine, Binary)
        and operation.tensor_program.combine.operator == operator
    )


def scalar_operand(operation: Op, index: int) -> object:
    arguments = operation.attributes.get("fx_args", ())
    return arguments[index] if index < len(arguments) else None


def cast_rounding_for(dtype: str) -> str:
    return "round_nearest_even" if dtype in {"torch.bfloat16", "torch.float16"} else "exact"


def ordered_operations(operations: tuple[Op, ...], graph: ComputeGraph) -> tuple[Op, ...]:
    by_id = {operation.id: operation for operation in operations}
    return tuple(
        operation for operation_id, operation in graph.ops.items() if operation_id in by_id
    )


def value_bytes(value_id: str, graph: ComputeGraph) -> int:
    value = graph.values[value_id]
    return prod(value.shape or ()) * DTYPE_BYTES.get(value.dtype or "", 0)


def has_illegal_effects(operation: Op) -> bool:
    return bool(operation.effects.get("unknown") or operation.effects.get("mutates_inputs"))


def is_contraction(operation: Op) -> bool:
    return (
        isinstance(operation.tensor_program, TensorProgram)
        and operation.tensor_program.family == "contraction"
    )


def producing_view(value_id: str, graph: ComputeGraph) -> Op | None:
    producer_id = graph.values[value_id].producer
    if producer_id is None:
        return None
    operation = graph.ops[producer_id]
    program = operation.tensor_program
    if (
        not isinstance(program, TensorProgram)
        or program.family != "view"
        or program.store.owns_storage
        or len(program.loads) != 1
        or value_id not in operation.results
    ):
        return None
    return operation


def substitute_loads(
    expression: ScalarExpression, replacements: Mapping[str, ScalarExpression]
) -> ScalarExpression:
    if isinstance(expression, Load):
        return replacements[expression.load_id]
    if isinstance(expression, Constant):
        return expression
    if isinstance(expression, Unary):
        return replace(expression, operand=substitute_loads(expression.operand, replacements))
    if isinstance(expression, Binary):
        return replace(
            expression,
            lhs=substitute_loads(expression.lhs, replacements),
            rhs=substitute_loads(expression.rhs, replacements),
        )
    if isinstance(expression, (AxisIndex, Accumulator)):
        return expression
    if isinstance(expression, Cast):
        return replace(expression, operand=substitute_loads(expression.operand, replacements))
    if isinstance(expression, Where):
        return replace(
            expression,
            condition=substitute_loads(expression.condition, replacements),
            true_value=substitute_loads(expression.true_value, replacements),
            false_value=substitute_loads(expression.false_value, replacements),
        )
    raise ValueError(f"unsupported scalar expression {expression!r}")
