"""RMSNorm: ``x * rsqrt(mean(x ** 2) + eps)``, optionally scaled by a weight
and added to a residual, recognized from its decomposed ops.

The ops become one row reduction: the sum of squares is the reduction's
combine, and the ops before and after the mean become its pre- and
post-reduction expressions, so the source's casts are kept.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..graph import ComputeGraph, Op, RegionCandidate
from ..ir import (
    AxisPredicate,
    Binary,
    Constant,
    IndexedLoad,
    IterationDomain,
    Load,
    ScalarExpression,
    TensorProgram,
    TensorStore,
)
from .matching import (
    external_inputs_of,
    has_illegal_effects,
    live_outputs_of,
    matches_target,
    producer_of,
    sole_user,
    substitute_loads,
    value_bytes,
)


@dataclass(frozen=True)
class RMSNorm:
    """A norm over the last axis of width elements."""

    width: int
    epsilon: float
    # The graph inputs of the weight multiply and the residual add, if fused.
    weight: str | None
    residual: str | None


def candidates(graph: ComputeGraph) -> tuple[RegionCandidate, ...]:
    """An RMSNorm region at each ``mean`` that starts the norm's chain."""
    candidates = []
    for operation in graph.ops.values():
        program = operation.tensor_program
        if (
            not isinstance(program, TensorProgram)
            or program.family != "reduction"
            or program.source_target != "aten.mean.dim"
        ):
            continue
        candidate = _rms_norm_candidate(operation, graph)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def _rms_norm_candidate(mean: Op, graph: ComputeGraph) -> RegionCandidate | None:
    chain = _rms_norm_chain(mean, graph)
    if chain is None:
        return None
    operations, x_id, weight_id, epsilon = chain
    tail = operations[-1]
    tail_value = graph.values[tail.results[0]]
    if tail_value.shape is None or tail_value.dtype is None or tail_value.strides is None:
        return None
    x_value = graph.values[x_id]
    if x_value.shape is None or x_value.dtype is None or x_value.strides is None:
        return None
    if len(x_value.shape) < 2 or x_value.shape[-1] <= 0:
        return None

    output_domains = tuple(
        IterationDomain(f"o{index}", size) for index, size in enumerate(tail_value.shape)
    )
    reduction_domain = IterationDomain("r0", x_value.shape[-1])
    output_axes = tuple(domain.axis for domain in output_domains)
    leaf_loads: list[IndexedLoad] = []

    def leaf_load(value_id: str, axes: tuple[object, ...]) -> Load:
        value = graph.values[value_id]
        load = IndexedLoad(
            id=f"load:{len(leaf_loads)}",
            value_id=value_id,
            axes=axes,
            strides=value.strides or (),
            dtype=value.dtype or "",
            source_shape=value.shape,
        )
        leaf_loads.append(load)
        return Load(load.id, load.dtype)

    x_reduction_load = leaf_load(x_id, (*output_axes[:-1], reduction_domain.axis))
    x_post_load = leaf_load(x_id, output_axes)
    weight_reduction_load = (
        leaf_load(weight_id, (reduction_domain.axis,)) if weight_id is not None else None
    )
    weight_post_load = leaf_load(weight_id, output_axes[-1:]) if weight_id is not None else None
    residual_id = None
    if (
        tail.tensor_program is not None
        and isinstance(tail.tensor_program.combine, Binary)
        and tail.tensor_program.combine.operator == "add"
    ):
        for operand_id in tail.operands:
            if operand_id != operations[-2].results[0]:
                residual_id = operand_id
    residual_load = leaf_load(residual_id, output_axes) if residual_id is not None else None

    expressions: dict[str, ScalarExpression] = {}
    post_expressions: dict[str, ScalarExpression] = {}
    x_expression_ids = {x_id}
    if (
        operations[0].attributes.get("target") == "prims.convert_element_type.default"
        and operations[0].operands[0] == x_id
    ):
        x_expression_ids.add(operations[0].results[0])
    after_mean = False
    for operation in operations:
        program = operation.tensor_program
        if not isinstance(program, TensorProgram):
            return None
        if operation is mean:
            input_expression = expressions.get(program.loads[0].value_id)
            if input_expression is None:
                return None
            combine = substitute_loads(
                program.combine,
                {program.loads[0].id: input_expression},
            )
            expressions[operation.results[0]] = program.post
            post_expressions[operation.results[0]] = program.post
            after_mean = True
            continue
        target_expressions = post_expressions if after_mean else expressions
        replacements: dict[str, ScalarExpression] = {}
        for load in program.loads:
            expression = target_expressions.get(load.value_id)
            if expression is None:
                if load.value_id in x_expression_ids:
                    expression = x_post_load if after_mean else x_reduction_load
                elif weight_reduction_load is not None and load.value_id == weight_id:
                    expression = weight_post_load if after_mean else weight_reduction_load
                elif residual_load is not None and load.value_id == residual_id:
                    expression = residual_load
                else:
                    return None
            replacements[load.id] = expression
        target_expressions[operation.results[0]] = substitute_loads(program.combine, replacements)

    program = TensorProgram(
        family="reduction",
        source_target="rms_norm",
        decomposition="row sum of squares with generated pre- and post-reduction expressions",
        iteration_domains=(*output_domains, reduction_domain),
        reduction_axes=(reduction_domain.axis,),
        loads=tuple(leaf_loads),
        init=mean.tensor_program.init if isinstance(mean.tensor_program, TensorProgram) else None,
        combine=combine,
        post=post_expressions[tail.results[0]],
        accumulator_dtype=mean.tensor_program.accumulator_dtype
        if isinstance(mean.tensor_program, TensorProgram)
        else None,
        store=TensorStore(
            tail_value.id,
            tail_value.shape,
            tail_value.dtype,
            tail_value.strides,
            True,
        ),
        predicates=tuple(
            AxisPredicate(domain.axis, domain.size)
            for domain in (*output_domains, reduction_domain)
        ),
    )
    operation_ids = {operation.id for operation in operations}
    rejections: list[str] = []
    if any(has_illegal_effects(operation) for operation in operations):
        rejections.append("source effects do not fit a fused norm body")
    if any(
        any(user not in operation_ids for user in graph.values[operation.results[0]].users)
        for operation in operations[:-1]
    ):
        rejections.append("a norm intermediate value has an external use")

    return RegionCandidate(
        id=f"region:{tail.id.removeprefix('op:')}:rms_norm",
        source_op_ids=tuple(operation.id for operation in operations),
        external_inputs=external_inputs_of(tuple(operations), graph),
        live_outputs=live_outputs_of(tuple(operations), graph),
        tensor_program=program,
        semantics=RMSNorm(x_value.shape[-1], epsilon, weight_id, residual_id),
        cost_bytes=sum(value_bytes(operation.results[0], graph) for operation in operations[:-1]),
        rejections=tuple(rejections),
    )


def _rms_norm_chain(
    mean: Op, graph: ComputeGraph
) -> tuple[tuple[Op, ...], str, str | None, float] | None:
    mean_program = mean.tensor_program
    if not isinstance(mean_program, TensorProgram):
        return None
    if len(mean_program.reduction_axes) != 1:
        return None
    pow_operation = producer_of(mean.operands[0], graph)
    if pow_operation is None or not matches_target(pow_operation, "aten.pow.Tensor_Scalar"):
        return None
    cast_operation = producer_of(pow_operation.operands[0], graph)
    if cast_operation is not None and matches_target(
        cast_operation, "prims.convert_element_type.default"
    ):
        x_id = cast_operation.operands[0]
    else:
        x_id = pow_operation.operands[0]
        cast_operation = None

    add = sole_user(mean, graph)
    if add is None or not (
        matches_target(add, "aten.add.Scalar") or matches_target(add, "aten.add.Tensor")
    ):
        return None
    rsqrt = sole_user(add, graph)
    if rsqrt is None or not matches_target(rsqrt, "aten.rsqrt.default"):
        return None
    scale = sole_user(rsqrt, graph)
    if scale is None or not matches_target(scale, "aten.mul.Tensor"):
        return None
    scale_x_id = cast_operation.results[0] if cast_operation is not None else x_id
    if scale_x_id not in scale.operands or rsqrt.results[0] not in scale.operands:
        return None

    def output_cast_of(operation: Op) -> Op | None:
        user = sole_user(operation, graph)
        if (
            user is not None
            and matches_target(user, "prims.convert_element_type.default")
            and graph.values[user.operands[0]].dtype != graph.values[user.results[0]].dtype
        ):
            return user
        return None

    def affine_of(operation: Op) -> tuple[Op, str] | None:
        user = sole_user(operation, graph)
        if (
            user is None
            or not matches_target(user, "aten.mul.Tensor")
            or operation.results[0] not in user.operands
        ):
            return None
        weight = next(
            (operand_id for operand_id in user.operands if operand_id != operation.results[0]),
            None,
        )
        return (user, weight) if weight in graph.inputs else None

    # The weight multiply may come before the output cast or, as in Hugging
    # Face's LlamaRMSNorm (`weight * hidden.to(input_dtype)`), after it.
    tail = scale
    affine: Op | None = None
    weight_id: str | None = None
    output_cast = output_cast_of(tail)
    if output_cast is not None:
        tail = output_cast
        matched = affine_of(tail)
        if matched is not None:
            affine, weight_id = matched
            tail = affine
    else:
        matched = affine_of(tail)
        if matched is not None:
            affine, weight_id = matched
            tail = affine
            output_cast = output_cast_of(tail)
            if output_cast is not None:
                tail = output_cast

    residual = sole_user(tail, graph)
    if not (
        residual is not None
        and matches_target(residual, "aten.add.Tensor")
        and tail.results[0] in residual.operands
    ):
        residual = None
    else:
        residual_id = next(
            operand_id for operand_id in residual.operands if operand_id != tail.results[0]
        )
        if (
            residual_id not in graph.inputs
            or graph.values[residual_id].shape != graph.values[tail.results[0]].shape
        ):
            residual = None
        else:
            tail = residual

    operations = [
        operation
        for operation in graph.ops.values()
        if operation.id
        in {
            operation.id
            for operation in (
                cast_operation,
                pow_operation,
                mean,
                add,
                rsqrt,
                scale,
                affine,
                output_cast,
                residual,
            )
            if operation is not None
        }
    ]
    add_program = add.tensor_program
    if (
        not isinstance(add_program, TensorProgram)
        or not isinstance(add_program.combine, Binary)
        or not isinstance(add_program.combine.rhs, Constant)
        or not isinstance(add_program.combine.rhs.value, float)
    ):
        return None
    return tuple(operations), x_id, weight_id, add_program.combine.rhs.value
