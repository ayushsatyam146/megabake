"""Reduction prologues: elementwise work feeding a reduction, recomputed
inside it.

A reduction that reads the result of an elementwise chain (``sum(a * b)``,
``sum(state * key)``, ``amax(scores * scale)``) otherwise runs the chain as
a body of its own that writes the intermediate to memory, and the
reduction reads it back. Here the chain's expression replaces the
reduction's load of that intermediate: each of the chain's loads is
re-indexed through the reduction's load (output dimension j of the
intermediate is the reduction coordinate the load reads it at), so the
fused reduction reads the chain's inputs directly.

Generic: any reduction program whose input is an elementwise chain
(``elementwise.scalar_fusion_operations``) used only by that reduction,
with plain or affine indices. A chain whose index maps cannot be
re-expressed is left unfused.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..graph import ComputeGraph, Op, RegionCandidate
from ..ir import (
    AxisAffine,
    AxisIndex,
    AxisTerm,
    Binary,
    Cast,
    IndexedLoad,
    Load,
    ScalarExpression,
    TensorProgram,
    Unary,
    Where,
)
from .elementwise import compose_scalar_chain, scalar_fusion_operations
from .matching import external_inputs_of, has_illegal_effects, substitute_loads, value_bytes


@dataclass(frozen=True)
class ReductionPrologue:
    """A reduction with the elementwise chains it reads fused in front."""

    # Bytes the unfused chains write and the reduction reads back.
    saved_bytes: int


class _Unmappable(Exception):
    pass


def candidates(graph: ComputeGraph) -> tuple[RegionCandidate, ...]:
    """MEGABAKE_REDUCTION_PROLOGUE=0 offers none."""
    import os

    if os.environ.get("MEGABAKE_REDUCTION_PROLOGUE", "1") != "1":
        return ()
    return tuple(
        candidate
        for operation in graph.ops.values()
        if (candidate := _prologue_candidate(operation, graph)) is not None
    )


def _rename_axis(axis, mapping: dict[str, object]):
    """``axis`` (an index of a chain load) in the reduction's coordinates."""
    if isinstance(axis, int):
        return axis
    if isinstance(axis, str):
        if axis not in mapping:
            raise _Unmappable(axis)
        return mapping[axis]
    if isinstance(axis, AxisAffine):
        terms: list[AxisTerm] = []
        offset = axis.offset
        for term in axis.terms:
            target = mapping.get(term.axis)
            if isinstance(target, str):
                terms.append(replace(term, axis=target))
            elif isinstance(target, int) and term.divisor == 1 and term.modulus == 0:
                offset += target * term.scale
            else:
                raise _Unmappable(term.axis)
        return AxisAffine(tuple(terms), offset) if terms else offset
    raise _Unmappable(repr(axis))


def _rename_expression(expression: ScalarExpression, mapping: dict[str, object]) -> ScalarExpression:
    if isinstance(expression, AxisIndex):
        target = mapping.get(expression.axis)
        if not isinstance(target, str):
            raise _Unmappable(expression.axis)
        return replace(expression, axis=target)
    if isinstance(expression, Unary):
        return replace(expression, operand=_rename_expression(expression.operand, mapping))
    if isinstance(expression, Cast):
        return replace(expression, operand=_rename_expression(expression.operand, mapping))
    if isinstance(expression, Binary):
        return replace(
            expression,
            lhs=_rename_expression(expression.lhs, mapping),
            rhs=_rename_expression(expression.rhs, mapping),
        )
    if isinstance(expression, Where):
        return replace(
            expression,
            condition=_rename_expression(expression.condition, mapping),
            true_value=_rename_expression(expression.true_value, mapping),
            false_value=_rename_expression(expression.false_value, mapping),
        )
    return expression


def _load_ids(expression: ScalarExpression) -> set[str]:
    if isinstance(expression, Load):
        return {expression.load_id}
    found: set[str] = set()
    for name in ("operand", "lhs", "rhs", "condition", "true_value", "false_value"):
        child = getattr(expression, name, None)
        if child is not None:
            found |= _load_ids(child)
    return found


def _prologue_candidate(reduction: Op, graph: ComputeGraph) -> RegionCandidate | None:
    program = reduction.tensor_program
    if not isinstance(program, TensorProgram) or program.family != "reduction":
        return None
    if program.state_accesses or has_illegal_effects(reduction):
        return None
    loads = list(program.loads)
    replacements: dict[str, ScalarExpression] = {}
    # The post expression runs without the reduction axes: a load it reads
    # stays a load.
    post_loads = _load_ids(program.post) if program.post is not None else set()
    fused_ops: list[Op] = []
    saved = 0
    for load in program.loads:
        producer_id = graph.values[load.value_id].producer
        if producer_id is None:
            continue
        producer = graph.ops[producer_id]
        result = producer.results[0] if len(producer.results) == 1 else None
        if (
            result is None
            or not isinstance(producer.tensor_program, TensorProgram)
            or producer.tensor_program.family not in {"elementwise", "copy"}
            or result in graph.outputs
            or set(graph.values[result].users) != {reduction.id}
            or any(load.value_id == other.value_id for other in program.loads if other is not load)
            or any(isinstance(axis, AxisAffine) or not isinstance(axis, (str, int)) for axis in load.axes)
            or load.id in post_loads
        ):
            continue
        chain = scalar_fusion_operations(producer, graph)
        if any(has_illegal_effects(operation) for operation in chain):
            continue
        composed = compose_scalar_chain(chain, graph)
        if composed is None:
            continue
        domains = composed.iteration_domains
        shape = graph.values[result].shape
        if (
            composed.reduction_axes
            or shape is None
            or len(domains) != len(load.axes)
            or tuple(domain.size for domain in domains) != tuple(shape)
        ):
            continue
        mapping = {domain.axis: axis for domain, axis in zip(domains, load.axes, strict=True)}
        try:
            prefix = f"{load.id}:"
            renamed = {
                inner.id: IndexedLoad(
                    id=prefix + inner.id,
                    value_id=inner.value_id,
                    axes=tuple(_rename_axis(axis, mapping) for axis in inner.axes),
                    strides=inner.strides,
                    dtype=inner.dtype,
                    source_shape=inner.source_shape,
                )
                for inner in composed.loads
            }
            expression = _rename_expression(composed.combine, mapping)
        except _Unmappable:
            continue
        expression = substitute_loads(
            expression, {inner_id: Load(new.id, new.dtype) for inner_id, new in renamed.items()}
        )
        replacements[load.id] = expression
        loads.remove(load)
        loads.extend(renamed.values())
        fused_ops.extend(chain)
        saved += value_bytes(result, graph) * 2
    if not replacements:
        return None

    def substitute(expression: ScalarExpression | None) -> ScalarExpression | None:
        if expression is None:
            return None
        return substitute_loads(
            expression,
            {**{load.id: Load(load.id, load.dtype) for load in loads}, **replacements},
        )

    fused = replace(
        program,
        loads=tuple(loads),
        combine=substitute(program.combine),
        post=substitute(program.post),
        init=program.init,
    )
    unique = {operation.id: operation for operation in (*fused_ops, reduction)}
    operations = tuple(unique.values())
    return RegionCandidate(
        id=f"region:{reduction.id.removeprefix('op:')}:prologue",
        source_op_ids=tuple(operation.id for operation in operations),
        external_inputs=external_inputs_of(operations, graph),
        live_outputs=(reduction.results[0],),
        tensor_program=fused,
        semantics=ReductionPrologue(saved),
        cost_bytes=saved,
    )
