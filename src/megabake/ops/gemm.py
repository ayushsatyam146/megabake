"""GEMMs: contractions read through aliasing views, and contractions fused
with the elementwise chain that consumes their results (epilogues)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import prod
from typing import Any

from ..graph import ComputeGraph, Op, RegionCandidate
from ..ir import (
    Accumulator,
    AxisAffine,
    AxisIndex,
    AxisTerm,
    Binary,
    Cast,
    IndexedLoad,
    Load,
    ScalarExpression,
    TensorProgram,
    TensorStore,
    Unary,
    Where,
)
from .elementwise import (
    compose_scalar_chain,
    is_scalar_operation,
    same_chain_layout,
    scalar_fusion_operations,
)
from .matching import (
    cast_rounding_for,
    has_illegal_effects,
    is_contraction,
    producer_of,
    producing_view,
    substitute_loads,
    value_bytes,
)


@dataclass(frozen=True)
class ViewPlusGemm:
    """The contraction reads ``source`` through ``view``, which only aliases it."""

    view: str
    source: str


@dataclass(frozen=True)
class GemmEpilogue:
    """One contraction, or two paired ones (gate and up projections sharing a
    left operand), whose consumer chain runs as their epilogue."""

    contractions: tuple[str, ...]

    @property
    def paired(self) -> bool:
        return len(self.contractions) == 2


def view_semantics(operation: Op, graph: ComputeGraph) -> ViewPlusGemm | None:
    """A contraction that reads an operand through an aliasing view."""
    program = operation.tensor_program
    if not isinstance(program, TensorProgram) or program.family != "contraction":
        return None

    for load in program.loads:
        view_operation = producing_view(load.value_id, graph)
        if view_operation is None:
            continue
        view_program = view_operation.tensor_program
        assert isinstance(view_program, TensorProgram)
        view_load = view_program.loads[0]
        if load.strides != view_load.strides:
            continue
        return ViewPlusGemm(view_operation.id, view_load.value_id)
    return None


def epilogue_candidates(graph: ComputeGraph) -> tuple[RegionCandidate, ...]:
    """GEMMs fused with the elementwise chain that consumes their results.

    The chain becomes the contraction's ``post`` expression: every read of
    the GEMM result at the output position is replaced by the GEMM's own
    result expression (the accumulator rounded to the result dtype), and the
    chain's other operands become loads indexed by the GEMM's output axes. A
    chain over two GEMMs of the same left operand and shape (gate and up
    projections) forms a paired contraction: accumulator 1 multiplies
    ``loads[2]`` as accumulator 0 multiplies ``loads[1]``.
    """
    candidates = []
    for tail in graph.ops.values():
        if not is_scalar_operation(tail) or _absorbed_by_consumer_chain(tail, graph):
            continue
        candidate = _gemm_epilogue_candidate(tail, graph)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def _absorbed_by_consumer_chain(operation: Op, graph: ComputeGraph) -> bool:
    """Whether ``operation``'s only consumer also reads a GEMM result, so the
    epilogue (a paired one) forms at that consumer instead. Fusing here
    alone would make the other GEMM wait for this one."""
    result = operation.results[0]
    users = graph.values[result].users
    if result in graph.outputs or len(users) != 1:
        return False
    consumer = graph.ops[users[0]]
    return is_scalar_operation(consumer) and any(
        _viewed_contraction(operand_id, graph) is not None
        for operand_id in consumer.operands
        if operand_id != result
    )


def _gemm_epilogue_candidate(tail: Op, graph: ComputeGraph) -> RegionCandidate | None:
    chain = _epilogue_chain(tail, graph)
    chain_ids = {operation.id for operation in chain}
    if any(has_illegal_effects(operation) for operation in chain):
        return None
    if any(
        operation.results[0] in graph.outputs
        or any(user not in chain_ids for user in graph.values[operation.results[0]].users)
        for operation in chain[:-1]
    ):
        return None
    if not all(
        same_chain_layout(operation.results[0], tail.results[0], graph) for operation in chain
    ):
        return None
    composed = compose_scalar_chain(chain, graph)
    if composed is None or _has_axis_index(composed.combine):
        return None

    leaves = {
        load.id: found
        for load in composed.loads
        if (found := _viewed_contraction(load.value_id, graph)) is not None
    }
    contraction_ids = list(dict.fromkeys(contraction.id for contraction, _ in leaves.values()))
    if not 1 <= len(contraction_ids) <= 2:
        return None
    order = list(graph.ops)
    contractions = [graph.ops[op_id] for op_id in sorted(contraction_ids, key=order.index)]
    first = contractions[0].tensor_program
    assert isinstance(first, TensorProgram)
    rows, columns = first.store.shape if len(first.store.shape) == 2 else (0, 0)
    store = composed.store
    if not _is_row_major_gemm(first) or prod(store.shape) != rows * columns:
        return None
    if not store.owns_storage or _significant_strides(
        store.strides, store.shape
    ) != _significant_strides(_contiguous_strides(store.shape), store.shape):
        return None
    for contraction in contractions:
        views = [
            view for found, path in leaves.values() if found.id == contraction.id for view in path
        ]
        allowed = chain_ids | {view.id for view in views}
        path_values = (contraction.results[0], *(view.results[0] for view in views))
        if any(
            value_id in graph.outputs
            or any(user not in allowed for user in graph.values[value_id].users)
            for value_id in path_values
        ):
            return None

    base_loads = first.loads
    posts = {contractions[0].id: first.post}
    if len(contractions) == 2:
        second = contractions[1].tensor_program
        assert isinstance(second, TensorProgram)
        if not _pairs_with(first, second, graph):
            return None
        base_loads = (*first.loads, replace(second.loads[1], id=f"load:{len(first.loads)}"))
        posts[contractions[1].id] = _with_accumulator(second.post, 1)

    chain_axes = tuple(domain.axis for domain in composed.iteration_domains)
    coordinates = _gemm_coordinates(store.shape, rows, columns)
    if coordinates is None:
        return None
    axis_map = dict(zip(chain_axes, coordinates, strict=True))
    extra_loads: list[IndexedLoad] = []
    replacements: dict[str, ScalarExpression] = {}
    for load in composed.loads:
        if load.id in leaves:
            contraction, _ = leaves[load.id]
            result = graph.values[load.value_id]
            source = graph.values[contraction.results[0]]
            if (
                len(load.axes) != len(chain_axes)
                or any(
                    axis != chain_axis and not (size == 1 and axis == 0)
                    for axis, chain_axis, size in zip(load.axes, chain_axes, store.shape)
                )
                or load.dtype != first.store.dtype
                or result.storage_offset != source.storage_offset
                or _significant_strides(load.strides, store.shape)
                != _significant_strides(_contiguous_strides(store.shape), store.shape)
            ):
                # Only reads at the output position come from the accumulators.
                return None
            replacements[load.id] = posts[contraction.id]
            continue
        axes = _gemm_load_axes(load.axes, axis_map)
        if axes is None:
            return None
        load_id = f"load:{len(base_loads) + len(extra_loads)}"
        extra_loads.append(replace(load, id=load_id, axes=axes))
        replacements[load.id] = Load(load_id, load.dtype)
    post = substitute_loads(composed.combine, replacements)
    if post.dtype != store.dtype:
        post = Cast(post, store.dtype, cast_rounding_for(store.dtype))

    program = replace(
        first,
        decomposition="indexed multiply-accumulate with a fused elementwise epilogue",
        loads=(*base_loads, *extra_loads),
        post=post,
        store=TensorStore(store.value_id, (rows, columns), store.dtype, (columns, 1), True),
    )
    operations = (*contractions, *chain)
    return RegionCandidate(
        id=f"region:{tail.id.removeprefix('op:')}:gemm_epilogue",
        source_op_ids=tuple(operation.id for operation in operations),
        external_inputs=tuple(dict.fromkeys(load.value_id for load in program.loads)),
        live_outputs=(tail.results[0],),
        tensor_program=program,
        semantics=GemmEpilogue(tuple(contraction.id for contraction in contractions)),
        cost_bytes=sum(
            2 * value_bytes(contraction.results[0], graph) for contraction in contractions
        )
        + sum(2 * value_bytes(operation.results[0], graph) for operation in chain[:-1]),
    )


def _epilogue_chain(tail: Op, graph: ComputeGraph) -> tuple[Op, ...]:
    """The scalar producers of ``tail`` whose results stay inside the chain.

    A producer whose result is also read elsewhere (the previous residual
    sum, say) stays materialized and becomes a load of the epilogue.
    """
    operations = {operation.id: operation for operation in scalar_fusion_operations(tail, graph)}
    while True:
        escaping = {
            operation_id
            for operation_id, operation in operations.items()
            if operation_id != tail.id
            and (
                operation.results[0] in graph.outputs
                or any(user not in operations for user in graph.values[operation.results[0]].users)
            )
        }
        if not escaping:
            break
        reachable: set[str] = set()
        pending = [tail.id]
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            reachable.add(current)
            for operand_id in graph.ops[current].operands:
                producer_id = graph.values[operand_id].producer
                if producer_id in operations and producer_id not in escaping:
                    pending.append(producer_id)
        operations = {
            operation_id: operation
            for operation_id, operation in operations.items()
            if operation_id in reachable
        }
    return tuple(
        operation
        for operation in scalar_fusion_operations(tail, graph)
        if operation.id in operations
    )


def _viewed_contraction(value_id: str, graph: ComputeGraph) -> tuple[Op, tuple[Op, ...]] | None:
    """The contraction whose result ``value_id`` is, through aliasing views,
    and those views."""
    views: list[Op] = []
    while (view := producing_view(value_id, graph)) is not None:
        views.append(view)
        view_program = view.tensor_program
        assert isinstance(view_program, TensorProgram)
        value_id = view_program.loads[0].value_id
    producer = producer_of(value_id, graph)
    if producer is None or not is_contraction(producer):
        return None
    return producer, tuple(views)


def _is_row_major_gemm(program: TensorProgram) -> bool:
    if len(program.store.shape) != 2 or len(program.reduction_axes) != 1:
        return False
    _, columns = program.store.shape
    return program.store.strides == (columns, 1) and program.store.owns_storage


def _pairs_with(first: TensorProgram, second: TensorProgram, graph: ComputeGraph) -> bool:
    """Whether two unbiased GEMMs share their left operand and shape, so one
    pass over the left operand can feed both."""
    if len(first.loads) != 2 or len(second.loads) != 2 or not _is_row_major_gemm(second):
        return False
    first_left, second_left = first.loads[0], second.loads[0]
    first_right, second_right = first.loads[1], second.loads[1]
    # Each GEMM may read the operand through its own reshape of one tensor.
    left_values = graph.values[first_left.value_id], graph.values[second_left.value_id]
    return (
        first.iteration_domains == second.iteration_domains
        and (first_left.axes, first_left.strides, first_left.dtype, first_left.source_shape)
        == (second_left.axes, second_left.strides, second_left.dtype, second_left.source_shape)
        and left_values[0].storage_relation.storage_id is not None
        and left_values[0].storage_relation.storage_id == left_values[1].storage_relation.storage_id
        and left_values[0].storage_offset == left_values[1].storage_offset
        and first.combine == second.combine
        and first.post == second.post
        and first.store.shape == second.store.shape
        and first.store.dtype == second.store.dtype
        and (first_right.axes, first_right.strides, first_right.dtype)
        == (second_right.axes, second_right.strides, second_right.dtype)
    )


def _with_accumulator(expression: ScalarExpression, index: int) -> ScalarExpression:
    if isinstance(expression, Accumulator):
        return replace(expression, index=index)
    if isinstance(expression, (Unary, Cast)):
        return replace(expression, operand=_with_accumulator(expression.operand, index))
    if isinstance(expression, Binary):
        return replace(
            expression,
            lhs=_with_accumulator(expression.lhs, index),
            rhs=_with_accumulator(expression.rhs, index),
        )
    if isinstance(expression, Where):
        return replace(
            expression,
            condition=_with_accumulator(expression.condition, index),
            true_value=_with_accumulator(expression.true_value, index),
            false_value=_with_accumulator(expression.false_value, index),
        )
    return expression


def _has_axis_index(expression: ScalarExpression) -> bool:
    if isinstance(expression, AxisIndex):
        return True
    if isinstance(expression, (Unary, Cast)):
        return _has_axis_index(expression.operand)
    if isinstance(expression, Binary):
        return _has_axis_index(expression.lhs) or _has_axis_index(expression.rhs)
    if isinstance(expression, Where):
        return any(
            _has_axis_index(part)
            for part in (expression.condition, expression.true_value, expression.false_value)
        )
    return False


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    return tuple(reversed(strides))


def _significant_strides(strides: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    # A size-1 axis never moves, so its stride does not matter.
    return tuple(stride for stride, size in zip(strides, shape, strict=True) if size != 1)


def _gemm_coordinates(
    shape: tuple[int, ...], rows: int, columns: int
) -> list[int | str | AxisAffine] | None:
    """Each axis of a contiguous ``shape`` as a function of the GEMM output
    coordinates ``o0`` (rows) and ``o1`` (columns), when the trailing axes
    hold the columns and the leading axes the rows."""
    split = len(shape)
    inner = 1
    while split > 0 and inner < columns:
        split -= 1
        inner *= shape[split]
    if inner != columns or prod(shape[:split]) != rows:
        return None
    coordinates: list[int | str | AxisAffine] = []
    for index, size in enumerate(shape):
        axis, end, extent = ("o1", len(shape), columns) if index >= split else ("o0", split, rows)
        below = prod(shape[index + 1 : end])
        if size == 1:
            coordinates.append(0)
        elif below == 1 and size == extent:
            coordinates.append(axis)
        else:
            modulus = 0 if size * below == extent else size
            coordinates.append(AxisAffine((AxisTerm(axis, 1, below, modulus),)))
    return coordinates


def _gemm_load_axes(
    axes: tuple[Any, ...], axis_map: Mapping[str, int | str | AxisAffine]
) -> tuple[Any, ...] | None:
    """A chain load's axes rewritten over the GEMM output axes, or None when
    an index does not compose."""
    result: list[Any] = []
    for axis in axes:
        if isinstance(axis, int):
            result.append(axis)
        elif isinstance(axis, str):
            if axis not in axis_map:
                return None
            result.append(axis_map[axis])
        elif isinstance(axis, AxisAffine):
            terms: list[AxisTerm] = []
            for term in axis.terms:
                mapped = axis_map.get(term.axis)
                if mapped is None:
                    return None
                if isinstance(mapped, int):
                    continue
                if isinstance(mapped, str):
                    terms.append(replace(term, axis=mapped))
                elif term.divisor == 1 and term.modulus == 0 and len(mapped.terms) == 1:
                    terms.append(replace(mapped.terms[0], scale=term.scale))
                else:
                    return None
            result.append(AxisAffine(tuple(terms), axis.offset))
        else:
            return None
    return tuple(result)
