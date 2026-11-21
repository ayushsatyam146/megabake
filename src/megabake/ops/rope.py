"""Rotary position embeddings (RoPE) over the last axis, recognized from
their decomposed multiplies, adds and concat.

The region is one elementwise map: the concat becomes a select on the
last-axis index between offset loads of x, and the multiplies and adds keep
their own rounding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from ..graph import ComputeGraph, Op, RegionCandidate
from ..ir import (
    AxisAffine,
    AxisIndex,
    AxisPredicate,
    AxisTerm,
    Binary,
    Cast,
    Constant,
    IndexedLoad,
    IterationDomain,
    Load,
    ScalarExpression,
    TensorProgram,
    TensorStore,
    Where,
)
from .matching import (
    binary_operator,
    external_inputs_of,
    has_illegal_effects,
    live_outputs_of,
    matches_target,
    producer_of,
    producing_view,
    sole_user,
    substitute_loads,
    value_bytes,
)


@dataclass(frozen=True)
class Rope:
    """Element i of the first ``rotary_width`` rotates with element
    i + rotary_width / 2; the rest of the ``width`` elements pass through."""

    width: int
    rotary_width: int
    # "rotate_half": Hugging Face's x * cos + rotate_half(x) * sin;
    # "split_halves": each half computed on its own, then concatenated.
    form: Literal["rotate_half", "split_halves"]


def candidates(graph: ComputeGraph) -> tuple[RegionCandidate, ...]:
    """A RoPE region at each last-axis concat that joins a rotation's halves."""
    candidates = []
    for operation in graph.ops.values():
        program = operation.tensor_program
        if program is None or program.family != "concat":
            continue
        candidate = _split_halves_candidate(operation, graph) or _rotate_half_candidate(
            operation, graph
        )
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def _rotate_half_candidate(concat: Op, graph: ComputeGraph) -> RegionCandidate | None:
    """Match ``x * cos + cat(-x[..., h:], x[..., :h]) * sin``.

    This is Hugging Face's ``apply_rotary_pos_emb`` with ``rotate_half``. The
    multiplies and the add keep their own captured programs; only the concat
    becomes a select on the last-axis index between offset loads of ``x``.
    """
    concat_program = concat.tensor_program
    if not isinstance(concat_program, TensorProgram) or len(concat.operands) != 2:
        return None
    output = graph.values[concat.results[0]]
    if output.shape is None or output.dtype is None or output.shape[-1] % 2:
        return None
    rank = len(output.shape)
    half = output.shape[-1] // 2
    negate = producer_of(concat.operands[0], graph)
    if negate is None or not matches_target(negate, "aten.neg.default"):
        return None
    first = _last_axis_slice(negate.operands[0], graph)
    second = _last_axis_slice(concat.operands[1], graph)
    if first is None or second is None or first[0] != second[0]:
        return None
    x_id = first[0]
    if first[1] != half or second[1] != 0:
        return None
    if any(
        graph.values[value_id].shape is None or graph.values[value_id].shape[-1] != half
        for value_id in (negate.results[0], concat.operands[1])
    ):
        return None
    x = graph.values[x_id]
    if x.shape != output.shape or x.strides is None or x.dtype != output.dtype:
        return None

    rotate_multiply = sole_user(concat, graph)
    if rotate_multiply is None or not binary_operator(rotate_multiply, "mul"):
        return None
    add = sole_user(rotate_multiply, graph)
    if add is None or not binary_operator(add, "add"):
        return None
    x_multiply = producer_of(
        next(
            (operand for operand in add.operands if operand != rotate_multiply.results[0]),
            rotate_multiply.results[0],
        ),
        graph,
    )
    if (
        x_multiply is None
        or x_multiply.id == rotate_multiply.id
        or not binary_operator(x_multiply, "mul")
        or x_id not in x_multiply.operands
    ):
        return None
    operations = (negate, concat, x_multiply, rotate_multiply, add)
    programs = [operation.tensor_program for operation in operations]
    if any(not isinstance(program, TensorProgram) for program in programs):
        return None
    if any(
        len(program.iteration_domains) != rank
        for program in (
            x_multiply.tensor_program,
            rotate_multiply.tensor_program,
            add.tensor_program,
        )
    ):
        return None

    output_axes = tuple(domain.axis for domain in add.tensor_program.iteration_domains)
    final_axis = output_axes[-1]
    # The captured load of x gives its axis map (unit axes are literal 0).
    x_axes = next(
        (tuple(load.axes) for load in x_multiply.tensor_program.loads if load.value_id == x_id),
        None,
    )
    if x_axes is None or x_axes[-1] != final_axis:
        return None
    leaf_loads: list[IndexedLoad] = []
    leaf_ids: dict[tuple[str, tuple[Any, ...]], str] = {}

    def leaf(
        value_id: str, axes: tuple[Any, ...], strides: tuple[int, ...], dtype: str, shape
    ) -> Load:
        key = (value_id, axes)
        if key not in leaf_ids:
            leaf_ids[key] = f"load:{len(leaf_loads)}"
            leaf_loads.append(IndexedLoad(leaf_ids[key], value_id, axes, strides, dtype, shape))
        return Load(leaf_ids[key], dtype)

    def x_at(offset: int) -> Load:
        axis = final_axis if offset == 0 else AxisAffine((AxisTerm(final_axis),), offset=offset)
        return leaf(x_id, (*x_axes[:-1], axis), x.strides, x.dtype, x.shape)

    def compose(operation: Op, known: Mapping[str, ScalarExpression]) -> ScalarExpression | None:
        program = operation.tensor_program
        replacements: dict[str, ScalarExpression] = {}
        for load in program.loads:
            if load.value_id in known:
                replacements[load.id] = known[load.value_id]
            elif load.value_id == x_id and tuple(load.axes) == x_axes:
                replacements[load.id] = x_at(0)
            elif load.value_id != x_id and all(
                axis in output_axes or isinstance(axis, int) for axis in load.axes
            ):
                replacements[load.id] = leaf(
                    load.value_id,
                    tuple(load.axes),
                    tuple(load.strides),
                    load.dtype,
                    load.source_shape,
                )
            else:
                return None
        return substitute_loads(program.combine, replacements)

    negated = compose(negate, {negate.operands[0]: x_at(half)})
    if negated is None:
        return None
    rotated = Where(
        Binary(
            "lt",
            AxisIndex(final_axis, "torch.int64"),
            Constant(half, "torch.int64"),
            "torch.bool",
        ),
        negated,
        x_at(-half),
        output.dtype,
    )
    rotated_product = compose(rotate_multiply, {concat.results[0]: rotated})
    x_product = compose(x_multiply, {})
    if rotated_product is None or x_product is None:
        return None
    combined = compose(
        add,
        {
            rotate_multiply.results[0]: rotated_product,
            x_multiply.results[0]: x_product,
        },
    )
    if combined is None:
        return None
    fused_program = replace(
        add.tensor_program,
        source_target="rope",
        decomposition="rotate_half rotation: offset loads of x selected by the last-axis index",
        loads=tuple(leaf_loads),
        combine=combined,
    )
    operation_ids = {operation.id for operation in operations}
    rejections: list[str] = []
    if any(has_illegal_effects(operation) for operation in operations):
        rejections.append("source effects do not fit a fused RoPE body")
    if any(
        any(user not in operation_ids for user in graph.values[operation.results[0]].users)
        for operation in operations[:-1]
    ):
        rejections.append("a RoPE intermediate value has an external use")
    ordered = tuple(operation for operation in graph.ops.values() if operation.id in operation_ids)
    return RegionCandidate(
        id=f"region:{add.id.removeprefix('op:')}:rope",
        source_op_ids=tuple(operation.id for operation in ordered),
        external_inputs=external_inputs_of(ordered, graph),
        live_outputs=live_outputs_of(ordered, graph),
        tensor_program=fused_program,
        semantics=Rope(output.shape[-1], output.shape[-1], "rotate_half"),
        cost_bytes=sum(
            value_bytes(operation.results[0], graph)
            for operation in ordered[:-1]
            if isinstance(operation.tensor_program, TensorProgram)
            and operation.tensor_program.store.owns_storage
        ),
        rejections=tuple(rejections),
    )


def _split_halves_candidate(concat: Op, graph: ComputeGraph) -> RegionCandidate | None:
    """Match ``cat(x1 * cos - x2 * sin, x2 * cos + x1 * sin)`` over the halves
    of the rotary width, with any further elements of x passed through."""
    program = concat.tensor_program
    if program is None or program.family != "concat" or len(concat.operands) != 2:
        return None
    if not isinstance(program, TensorProgram):
        return None
    concat_axis = next(
        (index for index, axis in enumerate(program.loads[0].axes) if isinstance(axis, AxisAffine)),
        None,
    )
    if concat_axis != len(graph.values[concat.results[0]].shape or ()) - 1:
        return None
    sub = producer_of(concat.operands[0], graph)
    add = producer_of(concat.operands[1], graph)
    if sub is None or add is None:
        return None
    if not (
        binary_operator(sub, "sub")
        and binary_operator(add, "add")
        and binary_operator(producer_of(sub.operands[0], graph), "mul")
        and binary_operator(producer_of(sub.operands[1], graph), "mul")
        and binary_operator(producer_of(add.operands[0], graph), "mul")
        and binary_operator(producer_of(add.operands[1], graph), "mul")
    ):
        return None

    roots: dict[str, int] = {}
    multiplies = (
        producer_of(sub.operands[0], graph),
        producer_of(sub.operands[1], graph),
        producer_of(add.operands[0], graph),
        producer_of(add.operands[1], graph),
    )
    for multiply in multiplies:
        if multiply is None or len(multiply.operands) != 2:
            return None
        for operand_id in multiply.operands:
            root = _view_root(operand_id, graph)
            if root is None:
                return None
            roots[root] = roots.get(root, 0) + 1
    if len(roots) != 3 or sorted(roots.values()) != [2, 2, 4]:
        return None
    x_id = next(value_id for value_id, count in roots.items() if count == 4)
    trig_ids = tuple(value_id for value_id, count in roots.items() if count == 2)
    cos_id, sin_id = trig_ids
    if any(
        (producer := producer_of(root_id, graph)) is not None
        and matches_target(producer, "prims.convert_element_type.default")
        for root_id in roots
    ):
        return None
    operations = _rope_operations(concat, (x_id, cos_id, sin_id), graph)
    if any("convert_element_type" in str(op.attributes.get("target")) for op in operations):
        return None

    output = graph.values[concat.results[0]]
    x = graph.values[x_id]
    cos = graph.values[cos_id]
    sin = graph.values[sin_id]
    if (
        output.shape is None
        or output.strides is None
        or x.shape != output.shape
        or cos.shape is None
        or sin.shape is None
        or len(cos.shape) != len(sin.shape)
        or cos.shape != sin.shape
        or cos.shape[-1] % 2
        or output.shape[-1] < cos.shape[-1]
        or output.dtype not in {"torch.float32", "torch.bfloat16"}
        or {x.dtype, cos.dtype, sin.dtype, output.dtype} != {output.dtype}
        or any(stride < 0 for stride in (*x.strides, *cos.strides, *sin.strides, *output.strides))
    ):
        return None
    rank = len(output.shape)
    rotary_width = cos.shape[-1]
    half_width = rotary_width // 2
    if any(cos.shape[index] != output.shape[index] for index in range(len(cos.shape) - 1)):
        return None

    output_axes = tuple(f"o{index}" for index in range(rank))
    final_axis = output_axes[-1]
    leading_axes = output_axes[:-1]
    # Each half reads x and the trig tables at its own source slice; derive
    # those last-axis starts instead of assuming one indexing convention.
    first_width = graph.values[concat.operands[0]].shape
    if first_width is None or first_width[-1] != half_width:
        return None
    load_ids: dict[tuple[str, int], str] = {}
    loads_list: list[IndexedLoad] = []

    def operand_load(operand_id: str, branch_start: int) -> Load | None:
        located = _last_axis_slice(operand_id, graph)
        if located is None:
            return None
        root_id, start = located
        root = graph.values[root_id]
        if root.shape is None or root.strides is None or len(root.shape) != rank:
            return None
        offset = start - branch_start
        key = (root_id, offset)
        if key not in load_ids:
            load_ids[key] = f"load:{len(loads_list)}"
            axis = final_axis if offset == 0 else AxisAffine((AxisTerm(final_axis),), offset=offset)
            loads_list.append(
                IndexedLoad(
                    load_ids[key],
                    root_id,
                    (*leading_axes, axis),
                    root.strides,
                    root.dtype or "",
                    root.shape,
                )
            )
        return Load(load_ids[key], root.dtype or "")

    branch_loads = []
    for multiply, branch_start in zip(multiplies, (0, 0, half_width, half_width)):
        operands = tuple(operand_load(operand_id, branch_start) for operand_id in multiply.operands)
        if any(operand is None for operand in operands):
            return None
        branch_loads.append(operands)
    passthrough = operand_load(x_id, 0)
    if passthrough is None:
        return None
    loads = tuple(loads_list)
    dtype = output.dtype or ""
    compute_dtype = "torch.float32" if dtype in {"torch.bfloat16", "torch.float16"} else dtype

    def source_binary(
        operator: str, lhs: ScalarExpression, rhs: ScalarExpression
    ) -> ScalarExpression:
        if compute_dtype == dtype:
            return Binary(operator, lhs, rhs, dtype)
        lhs = Cast(lhs, compute_dtype, "round_nearest_even")
        rhs = Cast(rhs, compute_dtype, "round_nearest_even")
        return Cast(
            Binary(operator, lhs, rhs, compute_dtype),
            dtype,
            "round_nearest_even",
        )

    axis_index = AxisIndex(final_axis, "torch.int64")
    products = [source_binary("mul", *operands) for operands in branch_loads]
    first = source_binary("sub", products[0], products[1])
    second = source_binary("add", products[2], products[3])
    expression = Where(
        Binary(
            "lt",
            axis_index,
            Constant(half_width, "torch.int64"),
            "torch.bool",
        ),
        first,
        Where(
            Binary(
                "lt",
                axis_index,
                Constant(rotary_width, "torch.int64"),
                "torch.bool",
            ),
            second,
            passthrough,
            dtype,
        ),
        dtype,
    )
    fused_program = TensorProgram(
        family="elementwise",
        source_target="rope",
        decomposition="paired half-width rotation with an optional pass-through tail",
        iteration_domains=tuple(
            IterationDomain(axis, size) for axis, size in zip(output_axes, output.shape)
        ),
        reduction_axes=(),
        loads=loads,
        init=None,
        combine=expression,
        post=None,
        accumulator_dtype=None,
        store=TensorStore(output.id, output.shape, output.dtype or "", output.strides, True),
        predicates=tuple(
            AxisPredicate(axis, size) for axis, size in zip(output_axes, output.shape)
        ),
    )
    operation_ids = {operation.id for operation in operations}
    rejections: list[str] = []
    if any(has_illegal_effects(operation) for operation in operations):
        rejections.append("source effects do not fit a fused RoPE body")
    if any(
        any(user not in operation_ids for user in graph.values[operation.results[0]].users)
        for operation in operations[:-1]
    ):
        rejections.append("a RoPE intermediate value has an external use")
    return RegionCandidate(
        id=f"region:{concat.id.removeprefix('op:')}:rope",
        source_op_ids=tuple(operation.id for operation in operations),
        external_inputs=external_inputs_of(tuple(operations), graph),
        live_outputs=live_outputs_of(tuple(operations), graph),
        tensor_program=fused_program,
        semantics=Rope(output.shape[-1], rotary_width, "split_halves"),
        cost_bytes=sum(
            value_bytes(operation.results[0], graph)
            for operation in operations[:-1]
            if isinstance(operation.tensor_program, TensorProgram)
            and operation.tensor_program.store.owns_storage
        ),
        rejections=tuple(rejections),
    )


def _rope_operations(
    concat: Op, root_ids: tuple[str, str, str], graph: ComputeGraph
) -> tuple[Op, ...]:
    operation_ids = {concat.id}
    frontier = list(concat.operands)
    roots = set(root_ids)
    while frontier:
        value_id = frontier.pop()
        if value_id in roots:
            continue
        producer = producer_of(value_id, graph)
        if producer is None or producer.id in operation_ids:
            continue
        operation_ids.add(producer.id)
        frontier.extend(producer.operands)
    # Aliasing views cost nothing and need no region; leaving them out lets
    # several RoPE regions read one shared view (of merged cosines, say).
    return tuple(
        candidate
        for candidate in graph.ops.values()
        if candidate.id in operation_ids
        and (candidate is concat or producing_view(candidate.results[0], graph) is None)
    )


def _last_axis_slice(value_id: str, graph: ComputeGraph) -> tuple[str, int] | None:
    """Return the view root and start offset of a unit-step last-axis slice."""
    start = 0
    while True:
        producer = producer_of(value_id, graph)
        if producer is None:
            return value_id, start
        if matches_target(producer, "aten.unsqueeze.default"):
            source = graph.values[producer.operands[0]]
            arguments = producer.attributes.get("fx_args", ())
            dimension = arguments[1] if len(arguments) > 1 else None
            if (
                source.shape is None
                or not isinstance(dimension, int)
                or dimension % (len(source.shape) + 1) == len(source.shape)
            ):
                return None
        elif matches_target(producer, "aten.slice.Tensor"):
            source = graph.values[producer.operands[0]]
            arguments = tuple(producer.attributes.get("fx_args", ()))
            dimension, slice_start, _, step = (
                *arguments[1:5],
                *(0, 0, None, 1)[len(arguments[1:5]) :],
            )
            slice_start = 0 if slice_start is None else slice_start
            if (
                source.shape is None
                or not isinstance(dimension, int)
                or not isinstance(slice_start, int)
                or step not in (None, 1)
            ):
                return None
            if dimension % len(source.shape) == len(source.shape) - 1:
                if slice_start < 0:
                    slice_start += source.shape[-1]
                start += max(0, min(slice_start, source.shape[-1]))
        else:
            return value_id, start
        value_id = producer.operands[0]


def _view_root(value_id: str, graph: ComputeGraph) -> str | None:
    while True:
        producer = producer_of(value_id, graph)
        if producer is None:
            return value_id
        if not matches_target(producer, "aten.slice.Tensor") and not matches_target(
            producer, "aten.unsqueeze.default"
        ):
            return value_id
        value_id = producer.operands[0]
