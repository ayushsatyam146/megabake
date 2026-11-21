"""Elementwise chains: a producer-to-consumer chain of elementwise ops (and
copies) with one layout, fused into one map whose expression composes theirs.

The chain's source casts are kept: each op's expression is substituted into
its consumer's, so the fused body rounds where the ops did.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..graph import ComputeGraph, Op, RegionCandidate
from ..ir import IndexedLoad, Load, ScalarExpression, TensorProgram
from .matching import (
    external_inputs_of,
    has_illegal_effects,
    substitute_loads,
    value_bytes,
)


@dataclass(frozen=True)
class ScalarChain:
    """A fused elementwise chain; its intermediates never reach memory."""

    # Bytes the unfused chain writes and reads back for its intermediates.
    saved_bytes: int


def candidates(
    graph: ComputeGraph, reserved: frozenset[str] = frozenset()
) -> tuple[RegionCandidate, ...]:
    """The maximal chain ending at each elementwise op: a chain inside a
    longer one is left out. ``reserved`` ops belong to other pattern
    candidates (norms, RoPE); a chain that absorbs one also comes in a
    variant that reads it as an external input, so the planner, which takes
    the largest non-overlapping candidates first, can still fuse the rest
    of the chain after the pattern claims its op."""
    chains = [
        candidate
        for operation in graph.ops.values()
        if (candidate := _chain_candidate(operation, graph)) is not None
    ]
    maximal = [
        candidate
        for candidate in chains
        if not any(
            set(candidate.source_op_ids) < set(other.source_op_ids) for other in chains
        )
    ]
    variants = [
        candidate
        for chain in maximal
        if set(chain.source_op_ids[:-1]) & reserved
        and (
            candidate := _chain_candidate(
                graph.ops[chain.source_op_ids[-1]], graph, reserved, suffix=":unreserved"
            )
        )
        is not None
    ]
    return (*maximal, *variants)


def _chain_candidate(
    tail: Op, graph: ComputeGraph, excluded: frozenset[str] = frozenset(), suffix: str = ""
) -> RegionCandidate | None:
    if not isinstance(tail.tensor_program, TensorProgram):
        return None
    if tail.tensor_program.family not in {"elementwise", "copy"}:
        return None

    operations = scalar_fusion_operations(tail, graph, excluded)
    if len(operations) < 2:
        return None

    operation_ids = {operation.id for operation in operations}
    rejections: list[str] = []
    if any(has_illegal_effects(operation) for operation in operations):
        rejections.append("source effects do not fit a vector-chain body")
    if any(
        operation.results[0] in graph.outputs
        or any(user not in operation_ids for user in graph.values[operation.results[0]].users)
        for operation in operations[:-1]
    ):
        rejections.append("an intermediate value has an external use")
    if not all(
        same_chain_layout(operation.results[0], tail.results[0], graph) for operation in operations
    ):
        rejections.append("source ownership or broadcast contract changes")

    fused_program = compose_scalar_chain(operations, graph)
    external_inputs = (
        tuple(load.value_id for load in fused_program.loads)
        if fused_program is not None
        else external_inputs_of(operations, graph)
    )
    saved_bytes = sum(value_bytes(operation.results[0], graph) * 2 for operation in operations[:-1])
    return RegionCandidate(
        id=f"region:{tail.id.removeprefix('op:')}:fused{suffix}",
        source_op_ids=tuple(operation.id for operation in operations),
        external_inputs=external_inputs,
        live_outputs=(tail.results[0],),
        tensor_program=fused_program,
        semantics=ScalarChain(saved_bytes),
        cost_bytes=saved_bytes,
        rejections=tuple(rejections),
    )


def scalar_fusion_operations(
    tail: Op, graph: ComputeGraph, excluded: frozenset[str] = frozenset()
) -> tuple[Op, ...]:
    """``tail`` and the elementwise producers that can fold into it, in
    producer-first order.

    A producer joins when its result has the tail's layout, is no graph
    output, and every use of it is inside the chain; anything else is an
    external input. Growing the chain only through such producers keeps it
    local: following every elementwise producer instead let a chain run the
    whole residual stream of a model (Gemma 3's residual adds connect every
    layer through elementwise ops), which then failed its layout and
    external-use checks as a whole, so nothing around it fused."""
    members: set[str] = {tail.id}
    changed = True
    while changed:
        changed = False
        for member_id in tuple(members):
            program = graph.ops[member_id].tensor_program
            if not isinstance(program, TensorProgram):
                continue
            for load in program.loads:
                producer_id = graph.values[load.value_id].producer
                if producer_id is None or producer_id in members or producer_id in excluded:
                    continue
                producer = graph.ops[producer_id]
                result = producer.results[0] if len(producer.results) == 1 else None
                if (
                    result is None
                    or not is_scalar_operation(producer)
                    or has_illegal_effects(producer)
                    or result in graph.outputs
                    or not same_chain_layout(result, tail.results[0], graph)
                    or not all(user in members for user in graph.values[result].users)
                ):
                    continue
                members.add(producer_id)
                changed = True

    operations: list[Op] = []
    visited: set[str] = set()

    def visit(operation: Op) -> None:
        if operation.id in visited:
            return
        visited.add(operation.id)
        program = operation.tensor_program
        if isinstance(program, TensorProgram):
            for load in program.loads:
                producer_id = graph.values[load.value_id].producer
                if producer_id in members:
                    visit(graph.ops[producer_id])
        operations.append(operation)

    visit(tail)
    return tuple(operations)


def is_scalar_operation(operation: Op) -> bool:
    program = operation.tensor_program
    return isinstance(program, TensorProgram) and program.family in {
        "elementwise",
        "copy",
    }


def compose_scalar_chain(chain: tuple[Op, ...], graph: ComputeGraph) -> TensorProgram | None:
    tail = chain[-1]
    tail_program = tail.tensor_program
    if not isinstance(tail_program, TensorProgram):
        return None

    leaf_loads: list[IndexedLoad] = []
    leaf_ids: dict[tuple, str] = {}
    expressions: dict[str, ScalarExpression] = {}
    for operation in chain:
        program = operation.tensor_program
        if not isinstance(program, TensorProgram):
            return None
        replacements: dict[str, ScalarExpression] = {}
        for load in program.loads:
            if load.value_id in expressions:
                replacements[load.id] = expressions[load.value_id]
                continue
            # A value read through different index maps (a broadcast and a
            # full read) needs one leaf load per map.
            leaf_key = (load.value_id, tuple(load.axes), tuple(load.strides))
            leaf_id = leaf_ids.get(leaf_key)
            if leaf_id is None:
                leaf_id = f"load:{len(leaf_loads)}"
                leaf_loads.append(
                    IndexedLoad(
                        id=leaf_id,
                        value_id=load.value_id,
                        axes=load.axes,
                        strides=load.strides,
                        dtype=load.dtype,
                        source_shape=load.source_shape,
                    )
                )
                leaf_ids[leaf_key] = leaf_id
            replacements[load.id] = Load(leaf_id, load.dtype)
        expressions[operation.results[0]] = substitute_loads(program.combine, replacements)

    return replace(
        tail_program,
        loads=tuple(leaf_loads),
        combine=expressions[tail.results[0]],
    )


def same_chain_layout(left_id: str, right_id: str, graph: ComputeGraph) -> bool:
    left = graph.values[left_id]
    right = graph.values[right_id]
    return left.shape == right.shape and left.strides == right.strides
