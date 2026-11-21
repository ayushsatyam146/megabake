from __future__ import annotations

import re

from contextvars import ContextVar
from dataclasses import dataclass, replace
from math import prod
from typing import Any, Mapping, Sequence

from ...graph import DTYPE_BYTES, Region
from ...ir import (
    Accumulator,
    AxisAffine,
    AxisIndex,
    Binary,
    Cast,
    Constant,
    DataIndex,
    IndexedLoad,
    Load,
    ScalarExpression,
    TensorProgram,
    Unary,
    Where,
)
from ...ops import GemmEpilogue, ScalarChain
from ..interface import (
    BodyConfig,
    CandidateSet,
    CompletionContract,
    LayoutContract,
    ParticipantContract,
    PipelineContract,
    ProviderRejection,
    ResourceEvidence,
    ScratchContract,
    SelectionPolicy,
    SemanticContract,
    TargetFacts,
    TensorLayout,
    TileContract,
    access_effects,
    validate_body_config,
    widened_participants,
)


@dataclass(frozen=True)
class GeneratedMapBody:
    """A device-callable map body and its required work-iterator encoding."""

    source: str
    function_name: str = "run"
    threads_per_cta: int = 128
    tile_elements: int = 128
    tile_count: int = 1
    # Work-iterator stride between tiles (``GeneratedBody.work_step``).
    work_step: int = 1
    work_iterator_dtype: str = "torch.int64"
    work_iterator_encoding: str = "one flat output-tile base index per tile"
    arguments: tuple[str, ...] = (
        "work_iterator",
        "tensor_bindings",
        "scratch",
        "epilogue",
    )


@dataclass(frozen=True)
class GeneratedReductionBody:
    """A device-callable reduction body and its work-iterator encoding."""

    source: str
    function_name: str = "run"
    threads_per_cta: int = 128
    reduction_tile: int = 128
    tile_count: int = 1
    # Work-iterator stride between tiles (``GeneratedBody.work_step``).
    work_step: int = 1
    scratch_elements: int = 4
    work_iterator_dtype: str = "torch.int64"
    work_iterator_encoding: str = "one flat output index per CTA"
    scratch_encoding: str = "scratch[block, warp] holds one accumulator partial per warp"
    phases: tuple[str, ...] = ("thread_partial", "warp_partial", "cta_merge", "epilogue")
    arguments: tuple[str, ...] = (
        "work_iterator",
        "tensor_bindings",
        "scratch",
        "epilogue",
    )


@dataclass(frozen=True)
class GeneratedContractionBody:
    """A device-callable baseline contraction and its work-iterator encoding."""

    source: str
    function_name: str = "run"
    threads_per_cta: int = 128
    output_tile_elements: int = 128
    reduction_chunk: int = 32
    tile_count: int = 1
    # Work-iterator stride between tiles (``GeneratedBody.work_step``).
    work_step: int = 1
    work_iterator_dtype: str = "torch.int64"
    work_iterator_encoding: str = (
        "one flat output-tile base index per CTA; thread adds its lane"
    )
    phases: tuple[str, ...] = (
        "output_tile",
        "bounded_reduction",
        "epilogue",
    )
    arguments: tuple[str, ...] = (
        "work_iterator",
        "tensor_bindings",
        "scratch",
        "epilogue",
    )


@dataclass(frozen=True)
class _TypedExpression:
    code: str
    dtype: str


class UnsupportedMapBody(ValueError):
    pass


UnsupportedReductionBody = UnsupportedMapBody
UnsupportedContractionBody = UnsupportedMapBody


_HOISTED_LOADS: ContextVar[Mapping[str, str]] = ContextVar(
    "megabake_hoisted_loads", default={}
)

_CUTLASS_TYPES = {
    "torch.bool": "cutlass.Boolean",
    "torch.int8": "cutlass.Int8",
    "torch.int16": "cutlass.Int16",
    "torch.int32": "cutlass.Int32",
    "torch.int64": "cutlass.Int64",
    "torch.float16": "cutlass.Float16",
    "torch.bfloat16": "cutlass.BFloat16",
    "torch.float32": "cutlass.Float32",
    "torch.float64": "cutlass.Float64",
}

_BINARY_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "bitwise_and": "&",
    # Integer coordinates only; float remainder follows Python and differs.
    "remainder": "%",
    "bitwise_or": "|",
    "bitwise_xor": "^",
    # Logical operators are lowered for boolean operands only.
    "logical_and": "&",
    "logical_or": "|",
    "logical_xor": "^",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}
_ARITHMETIC = frozenset({"add", "sub", "mul", "div"})
_HALF_DTYPES = frozenset({"torch.bfloat16", "torch.float16"})


def emit_map_body(program: TensorProgram, config: BodyConfig) -> GeneratedMapBody:
    _validate_map_body(program, config)

    threads = config.participants.participating_threads
    axis_coordinates = {
        domain.axis: f"o{index}" for index, domain in enumerate(program.iteration_domains)
    }
    input_value_ids = tuple(dict.fromkeys(load.value_id for load in program.loads))
    binding_indices = {value_id: index for index, value_id in enumerate(input_value_ids)}
    load_indices = {load.id: binding_indices[load.value_id] for load in program.loads}
    # Bind each load once so deep expressions stay within the DSL
    # preprocessor's recursion limit.
    hoisted_names = {load.id: f"load_{index}" for index, load in enumerate(program.loads)}
    index_load_ids = {
        axis.load_id
        for load in program.loads
        for axis in load.axes
        if isinstance(axis, DataIndex)
    }
    ordered_loads = sorted(
        program.loads, key=lambda load: load.id not in index_load_ids
    )
    # Each thread covers MAP_ELEMENTS_PER_THREAD elements of a tile, strided by
    # the CTA size. Every element's loads are issued before any store, so a
    # tile costs one round of memory latency; a store between them would
    # serialize the loads (the pointers may alias). Coordinates wrap modulo
    # each axis, so an element past the end loads in bounds; only its store
    # is guarded.
    elements = MAP_ELEMENTS_PER_THREAD
    tile_elements = threads * elements
    load_lines = ""
    value_lines = ""
    store_lines = ""
    for element in range(elements):
        names = {axis: f"{name}_{element}" for axis, name in axis_coordinates.items()}
        hoisted = {load_id: f"{name}_{element}" for load_id, name in hoisted_names.items()}
        index = f"linear_{element}"
        load_lines += f"        {index} = base + {element * threads}\n"
        load_lines += f"        remainder_{element} = {index}\n"
        coordinates = _coordinate_lines(
            program.store.shape, program.store.strides, f"remainder_{element}"
        )
        load_lines += re.sub(r"\bo(\d+)\b", rf"o\1_{element}", coordinates).replace(
            "            ", "        "
        )
        for load in ordered_loads:
            if load.id not in index_load_ids:
                load_lines += _gather_bounds_lines(load, hoisted).replace(
                    "            ", "        ", 1
                ).replace("\n            ", "\n        ")
            load_lines += (
                f"        {hoisted[load.id]} = "
                f"{_load_access(load, load_indices[load.id], program.loads, names).code}\n"
            )
        token = _HOISTED_LOADS.set(hoisted)
        try:
            value = _emit_expression(program.combine, load_indices, names, program.loads)
        finally:
            _HOISTED_LOADS.reset(token)
        value = _cast_expression(value, program.store.dtype)
        value_lines += f"        value_{element} = {value.code}\n"
        output_index = ", ".join(
            names[domain.axis] for domain in program.iteration_domains
        ) or "0"
        store_lines += (
            f"        if {index} < output_count:\n"
            f"            output[{output_index}] = value_{element}\n"
        )

    binding_type = ", ".join(
        "cute.Tensor" for _ in range(len(input_value_ids) + 1)
    )
    trap_import = (
        "from megabake.runtime import trap_if\n" if index_load_ids else ""
    )
    source = f'''import cutlass
import cutlass.cute as cute
{trap_import}

@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    output = tensor_bindings[{len(input_value_ids)}]
    output_count = cute.size(output)
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        base = work_iterator[tile] + thread
{load_lines}{value_lines}{store_lines}'''
    element_count = 1
    for size in program.store.shape:
        element_count *= size
    tile_count = (element_count + tile_elements - 1) // tile_elements
    return GeneratedMapBody(
        source=source,
        threads_per_cta=threads,
        tile_elements=tile_elements,
        tile_count=tile_count,
        work_step=tile_elements,
    )


def emit_reduction_body(
    program: TensorProgram, config: BodyConfig
) -> GeneratedReductionBody:
    _validate_reduction_body(program, config)

    threads = config.participants.participating_threads
    warps = threads // 32
    output_domains = [
        domain
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    ]
    reduction_domain = next(
        domain for domain in program.iteration_domains if domain.axis in program.reduction_axes
    )
    axis_coordinates = {
        domain.axis: f"o{index}" for index, domain in enumerate(output_domains)
    }
    reduction_coordinate = f"r{program.reduction_axes.index(reduction_domain.axis)}"
    input_value_ids = tuple(dict.fromkeys(load.value_id for load in program.loads))
    binding_indices = {value_id: index for index, value_id in enumerate(input_value_ids)}
    load_indices = {load.id: binding_indices[load.value_id] for load in program.loads}

    accumulator = _emit_expression(program.init, load_indices, axis_coordinates, program.loads)
    combine = _emit_expression(
        program.combine,
        load_indices,
        {**axis_coordinates, reduction_domain.axis: reduction_coordinate},
        program.loads,
    )
    assert program.post is not None
    post = _emit_expression(program.post, load_indices, axis_coordinates, program.loads)

    coordinate_lines = _coordinate_lines(
        program.store.shape, program.store.strides, "output_index"
    )
    output_coordinates = ", ".join(
        axis_coordinates[domain.axis] for domain in output_domains
    )
    binding_type = ", ".join(
        "cute.Tensor" for _ in range(len(input_value_ids) + 1)
    )
    accumulator_type = _cutlass_type(program.accumulator_dtype or "")
    reduction_operator = _ordered_reduction_operator(program.combine)
    # Output axes that only the post-reduction expression reads (the feature
    # axis of a norm) share one row reduction: reduce each row once, then let
    # every thread write that row's outputs.
    reduced_axes = _expression_axes(program.combine, program.loads)
    if reduced_axes is not None:
        reduced_axes |= _expression_axes(program.init, program.loads) or set()
    row_domains = [
        domain
        for domain in output_domains
        if reduced_axes is None or domain.axis in reduced_axes
    ]
    post_domains = [domain for domain in output_domains if domain not in row_domains]
    warp_reduction = (
        "cute.arch.warp_reduction(accumulator, lambda left, right: cute.arch.fmax(left, right, nan=True))"
        if reduction_operator == "maximum"
        else "cute.arch.warp_reduction_sum(accumulator)"
    )
    merge_partial = (
        "accumulator = cute.arch.fmax(accumulator, scratch[block, warp_index], nan=True)"
        if reduction_operator == "maximum"
        else "accumulator = accumulator + scratch[block, warp_index]"
    )
    if post_domains:
        return _emit_row_reduction_body(
            program,
            threads,
            row_domains,
            post_domains,
            axis_coordinates,
            reduction_coordinate,
            reduction_domain,
            input_value_ids,
            load_indices,
            accumulator,
            combine,
            reduction_operator,
        )
    source = f'''import cutlass
import cutlass.cute as cute


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch: cute.Tensor,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    warp = thread // 32
    lane = thread % 32
    warps = {warps}
    output = tensor_bindings[{len(input_value_ids)}]
    output_count = cute.size(output)
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        output_index = work_iterator[tile]
        if output_index < output_count:
{coordinate_lines}            accumulator = {accumulator_type}({accumulator.code})
            for reduction_index in {_strided_range("thread", reduction_domain.size, threads)}:
                {reduction_coordinate} = reduction_index
                accumulator = {combine.code}
            warp_partial = {warp_reduction}
            if lane == 0:
                scratch[block, warp] = warp_partial
            cute.arch.sync_threads()
            accumulator = {accumulator_type}({accumulator.code})
            for warp_index in cutlass.range(warps):
                {merge_partial}
            value = {post.code}
            if thread == 0:
                output[{output_coordinates}] = value
            cute.arch.sync_threads()
'''
    output_count = 1
    for size in program.store.shape:
        output_count *= size
    return GeneratedReductionBody(
        source=source,
        threads_per_cta=threads,
        reduction_tile=threads,
        tile_count=output_count,
        scratch_elements=warps,
    )


def _expression_axes(expression: Any, loads: Sequence[IndexedLoad]) -> set[str] | None:
    """Iteration axes an expression reads; None when an index is data dependent."""
    by_id = {load.id: load for load in loads}
    axes: set[str] = set()
    pending = [expression]
    while pending:
        node = pending.pop()
        if node is None or isinstance(node, (Constant, Accumulator)):
            continue
        if isinstance(node, Load):
            for axis in by_id[node.load_id].axes:
                if isinstance(axis, str):
                    axes.add(axis)
                elif isinstance(axis, AxisAffine):
                    axes.update(term.axis for term in axis.terms)
                elif not isinstance(axis, int):
                    return None
        elif isinstance(node, AxisIndex):
            axes.add(node.axis)
        elif isinstance(node, (Unary, Cast)):
            pending.append(node.operand)
        elif isinstance(node, Binary):
            pending.extend((node.lhs, node.rhs))
        elif isinstance(node, Where):
            pending.extend((node.condition, node.true_value, node.false_value))
        else:
            return None
    return axes


# Elements each thread of a map body covers per tile (see emit_map_body).
MAP_ELEMENTS_PER_THREAD = 4


def reduced_row_axes(program: TensorProgram) -> set[str] | None:
    """Output axes a reduction's combine and init read: one reduction runs
    per coordinate of these. The other output axes only take the post
    expression. None when an index is data dependent."""
    axes = _expression_axes(program.combine, program.loads)
    if axes is None:
        return None
    return axes | (_expression_axes(program.init, program.loads) or set())


def single_cta_rounds(program: TensorProgram, threads: int) -> int:
    """Rounds of dependent memory latency one CTA spends running every tile
    of a map or reduction body: map tiles, or reduced rows."""
    elements = prod(program.store.shape)
    if program.family != "reduction":
        return -(-elements // (threads * MAP_ELEMENTS_PER_THREAD))
    output_domains = [
        domain
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    ]
    reduced_axes = reduced_row_axes(program)
    row_domains = [
        domain
        for domain in output_domains
        if reduced_axes is None or domain.axis in reduced_axes
    ]
    if len(row_domains) == len(output_domains):
        return elements
    return prod(domain.size for domain in row_domains)


# Strided per-thread loops with at most this many iterations unroll fully, so
# their independent loads are in flight together instead of one per trip.
_UNROLL_ITERATIONS = 16


def _strided_range(start: str, size: int, threads: int) -> str:
    """``cutlass.range`` over ``start, start + threads, ...`` below ``size``."""
    unroll = ", unroll_full=True" if -(-size // threads) <= _UNROLL_ITERATIONS else ""
    return f"cutlass.range({start}, {size}, {threads}{unroll})"


def _row_major_lines(domains: Sequence[Any], names: Mapping[str, str], value: str, indent: str) -> str:
    # Each coordinate comes straight from the index; a remainder variable
    # reassigned once per axis sends the DSL preprocessor into unbounded
    # recursion at three axes.
    lines = []
    stride = 1
    for domain in reversed(domains):
        lines.append(f"{indent}{names[domain.axis]} = ({value} // {stride}) % {domain.size}")
        stride *= domain.size
    return "\n".join(lines) + "\n"


def _emit_row_reduction_body(
    program: TensorProgram,
    threads: int,
    row_domains: Sequence[Any],
    post_domains: Sequence[Any],
    axis_coordinates: Mapping[str, str],
    reduction_coordinate: str,
    reduction_domain: Any,
    input_value_ids: Sequence[str],
    load_indices: Mapping[str, int],
    accumulator: Any,
    combine: Any,
    reduction_operator: str,
) -> GeneratedReductionBody:
    warps = threads // 32
    # Bind each post-reduction load once, as map bodies do: several bounds-
    # checked loads in one statement inside these loops exceed the DSL
    # preprocessor's recursion limit.
    post_load_ids = []
    pending = [program.post]
    while pending:
        node = pending.pop()
        if isinstance(node, Load):
            post_load_ids.append(node.load_id)
        elif isinstance(node, (Unary, Cast)):
            pending.append(node.operand)
        elif isinstance(node, Binary):
            pending.extend((node.lhs, node.rhs))
        elif isinstance(node, Where):
            pending.extend((node.condition, node.true_value, node.false_value))
    hoisted_names = {
        load_id: f"post_load_{index}"
        for index, load_id in enumerate(dict.fromkeys(post_load_ids))
    }
    load_lines = "".join(
        f"                {name} = "
        f"{_load_access(load, load_indices[load.id], program.loads, axis_coordinates).code}\n"
        for load in program.loads
        for name in (hoisted_names.get(load.id),)
        if name is not None
    )
    token = _HOISTED_LOADS.set(hoisted_names)
    try:
        post = _emit_expression(program.post, load_indices, axis_coordinates, program.loads)
    finally:
        _HOISTED_LOADS.reset(token)
    row_count = prod(domain.size for domain in row_domains)
    post_count = prod(domain.size for domain in post_domains)
    output_coordinates = ", ".join(
        axis_coordinates[domain.axis]
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    )
    post_steps = -(-post_count // threads)
    if post_steps <= _UNROLL_ITERATIONS:
        # Straight-line output loop: every step's loads first, then every
        # step's stores, so the loads are in flight together; a store before
        # a later load would serialize them (the pointers may alias).
        post_loop = _staged_post_lines(
            program, threads, post_steps, post_count, post_domains, axis_coordinates,
            hoisted_names, load_indices,
        )
    else:
        post_loop = (
            f"            for post_index in {_strided_range('thread', post_count, threads)}:\n"
            f"{_row_major_lines(post_domains, axis_coordinates, 'post_index', '                ')}"
            f"{load_lines}"
            f"                output[{output_coordinates}] = {post.code}\n"
        )
    binding_type = ", ".join("cute.Tensor" for _ in range(len(input_value_ids) + 1))
    accumulator_type = _cutlass_type(program.accumulator_dtype or "")
    warp_reduction = (
        "cute.arch.warp_reduction(accumulator, lambda left, right: cute.arch.fmax(left, right, nan=True))"
        if reduction_operator == "maximum"
        else "cute.arch.warp_reduction_sum(accumulator)"
    )
    merge_partial = (
        "accumulator = cute.arch.fmax(accumulator, scratch[block, warp_index], nan=True)"
        if reduction_operator == "maximum"
        else "accumulator = accumulator + scratch[block, warp_index]"
    )
    source = f'''import cutlass
import cutlass.cute as cute


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch: cute.Tensor,
    epilogue,
):
    # One CTA tile per reduced row: the reduction runs once and every thread
    # then evaluates the post-reduction expression for that row's outputs.
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    warp = thread // 32
    lane = thread % 32
    warps = {warps}
    output = tensor_bindings[{len(input_value_ids)}]
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        row_index = work_iterator[tile]
        if row_index < {row_count}:
{_row_major_lines(row_domains, axis_coordinates, "row_index", "            ")}            accumulator = {accumulator_type}({accumulator.code})
            for reduction_index in {_strided_range("thread", reduction_domain.size, threads)}:
                {reduction_coordinate} = reduction_index
                accumulator = {combine.code}
            warp_partial = {warp_reduction}
            if lane == 0:
                scratch[block, warp] = warp_partial
            cute.arch.sync_threads()
            accumulator = {accumulator_type}({accumulator.code})
            for warp_index in cutlass.range(warps):
                {merge_partial}
{post_loop}            cute.arch.sync_threads()
'''
    return GeneratedReductionBody(
        source=source,
        threads_per_cta=threads,
        reduction_tile=threads,
        tile_count=row_count,
        scratch_elements=warps,
    )


def _staged_post_lines(
    program: TensorProgram,
    threads: int,
    steps: int,
    post_count: int,
    post_domains: Sequence[Any],
    axis_coordinates: Mapping[str, str],
    hoisted_names: Mapping[str, str],
    load_indices: Mapping[str, int],
) -> str:
    """The row reduction's output loop, unrolled at generation time.

    Step ``s`` covers ``thread + s * threads``. Coordinates wrap modulo each
    axis, so the loads of a step past the end stay in bounds; only its store
    is guarded.
    """
    loads_by_id = {load.id: load for load in program.loads}
    output_axes = [
        domain.axis
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    ]
    post_axes = {domain.axis for domain in post_domains}
    load_lines: list[str] = []
    store_lines: list[str] = []
    for step in range(steps):
        names = {
            axis: f"{name}_{step}" if axis in post_axes else name
            for axis, name in axis_coordinates.items()
        }
        index = f"post_index_{step}"
        load_lines.append(f"            {index} = thread + {step * threads}\n")
        load_lines.append(_row_major_lines(post_domains, names, index, "            "))
        step_names = {load_id: f"{name}_{step}" for load_id, name in hoisted_names.items()}
        for load_id, name in step_names.items():
            access = _load_access(
                loads_by_id[load_id], load_indices[load_id], program.loads, names
            ).code
            load_lines.append(f"            {name} = {access}\n")
        token = _HOISTED_LOADS.set(step_names)
        try:
            value = _emit_expression(program.post, load_indices, names, program.loads)
        finally:
            _HOISTED_LOADS.reset(token)
        coordinates = ", ".join(names[axis] for axis in output_axes)
        guard = (step + 1) * threads > post_count
        if guard:
            store_lines.append(f"            if {index} < {post_count}:\n")
            store_lines.append(f"                output[{coordinates}] = {value.code}\n")
        else:
            store_lines.append(f"            output[{coordinates}] = {value.code}\n")
    return "".join(load_lines) + "".join(store_lines)


def emit_contraction_body(
    program: TensorProgram, config: BodyConfig
) -> GeneratedContractionBody:
    _validate_contraction_body(program, config)

    threads = config.participants.participating_threads
    reduction_chunk = config.tiles.reduction_tile[0]
    output_domains = [
        domain
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    ]
    reduction_domain = next(
        domain
        for domain in program.iteration_domains
        if domain.axis in program.reduction_axes
    )
    axis_coordinates = {
        domain.axis: f"o{index}" for index, domain in enumerate(output_domains)
    }
    reduction_coordinate = f"r{program.reduction_axes.index(reduction_domain.axis)}"
    load_indices = {load.id: index for index, load in enumerate(program.loads)}

    accumulator = _emit_expression(
        program.init, load_indices, axis_coordinates, program.loads
    )
    combine = _emit_expression(
        program.combine,
        load_indices,
        {**axis_coordinates, reduction_domain.axis: reduction_coordinate},
        program.loads,
    )
    assert program.post is not None
    post = _emit_expression(
        program.post, load_indices, axis_coordinates, program.loads
    )

    coordinate_lines = _coordinate_lines(
        program.store.shape, program.store.strides
    )
    output_coordinates = ", ".join(
        axis_coordinates[domain.axis] for domain in output_domains
    )
    binding_type = ", ".join(
        "cute.Tensor" for _ in range(len(program.loads) + 1)
    )
    accumulator_type = _cutlass_type(program.accumulator_dtype or "")
    output_cutlass_type = _cutlass_type(program.store.dtype)
    source = f'''import cutlass
import cutlass.cute as cute


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    output = tensor_bindings[{len(program.loads)}]
    output_count = cute.size(output)
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        linear_index = work_iterator[tile] + thread
        if linear_index < output_count:
{coordinate_lines}            accumulator = {accumulator_type}({accumulator.code})
            for reduction_base in cutlass.range(0, {reduction_domain.size}, {reduction_chunk}):
                for reduction_offset in cutlass.range({reduction_chunk}):
                    reduction_index = reduction_base + reduction_offset
                    if reduction_index < {reduction_domain.size}:
                        {reduction_coordinate} = reduction_index
                        accumulator = {combine.code}
            stored_value = {output_cutlass_type}({post.code})
            output[{output_coordinates}] = stored_value
'''
    element_count = 1
    for size in program.store.shape:
        element_count *= size
    tile_count = (element_count + threads - 1) // threads
    return GeneratedContractionBody(
        source=source,
        threads_per_cta=threads,
        output_tile_elements=threads,
        reduction_chunk=reduction_chunk,
        tile_count=tile_count,
        work_step=threads,
    )


def _validate_map_body(program: TensorProgram, config: BodyConfig) -> None:
    validate_body_config(config)
    if program.family not in {"elementwise", "copy", "gather", "concat"}:
        raise UnsupportedMapBody(
            f"map body generator supports elementwise, copy, and gather, not {program.family}"
        )
    if config.semantic.operation_family != program.family:
        raise UnsupportedMapBody("body config and tensor program families differ")
    if program.reduction_axes or program.init is not None or program.post is not None:
        raise UnsupportedMapBody("map body generator does not support reduction state")
    if program.accumulator_dtype is not None:
        raise UnsupportedMapBody("map body generator does not support an accumulator")

    threads = config.participants.participating_threads
    if threads <= 0:
        raise UnsupportedMapBody("map body requires participating threads")
    if config.tiles.cta_shape != (threads,) + (1,) * (
        len(program.iteration_domains) - 1
    ):
        raise UnsupportedMapBody("map body requires a flat one-element-per-thread tile")
    if config.tiles.work_partition != "one output element per thread":
        raise UnsupportedMapBody("map body requires one output element per thread")
    if config.scratch.owner != "none" or config.scratch.size_bytes != 0:
        raise UnsupportedMapBody("map body does not use scratch")

    if any(size <= 0 for size in program.store.shape):
        raise UnsupportedMapBody("map body requires positive output extents")
    if any(stride < 0 for stride in program.store.strides):
        raise UnsupportedMapBody("negative output strides are not supported")

    _validate_body_layouts(program, config, program.iteration_domains)


def _validate_reduction_body(program: TensorProgram, config: BodyConfig) -> None:
    validate_body_config(config)
    # shortcut: one reduction axis and ordered addition or maximum only
    if program.family != "reduction":
        raise UnsupportedReductionBody(
            f"reduction body generator supports reduction, not {program.family}"
        )
    if config.semantic.operation_family != program.family:
        raise UnsupportedReductionBody("body config and tensor program families differ")
    if len(program.reduction_axes) != 1:
        raise UnsupportedReductionBody(
            "reduction body generator supports one reduction axis"
        )
    if program.init is None or program.post is None or program.accumulator_dtype is None:
        raise UnsupportedReductionBody("reduction body requires complete reduction state")
    if _ordered_reduction_operator(program.combine) not in {"add", "maximum"}:
        raise UnsupportedReductionBody(
            "reduction body generator supports ordered addition or maximum"
        )
    if (
        _ordered_reduction_operator(program.combine) == "maximum"
        and program.accumulator_dtype != "torch.float32"
    ):
        raise UnsupportedReductionBody("maximum reduction requires a float32 accumulator")
    if program.init.dtype != program.accumulator_dtype:
        raise UnsupportedReductionBody("initializer and accumulator dtypes differ")

    threads = config.participants.participating_threads
    warps = threads // 32
    if threads <= 0 or threads % 32:
        raise UnsupportedReductionBody("reduction body requires whole warps")
    if config.tiles.cta_shape != (threads,) + (1,) * (
        len(program.iteration_domains) - len(program.reduction_axes) - 1
    ):
        raise UnsupportedReductionBody("reduction body requires one CTA per output row")
    if config.tiles.work_partition != "one output row per CTA":
        raise UnsupportedReductionBody("reduction body requires one output row per CTA")
    if config.tiles.reduction_tile != (threads,):
        raise UnsupportedReductionBody("reduction body requires one reduction strip per thread")
    if config.scratch.owner != "cta" or config.scratch.size_bytes < warps * _dtype_bytes(
        program.accumulator_dtype
    ):
        raise UnsupportedReductionBody("reduction body lacks warp-partial scratch")

    output_domains = tuple(
        domain
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    )
    if not output_domains or any(domain.size <= 0 for domain in output_domains):
        raise UnsupportedReductionBody("reduction body requires a nonempty output row")
    if any(domain.size <= 0 for domain in program.iteration_domains):
        raise UnsupportedReductionBody("reduction body requires a nonempty reduction extent")
    if any(stride < 0 for stride in program.store.strides):
        raise UnsupportedReductionBody("negative output strides are not supported")

    _validate_body_layouts(program, config, output_domains)


def _validate_contraction_body(program: TensorProgram, config: BodyConfig) -> None:
    validate_body_config(config)
    if program.family != "contraction":
        raise UnsupportedContractionBody(
            f"contraction body generator supports contraction, not {program.family}"
        )
    if config.semantic.operation_family != program.family:
        raise UnsupportedContractionBody(
            "body config and tensor program families differ"
        )
    if len(program.reduction_axes) != 1:
        raise UnsupportedContractionBody(
            "contraction body generator supports one reduction axis"
        )
    if program.init is None or program.post is None or program.accumulator_dtype is None:
        raise UnsupportedContractionBody(
            "contraction body requires complete reduction state"
        )
    if not _is_add_reduction(program.combine):
        raise UnsupportedContractionBody(
            "contraction body generator supports ordered multiply-accumulate"
        )
    if program.init.dtype != program.accumulator_dtype:
        raise UnsupportedContractionBody(
            "initializer and accumulator dtypes differ"
        )

    threads = config.participants.participating_threads
    if threads <= 0:
        raise UnsupportedContractionBody("contraction body requires participating threads")
    output_domains = tuple(
        domain
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    )
    if config.tiles.cta_shape != (threads,) + (1,) * (len(output_domains) - 1):
        raise UnsupportedContractionBody(
            "contraction body requires a flat one-output-element-per-thread tile"
        )
    if config.tiles.work_partition != "one output element per thread":
        raise UnsupportedContractionBody(
            "contraction body requires one output element per thread"
        )
    if (
        not config.tiles.reduction_tile
        or config.tiles.reduction_tile[0] <= 0
        or len(config.tiles.reduction_tile) != 1
    ):
        raise UnsupportedContractionBody(
            "contraction body requires one positive reduction chunk"
        )
    if config.scratch.owner != "none" or config.scratch.size_bytes != 0:
        raise UnsupportedContractionBody("contraction body does not use scratch")

    if not program.store.shape or any(size <= 0 for size in program.store.shape):
        raise UnsupportedContractionBody(
            "contraction body requires a nonempty positive-rank output"
        )
    if any(domain.size <= 0 for domain in program.iteration_domains):
        raise UnsupportedContractionBody(
            "contraction body requires a nonempty positive reduction extent"
        )
    if any(stride < 0 for stride in program.store.strides):
        raise UnsupportedContractionBody(
            "negative output strides are not supported"
        )

    _validate_body_layouts(program, config, output_domains)


def _validate_body_layouts(
    program: TensorProgram,
    config: BodyConfig,
    output_domains: tuple,
) -> None:
    expected_input_ids = {load.value_id for load in program.loads}
    if set(config.layouts.inputs) != expected_input_ids:
        raise UnsupportedMapBody("body input layouts do not match tensor program loads")
    if set(config.layouts.outputs) != {program.store.value_id}:
        raise UnsupportedMapBody("body output layouts do not match tensor program store")

    output_layout = config.layouts.outputs[program.store.value_id]
    if (
        output_layout.axes != tuple(domain.axis for domain in output_domains)
        or output_layout.strides != tuple(program.store.strides)
        or output_layout.dtype != program.store.dtype
    ):
        raise UnsupportedMapBody("body output layout does not match tensor program")

    loads_by_value: dict[str, list[Any]] = {}
    for load in program.loads:
        loads_by_value.setdefault(load.value_id, []).append(load)
    for load in program.loads:
        layout = config.layouts.inputs[load.value_id]
        if (
            layout.strides != tuple(load.strides)
            or layout.dtype != load.dtype
        ):
            raise UnsupportedMapBody("body input layout does not match tensor program")
        primary_load = loads_by_value[load.value_id][0]
        if load is primary_load and layout.axes != primary_load.axes:
            raise UnsupportedMapBody("body input layout does not match tensor program")
        if any(
            other.source_shape != load.source_shape
            or other.strides != load.strides
            or other.dtype != load.dtype
            for other in loads_by_value[load.value_id]
        ):
            raise UnsupportedMapBody(
                "repeated loads of one value must share shape, strides, and dtype"
            )
        if any(stride < 0 for stride in load.strides):
            raise UnsupportedMapBody("negative input strides are not supported")


def _is_add_reduction(expression: ScalarExpression) -> bool:
    if not isinstance(expression, Binary) or expression.operator != "add":
        return False
    left = _expression_has_accumulator(expression.lhs)
    right = _expression_has_accumulator(expression.rhs)
    return left != right


def _ordered_reduction_operator(expression: ScalarExpression) -> str | None:
    if not isinstance(expression, Binary):
        return None
    left = _expression_has_accumulator(expression.lhs)
    right = _expression_has_accumulator(expression.rhs)
    return expression.operator if left != right else None


def _maximum_code(left: str, right: str) -> str:
    return f"cute.arch.fmax({left}, {right}, nan=True)"


def _constant_code(value: int | float | bool) -> str:
    if isinstance(value, float) and value == float("-inf"):
        return 'float("-inf")'
    return repr(value)


def _expression_has_accumulator(expression: ScalarExpression) -> bool:
    if isinstance(expression, Accumulator):
        return True
    if isinstance(expression, Binary):
        return _expression_has_accumulator(expression.lhs) or _expression_has_accumulator(
            expression.rhs
        )
    if isinstance(expression, Unary):
        return _expression_has_accumulator(expression.operand)
    if isinstance(expression, Cast):
        return _expression_has_accumulator(expression.operand)
    if isinstance(expression, Where):
        return any(
            _expression_has_accumulator(value)
            for value in (expression.condition, expression.true_value, expression.false_value)
        )
    return False


def _coordinate_lines(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    value: str = "linear_index",
) -> str:
    if len(shape) != len(strides):
        raise UnsupportedMapBody("output shape and strides do not match")
    lines: list[str] = []
    remainder = value
    axes = sorted(
        range(len(shape)),
        key=lambda index: (strides[index] == 0, strides[index], index),
    )
    for position, index in enumerate(axes):
        size = shape[index]
        coordinate = f"o{index}"
        lines.append(f"            {coordinate} = {remainder} % {size}\n")
        if position < len(axes) - 1:
            lines.append(f"            {remainder} = {remainder} // {size}\n")
    return "".join(lines)


def _emit_expression(
    expression: ScalarExpression,
    load_indices: Mapping[str, int],
    axis_coordinates: Mapping[str, str],
    loads: tuple,
) -> _TypedExpression:
    if isinstance(expression, Load):
        index = load_indices.get(expression.load_id)
        if index is None:
            raise UnsupportedMapBody(f"unknown load {expression.load_id}")
        load = next(
            candidate for candidate in loads if candidate.id == expression.load_id
        )
        hoisted = _HOISTED_LOADS.get().get(expression.load_id)
        if hoisted is not None:
            return _TypedExpression(hoisted, load.dtype)
        return _load_access(load, index, loads, axis_coordinates)

    if isinstance(expression, Constant):
        return _TypedExpression(
            f"{_cutlass_type(expression.dtype)}({_constant_code(expression.value)})",
            expression.dtype,
        )
    if isinstance(expression, Accumulator):
        name = "accumulator" if expression.index == 0 else f"accumulator_{expression.index}"
        return _TypedExpression(name, expression.dtype)
    if isinstance(expression, AxisIndex):
        coordinate = axis_coordinates.get(expression.axis)
        if coordinate is None:
            raise UnsupportedMapBody(f"unknown axis {expression.axis}")
        return _cast_expression(
            _TypedExpression(coordinate, "torch.int64"), expression.dtype
        )
    if isinstance(expression, Unary):
        operand = _cast_expression(
            _emit_expression(
                expression.operand, load_indices, axis_coordinates, loads
            ),
            expression.dtype,
        )
        return _TypedExpression(
            _unary_code(expression.operator, operand), expression.dtype
        )
    if isinstance(expression, Binary):
        left = _emit_expression(expression.lhs, load_indices, axis_coordinates, loads)
        right = _emit_expression(expression.rhs, load_indices, axis_coordinates, loads)
        if expression.operator in {"eq", "ne", "lt", "le", "gt", "ge"}:
            comparison_dtype = _comparison_dtype(left.dtype, right.dtype)
            left = _cast_expression(left, comparison_dtype)
            right = _cast_expression(right, comparison_dtype)
        else:
            left = _cast_expression(left, expression.dtype)
            right = _cast_expression(right, expression.dtype)
        if expression.operator == "pow":
            code = f"cute.pow({left.code}, {right.code})"
        elif expression.operator == "maximum":
            code = _maximum_code(left.code, right.code)
        elif expression.operator == "minimum":
            code = f"cute.arch.fmin({left.code}, {right.code}, nan=True)"
        else:
            symbol = _BINARY_SYMBOLS.get(expression.operator)
            if symbol is None:
                raise UnsupportedMapBody(
                    f"unsupported binary operator {expression.operator}"
                )
            if expression.dtype in _HALF_DTYPES and expression.operator in _ARITHMETIC:
                # PyTorch computes half-precision arithmetic in float32 and
                # rounds each result. Doing the same keeps every source
                # rounding point when operations are fused, where native half
                # arithmetic could contract a multiply-add and skip one.
                left = _cast_expression(left, "torch.float32")
                right = _cast_expression(right, "torch.float32")
                return _cast_expression(
                    _TypedExpression(
                        f"({left.code} {symbol} {right.code})", "torch.float32"
                    ),
                    expression.dtype,
                )
            code = f"({left.code} {symbol} {right.code})"
        return _TypedExpression(code, expression.dtype)
    if isinstance(expression, Where):
        condition = _emit_expression(
            expression.condition, load_indices, axis_coordinates, loads
        )
        condition = _cast_expression(condition, "torch.bool")
        true_value = _cast_expression(
            _emit_expression(
                expression.true_value, load_indices, axis_coordinates, loads
            ),
            expression.dtype,
        )
        false_value = _cast_expression(
            _emit_expression(
                expression.false_value, load_indices, axis_coordinates, loads
            ),
            expression.dtype,
        )
        # Parenthesized as a whole: a select nested in a larger expression
        # must not absorb its neighbours (`a if c else b * x`).
        return _TypedExpression(
            f"(({true_value.code}) if ({condition.code}) else ({false_value.code}))",
            expression.dtype,
        )
    if isinstance(expression, Cast):
        return _cast_expression(
            _emit_expression(
                expression.operand, load_indices, axis_coordinates, loads
            ),
            expression.dtype,
        )
    raise UnsupportedMapBody(f"unsupported scalar expression {expression!r}")


def _gather_bounds_lines(load: Any, hoisted_names: Mapping[str, str]) -> str:
    # Out-of-range gathers trap inside the workload kernel, matching the
    # device-side assert of the eager source instead of a per-call host check.
    # The index is per lane, so the warp votes first (see trap_if).
    lines = ""
    for position, axis in enumerate(load.axes):
        if not isinstance(axis, DataIndex) or load.source_shape is None:
            continue
        index = hoisted_names[axis.load_id]
        size = load.source_shape[position]
        lines += f"            trap_if({index} < 0 or {index} >= {size})\n"
    return lines


def _load_access(
    load: Any,
    index: int,
    loads: tuple,
    axis_coordinates: Mapping[str, str],
) -> _TypedExpression:
    coordinates = []
    bounds = []
    for axis in load.axes:
        coordinate = _axis_code(axis, axis_coordinates, loads)
        coordinates.append(coordinate)
        if load.source_shape is not None and not isinstance(axis, DataIndex):
            size = load.source_shape[len(coordinates) - 1]
            bounds.append(f"0 <= ({coordinate}) < {size}")
    coordinate_code = ", ".join(coordinates) or "0"
    access = f"tensor_bindings[{index}][{coordinate_code}]"
    if bounds:
        zero = f"{_cutlass_type(load.dtype)}(0)"
        access = f"({access} if {' and '.join(bounds)} else {zero})"
    return _TypedExpression(access, load.dtype)


def _axis_code(
    axis: Any, axis_coordinates: Mapping[str, str], loads: tuple
) -> str:
    if isinstance(axis, str):
        return axis_coordinates[axis]
    if isinstance(axis, int):
        return str(axis)
    if isinstance(axis, DataIndex):
        for index, candidate in enumerate(loads):
            if candidate.id == axis.load_id:
                return _load_access(
                    candidate, index, loads, axis_coordinates
                ).code
        raise UnsupportedMapBody(f"unknown data index {axis.load_id}")
    if not isinstance(axis, AxisAffine):
        raise UnsupportedMapBody(f"unsupported indexed axis {axis!r}")
    code = str(axis.offset)
    for term in axis.terms:
        coordinate = axis_coordinates[term.axis]
        if term.divisor != 1:
            coordinate = f"({coordinate} // {term.divisor})"
        if term.modulus:
            coordinate = f"({coordinate} % {term.modulus})"
        if term.scale != 1:
            coordinate = f"({coordinate} * {term.scale})"
        code = f"{code} + {coordinate}"
    return f"({code})"


def _unary_code(operator: str, operand: _TypedExpression) -> str:
    if operator == "relu":
        zero = f"{_cutlass_type(operand.dtype)}(0)"
        return f"(({operand.code}) if (({operand.code}) > {zero}) else {zero})"
    if operator == "neg":
        return f"(-({operand.code}))"
    if operator == "logical_not":
        false = f"{_cutlass_type(operand.dtype)}(False)"
        return f"(({operand.code}) == {false})"
    function = _UNARY_FUNCTIONS.get(operator)
    if function is None:
        raise UnsupportedMapBody(f"unsupported unary operator {operator}")
    if operand.dtype in _HALF_DTYPES:
        # PyTorch's opmath: half-precision math runs in float32 and rounds once.
        widened = _cast_expression(operand, "torch.float32")
        return _cast_expression(
            _TypedExpression(_unary_function_code(function, widened), "torch.float32"),
            operand.dtype,
        ).code
    return _unary_function_code(function, operand)


_UNARY_FUNCTIONS = {
    "abs": "cute.abs",
    "sigmoid": "sigmoid",
    "reciprocal": "reciprocal",
    "tanh": "cute.tanh",
    "rsqrt": "cute.rsqrt",
    "sqrt": "cute.sqrt",
    "exp": "cute.exp",
    "exp2": "cute.math.exp2",
    "expm1": "cute.math.expm1",
    "log": "cute.math.log",
    "log1p": "cute.math.log1p",
    "log2": "cute.math.log2",
    "erf": "cute.math.erf",
    "floor": "cute.math.floor",
    "ceil": "cute.math.ceil",
    "trunc": "cute.math.trunc",
    "cos": "cute.cos",
    "sin": "cute.sin",
}


def _unary_function_code(function: str, operand: _TypedExpression) -> str:
    one = f"{_cutlass_type(operand.dtype)}(1.0)"
    if function == "sigmoid":
        return f"({one} / ({one} + cute.exp(-({operand.code}))))"
    if function == "reciprocal":
        return f"({one} / ({operand.code}))"
    return f"{function}({operand.code})"


def _cast_expression(value: _TypedExpression, dtype: str) -> _TypedExpression:
    cutlass_type = _cutlass_type(dtype)
    if value.dtype == dtype:
        return value
    return _TypedExpression(f"({value.code}).to({cutlass_type})", dtype)


def _cutlass_type(dtype: str) -> str:
    try:
        return _CUTLASS_TYPES[dtype]
    except KeyError as error:
        raise UnsupportedMapBody(f"unsupported dtype {dtype}") from error


def _comparison_dtype(left: str, right: str) -> str:
    if left == right:
        return left
    if _is_float(left) and _is_float(right):
        if {left, right} == {"torch.float16", "torch.bfloat16"}:
            return "torch.float32"
        return _larger_float(left, right)
    if _is_float(left):
        return left
    if _is_float(right):
        return right
    if left == "torch.bool":
        return right
    if right == "torch.bool":
        return left
    return _larger_integer(left, right)


def _is_float(dtype: str) -> bool:
    return dtype in {
        "torch.float16",
        "torch.bfloat16",
        "torch.float32",
        "torch.float64",
    }


def _float_rank(dtype: str) -> int:
    return ["torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"].index(
        dtype
    )


def _integer_rank(dtype: str) -> int:
    return ["torch.bool", "torch.int8", "torch.int16", "torch.int32", "torch.int64"].index(
        dtype
    )


def _larger_float(left: str, right: str) -> str:
    return left if _float_rank(left) > _float_rank(right) else right


def _larger_integer(left: str, right: str) -> str:
    return left if _integer_rank(left) > _integer_rank(right) else right


def _dtype_bytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as error:
        raise UnsupportedReductionBody(f"unsupported dtype {dtype}") from error


class GenericBodyProvider:
    provider_name = "generic"

    def get_candidates(
        self, region: Region, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        if not policy.allow_generic:
            return CandidateSet(
                (), (ProviderRejection(self.provider_name, "generic providers disabled"),)
            )
        if not region.tensor_programs:
            return CandidateSet(
                (),
                (ProviderRejection(self.provider_name, "region has no tensor program"),),
            )
        if len(region.tensor_programs) != 1:
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, "multi-program regions are not supported yet"
                    ),
                ),
            )

        program = region.tensor_programs[0]
        family = program.family
        if family == "view":
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, "aliasing view needs no device body"
                    ),
                ),
            )
        if isinstance(region.semantics, GemmEpilogue):
            # The fused region only pays off in a tuned GEMM body; without one
            # the planner keeps the unfused GEMM and its consumers.
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, "GEMM epilogue fusion needs a tuned GEMM body"
                    ),
                ),
            )
        if family not in {
            "elementwise",
            "copy",
            "gather",
            "concat",
            "reduction",
            "contraction",
        }:
            return CandidateSet(
                (),
                (
                    ProviderRejection(
                        self.provider_name, f"unsupported program family {family}"
                    ),
                ),
            )

        return CandidateSet((generic_config(region, program),), ())

    def emit(
        self,
        program: TensorProgram,
        config: BodyConfig,
        *,
        defer_length_publish: bool = False,
        workers: int | None = None,
    ) -> GeneratedMapBody | GeneratedReductionBody | GeneratedContractionBody:
        del defer_length_publish, workers
        if program.family in {"elementwise", "copy", "gather", "concat"}:
            return emit_map_body(program, config)
        if program.family == "reduction":
            return emit_reduction_body(program, config)
        if program.family == "contraction":
            return emit_contraction_body(program, config)
        raise UnsupportedMapBody(f"no generic body emits {program.family} programs")

    def widen(self, program: TensorProgram, config: BodyConfig, threads: int) -> BodyConfig:
        participants = widened_participants(config, threads)
        output_domains = tuple(
            domain
            for domain in program.iteration_domains
            if domain.axis not in program.reduction_axes
        )
        cta_shape = (threads,) + (1,) * max(len(output_domains) - 1, 0)
        if program.family in {"elementwise", "copy", "gather", "concat", "contraction"}:
            return replace(
                config, participants=participants, tiles=replace(config.tiles, cta_shape=cta_shape)
            )
        if program.family == "reduction":
            accumulator_size = DTYPE_BYTES[program.accumulator_dtype or ""]
            return replace(
                config,
                participants=participants,
                tiles=replace(config.tiles, cta_shape=cta_shape, reduction_tile=(threads,)),
                scratch=replace(
                    config.scratch,
                    size_bytes=max(config.scratch.size_bytes, (threads // 32) * accumulator_size),
                ),
            )
        raise UnsupportedMapBody(f"cannot widen generic {program.family} bodies")


def generic_config(region: Region, program: Any) -> BodyConfig:
    """The generic body contract for ``program``; tuned providers that run a
    generic-shaped body start from it."""
    output_axes = tuple(
        domain.axis
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    )
    is_reduction = program.family == "reduction"
    is_contraction = program.family == "contraction"
    is_fused_chain = isinstance(region.semantics, ScalarChain)
    threads = 128
    warp_count = threads // 32
    scratch_bytes = (
        warp_count * DTYPE_BYTES[program.accumulator_dtype]
        if is_reduction
        else 0
    )
    barriers = ("cta_reduction",) if is_reduction else ()
    cta_shape = (128,) + (1,) * max(len(output_axes) - 1, 0)
    completion = CompletionContract(
        outputs_ready="assigned output elements are stored when run returns",
        scratch_reusable="scratch is reusable after the CTA reduction barrier"
        if is_reduction
        else "scratch is reusable when run returns",
        register_fragments="none",
    )
    evidence = (
        ResourceEvidence(
            measured=False,
            source="generated CuTe reduction; composition informed by Mirage primitive-body and Hopper norm tasks",
            revision="mirage@f9eb70c254acefc9f3667b2a973d0dcf25471fce; task-07",
            resources={"threads": threads, "shared_bytes": scratch_bytes},
            measurements={},
        )
        if is_reduction
        else ResourceEvidence(
            measured=False,
            source=(
                "generated CuTe indexed contraction; accumulation and epilogue "
                "structure informed by Mirage forloop-accum and threadblock epilogues"
            ),
            revision="mirage@f9eb70c254acefc9f3667b2a973d0dcf25471fce; task-08",
            resources={
                "threads": threads,
                "shared_bytes": scratch_bytes,
                "reduction_chunk": 32,
            },
            measurements={},
        )
        if is_contraction
        else ResourceEvidence(
            measured=False,
            source="generated baseline contract",
            revision="task-05",
            resources={"threads": threads, "shared_bytes": scratch_bytes},
            measurements=(
                {"fusion_saved_bytes": region.semantics.saved_bytes}
                if is_fused_chain
                else {}
            ),
        )
    )
    return BodyConfig(
        provider=GenericBodyProvider.provider_name,
        name=f"generic_{program.family}_fused" if is_fused_chain else f"generic_{program.family}",
        semantic=SemanticContract(
            operation_family=program.family,
            source_op_ids=region.source_op_ids,
            numerical_policy="source_semantics",
            expression_hooks=("vector_chain",) if is_fused_chain else (),
            access_effects=access_effects(program),
        ),
        layouts=_layouts(program),
        tiles=TileContract(
            cta_shape=cta_shape,
            cluster_shape=(1,) * len(cta_shape),
            work_partition=(
                "one output row per CTA"
                if is_reduction
                else "one output element per thread"
            ),
            mma_family=None,
            reduction_tile=(threads,) if is_reduction else (32,)
            if is_contraction
            else (),
        ),
        participants=ParticipantContract(
            participating_threads=threads,
            waiting_threads=threads,
            warp_roles=("worker",) * (threads // 32),
            registers_per_thread=32,
        ),
        pipeline=PipelineContract(
            stage_count=1,
            barriers=barriers,
            async_operations=(),
            drain_rule="no asynchronous pipeline to drain",
        ),
        scratch=ScratchContract(
            owner="cta" if is_reduction else "none",
            size_bytes=scratch_bytes,
            alignment_bytes=16,
        ),
        completion=completion,
        evidence=evidence,
    )


def _layouts(program: Any, alignment_bytes: int | None = None) -> LayoutContract:
    domain_sizes = {
        domain.axis: domain.size for domain in program.iteration_domains
    }
    inputs: dict[str, TensorLayout] = {}
    for load in program.loads:
        if load.value_id in inputs:
            continue
        inputs[load.value_id] = TensorLayout(
            dtype=load.dtype,
            axes=tuple(load.axes),
            strides=tuple(load.strides),
            alignment_bytes=alignment_bytes or DTYPE_BYTES[load.dtype],
            shape=(
                tuple(load.source_shape)
                if load.source_shape is not None
                else tuple(
                    1 if isinstance(axis, int) else domain_sizes[axis]
                    for axis in load.axes
                )
            ),
        )
    store = program.store
    output_axes = tuple(
        domain.axis
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    )
    outputs = {
        store.value_id: TensorLayout(
            dtype=store.dtype,
            axes=output_axes,
            strides=tuple(store.strides),
            alignment_bytes=alignment_bytes or DTYPE_BYTES[store.dtype],
            shape=tuple(store.shape),
        )
    }
    return LayoutContract(inputs, outputs)
