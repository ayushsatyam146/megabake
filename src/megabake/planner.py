from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from itertools import chain, product
from math import prod
from typing import Any

from .bodies.catalog import get_candidates
from .bodies.interface import BodyConfig, SelectionPolicy, TargetFacts
from .graph import DTYPE_BYTES, ComputeGraph, Region
from .ir import AxisAffine, DataIndex, TensorProgram
from .ops import Attention
from .plan import (
    BodySelectionPlan,
    ConversionBody,
    CuTeLayout,
    InvalidTileMap,
    LogicalTile,
    ResourceRequirements,
    RejectionRecord,
    SelectedBody,
    TileAccess,
    TileGrid,
    TileMap,
    UnsupportedRegion,
)


def select_bodies(
    graph: ComputeGraph,
    target: TargetFacts,
    policy: SelectionPolicy | None = None,
) -> BodySelectionPlan:
    policy = _validated_policy(graph, policy)
    selections, rejections, unsupported = _enumerate_body_selections(
        graph, target, policy, max_plans=1
    )
    if not selections:
        return BodySelectionPlan(
            (),
            tuple(rejections),
            tuple(unsupported),
            (),
            graph,
            target,
        )
    return selections[0]


def enumerate_body_selections(
    graph: ComputeGraph,
    target: TargetFacts,
    policy: SelectionPolicy | None = None,
    max_plans: int = 8,
) -> tuple[BodySelectionPlan, ...]:
    if max_plans < 1:
        raise ValueError("max_plans must be positive")
    policy = _validated_policy(graph, policy)
    # The preflight and the measured search enumerate the same selections;
    # sharing them lets both reuse the same cached tile maps.
    key = (id(graph), target, policy, max_plans)
    entry = _SELECTION_CACHE.get(key)
    if entry is not None and entry[0] is graph:
        return entry[1]
    selections = _enumerate_body_selections(
        graph, target, policy, max_plans=max_plans
    )[0]
    if len(_SELECTION_CACHE) >= 8:
        _SELECTION_CACHE.pop(next(iter(_SELECTION_CACHE)))
    _SELECTION_CACHE[key] = (graph, selections)
    return selections


_SELECTION_CACHE: dict[tuple[Any, ...], tuple[ComputeGraph, tuple[BodySelectionPlan, ...]]] = {}


def _validated_policy(
    graph: ComputeGraph, policy: SelectionPolicy | None
) -> SelectionPolicy:
    selected_policy = policy or SelectionPolicy()
    if selected_policy.mode not in {"prefill", "decode"}:
        raise ValueError(
            f"selection mode {selected_policy.mode!r} must be 'prefill' or 'decode'"
        )
    stateful = any(
        isinstance(semantics, Attention) and semantics.cache.policy == "stateful"
        for region in graph.regions
        for semantics in (
            region.semantics,
            *(candidate.semantics for candidate in region.fusion_candidates),
        )
    )
    if stateful and selected_policy.mode != "decode":
        raise ValueError("prefill mode cannot compile a stateful KV cache update")
    return selected_policy


def _enumerate_body_selections(
    graph: ComputeGraph,
    target: TargetFacts,
    selected_policy: SelectionPolicy,
    max_plans: int,
):
    type Option = tuple[Region, TensorProgram, BodyConfig] | None
    options_by_region: list[list[Option]] = []
    rejections: list[RejectionRecord] = []
    unsupported: list[UnsupportedRegion] = []
    covered_by_candidates = {
        operation_id
        for region in graph.regions
        for candidate in region.fusion_candidates
        if candidate.executable
        for operation_id in candidate.source_op_ids
    }
    covered_empty_regions: list[int] = []

    for region in graph.regions:
        options: list[Option] = []
        for candidate in region.fusion_candidates:
            if not candidate.executable:
                rejections.append(
                    RejectionRecord(
                        candidate.id,
                        "fusion",
                        "; ".join(candidate.rejections) or "candidate has no tensor program",
                    )
                )
                continue
            assert candidate.tensor_program is not None
            candidate_region = Region(
                id=candidate.id,
                source_op_ids=candidate.source_op_ids,
                external_inputs=candidate.external_inputs,
                live_outputs=candidate.live_outputs,
                semantics=candidate.semantics,
                tensor_programs=(candidate.tensor_program,),
            )
            options.extend(
                _body_options(candidate_region, candidate.tensor_program, target, selected_policy, rejections)
            )

        if all(
            isinstance(program, TensorProgram) and program.family == "view"
            for program in region.tensor_programs
        ):
            options.append(None)
            options_by_region.append(options)
            continue
        if region.tensor_programs:
            program = region.tensor_programs[0]
            if isinstance(program, TensorProgram):
                options.extend(
                    _body_options(region, program, target, selected_policy, rejections)
                )
            else:
                unsupported.append(UnsupportedRegion(region.id, "tensor program is unknown"))
        else:
            unsupported.append(UnsupportedRegion(region.id, "region has no tensor program"))
        if not options:
            if all(
                operation_id in covered_by_candidates
                for operation_id in region.source_op_ids
            ):
                covered_empty_regions.append(len(options_by_region))
                options_by_region.append([None])
                continue
            unsupported.append(UnsupportedRegion(region.id, "no valid provider candidate"))
        options_by_region.append(options)

    selections: list[BodySelectionPlan] = []
    first_combination = [None] * len(options_by_region)
    covered_operations = set()
    all_options = [
        (index, option)
        for index, options in enumerate(options_by_region)
        for option in options
        if option is not None
    ]
    all_options.sort(
        key=lambda item: len(item[1][0].source_op_ids),
        reverse=True,
    )
    for index, option in all_options:
        region_operations = set(graph.regions[index].source_op_ids)
        candidate_operations = set(option[0].source_op_ids)
        if (
            not region_operations <= candidate_operations
            or covered_operations.intersection(candidate_operations)
        ):
            continue
        first_combination[index] = option
        covered_operations.update(candidate_operations)

    seed_coverage = {
        operation_id
        for option in first_combination
        if option is not None
        for operation_id in option[0].source_op_ids
    }
    seed_is_complete = all(
        first_combination[index] is not None
        or set(graph.regions[index].source_op_ids) <= seed_coverage
        or all(
            isinstance(program, TensorProgram) and program.family == "view"
            for program in graph.regions[index].tensor_programs
        )
        for index, options in enumerate(options_by_region)
    )
    if not seed_is_complete:
        first_combination = [
            next(
                (
                    option
                    for option in options
                    if option is not None and option[0].id == graph.regions[index].id
                ),
                None,
            )
            for index, options in enumerate(options_by_region)
        ]
        operation_indices = {
            operation_id: index
            for index, option in enumerate(first_combination)
            if option is not None
            for operation_id in option[0].source_op_ids
        }
        for index, options in enumerate(options_by_region):
            if first_combination[index] is not None:
                continue
            replacement = next(
                (
                    option
                    for option in options
                    if option is not None
                    and set(graph.regions[index].source_op_ids)
                    <= set(option[0].source_op_ids)
                ),
                None,
            )
            if replacement is None:
                continue
            replacement_operations = set(replacement[0].source_op_ids)
            replaced_indices = {
                operation_indices[operation_id]
                for operation_id in replacement_operations
            }
            replaced_operations = {
                operation_id
                for replaced_index in replaced_indices
                if first_combination[replaced_index] is not None
                for operation_id in first_combination[replaced_index][0].source_op_ids
            }
            if not replaced_operations <= replacement_operations:
                continue
            for replaced_index in replaced_indices:
                first_combination[replaced_index] = None
            first_combination[index] = replacement
            operation_indices.update(
                (operation_id, index)
                for operation_id in replacement_operations
            )

    combination_limit = max_plans * 16
    combination_count = 1
    for options in options_by_region:
        combination_count *= max(1, len(options))
        if combination_count > combination_limit:
            break
    seed = (tuple(first_combination),)
    combinations = (
        seed
        if combination_count > combination_limit
        else chain(seed, product(*options_by_region))
    )
    examined_combinations = 0
    for combination in combinations:
        examined_combinations += 1
        chosen = []
        covered_operations: set[str] = set()
        for option in sorted(
            (option for option in combination if option is not None),
            key=lambda option: -len(option[0].source_op_ids),
        ):
            if covered_operations.intersection(option[0].source_op_ids):
                continue
            chosen.append(option)
            covered_operations.update(option[0].source_op_ids)
        if any(
            not combination[index]
            and not set(graph.regions[index].source_op_ids) <= covered_operations
            for index in covered_empty_regions
        ):
            continue
        if not chosen:
            continue
        selected_bodies = [_selected_body(*option) for option in chosen]
        region_ids_by_operation = {
            operation_id: selected.region_id
            for selected in selected_bodies
            for operation_id in selected.source_op_ids
        }
        producer_regions_by_value = {
            value_id: region_ids_by_operation[graph.values[value_id].producer]
            for value_id in graph.values
            if graph.values[value_id].producer in region_ids_by_operation
        }
        for value in graph.values.values():
            relation = value.storage_relation
            if relation.storage_id in producer_regions_by_value:
                producer_regions_by_value.setdefault(
                    value.id, producer_regions_by_value[relation.storage_id]
                )
        selected_bodies = [
            replace(
                selected,
                dependencies=tuple(
                    dict.fromkeys(
                        producer_regions_by_value[value_id]
                        for value_id in selected.inputs
                        if value_id in producer_regions_by_value
                        and producer_regions_by_value[value_id] != selected.region_id
                    )
                ),
            )
            for selected in selected_bodies
        ]
        selected_bodies = _topological_sort_bodies(selected_bodies)
        conversions = _insert_layout_conversions(tuple(selected_bodies))
        selections.append(
            BodySelectionPlan(
                tuple(selected_bodies),
                tuple(rejections),
                (),
                conversions,
                graph,
                target,
            )
        )
        if len(selections) >= max_plans:
            break
        if examined_combinations >= max_plans * 16:
            break
    return tuple(selections), rejections, unsupported


def _body_options(
    region: Region,
    program: TensorProgram,
    target: TargetFacts,
    policy: SelectionPolicy,
    rejections: list[RejectionRecord],
) -> list[tuple[Region, TensorProgram, BodyConfig]]:
    candidates = get_candidates(region, target, policy)
    rejections.extend(
        RejectionRecord(region.id, rejection.provider, rejection.reason)
        for rejection in candidates.rejections
    )
    return [
        (region, program, body)
        for body in sorted(
            candidates.configurations,
            key=lambda body: _body_order(body, region, program, policy),
        )
    ]


def _selected_body(
    region: Region, program: TensorProgram, body: BodyConfig
) -> SelectedBody:
    return SelectedBody(
        region_id=region.id,
        source_op_ids=region.source_op_ids,
        body=body,
        program=program,
        inputs=tuple(load.value_id for load in program.loads),
        outputs=(program.store.value_id,),
        dependencies=(),
        participants=body.participants,
        resources=_resource_requirements(body),
    )


def _topological_sort_bodies(
    bodies: Sequence[SelectedBody],
) -> tuple[SelectedBody, ...]:
    positions = {body.region_id: position for position, body in enumerate(bodies)}
    consumers = {region_id: [] for region_id in positions}
    indegree = {region_id: 0 for region_id in positions}
    for body in bodies:
        for producer_id in body.dependencies:
            if producer_id in consumers:
                consumers[producer_id].append(body.region_id)
                indegree[body.region_id] += 1
    import heapq

    ready = [positions[region_id] for region_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        position = heapq.heappop(ready)
        region_id = bodies[position].region_id
        order.append(position)
        for consumer in consumers[region_id]:
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                heapq.heappush(ready, positions[consumer])
    if len(order) != len(bodies):
        raise ValueError("body selection contains a dependency cycle")
    return tuple(bodies[position] for position in order)


def _body_order(
    config: BodyConfig,
    region: Region,
    program: TensorProgram,
    policy: SelectionPolicy,
) -> tuple[int, ...]:
    if program.family == "attention":
        return _attention_body_order(config, region, policy)
    if program.family == "contraction" and program.store.shape:
        return _contraction_body_order(config, program, policy)
    return (0 if config.evidence.measured else 1, *_body_cost(config))


def _attention_body_order(
    config: BodyConfig, region: Region, policy: SelectionPolicy
) -> tuple[int, ...]:
    loads = {load.id: load for load in region.tensor_programs[0].loads}
    grouping = region.semantics.grouping
    query_length = region.semantics.q.source_shape[1]
    kv_length = loads["attention:k"].source_shape[1]
    parallelism = grouping.batch * grouping.query_heads * query_length
    decode_work = query_length == 1 and kv_length > 0 and parallelism > 0
    mode_priority = 0 if (policy.mode == "decode") == decode_work else 1
    return (
        0 if (config.name == "attention_decode_rows") == decode_work else 1,
        mode_priority,
        *_body_cost(config),
    )


def _contraction_body_order(
    config: BodyConfig, program: TensorProgram, policy: SelectionPolicy
) -> tuple[int, ...]:
    small_m = program.store.shape[0] <= 16
    mode_priority = 0 if (policy.mode == "decode") == small_m else 1
    small_tile = config.tiles.cta_shape[-1] <= 64
    return (
        0 if config.evidence.measured else 1,
        0 if config.tiles.mma_family else 1,
        0 if small_tile == small_m else 1,
        mode_priority,
        *_body_cost(config),
    )


def make_tile_maps(selection: BodySelectionPlan) -> tuple[TileMap, ...]:
    return tuple(_cached_tile_map(selected, selection.conversions) for selected in selection.selections)


# Tile maps enumerate every output tile and its input tiles, which for large
# graphs costs minutes, and plan search, scheduling, and storage planning each
# ask for the same region's map. A map depends only on the selected body and
# the conversions into it, so it is built once per (body, conversions) pair.
# Entries hold their keys' objects, so an id is never reused while cached.
_TILE_MAP_CACHE: dict[tuple[int, tuple[int, ...]], tuple[Any, tuple[Any, ...], TileMap]] = {}
_TILE_MAP_CACHE_LIMIT = 8192


def _cached_tile_map(
    selected: SelectedBody, conversions: Sequence[ConversionBody]
) -> TileMap:
    relevant = tuple(
        conversion
        for conversion in conversions
        if conversion.target_region_id == selected.region_id
        or conversion.source_region_id == selected.region_id
    )
    key = (id(selected), tuple(id(conversion) for conversion in relevant))
    entry = _TILE_MAP_CACHE.get(key)
    if entry is not None and entry[0] is selected and all(
        old is new for old, new in zip(entry[1], relevant, strict=True)
    ):
        return entry[2]
    tile_map = _make_tile_map(selected, conversions)
    if len(_TILE_MAP_CACHE) >= _TILE_MAP_CACHE_LIMIT:
        _TILE_MAP_CACHE.pop(next(iter(_TILE_MAP_CACHE)))
    _TILE_MAP_CACHE[key] = (selected, relevant, tile_map)
    return tile_map


def _body_cost(config: BodyConfig) -> tuple[int, int]:
    base_cost = (
        config.participants.participating_threads
        * config.participants.registers_per_thread
        + config.scratch.size_bytes
    )
    saved_bytes = config.evidence.measurements.get("fusion_saved_bytes", 0)
    return (
        max(0, base_cost - saved_bytes) if isinstance(saved_bytes, int) else base_cost,
        -len(config.evidence.measurements),
    )


def _resource_requirements(config: BodyConfig) -> ResourceRequirements:
    active_blocks = config.evidence.resources.get("active_blocks_per_sm")
    return ResourceRequirements(
        shared_memory_bytes=config.scratch.size_bytes,
        pipeline_stages=config.pipeline.stage_count,
        barriers=config.pipeline.barriers,
        maximum_active_blocks=active_blocks if isinstance(active_blocks, int) else None,
    )


def _insert_layout_conversions(
    selections: Sequence[SelectedBody],
) -> tuple[ConversionBody, ...]:
    output_layouts: dict[str, tuple[str, CuTeLayout]] = {}
    for selected in selections:
        value_id = selected.program.store.value_id
        contract = selected.body.layouts.outputs.get(value_id)
        if contract is None:
            raise InvalidTileMap(f"{selected.region_id} has no output layout for {value_id}")
        output_layouts[value_id] = (selected.region_id, CuTeLayout.from_contract(contract))

    conversions: list[ConversionBody] = []
    for selected in selections:
        for value_id in selected.inputs:
            contract = selected.body.layouts.inputs.get(value_id)
            if contract is None:
                raise InvalidTileMap(f"{selected.region_id} has no input layout for {value_id}")
            produced = output_layouts.get(value_id)
            if produced is None:
                continue
            source_region_id, source_layout = produced
            target_layout = CuTeLayout.from_contract(contract)
            if not source_layout.needs_conversion_to(target_layout):
                continue
            conversions.append(
                ConversionBody(
                    value_id=value_id,
                    source_region_id=source_region_id,
                    target_region_id=selected.region_id,
                    source_layout=source_layout,
                    target_layout=target_layout,
                    cost_bytes=_conversion_cost(target_layout),
                    dependencies=(source_region_id,),
                )
            )
    return tuple(conversions)


def _make_tile_map(
    selected: SelectedBody,
    conversions: Sequence[ConversionBody],
) -> TileMap:
    program = selected.program
    body = selected.body
    output_contract = body.layouts.outputs.get(program.store.value_id)
    if output_contract is None:
        raise InvalidTileMap(f"{selected.region_id} has no output layout")
    output_layout = CuTeLayout.from_contract(output_contract)
    output_domains = tuple(
        domain
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    )
    output_axes = tuple(domain.axis for domain in output_domains)
    if tuple(output_contract.axes) != output_axes:
        raise InvalidTileMap(f"{selected.region_id} output axes do not match its domains")

    input_layouts: list[tuple[str, CuTeLayout]] = []
    for load in program.loads:
        contract = body.layouts.inputs.get(load.value_id)
        if contract is None:
            raise InvalidTileMap(f"{selected.region_id} has no layout for {load.value_id}")
        conversion = next(
            (
                candidate
                for candidate in conversions
                if candidate.value_id == load.value_id
                and candidate.target_region_id == selected.region_id
            ),
            None,
        )
        if contract.dtype != load.dtype or (
            conversion is None and tuple(contract.strides) != tuple(load.strides)
        ):
            raise InvalidTileMap(f"{selected.region_id} input layout does not match {load.value_id}")
        base_layout = (
            conversion.target_layout
            if conversion is not None
            else CuTeLayout.from_contract(contract)
        )
        if tuple(contract.axes) == tuple(load.axes):
            layout = base_layout
        else:
            domain_sizes = {
                domain.axis: domain.size for domain in program.iteration_domains
            }
            shape = load.source_shape or tuple(
                1 if isinstance(axis, int) else domain_sizes[axis]
                for axis in load.axes
            )
            layout = CuTeLayout(
                load.dtype,
                shape,
                base_layout.strides,
                contract.alignment_bytes,
            )
        input_layouts.append((load.value_id, layout))

    grid = TileGrid(program.store.shape, _output_tile_shape(program, body))
    reduction_size = prod(
        domain.size for domain in program.iteration_domains
        if domain.axis in program.reduction_axes
    )
    reduction_range = (0, reduction_size) if program.reduction_axes else None
    accesses = tuple(
        TileAccess(
            output=_logical_tile(
                program.store.value_id,
                output_axes,
                output_layout,
                origin,
                grid.tile_shape,
            ),
            inputs=tuple(
                _input_tile(load, layout, origin, grid.tile_shape, output_axes, program)
                for load, (_, layout) in zip(program.loads, input_layouts, strict=True)
            ),
            reduction_range=reduction_range,
        )
        for origin in grid.origins()
    )
    return TileMap(
        region_id=selected.region_id,
        body=body,
        participants=selected.participants,
        resources=selected.resources,
        output_grid=grid,
        output_layout=output_layout,
        input_layouts=tuple(input_layouts),
        accesses=accesses,
        conversion_bodies=tuple(
            conversion
            for conversion in conversions
            if conversion.target_region_id == selected.region_id
        ),
    )


def _output_tile_shape(program: TensorProgram, body: BodyConfig) -> tuple[int, ...]:
    output_shape = program.store.shape
    if program.family == "reduction":
        # One tile per reduced row, as the reduction body runs: axes that only
        # the post-reduction expression reads (a norm's feature axis) span
        # the whole tile.
        from .bodies.cuda.generic import reduced_row_axes

        row_axes = reduced_row_axes(program)
        output_axes = [
            domain.axis
            for domain in program.iteration_domains
            if domain.axis not in program.reduction_axes
        ]
        return tuple(
            1 if row_axes is None or axis in row_axes else size
            for axis, size in zip(output_axes, output_shape, strict=True)
        )
    if body.tiles.work_partition == "one output element per thread":
        return _row_major_tile_shape(output_shape, body.participants.participating_threads)
    if len(body.tiles.cta_shape) != len(output_shape):
        raise InvalidTileMap("CTA tile rank does not match the output rank")
    return body.tiles.cta_shape


def _row_major_tile_shape(
    shape: Sequence[int], element_limit: int
) -> tuple[int, ...]:
    tile_shape = [1] * len(shape)
    remaining = element_limit
    for dimension in reversed(range(len(shape))):
        extent = min(shape[dimension], remaining)
        tile_shape[dimension] = extent
        remaining //= extent
        if extent < shape[dimension]:
            break
    return tuple(tile_shape)


def _input_tile(
    load: Any,
    layout: CuTeLayout,
    output_origin: tuple[int, ...],
    output_tile_shape: tuple[int, ...],
    output_axes: tuple[str, ...],
    program: TensorProgram,
) -> LogicalTile:
    origin: list[int] = []
    shape: list[int] = []
    reduction_domains = {
        domain.axis: domain.size
        for domain in program.iteration_domains
        if domain.axis in program.reduction_axes
    }
    axis_ranges = {
        **{
            axis: (output_origin[index], output_origin[index] + output_tile_shape[index] - 1)
            for index, axis in enumerate(output_axes)
        },
        **{axis: (0, size - 1) for axis, size in reduction_domains.items()},
    }
    for axis in load.axes:
        if isinstance(axis, DataIndex):
            origin.append(0)
            shape.append(layout.shape[len(origin) - 1])
            continue
        if isinstance(axis, int):
            origin.append(axis)
            shape.append(1)
            continue
        if axis in output_axes:
            output_dimension = output_axes.index(axis)
            origin.append(output_origin[output_dimension])
            shape.append(output_tile_shape[output_dimension])
            continue
        if axis in reduction_domains:
            origin.append(0)
            shape.append(reduction_domains[axis])
            continue
        if not isinstance(axis, AxisAffine):
            raise InvalidTileMap(
                f"input axis {axis} is not connected to the iteration domains"
            )
        minimum, maximum = _affine_bounds(axis, axis_ranges)
        if maximum < 0 or minimum >= layout.shape[len(origin)]:
            origin.append(0)
            shape.append(1)
        else:
            origin.append(max(minimum, 0))
            shape.append(max(maximum - max(minimum, 0) + 1, 1))
    return _logical_tile(load.value_id, tuple(load.axes), layout, tuple(origin), tuple(shape))


def _affine_bounds(
    axis: AxisAffine, ranges: dict[str, tuple[int, int]]
) -> tuple[int, int]:
    if any(
        term.scale <= 0 or term.divisor <= 0 or term.modulus < 0
        for term in axis.terms
    ):
        raise InvalidTileMap("affine input axis is not monotonic")
    values = []
    for bound in (0, 1):
        value = axis.offset
        for term in axis.terms:
            coordinate = ranges[term.axis][bound] // term.divisor
            if term.modulus:
                coordinate %= term.modulus
            value += coordinate * term.scale
        values.append(value)
    return min(values), max(values)


def _logical_tile(
    value_id: str,
    axes: tuple[str | int, ...],
    layout: CuTeLayout,
    origin: tuple[int, ...],
    tile_shape: tuple[int, ...],
) -> LogicalTile:
    clipped_shape = tuple(
        min(extent, dimension - begin)
        for extent, dimension, begin in zip(
            tile_shape, layout.shape, origin, strict=True
        )
    )
    if any(extent <= 0 for extent in clipped_shape):
        raise InvalidTileMap(f"tile for {value_id} is empty")
    return LogicalTile(value_id, axes, layout, origin, clipped_shape)


def _conversion_cost(layout: CuTeLayout) -> int:
    return 2 * prod(layout.shape) * DTYPE_BYTES.get(layout.dtype, 0)

