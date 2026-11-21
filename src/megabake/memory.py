from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .bodies.interface import TargetFacts
from .graph import DTYPE_BYTES, ComputeGraph
from .plan import (
    BodySelectionPlan,
    CuTeLayout,
    KernelPlan,
    TileMap,
)
from .planner import make_tile_maps
from .schedule import PhaseSchedule, ScheduledPhase


class InvalidStorage(ValueError):
    pass


@dataclass(frozen=True)
class StorageLifetime:
    start_phase: int
    end_phase: int
    reader_region_ids: tuple[str, ...]
    release_kind: str


@dataclass(frozen=True)
class GlobalStorageAssignment:
    storage_id: str
    value_id: str
    producer_region_id: str
    offset_bytes: int
    size_bytes: int
    required_bytes: int
    alignment_bytes: int
    layout: CuTeLayout
    lifetime: StorageLifetime


@dataclass(frozen=True)
class TensorBinding:
    value_id: str
    storage_id: str
    kind: str
    layout: CuTeLayout | None
    region_ids: tuple[str, ...] = ()
    binding_name: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def boundary(self) -> bool:
        return self.kind in {"input", "parameter", "output", "state", "alias"}


@dataclass(frozen=True)
class WorkerScratchPhase:
    region_id: str
    phase_index: int
    size_bytes: int
    alignment_bytes: int
    body_data_offset_bytes: int
    barrier_offsets: Mapping[str, int]
    descriptor_offsets: Mapping[str, int]
    phase_initialization: Mapping[str, int]
    lifetime: StorageLifetime


@dataclass(frozen=True)
class WorkerScratch:
    worker_id: int
    offset_bytes: int
    size_bytes: int
    alignment_bytes: int
    phases: tuple[WorkerScratchPhase, ...]


@dataclass(frozen=True)
class StateStorageAssignment:
    key_cache_value_id: str
    value_cache_value_id: str
    valid_length_value_id: str
    storage_identity: tuple[str, str]
    capacity: int
    query_length: int
    key_layout: CuTeLayout
    value_layout: CuTeLayout
    read_regions: tuple[str, ...]
    write_regions: tuple[str, ...]
    read_range: tuple[int | str, int | str]
    write_range: tuple[int | str, int | str]
    source_order: tuple[str, ...]
    lifetime: str = "boundary_state"


@dataclass(frozen=True)
class StoragePlan:
    workspace: tuple[GlobalStorageAssignment, ...]
    bindings: tuple[TensorBinding, ...]
    worker_scratch: tuple[WorkerScratch, ...]
    state: tuple[StateStorageAssignment, ...] = ()

    @property
    def workspace_bytes(self) -> int:
        return max(
            (assignment.offset_bytes + assignment.size_bytes for assignment in self.workspace),
            default=0,
        )

    @property
    def worker_scratch_bytes(self) -> int:
        return max((scratch.size_bytes for scratch in self.worker_scratch), default=0)

    @property
    def peak_live_workspace_bytes(self) -> int:
        phase_count = max(
            (assignment.lifetime.end_phase for assignment in self.workspace),
            default=0,
        )
        return max(
            (
                sum(
                    assignment.size_bytes
                    for assignment in self.workspace
                    if assignment.lifetime.start_phase <= phase < assignment.lifetime.end_phase
                )
                for phase in range(phase_count)
            ),
            default=0,
        )


@dataclass(frozen=True)
class LaunchRequirements:
    architecture: str
    cooperative: bool
    grid_size: int
    threads_per_block: int
    dynamic_shared_memory_bytes: int
    registers_per_thread: int
    max_active_blocks_per_sm: int | None
    required_active_blocks: int


def assign_storage(plan: PhaseSchedule, *, reuse_workspace: bool = True) -> KernelPlan:
    """Assign storage to every value; ``reuse_workspace=False`` keeps each
    intermediate in its own slice so diagnostics can read it after the call."""
    if not isinstance(plan, PhaseSchedule):
        raise InvalidStorage("assign_storage expects a PhaseSchedule")
    if plan.selection is None:
        raise InvalidStorage("phase schedule is missing its body selection")
    selection = plan.selection
    if not selection.report.complete:
        raise InvalidStorage("cannot assign storage to unsupported regions")
    if selection.target is None:
        raise InvalidStorage("body selection is missing target facts")
    if selection.graph is None:
        raise InvalidStorage("body selection is missing its source graph")
    if plan.dependency_plan is None:
        raise InvalidStorage("phase schedule is missing its dependency plan")

    tile_maps = {tile_map.region_id: tile_map for tile_map in _tile_maps(selection)}
    workspace_requests: list[_WorkspaceRequest] = []
    bindings: list[TensorBinding] = []

    consumers = _consumers(selection)
    output_roots = _output_storage_roots(selection.graph)
    for selected in selection.selections:
        for value_id in selected.outputs:
            if value_id in selection.graph.outputs or _aliases_boundary(
                value_id, selection.graph, _direct_boundary_ids(selection.graph)
            ):
                continue
            if value_id in output_roots:
                # A graph output views this value, so it must not share
                # reusable workspace; give it per-call output storage.
                bindings.append(
                    TensorBinding(
                        value_id=value_id,
                        storage_id=value_id,
                        kind="output",
                        layout=_value_shaped(
                            tile_maps[selected.region_id].output_layout,
                            selection.graph.values[value_id],
                        ),
                        region_ids=(selected.region_id, *consumers.get(value_id, ())),
                    )
                )
                continue

            source_layout = _value_shaped(
                tile_maps[selected.region_id].output_layout,
                selection.graph.values[value_id],
            )
            conversions = tuple(
                conversion
                for conversion in selection.conversions
                if conversion.value_id == value_id
            )
            consumer_layouts = (
                ()
                if conversions
                else tuple(
                    layout
                    for region_id in consumers.get(value_id, ())
                    for candidate_id, layout in tile_maps[region_id].input_layouts
                    if candidate_id == value_id
                )
            )
            workspace_requests.append(
                _make_workspace_request(
                    value_id,
                    value_id,
                    selected.region_id,
                    (source_layout, *consumer_layouts),
                    consumers.get(value_id, ()),
                    plan.phases,
                )
            )
            region_ids = (selected.region_id, *consumers.get(value_id, ()))
            if conversions:
                region_ids = (selected.region_id,)
            bindings.append(
                TensorBinding(
                    value_id=value_id,
                    storage_id=value_id,
                    kind="workspace",
                    layout=source_layout,
                    region_ids=region_ids,
                )
            )

            for conversion in conversions:
                converted_id = f"{value_id}:converted:{conversion.target_region_id}"
                workspace_requests.append(
                    _make_workspace_request(
                        converted_id,
                        value_id,
                        conversion.source_region_id,
                        (conversion.target_layout,),
                        (conversion.target_region_id,),
                        plan.phases,
                    )
                )
                bindings.append(
                    TensorBinding(
                        value_id=value_id,
                        storage_id=converted_id,
                        kind="conversion",
                        layout=conversion.target_layout,
                        region_ids=(conversion.target_region_id,),
                    )
                )

    bindings.extend(_boundary_bindings(selection, tile_maps))
    bindings = _deduplicate_bindings(bindings)
    worker_scratch = _worker_scratch(plan)
    offsets, _workspace_bytes = (
        _allocate_workspace(workspace_requests)
        if reuse_workspace
        else _allocate_distinct_workspace(workspace_requests)
    )
    workspace = tuple(
        GlobalStorageAssignment(
            request.storage_id,
            request.value_id,
            request.producer_region_id,
            offsets[request.storage_id],
            request.size_bytes,
            request.required_bytes,
            request.alignment_bytes,
            request.layout,
            request.lifetime,
        )
        for request in workspace_requests
    )
    storage = StoragePlan(
        workspace,
        tuple(bindings),
        worker_scratch,
        _state_assignments(selection),
    )
    launch = _launch_requirements(plan, selection.target, storage)
    return KernelPlan(
        selection=selection,
        dependencies=plan.dependency_plan,
        phases=plan.phases,
        tile_maps=tuple(tile_maps.values()),
        storage=storage,
        launch=launch,
    )


def distinct_workspace(plan: KernelPlan) -> KernelPlan:
    """The same plan with every workspace value in its own slice.

    Table launches read workspace addresses from the device address table, so
    the compiled kernel runs unchanged on this storage, and every intermediate
    is still readable after the call."""
    workspace = []
    end = 0
    for assignment in plan.storage.workspace:
        offset = _round_up(end, assignment.alignment_bytes)
        workspace.append(replace(assignment, offset_bytes=offset))
        end = offset + assignment.size_bytes
    distinct = replace(plan, storage=replace(plan.storage, workspace=tuple(workspace)))
    verify_plan(distinct)
    return distinct


def verify_plan(plan: KernelPlan) -> None:
    if not isinstance(plan, KernelPlan):
        raise InvalidStorage("verify_plan expects a KernelPlan")
    if not plan.selection.report.complete:
        raise InvalidStorage("kernel plan contains unsupported regions")
    if plan.selection.target is None or plan.selection.graph is None:
        raise InvalidStorage("kernel plan is missing target or source graph")

    phase_regions = tuple(phase.region_id for phase in plan.phases)
    selected_regions = tuple(selected.region_id for selected in plan.selection.selections)
    if phase_regions != selected_regions:
        raise InvalidStorage("scheduled phases do not cover the selected bodies")
    if tuple(tile_map.region_id for tile_map in plan.tile_maps) != selected_regions:
        raise InvalidStorage("tile maps do not cover the selected bodies")
    if plan.dependencies is None:
        raise InvalidStorage("kernel plan is missing its dependency plan")

    for value_id in plan.selection.graph.outputs:
        if not any(
            binding.value_id == value_id and binding.kind == "output"
            for binding in plan.storage.bindings
        ):
            raise InvalidStorage(f"missing output binding for {value_id}")

    completed_regions: set[str] = set()
    for phase in plan.phases:
        expected = tuple(sorted(plan.dependencies.prerequisites_for(phase.region_id)))
        if phase.dependencies != expected:
            raise InvalidStorage(f"{phase.region_id} phase dependencies are inconsistent")
        if any(prerequisite not in completed_regions for prerequisite in phase.dependencies):
            raise InvalidStorage(f"{phase.region_id} has a prerequisite in a later phase")
        completed_regions.add(phase.region_id)

    binding_by_region_value = {
        (region_id, binding.value_id): binding
        for binding in plan.storage.bindings
        for region_id in binding.region_ids
    }
    for phase in plan.phases:
        for value_id in _phase_values(phase, plan.selection):
            if (phase.region_id, value_id) not in binding_by_region_value:
                raise InvalidStorage(f"{phase.region_id} has no storage binding for {value_id}")

    workspace_ids = {assignment.storage_id for assignment in plan.storage.workspace}
    for binding in plan.storage.bindings:
        if binding.boundary and binding.storage_id in workspace_ids:
            raise InvalidStorage(f"boundary binding {binding.value_id} uses workspace storage")

    phase_by_region = {phase.region_id: phase for phase in plan.phases}
    for assignment in plan.storage.workspace:
        if assignment.offset_bytes % assignment.alignment_bytes:
            raise InvalidStorage(f"workspace slice {assignment.storage_id} is misaligned")
        if assignment.size_bytes < assignment.required_bytes:
            raise InvalidStorage(f"workspace slice {assignment.storage_id} is undersized")
        if assignment.offset_bytes + assignment.size_bytes > plan.storage.workspace_bytes:
            raise InvalidStorage(f"workspace slice {assignment.storage_id} exceeds workspace")

        lifetime = assignment.lifetime
        if lifetime.start_phase < 0 or lifetime.end_phase > len(plan.phases):
            raise InvalidStorage(f"workspace slice {assignment.storage_id} has an invalid lifetime")
        if lifetime.start_phase >= lifetime.end_phase:
            raise InvalidStorage(f"workspace slice {assignment.storage_id} has an empty lifetime")
        producer = phase_by_region.get(assignment.producer_region_id)
        if producer is None or producer.index != lifetime.start_phase:
            raise InvalidStorage(f"workspace slice {assignment.storage_id} has an invalid producer")
        for region_id in lifetime.reader_region_ids:
            reader = phase_by_region.get(region_id)
            if reader is None or not lifetime.start_phase <= reader.index < lifetime.end_phase:
                raise InvalidStorage(
                    f"workspace slice {assignment.storage_id} has an invalid reader"
                )

    for index, left in enumerate(plan.storage.workspace):
        for right in plan.storage.workspace[index + 1 :]:
            lifetime_overlap = (
                left.lifetime.start_phase < right.lifetime.end_phase
                and right.lifetime.start_phase < left.lifetime.end_phase
            )
            byte_overlap = (
                left.offset_bytes < right.offset_bytes + right.size_bytes
                and right.offset_bytes < left.offset_bytes + left.size_bytes
            )
            if lifetime_overlap and byte_overlap:
                raise InvalidStorage(
                    f"workspace slice {left.storage_id} overlaps {right.storage_id}"
                )

    _verify_worker_scratch(plan)
    _verify_launch(plan)
    _verify_collective_participation(plan)
    _verify_state_storage(plan)


def _tile_maps(selection: BodySelectionPlan) -> tuple[TileMap, ...]:
    return make_tile_maps(selection)


def _consumers(selection: BodySelectionPlan) -> dict[str, tuple[str, ...]]:
    # Boundary bindings ask once per value; build the map once per selection.
    cached = _CONSUMERS_CACHE.get(id(selection))
    if cached is not None and cached[0] is selection:
        return cached[1]
    result = _build_consumers(selection)
    _CONSUMERS_CACHE.clear()
    _CONSUMERS_CACHE[id(selection)] = (selection, result)
    return result


_CONSUMERS_CACHE: dict[int, tuple[BodySelectionPlan, dict[str, tuple[str, ...]]]] = {}


def _build_consumers(selection: BodySelectionPlan) -> dict[str, tuple[str, ...]]:
    consumers: dict[str, list[str]] = {}
    graph = selection.graph
    for selected in selection.selections:
        for value_id in selected.inputs:
            for consumer_id in _storage_value_ids(value_id, graph):
                consumers.setdefault(consumer_id, []).append(selected.region_id)
    return {value_id: tuple(region_ids) for value_id, region_ids in consumers.items()}


def _storage_value_ids(
    value_id: str, graph: ComputeGraph, seen: set[str] | None = None
) -> tuple[str, ...]:
    seen = seen or set()
    if value_id in seen:
        return ()
    seen.add(value_id)
    value = graph.values.get(value_id)
    if value is None:
        return (value_id,)
    relation = value.storage_relation
    related_ids = (value_id, relation.storage_id, *relation.aliases)
    return tuple(
        dict.fromkeys(
            related_id
            for related_id in related_ids
            if related_id is not None
            for related_id in (related_id, *_storage_value_ids(related_id, graph, seen))
        )
    )


def _output_storage_roots(graph: ComputeGraph) -> frozenset[str]:
    roots: set[str] = set()
    pending = list(graph.outputs)
    while pending:
        value = graph.values.get(pending.pop())
        if value is None:
            continue
        relation = value.storage_relation
        for related in (*relation.aliases, relation.storage_id):
            if related is not None and related != value.id and related not in roots:
                roots.add(related)
                pending.append(related)
    return frozenset(roots)


def _direct_boundary_ids(graph: ComputeGraph) -> frozenset[str]:
    parameter_names = set(getattr(graph.runtime_context, "parameter_names", None) or ())
    return frozenset(
        value_id for value_id in (*graph.inputs, *graph.outputs) if value_id in graph.values
    ) | frozenset(
        value_id
        for value_id, value in graph.values.items()
        if value.binding_name in parameter_names
    )


def _aliases_boundary(
    value_id: str, graph: ComputeGraph, boundary_ids: frozenset[str], seen: set[str] | None = None
) -> bool:
    seen = seen or set()
    if value_id in boundary_ids or value_id in seen:
        return value_id in boundary_ids
    seen.add(value_id)
    value = graph.values.get(value_id)
    if value is None:
        return False
    return any(
        _aliases_boundary(alias, graph, boundary_ids, seen)
        for alias in value.storage_relation.aliases
    )


@dataclass(frozen=True)
class _WorkspaceRequest:
    storage_id: str
    value_id: str
    producer_region_id: str
    size_bytes: int
    required_bytes: int
    alignment_bytes: int
    layout: CuTeLayout
    lifetime: StorageLifetime


def _make_workspace_request(
    storage_id: str,
    value_id: str,
    producer_region_id: str,
    layouts: tuple[CuTeLayout, ...],
    reader_region_ids: tuple[str, ...],
    phases: tuple[ScheduledPhase, ...],
) -> _WorkspaceRequest:
    if not layouts:
        raise InvalidStorage(f"workspace value {value_id} has no layout")
    required_bytes = max(_layout_size_bytes(layout) for layout in layouts)
    alignment_bytes = max(128, *(layout.alignment_bytes for layout in layouts))
    size = _round_up(required_bytes, 128)
    phase_by_region = {phase.region_id: phase for phase in phases}
    producer = phase_by_region.get(producer_region_id)
    if producer is None:
        raise InvalidStorage(f"workspace value {value_id} has no producer phase")
    readers = tuple(
        sorted(
            (phase_by_region[region_id] for region_id in reader_region_ids),
            key=lambda phase: phase.index,
        )
    )
    last_use = max((reader.index for reader in readers), default=producer.index)
    last_use = max(last_use, producer.index)
    # Phases between grid barriers may run concurrently on different CTAs, so
    # storage is released at the barrier that closes its last use, not after
    # the using phase.
    end_index = last_use
    while end_index + 1 < len(phases) and phases[end_index].barrier is None:
        end_index += 1
    end_phase = end_index + 1
    release_phase = phases[end_index]
    return _WorkspaceRequest(
        storage_id,
        value_id,
        producer_region_id,
        size,
        required_bytes,
        alignment_bytes,
        layouts[0],
        StorageLifetime(
            producer.index,
            end_phase,
            tuple(dict.fromkeys(reader_region_ids)),
            "phase_barrier" if release_phase.barrier is not None else "body_completion",
        ),
    )


def _allocate_distinct_workspace(
    requests: Sequence[_WorkspaceRequest],
) -> tuple[Mapping[str, int], int]:
    offsets: dict[str, int] = {}
    end = 0
    for request in requests:
        offsets[request.storage_id] = _round_up(end, request.alignment_bytes)
        end = offsets[request.storage_id] + request.size_bytes
    return offsets, end


def _allocate_workspace(
    requests: Sequence[_WorkspaceRequest],
) -> tuple[Mapping[str, int], int]:
    events: list[tuple[int, int, str]] = []
    for request in requests:
        events.extend(
            (
                (request.lifetime.start_phase, 1, request.storage_id),
                (request.lifetime.end_phase, 0, request.storage_id),
            )
        )
    events.sort()

    request_by_id = {request.storage_id: request for request in requests}
    offsets: dict[str, int] = {}
    free_ranges: list[tuple[int, int]] = []
    peak = 0
    for _, event_kind, storage_id in events:
        request = request_by_id[storage_id]
        if event_kind == 0:
            free_ranges.append((offsets[storage_id], offsets[storage_id] + request.size_bytes))
            free_ranges.sort()
            merged: list[tuple[int, int]] = []
            for start, end in free_ranges:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            free_ranges = merged
            continue

        candidates = []
        for range_index, (start, end) in enumerate(free_ranges):
            offset = _round_up(start, request.alignment_bytes)
            if offset + request.size_bytes <= end:
                candidates.append((end - offset - request.size_bytes, offset, range_index))
        if candidates:
            _, offset, range_index = min(candidates)
            start, end = free_ranges.pop(range_index)
            if start < offset:
                free_ranges.append((start, offset))
            if offset + request.size_bytes < end:
                free_ranges.append((offset + request.size_bytes, end))
            free_ranges.sort()
        else:
            offset = _round_up(peak, request.alignment_bytes)
            peak = offset + request.size_bytes
        offsets[storage_id] = offset
    return offsets, _round_up(peak, 128)


def _boundary_bindings(
    selection: BodySelectionPlan, tile_maps: Mapping[str, TileMap]
) -> list[TensorBinding]:
    graph = selection.graph
    parameter_names = set(getattr(graph.runtime_context, "parameter_names", None) or ())
    state_value_ids = _state_value_ids(graph)
    bindings: list[TensorBinding] = []

    for value_id in graph.inputs:
        bindings.append(
            _make_boundary_binding(
                value_id,
                selection,
                tile_maps,
                "state" if value_id in state_value_ids else "input",
            )
        )

    for value_id, value in graph.values.items():
        if value.binding_name in parameter_names:
            bindings.append(_make_boundary_binding(value_id, selection, tile_maps, "parameter"))

    for value_id in graph.outputs:
        bindings.append(_make_boundary_binding(value_id, selection, tile_maps, "output"))

    selected_outputs = {
        value_id for selected in selection.selections for value_id in selected.outputs
    }
    for selected in selection.selections:
        for value_id in selected.inputs:
            if value_id not in selected_outputs and not any(
                binding.value_id == value_id for binding in bindings
            ):
                bindings.append(
                    _make_boundary_binding(
                        value_id,
                        selection,
                        tile_maps,
                        "state" if value_id in state_value_ids else "alias",
                    )
                )
    # A region output that aliases a boundary tensor and has no reader, such
    # as the result of an in-place state write, still needs its binding.
    direct_boundary_ids = _direct_boundary_ids(graph)
    bound = {binding.value_id for binding in bindings}
    for selected in selection.selections:
        for value_id in selected.outputs:
            if value_id not in bound and _aliases_boundary(value_id, graph, direct_boundary_ids):
                bindings.append(_make_boundary_binding(value_id, selection, tile_maps, "alias"))
                bound.add(value_id)
    return bindings


def _state_value_ids(graph: ComputeGraph) -> frozenset[str]:
    state_ids: set[str] = set()
    for operation in graph.ops.values():
        if not (
            operation.effects.get("mutates_inputs") or operation.effects.get("may_mutate_inputs")
        ):
            continue
        written = operation.effects.get("mutated_operands", operation.operands)
        state_ids.update(value_id for value_id in written if value_id in graph.inputs)
    return frozenset(state_ids)


def _state_assignments(
    selection: BodySelectionPlan,
) -> tuple[StateStorageAssignment, ...]:
    graph = selection.graph
    if graph is None:
        raise InvalidStorage("body selection is missing its source graph")
    assignments: dict[tuple[str, str], StateStorageAssignment] = {}
    for selected in selection.selections:
        for access in selected.program.state_accesses:
            key = access.storage_identity
            existing = assignments.get(key)
            read_regions = tuple(
                dict.fromkeys((*(existing.read_regions if existing else ()), selected.region_id))
            )
            write_regions = tuple(
                dict.fromkeys((*(existing.write_regions if existing else ()), selected.region_id))
            )
            assignments[key] = StateStorageAssignment(
                key_cache_value_id=access.key_cache_value_id,
                value_cache_value_id=access.value_cache_value_id,
                valid_length_value_id=access.valid_length_value_id,
                storage_identity=access.storage_identity,
                capacity=access.capacity,
                query_length=access.query_length,
                key_layout=_state_layout(graph, access.key_cache_value_id),
                value_layout=_state_layout(graph, access.value_cache_value_id),
                read_regions=read_regions,
                write_regions=write_regions,
                read_range=access.read_range,
                write_range=access.write_range,
                source_order=access.source_order,
            )
    return tuple(assignments.values())


def _state_layout(graph: ComputeGraph, value_id: str) -> CuTeLayout:
    value = graph.values[value_id]
    if value.shape is None or value.strides is None or value.dtype is None:
        raise InvalidStorage(f"state value {value_id} lacks a layout")
    return CuTeLayout(
        value.dtype,
        value.shape,
        value.strides,
        max(16, DTYPE_BYTES.get(value.dtype, 1)),
    )


def _verify_state_storage(plan: KernelPlan) -> None:
    workspace_ids = {assignment.storage_id for assignment in plan.storage.workspace}
    for state in plan.storage.state:
        if (
            state.lifetime != "boundary_state"
            or state.capacity <= 0
            or state.query_length <= 0
            or state.capacity < state.query_length
            or not state.read_regions
            or not state.write_regions
            or set(state.storage_identity) & workspace_ids
        ):
            raise InvalidStorage(f"KV state {state.storage_identity} has an invalid plan")
        for value_id in (
            state.key_cache_value_id,
            state.value_cache_value_id,
            state.valid_length_value_id,
        ):
            binding = next(
                (binding for binding in plan.storage.bindings if binding.value_id == value_id),
                None,
            )
            if binding is None or binding.kind != "state":
                raise InvalidStorage(f"KV state value {value_id} is not a state binding")


def _make_boundary_binding(
    value_id: str,
    selection: BodySelectionPlan,
    tile_maps: Mapping[str, TileMap],
    kind: str,
) -> TensorBinding:
    graph = selection.graph
    value = graph.values[value_id]
    layout = None
    for selected in selection.selections:
        if value_id not in selected.inputs:
            continue
        for candidate_id, candidate_layout in tile_maps[selected.region_id].input_layouts:
            if candidate_id == value_id:
                layout = candidate_layout
                break
        if layout is not None:
            break
    if layout is None and value.shape is not None and value.strides is not None:
        layout = CuTeLayout(
            dtype=value.dtype or "unknown",
            shape=value.shape,
            strides=value.strides,
            alignment_bytes=DTYPE_BYTES.get(value.dtype or "", 1),
        )
    return TensorBinding(
        value_id=value_id,
        storage_id=value_id,
        kind=kind,
        layout=layout,
        region_ids=tuple(
            dict.fromkeys(
                (
                    *_consumers(selection).get(value_id, ()),
                    *(
                        selected.region_id
                        for selected in selection.selections
                        if value_id in selected.outputs
                    ),
                )
            )
        ),
        binding_name=value.binding_name,
        aliases=value.storage_relation.aliases,
    )


def _value_shaped(layout: CuTeLayout, value: Any) -> CuTeLayout:
    """``layout`` in ``value``'s own shape when it is a dense reshape of it:
    a GEMM with a fused epilogue writes its consumer's result as the 2-D GEMM
    output, while other regions index the value in its graph shape."""
    if value.shape is None or value.strides is None or tuple(value.shape) == layout.shape:
        return layout
    shaped = CuTeLayout(
        layout.dtype, tuple(value.shape), tuple(value.strides), layout.alignment_bytes
    )
    return shaped if layout.same_dense_memory(shaped) else layout


def _deduplicate_bindings(bindings: list[TensorBinding]) -> list[TensorBinding]:
    unique: dict[tuple[str, str, str], TensorBinding] = {}
    for binding in bindings:
        key = (binding.value_id, binding.storage_id, binding.kind)
        existing = unique.get(key)
        if existing is None:
            unique[key] = binding
            continue
        unique[key] = replace(
            existing,
            region_ids=tuple(dict.fromkeys((*existing.region_ids, *binding.region_ids))),
        )
    return list(unique.values())


def _worker_scratch(schedule: PhaseSchedule) -> tuple[WorkerScratch, ...]:
    phase_layouts = tuple(_scratch_phase(phase) for phase in schedule.phases)
    size = max((phase.size_bytes for phase in phase_layouts), default=0)
    alignment = max((phase.alignment_bytes for phase in phase_layouts), default=1)
    return tuple(
        WorkerScratch(worker_id, 0, size, alignment, phase_layouts)
        for worker_id in range(schedule.worker_count)
    )


def _scratch_phase(phase: ScheduledPhase) -> WorkerScratchPhase:
    body = phase.body
    offset = 0
    barrier_offsets: dict[str, int] = {}
    for barrier_name in body.pipeline.barriers:
        barrier_offsets[barrier_name] = _round_up(offset, 8)
        offset = barrier_offsets[barrier_name] + 8

    descriptor_offsets: dict[str, int] = {}
    for descriptor_name in body.pipeline.descriptors:
        descriptor_offsets[descriptor_name] = _round_up(offset, 128)
        offset = descriptor_offsets[descriptor_name] + 128

    alignment = max(1, body.scratch.alignment_bytes)
    body_offset = _round_up(offset, alignment)
    size = _round_up(body_offset + body.scratch.size_bytes, alignment)
    return WorkerScratchPhase(
        region_id=phase.region_id,
        phase_index=phase.index,
        size_bytes=size,
        alignment_bytes=alignment,
        body_data_offset_bytes=body_offset,
        barrier_offsets=barrier_offsets,
        descriptor_offsets=descriptor_offsets,
        phase_initialization={barrier_name: 0 for barrier_name in body.pipeline.barriers},
        lifetime=StorageLifetime(
            phase.index,
            phase.index + 1,
            (),
            "phase_barrier" if phase.barrier is not None else "body_completion",
        ),
    )


def _launch_requirements(
    schedule: PhaseSchedule, target: TargetFacts, storage: StoragePlan
) -> LaunchRequirements:
    threads = max(
        (phase.participants.participating_threads for phase in schedule.phases), default=1
    )
    registers = max(
        (phase.participants.registers_per_thread for phase in schedule.phases), default=1
    )
    active_blocks = min(
        (
            block_limit
            for block_limit in (phase.resources.maximum_active_blocks for phase in schedule.phases)
            if block_limit is not None
        ),
        default=None,
    )
    return LaunchRequirements(
        architecture=target.architecture,
        cooperative=True,
        grid_size=schedule.worker_count,
        threads_per_block=threads,
        dynamic_shared_memory_bytes=storage.worker_scratch_bytes,
        registers_per_thread=registers,
        max_active_blocks_per_sm=active_blocks,
        required_active_blocks=schedule.worker_count,
    )


def _verify_worker_scratch(plan: KernelPlan) -> None:
    target = plan.selection.target
    scratch_by_worker = {scratch.worker_id: scratch for scratch in plan.storage.worker_scratch}
    if tuple(sorted(scratch_by_worker)) != tuple(range(plan.launch.grid_size)):
        raise InvalidStorage("worker scratch assignments do not cover every worker")
    for worker in plan.storage.worker_scratch:
        if worker.offset_bytes != 0:
            raise InvalidStorage(f"worker {worker.worker_id} scratch is not CTA-local")
        if len(worker.phases) != len(plan.phases):
            raise InvalidStorage(f"worker {worker.worker_id} lacks a phase scratch layout")
        for phase_layout, phase in zip(worker.phases, plan.phases, strict=True):
            if phase_layout.region_id != phase.region_id:
                raise InvalidStorage(f"worker {worker.worker_id} scratch phase order is invalid")
            if worker.size_bytes < phase_layout.size_bytes:
                raise InvalidStorage(f"worker {worker.worker_id} scratch is undersized")
            if worker.alignment_bytes % phase_layout.alignment_bytes:
                raise InvalidStorage(f"worker {worker.worker_id} scratch alignment is incompatible")
            if tuple(phase_layout.barrier_offsets) != phase.body.pipeline.barriers:
                raise InvalidStorage(
                    f"{phase.region_id} scratch barrier assignment is incompatible"
                )
            if tuple(phase_layout.descriptor_offsets) != tuple(phase.body.pipeline.descriptors):
                raise InvalidStorage(
                    f"{phase.region_id} scratch descriptor assignment is incompatible"
                )
            lifetime = phase_layout.lifetime
            if (
                lifetime.start_phase != phase.index
                or lifetime.end_phase != phase.index + 1
                or lifetime.reader_region_ids
                or lifetime.release_kind not in {"phase_barrier", "body_completion"}
            ):
                raise InvalidStorage(f"{phase.region_id} scratch lifetime is invalid")
        if target is not None and worker.size_bytes > target.max_shared_memory_per_cta:
            raise InvalidStorage(
                f"worker {worker.worker_id} scratch exceeds shared-memory capacity"
            )


def _verify_launch(plan: KernelPlan) -> None:
    target = plan.selection.target
    launch = plan.launch
    if target is None:
        return
    if launch.architecture != target.architecture or not launch.cooperative:
        raise InvalidStorage("launch requirements do not match the target")
    if launch.grid_size != len(plan.storage.worker_scratch):
        raise InvalidStorage("launch grid does not match worker scratch")
    if launch.threads_per_block > target.max_threads_per_cta:
        raise InvalidStorage("launch requires too many threads per CTA")
    if launch.registers_per_thread > target.max_registers_per_thread:
        raise InvalidStorage("launch requires too many registers per thread")
    if launch.dynamic_shared_memory_bytes != plan.storage.worker_scratch_bytes:
        raise InvalidStorage("launch shared-memory size does not match worker scratch")
    capacity = launch.max_active_blocks_per_sm
    if (
        capacity is not None
        and launch.required_active_blocks > capacity * target.multiprocessor_count
    ):
        raise InvalidStorage("launch cannot sustain the required resident workers")


def _verify_collective_participation(plan: KernelPlan) -> None:
    all_workers = tuple(range(plan.launch.grid_size))
    for phase in plan.phases:
        worker_ids = tuple(worker.worker_id for worker in phase.workers)
        if len(set(worker_ids)) != len(worker_ids) or any(
            worker_id not in all_workers for worker_id in worker_ids
        ):
            raise InvalidStorage(f"{phase.region_id} has invalid worker participation")
        participating_ids = tuple(
            worker.worker_id for worker in phase.workers if worker.participates_in_barrier
        )
        if phase.barrier is None:
            if participating_ids not in ((), all_workers):
                raise InvalidStorage(f"{phase.region_id} has invalid terminal barrier membership")
        elif phase.barrier.participating_workers != all_workers or participating_ids != all_workers:
            raise InvalidStorage(f"{phase.region_id} barrier does not include every worker")
        if (
            phase.participants.waiting_threads < 0
            or phase.participants.waiting_threads > phase.participants.participating_threads
        ):
            raise InvalidStorage(f"{phase.region_id} has invalid thread participation")


def _phase_values(phase: ScheduledPhase, selection: BodySelectionPlan) -> tuple[str, ...]:
    selected = next(
        selected for selected in selection.selections if selected.region_id == phase.region_id
    )
    return tuple(dict.fromkeys((*selected.inputs, *selected.outputs)))


def _layout_size_bytes(layout: CuTeLayout) -> int:
    element_count = 1 + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(layout.shape, layout.strides, strict=True)
    )
    return element_count * DTYPE_BYTES.get(layout.dtype, 0)


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
