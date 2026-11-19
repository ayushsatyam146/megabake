from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from bisect import bisect_left, bisect_right
import heapq

from .bodies.interface import BodyConfig, ParticipantContract, WorkItem
from .graph import ComputeGraph
from .plan import (
    BodySelectionPlan,
    ConversionBody,
    ResourceRequirements,
    SelectedBody,
    TileMap,
)
from .planner import make_tile_maps


class InvalidSchedule(ValueError):
    pass


class DependencyKind(StrEnum):
    PRODUCER_TO_CONSUMER = "producer_to_consumer"
    LAYOUT_CONVERSION = "layout_conversion"
    STATE_ACCESS = "state_access"
    WRITE_AFTER_READ = "write_after_read"
    WRITE_AFTER_WRITE = "write_after_write"


@dataclass(frozen=True)
class RegionDependency:
    producer_region_id: str
    consumer_region_id: str
    value_id: str
    kind: DependencyKind
    producer_tiles: tuple[tuple[int, ...], ...]
    consumer_tiles: tuple[tuple[int, ...], ...]
    conservative: bool = False
    reason: str = ""


@dataclass(frozen=True)
class PrerequisiteSet:
    region_id: str
    prerequisites: tuple[str, ...]


@dataclass(frozen=True)
class DependencyPlan:
    dependencies: tuple[RegionDependency, ...]
    prerequisite_sets: tuple[PrerequisiteSet, ...]

    @property
    def conservative(self) -> bool:
        return any(dependency.conservative for dependency in self.dependencies)

    def prerequisites_for(self, region_id: str) -> frozenset[str]:
        for record in self.prerequisite_sets:
            if record.region_id == region_id:
                return frozenset(record.prerequisites)
        raise InvalidSchedule(f"region {region_id} has no prerequisite set")


@dataclass(frozen=True)
class PhaseBarrier:
    id: str
    participating_workers: tuple[int, ...]
    kind: str = "grid"


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    region_id: str
    work_items: tuple[WorkItem, ...]
    participates_in_barrier: bool = True


@dataclass(frozen=True)
class ScheduledPhase:
    index: int
    region_id: str
    body: BodyConfig
    participants: ParticipantContract
    resources: ResourceRequirements
    work_items: tuple[WorkItem, ...]
    workers: tuple[WorkerAssignment, ...]
    dependencies: tuple[str, ...]
    conversion_bodies: tuple[ConversionBody, ...] = ()
    barrier: PhaseBarrier | None = None
    # Prerequisites that run earlier in this phase's own barrier interval: the
    # producer publishes a counter per output tile and this phase waits on
    # the tiles it reads (see _count_gemm_inputs).
    counted: tuple[str, ...] = ()
    # Every worker runs all of this phase's work items before the other
    # phases of its barrier interval, which may then depend on it. Small
    # producers fold into the interval of their consumers this way instead
    # of costing a barrier interval of their own (see _fold_small_intervals).
    recompute: bool = False


@dataclass(frozen=True)
class PhaseSchedule:
    phases: tuple[ScheduledPhase, ...]
    worker_count: int
    prerequisite_sets: tuple[PrerequisiteSet, ...]
    selection: BodySelectionPlan | None = None
    dependency_plan: DependencyPlan | None = None

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(phase.region_id for phase in self.phases)

    @property
    def barriers(self) -> tuple[PhaseBarrier, ...]:
        return tuple(phase.barrier for phase in self.phases if phase.barrier)

    def prerequisites_for(self, region_id: str) -> frozenset[str]:
        for record in self.prerequisite_sets:
            if record.region_id == region_id:
                return frozenset(record.prerequisites)
        raise InvalidSchedule(f"region {region_id} has no prerequisite set")


@dataclass(frozen=True)
class ScheduleSimulation:
    phase_order: tuple[str, ...]
    barrier_order: tuple[str, ...]
    work_item_counts: Mapping[str, int]
    idle_worker_counts: Mapping[str, int]


@dataclass(frozen=True)
class TilePrerequisite:
    consumer_task_id: str
    producer_task_ids: tuple[str, ...]
    dependencies: tuple[RegionDependency, ...]
    conservative_analysis: bool = False

    @property
    def conservative(self) -> bool:
        return self.conservative_analysis or any(
            dependency.conservative for dependency in self.dependencies
        )


@dataclass(frozen=True)
class EventScheduleEvent:
    id: int
    producer_task_ids: tuple[str, ...]
    consumer_task_ids: tuple[str, ...]
    trigger_count: int
    prerequisites: tuple[TilePrerequisite, ...]


@dataclass(frozen=True)
class ScheduledTask:
    id: str
    order: int
    region_id: str
    work_item: WorkItem
    worker_id: int
    wait_event_ids: tuple[int, ...]
    release_event_ids: tuple[int, ...]
    work_items: tuple[WorkItem, ...] = ()


@dataclass(frozen=True)
class EventWorker:
    worker_id: int
    task_ids: tuple[str, ...]


@dataclass(frozen=True)
class EventSchedule:
    tasks: tuple[ScheduledTask, ...]
    events: tuple[EventScheduleEvent, ...]
    workers: tuple[EventWorker, ...]
    worker_count: int
    selection: BodySelectionPlan
    dependency_plan: DependencyPlan
    phase_schedule: PhaseSchedule
    prerequisites: tuple[TilePrerequisite, ...]

    def task_for(self, task_id: str) -> ScheduledTask:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise InvalidSchedule(f"event schedule has no task {task_id}")

    def event_for(self, event_id: int) -> EventScheduleEvent:
        for event in self.events:
            if event.id == event_id:
                return event
        raise InvalidSchedule(f"event schedule has no event {event_id}")

@dataclass(frozen=True)
class EventScheduleSimulation:
    task_order: tuple[str, ...]
    event_order: tuple[int, ...]
    finish_time: int
    phase_finish_time: int
    completed_task_ids: tuple[str, ...]
    event_completion_times: Mapping[int, int]
    worker_idle_ticks: Mapping[int, int]
    output_tiles_by_region: Mapping[str, tuple[tuple[int, ...], ...]]
    outputs_match_phase: bool

    @property
    def speedup_ticks(self) -> int:
        return self.phase_finish_time - self.finish_time


def derive_dependencies(selection: BodySelectionPlan) -> DependencyPlan:
    """Dependencies of ``selection``'s regions; computed once per selection
    object, since plan search derives them for every worker count."""
    cached = _DEPENDENCY_CACHE.get(id(selection))
    if cached is not None and cached[0] is selection:
        return cached[1]
    result = _derive_dependencies(selection)
    if len(_DEPENDENCY_CACHE) >= 8:
        _DEPENDENCY_CACHE.pop(next(iter(_DEPENDENCY_CACHE)))
    _DEPENDENCY_CACHE[id(selection)] = (selection, result)
    return result


_DEPENDENCY_CACHE: dict[int, tuple[BodySelectionPlan, DependencyPlan]] = {}


def _derive_dependencies(selection: BodySelectionPlan) -> DependencyPlan:
    if not selection.report.complete:
        raise InvalidSchedule("cannot schedule a plan with unsupported regions")
    if selection.graph is None:
        raise InvalidSchedule("body selection is missing its source graph")

    selected_by_region = {
        selected.region_id: selected for selected in selection.selections
    }
    tile_maps = {
        tile_map.region_id: tile_map for tile_map in make_tile_maps(selection)
    }
    dependencies: list[RegionDependency] = []

    for consumer in selection.selections:
        for producer_region_id in consumer.dependencies:
            producer = selected_by_region.get(producer_region_id)
            if producer is None:
                continue
            for value_id in consumer.inputs:
                aliases = {
                    value_id,
                    *selection.graph.values[value_id].storage_relation.aliases,
                    selection.graph.values[value_id].storage_relation.storage_id,
                }
                if not aliases.intersection(producer.outputs):
                    continue
                dependencies.append(
                    _tile_dependency(
                        producer_region_id,
                        consumer.region_id,
                        value_id,
                        tile_maps[producer_region_id],
                        tile_maps[consumer.region_id],
                    )
                )

    for conversion in selection.conversions:
        dependencies.append(
            RegionDependency(
                producer_region_id=conversion.source_region_id,
                consumer_region_id=conversion.target_region_id,
                value_id=conversion.value_id,
                kind=DependencyKind.LAYOUT_CONVERSION,
                producer_tiles=_output_tiles(tile_maps[conversion.source_region_id]),
                consumer_tiles=_consumer_tiles(
                    tile_maps[conversion.target_region_id], conversion.value_id
                ),
            )
        )

    dependencies.extend(_storage_order_dependencies(selection, tile_maps))
    dependencies.extend(_state_access_dependencies(selection, tile_maps))
    plan = _dependency_plan(tuple(dependencies), selected_by_region)
    _topological_order(selection.selections, plan)
    return plan


def make_phase_schedule(
    selection: BodySelectionPlan,
    dependencies: DependencyPlan,
    *,
    worker_count: int | None = None,
    fold_small_intervals: bool = False,
) -> PhaseSchedule:
    if not selection.report.complete:
        raise InvalidSchedule("cannot schedule a plan with unsupported regions")
    if not isinstance(dependencies, DependencyPlan):
        raise InvalidSchedule("dependencies must come from derive_dependencies")

    selected_by_region = {
        selected.region_id: selected for selected in selection.selections
    }
    for dependency in dependencies.dependencies:
        if dependency.producer_region_id not in selected_by_region:
            raise InvalidSchedule(
                f"dependency names unscheduled producer {dependency.producer_region_id}"
            )
        if dependency.consumer_region_id not in selected_by_region:
            raise InvalidSchedule(
                f"dependency names unscheduled consumer {dependency.consumer_region_id}"
            )

    order, barrier_after = _level_order(
        _topological_order(selection.selections, dependencies), dependencies
    )
    recompute: frozenset[str] = frozenset()
    if fold_small_intervals:
        order, barrier_after, recompute = _fold_small_intervals(
            order, barrier_after, selection, selected_by_region
        )
    # Plans list bodies in phase order.
    selection = replace(
        selection, selections=tuple(selected_by_region[region_id] for region_id in order)
    )
    tile_maps = {
        tile_map.region_id: tile_map for tile_map in make_tile_maps(selection)
    }
    default_worker_count = _worker_count(selection, tile_maps)
    worker_capacity = (
        selection.target.multiprocessor_count
        * min(
            selected.resources.maximum_active_blocks or 1
            for selected in selection.selections
        )
        if selection.target is not None and selection.selections
        else None
    )
    if worker_count is None:
        worker_count = default_worker_count
    elif worker_count < 1:
        raise InvalidSchedule(f"worker count {worker_count} is not positive")
    elif worker_capacity is not None and worker_count > worker_capacity:
        raise InvalidSchedule(
            f"worker count {worker_count} exceeds capacity {worker_capacity}"
        )
    phases: list[ScheduledPhase] = []

    for phase_index, region_id in enumerate(order):
        selected = selected_by_region[region_id]
        tile_map = tile_maps[region_id]
        work_items = tile_map.work_items
        folded = region_id in recompute
        barrier = (
            PhaseBarrier(
                f"phase:{phase_index}:grid",
                tuple(range(worker_count)),
            )
            if barrier_after[phase_index]
            else None
        )
        phases.append(
            ScheduledPhase(
                index=phase_index,
                region_id=region_id,
                body=selected.body,
                participants=selected.participants,
                resources=selected.resources,
                work_items=work_items,
                workers=(
                    _recompute_assignments(region_id, work_items, worker_count)
                    if folded
                    else _worker_assignments(region_id, work_items, worker_count)
                ),
                dependencies=tuple(
                    sorted(dependencies.prerequisites_for(region_id))
                ),
                conversion_bodies=tuple(
                    conversion
                    for conversion in selection.conversions
                    if conversion.source_region_id == region_id
                ),
                barrier=barrier,
                recompute=folded,
            )
        )

    if counted_gemm_enabled():
        phases = _count_gemm_inputs(phases, dependencies, selected_by_region)
    return PhaseSchedule(
        tuple(phases),
        worker_count,
        dependencies.prerequisite_sets,
        selection,
        dependencies,
    )


def counted_gemm_enabled() -> bool:
    """A decode GEMM whose activation is exactly another GEMM's output joins
    that GEMM's interval and waits per K tile on the producer's tile counters
    (MEGABAKE_COUNTED_GEMM=0: a grid barrier as before)."""
    import os

    return os.environ.get("MEGABAKE_COUNTED_GEMM", "1") == "1"


def _count_gemm_inputs(
    phases: list[ScheduledPhase], dependencies: DependencyPlan, selected_by_region
) -> list[ScheduledPhase]:
    """Drop the grid barrier in front of a GEMM whose prerequisites in the
    previous interval are one unsplit GEMM producing its whole activation
    (_counted_gemm_consumer); the consumer then waits on the producer's tile
    counters, K tile by K tile, and overlaps the producer's last tiles.

    Measured on Llama-3.2-1B decode (2026-11-19, 60 SMs): gate.up -> down,
    1,529 -> 1,454 us. Doing the same for decode attention after q/k/v
    (waiting on its heads' tiles) was 1,564 us and was removed. The memory
    planner releases storage at the barrier that closes its last use, so the
    merged interval keeps every buffer it touches distinct."""
    phases = list(phases)
    for index, phase in enumerate(phases):
        if index == 0 or phases[index - 1].barrier is None:
            continue
        if not _counted_gemm_consumer(phase, dependencies, selected_by_region):
            continue
        # The interval that ends at index - 1 began after the barrier before it.
        start = max(
            (position + 1 for position in range(index - 1) if phases[position].barrier is not None),
            default=0,
        )
        interval = {item.region_id: item for item in phases[start:index]}
        needed = dependencies.prerequisites_for(phase.region_id) & set(interval)
        if len(needed) == 1 and _counted_producer(interval[next(iter(needed))]):
            phases[index - 1] = replace(phases[index - 1], barrier=None)
            phases[index] = replace(phase, counted=tuple(needed))
    return phases


def _counted_gemm_consumer(phase: ScheduledPhase, dependencies, selected_by_region) -> bool:
    """An unsplit swapAB GEMM with one token row whose activation (CUTLASS B,
    the program's first load) is one GEMM's whole output, its K tile k that
    GEMM's output tile k."""
    if selected_by_region is None or not _counted_producer(phase):
        return False
    program = selected_by_region[phase.region_id].program
    if program.store.shape[0] != 1:
        return False
    activation = program.loads[0].value_id
    sources = [
        dependency
        for dependency in dependencies.dependencies
        if dependency.consumer_region_id == phase.region_id and dependency.value_id == activation
    ]
    if len(sources) != 1:
        return False
    producer = selected_by_region.get(sources[0].producer_region_id)
    if producer is None or producer.body.provider != "g0_range":
        return False
    reduction = next(
        domain.size for domain in program.iteration_domains if domain.axis in program.reduction_axes
    )
    return tuple(producer.program.store.shape) == (1, reduction)


def _counted_producer(phase: ScheduledPhase) -> bool:
    import re

    return (
        phase.body.provider == "g0_range"
        and phase.body.name.startswith("g0_range_swapab_")
        and re.search(r"_k\d+(?=_|$)", phase.body.name) is None
        and not phase.recompute
    )


def simulate_phase_schedule(schedule: PhaseSchedule) -> ScheduleSimulation:
    # Results become visible to other workers only at a grid barrier; phases
    # in one barrier interval may run concurrently and must be independent.
    completed: set[str] = set()
    interval: set[str] = set()
    phase_order: list[str] = []
    barrier_order: list[str] = []
    work_counts: dict[str, int] = {}
    idle_counts: dict[str, int] = {}

    # Recompute phases of the current interval, which every worker has run.
    recomputed: set[str] = set()
    for index, phase in enumerate(schedule.phases):
        if phase.recompute and interval - recomputed:
            raise InvalidSchedule(
                f"recompute phase {phase.index} follows a distributed phase of its interval"
            )
        counted = set(phase.counted)
        if counted - interval:
            raise InvalidSchedule(
                f"phase {phase.index} counts on {sorted(counted - interval)} outside its interval"
            )
        concurrent = schedule.prerequisites_for(phase.region_id) & (
            interval - recomputed - counted
        )
        if concurrent:
            raise InvalidSchedule(
                f"phase {phase.index} shares a barrier interval with its prerequisites "
                f"{sorted(concurrent)}"
            )
        missing = schedule.prerequisites_for(phase.region_id) - completed - recomputed - counted
        if missing:
            raise InvalidSchedule(
                f"phase {phase.index} has unfinished prerequisites {sorted(missing)}"
            )
        if phase.index != index:
            raise InvalidSchedule("phase indices are not contiguous")
        if len(phase.workers) != schedule.worker_count:
            raise InvalidSchedule(f"phase {phase.index} does not cover every worker")

        worker_ids = [worker.worker_id for worker in phase.workers]
        if worker_ids != list(range(schedule.worker_count)):
            raise InvalidSchedule(f"phase {phase.index} has invalid worker IDs")
        if phase.recompute:
            if any(worker.work_items != phase.work_items for worker in phase.workers):
                raise InvalidSchedule(
                    f"recompute phase {phase.index} does not give every worker every item"
                )
        else:
            assigned = Counter(
                work_item
                for worker in phase.workers
                for work_item in worker.work_items
            )
            if assigned != Counter(phase.work_items):
                raise InvalidSchedule(f"phase {phase.index} has invalid work assignments")
        if any(not worker.participates_in_barrier for worker in phase.workers):
            raise InvalidSchedule(f"phase {phase.index} has a worker outside its barrier")

        is_final = index + 1 == len(schedule.phases)
        if phase.barrier is not None:
            if is_final:
                raise InvalidSchedule("the final phase has an unnecessary barrier")
            if phase.barrier.kind != "grid":
                raise InvalidSchedule(f"phase {phase.index} has an invalid barrier")
            if phase.barrier.participating_workers != tuple(
                range(schedule.worker_count)
            ):
                raise InvalidSchedule(f"phase {phase.index} has an incomplete barrier")
            barrier_order.append(phase.barrier.id)

        phase_order.append(phase.region_id)
        work_counts[phase.region_id] = len(phase.work_items)
        idle_counts[phase.region_id] = sum(
            not worker.work_items for worker in phase.workers
        )
        interval.add(phase.region_id)
        if phase.recompute:
            recomputed.add(phase.region_id)
        if phase.barrier is not None or is_final:
            completed |= interval
            interval = set()
            recomputed = set()

    return ScheduleSimulation(
        tuple(phase_order),
        tuple(barrier_order),
        work_counts,
        idle_counts,
    )


def make_event_schedule(
    selection: BodySelectionPlan,
    dependencies: DependencyPlan,
    *,
    worker_count: int | None = None,
) -> EventSchedule:
    if not selection.report.complete:
        raise InvalidSchedule("cannot schedule a plan with unsupported regions")
    phase_schedule = make_phase_schedule(
        selection, dependencies, worker_count=worker_count
    )
    tile_maps = {
        tile_map.region_id: tile_map
        for tile_map in make_tile_maps(selection)
    }
    selected_worker_count = phase_schedule.worker_count
    region_order = phase_schedule.region_ids
    task_records: list[ScheduledTask] = []
    for region_id in region_order:
        for access in tile_maps[region_id].accesses:
            order = len(task_records)
            task_records.append(
                ScheduledTask(
                    id=f"{region_id}:{access.output.origin}",
                    order=order,
                    region_id=region_id,
                    work_item=access.work_item,
                    worker_id=order % selected_worker_count,
                    wait_event_ids=(),
                    release_event_ids=(),
                )
            )
    tasks = tuple(task_records)
    if len({task.id for task in tasks}) != len(tasks):
        raise InvalidSchedule("event task IDs are not unique")

    prerequisites = _tile_prerequisites(tasks, dependencies, tile_maps)
    grouped: dict[tuple[str, ...], dict[str, list[RegionDependency]]] = {}
    for prerequisite in prerequisites:
        by_consumer = grouped.setdefault(prerequisite.producer_task_ids, {})
        by_consumer.setdefault(prerequisite.consumer_task_id, []).extend(
            prerequisite.dependencies
        )

    events: list[EventScheduleEvent] = []
    wait_events: dict[str, list[int]] = {}
    release_events: dict[str, list[int]] = {}
    for event_id, (producer_task_ids, by_consumer) in enumerate(grouped.items()):
        consumer_task_ids = tuple(sorted(by_consumer))
        members = tuple(
            TilePrerequisite(
                consumer_task_id,
                producer_task_ids,
                tuple(
                    dict.fromkeys(
                        by_consumer[consumer_task_id]
                    )
                ),
            )
            for consumer_task_id in consumer_task_ids
        )
        event = EventScheduleEvent(
            event_id,
            producer_task_ids,
            consumer_task_ids,
            len(producer_task_ids),
            members,
        )
        events.append(event)
        for consumer_task_id in consumer_task_ids:
            wait_events.setdefault(consumer_task_id, []).append(event_id)
        for producer_task_id in producer_task_ids:
            release_events.setdefault(producer_task_id, []).append(event_id)

    tasks = tuple(
        replace(
            task,
            wait_event_ids=tuple(wait_events.get(task.id, ())),
            release_event_ids=tuple(release_events.get(task.id, ())),
        )
        for task in tasks
    )
    workers = tuple(
        EventWorker(
            worker_id,
            tuple(task.id for task in tasks if task.worker_id == worker_id),
        )
        for worker_id in range(selected_worker_count)
    )
    schedule = EventSchedule(
        tasks,
        tuple(events),
        workers,
        selected_worker_count,
        selection,
        dependencies,
        phase_schedule,
        tuple(prerequisites),
    )
    simulate_event_schedule(schedule)
    return schedule


def make_region_event_schedule(
    selection: BodySelectionPlan,
    dependencies: DependencyPlan,
    *,
    worker_count: int | None = None,
) -> EventSchedule:
    phase_schedule = make_phase_schedule(
        selection, dependencies, worker_count=worker_count
    )
    tasks = tuple(
        ScheduledTask(
            id=f"{phase.region_id}:{worker_id}",
            order=phase.index * phase_schedule.worker_count + worker_id,
            region_id=phase.region_id,
            work_item=(
                phase.workers[worker_id].work_items or phase.work_items[:1]
            )[0],
            worker_id=worker_id,
            wait_event_ids=(),
            release_event_ids=(),
            work_items=phase.workers[worker_id].work_items or phase.work_items[:1],
        )
        for phase in phase_schedule.phases
        for worker_id in range(phase_schedule.worker_count)
    )
    task_ids_by_region = {
        region_id: tuple(task.id for task in tasks if task.region_id == region_id)
        for region_id in phase_schedule.region_ids
    }
    grouped: dict[tuple[str, ...], set[str]] = {}
    prerequisites: list[TilePrerequisite] = []
    for consumer in phase_schedule.phases:
        for producer_region_id in consumer.dependencies:
            producer_task_ids = task_ids_by_region[producer_region_id]
            grouped.setdefault(producer_task_ids, set()).update(
                task_ids_by_region[consumer.region_id]
            )
            source_dependencies = tuple(
                dependency
                for dependency in dependencies.dependencies
                if dependency.producer_region_id == producer_region_id
                and dependency.consumer_region_id == consumer.region_id
            )
            prerequisites.extend(
                TilePrerequisite(
                    task_id,
                    producer_task_ids,
                    source_dependencies,
                    True,
                )
                for task_id in task_ids_by_region[consumer.region_id]
            )

    events: list[EventScheduleEvent] = []
    wait_events: dict[str, list[int]] = {}
    release_events: dict[str, list[int]] = {}
    for event_id, (producer_task_ids, consumer_task_ids) in enumerate(
        grouped.items()
    ):
        event = EventScheduleEvent(
            event_id,
            producer_task_ids,
            tuple(sorted(consumer_task_ids)),
            len(producer_task_ids),
            tuple(
                prerequisite
                for prerequisite in prerequisites
                if prerequisite.producer_task_ids == producer_task_ids
            ),
        )
        events.append(event)
        for consumer_task_id in event.consumer_task_ids:
            wait_events.setdefault(consumer_task_id, []).append(event_id)
        for producer_task_id in event.producer_task_ids:
            release_events.setdefault(producer_task_id, []).append(event_id)

    tasks = tuple(
        replace(
            task,
            wait_event_ids=tuple(wait_events.get(task.id, ())),
            release_event_ids=tuple(release_events.get(task.id, ())),
        )
        for task in tasks
    )
    workers = tuple(
        EventWorker(
            worker_id,
            tuple(task.id for task in tasks if task.worker_id == worker_id),
        )
        for worker_id in range(phase_schedule.worker_count)
    )
    schedule = EventSchedule(
        tasks,
        tuple(events),
        workers,
        phase_schedule.worker_count,
        selection,
        dependencies,
        phase_schedule,
        tuple(prerequisites),
    )
    simulate_event_schedule(schedule)
    return schedule


def simulate_event_schedule(
    schedule: EventSchedule,
    task_delays: Mapping[str, int] | None = None,
) -> EventScheduleSimulation:
    _verify_event_schedule(schedule)
    delays = task_delays or {}
    unknown_delays = set(delays) - {task.id for task in schedule.tasks}
    if unknown_delays:
        raise InvalidSchedule(f"delays name unknown tasks {sorted(unknown_delays)}")
    if any(delay < 0 for delay in delays.values()):
        raise InvalidSchedule("task delays cannot be negative")

    task_by_id = {task.id: task for task in schedule.tasks}
    duration_by_id = {
        task_id: (len(task_by_id[task_id].work_items) or 1)
        + delays.get(task_id, 0)
        for task_id in task_by_id
    }
    worker_cursors = [0] * schedule.worker_count
    completion_times: dict[str, int] = {}
    event_counts = {event.id: 0 for event in schedule.events}
    event_completion_times: dict[int, int] = {}
    completed_order: list[str] = []
    event_order: list[int] = []
    idle_ticks = dict.fromkeys(range(schedule.worker_count), 0)
    running_tasks: dict[int, str] = {}
    phase_finish_time = _phase_finish_time(schedule, duration_by_id)
    time = 0

    while len(completed_order) < len(schedule.tasks):
        finished = sorted(
            task_id
            for task_id, completion_time in completion_times.items()
            if completion_time == time
        )
        for task_id in finished:
            completed_order.append(task_id)
            running_worker_id = next(
                worker_id
                for worker_id, running_task_id in running_tasks.items()
                if running_task_id == task_id
            )
            del running_tasks[running_worker_id]
            for event_id in task_by_id[task_id].release_event_ids:
                event_counts[event_id] += 1
                event = schedule.event_for(event_id)
                if event_counts[event_id] == event.trigger_count:
                    event_completion_times[event_id] = time
                    event_order.append(event_id)

        started: list[str] = []
        all_tasks_complete = len(completed_order) == len(schedule.tasks)
        for worker in schedule.workers:
            if worker.worker_id in running_tasks:
                continue
            if all_tasks_complete:
                continue
            cursor = worker_cursors[worker.worker_id]
            if cursor >= len(worker.task_ids):
                if time < phase_finish_time:
                    idle_ticks[worker.worker_id] += 1
                continue
            task_id = worker.task_ids[cursor]
            task = task_by_id[task_id]
            ready = all(
                event_completion_times.get(event_id, time + 1) <= time
                for event_id in task.wait_event_ids
            )
            if not ready:
                idle_ticks[worker.worker_id] += 1
                continue
            completion_times[task_id] = time + duration_by_id[task_id]
            worker_cursors[worker.worker_id] += 1
            running_tasks[worker.worker_id] = task_id
            started.append(task_id)

        if not finished and not started:
            future = [completion for completion in completion_times.values() if completion > time]
            if not future:
                raise InvalidSchedule("event schedule cannot make progress")
            time += 1
            continue
        time += 1

    finish_time = max(completion_times.values(), default=0)
    output_tiles = {
        region_id: tuple(
            dict.fromkeys(
                work_item.output_tile
                for task in schedule.tasks
                if task.region_id == region_id
                for work_item in (task.work_items or (task.work_item,))
            )
        )
        for region_id in schedule.phase_schedule.region_ids
    }
    phase_output_tiles = {
        region_id: tuple(
            work_item.output_tile
            for phase in schedule.phase_schedule.phases
            if phase.region_id == region_id
            for work_item in phase.work_items
        )
        for region_id in schedule.phase_schedule.region_ids
    }
    return EventScheduleSimulation(
        tuple(completed_order),
        tuple(event_order),
        finish_time,
        phase_finish_time,
        tuple(sorted(task_by_id, key=lambda task_id: task_by_id[task_id].order)),
        event_completion_times,
        idle_ticks,
        output_tiles,
        all(
            set(output_tiles[region_id]) == set(phase_output_tiles[region_id])
            for region_id in output_tiles
        ),
    )


def _tile_prerequisites(
    tasks: tuple[ScheduledTask, ...],
    dependencies: DependencyPlan,
    tile_maps: Mapping[str, TileMap],
) -> tuple[TilePrerequisite, ...]:
    tasks_by_region: dict[str, dict[tuple[int, ...], ScheduledTask]] = {}
    accesses_by_region: dict[str, dict[tuple[int, ...], object]] = {}
    for task in tasks:
        tasks_by_region.setdefault(task.region_id, {})[task.work_item.output_tile] = task
        accesses_by_region.setdefault(task.region_id, {})[
            task.work_item.output_tile
        ] = next(
            access
            for access in tile_maps[task.region_id].accesses
            if access.output.origin == task.work_item.output_tile
        )

    grouped: dict[
        tuple[str, tuple[str, ...]],
        tuple[tuple[RegionDependency, ...], bool],
    ] = {}
    for dependency in dependencies.dependencies:
        producer_tasks = tasks_by_region[dependency.producer_region_id]
        consumer_tasks = tasks_by_region[dependency.consumer_region_id]
        producer_ids = tuple(sorted(task.id for task in producer_tasks.values()))
        for consumer_task in consumer_tasks.values():
            consumer_task_id = consumer_task.id
            producer_ids_for_consumer = producer_ids
            analysis_conservative = dependency.conservative
            if not dependency.conservative:
                access = accesses_by_region[dependency.consumer_region_id][
                    consumer_task.work_item.output_tile
                ]
                input_tile = next(
                    input_tile
                    for input_tile in access.inputs
                    if input_tile.value_id == dependency.value_id
                )
                exact_ids: list[str] = []
                exact = True
                for producer_task in producer_tasks.values():
                    producer_task_id = producer_task.id
                    producer_access = accesses_by_region[dependency.producer_region_id][
                        producer_task.work_item.output_tile
                    ]
                    overlaps, is_exact = _logical_tiles_overlap(
                        producer_access.output,
                        input_tile,
                    )
                    exact = exact and is_exact
                    if overlaps:
                        exact_ids.append(producer_task_id)
                analysis_conservative = analysis_conservative or not exact
                if exact and not exact_ids:
                    raise InvalidSchedule(
                        f"{consumer_task_id} reads no producer tile for "
                        f"{dependency.value_id}"
                    )
                if exact:
                    producer_ids_for_consumer = tuple(sorted(exact_ids))
            key = (consumer_task_id, producer_ids_for_consumer)
            dependencies_record, conservative_record = grouped.get(
                key,
                ((), False),
            )
            grouped[key] = (
                tuple(
                    dependency
                    for existing in dependencies_record
                    if existing != dependency
                )
                + (dependency,),
                conservative_record or analysis_conservative,
            )

    return tuple(
        TilePrerequisite(
            consumer_task_id,
            producer_ids,
            dependencies_record,
            conservative_record,
        )
        for (
            (consumer_task_id, producer_ids),
            (dependencies_record, conservative_record),
        ) in grouped.items()
    )


def _verify_event_schedule(schedule: EventSchedule) -> None:
    task_ids = {task.id for task in schedule.tasks}
    if [task.order for task in schedule.tasks] != list(range(len(schedule.tasks))):
        raise InvalidSchedule("event tasks do not use a common topological order")
    for worker in schedule.workers:
        worker_tasks = [schedule.task_for(task_id) for task_id in worker.task_ids]
        if [task.order for task in worker_tasks] != sorted(
            task.order for task in worker_tasks
        ):
            raise InvalidSchedule(f"worker {worker.worker_id} task order is not topological")
    for event in schedule.events:
        if not event.producer_task_ids or not event.consumer_task_ids:
            raise InvalidSchedule(f"event {event.id} has no producers or consumers")
        if event.trigger_count != len(set(event.producer_task_ids)):
            raise InvalidSchedule(f"event {event.id} trigger count is not unique")
        if any(
            producer_task_id not in task_ids
            for producer_task_id in event.producer_task_ids
        ):
            raise InvalidSchedule(f"event {event.id} names an unknown producer")
        if any(
            consumer_task_id not in task_ids
            for consumer_task_id in event.consumer_task_ids
        ):
            raise InvalidSchedule(f"event {event.id} names an unknown consumer")

    expected_waits: dict[str, set[int]] = {task.id: set() for task in schedule.tasks}
    expected_releases: dict[str, set[int]] = {task.id: set() for task in schedule.tasks}
    for event in schedule.events:
        for consumer_task_id in event.consumer_task_ids:
            expected_waits[consumer_task_id].add(event.id)
        for producer_task_id in event.producer_task_ids:
            expected_releases[producer_task_id].add(event.id)
    for task in schedule.tasks:
        if set(task.wait_event_ids) != expected_waits[task.id]:
            raise InvalidSchedule(f"{task.id} has incomplete event waits")
        if set(task.release_event_ids) != expected_releases[task.id]:
            raise InvalidSchedule(f"{task.id} has incomplete event releases")

    task_order = {task.id: task.order for task in schedule.tasks}
    for event in schedule.events:
        producer_orders = [task_order[task_id] for task_id in event.producer_task_ids]
        consumer_orders = [task_order[task_id] for task_id in event.consumer_task_ids]
        if max(producer_orders, default=-1) >= min(consumer_orders, default=0):
            raise InvalidSchedule(f"event {event.id} violates the common task order")

    expected_groups: dict[tuple[str, ...], set[str]] = {}
    for prerequisite in schedule.prerequisites:
        expected_groups.setdefault(
            prerequisite.producer_task_ids, set()
        ).add(prerequisite.consumer_task_id)
    actual_groups = {
        event.producer_task_ids: set(event.consumer_task_ids)
        for event in schedule.events
    }
    if actual_groups != expected_groups:
        raise InvalidSchedule("compressed events do not match concrete prerequisites")


def _phase_finish_time(
    schedule: EventSchedule, duration_by_id: Mapping[str, int]
) -> int:
    finish_time = 0
    for phase in schedule.phase_schedule.phases:
        phase_duration = max(
            (
                sum(
                    duration_by_id[task.id]
                    for task in schedule.tasks
                    if task.region_id == phase.region_id
                    and task.worker_id == worker.worker_id
                )
                for worker in phase.workers
            ),
            default=0,
        )
        finish_time += phase_duration
    return finish_time


def _tile_dependency(
    producer_region_id: str,
    consumer_region_id: str,
    value_id: str,
    producer_map: TileMap,
    consumer_map: TileMap,
) -> RegionDependency:
    producer_output_tiles = tuple(
        access.output for access in producer_map.accesses
    )
    consumer_accesses = tuple(
        access
        for access in consumer_map.accesses
        if any(input_tile.value_id == value_id for input_tile in access.inputs)
    )
    if not consumer_accesses:
        raise InvalidSchedule(
            f"{consumer_region_id} has no mapped access to {value_id}"
        )
    if all(not tile.origin for tile in producer_output_tiles):
        return RegionDependency(
            producer_region_id,
            consumer_region_id,
            value_id,
            DependencyKind.PRODUCER_TO_CONSUMER,
            ((),),
            tuple(sorted(access.output.origin for access in consumer_accesses)),
        )

    producer_origins: set[tuple[int, ...]] = set()
    consumer_tiles: set[tuple[int, ...]] = set()
    exact = True
    # Index producer tiles along the axis that separates them most. Indexing
    # the first axis alone was quadratic for [1, heads, rows, columns] state,
    # where every tile starts at batch 0.
    rank = len(producer_output_tiles[0].origin)
    axis = max(
        range(rank),
        key=lambda position: len({tile.origin[position] for tile in producer_output_tiles}),
    )
    producer_by_origin: dict[int, list[object]] = {}
    for tile in producer_output_tiles:
        producer_by_origin.setdefault(tile.origin[axis], []).append(tile)
    origins = tuple(sorted(producer_by_origin))
    maximum_extent = max(tile.shape[axis] for tile in producer_output_tiles)
    # A consumer tile that reads the producer's whole value (a GEMM reading
    # its activation, a norm its row) depends on every producer tile; one
    # overlap check settles exactness instead of one per producer tile.
    extent = tuple(
        max(tile.origin[position] + tile.shape[position] for tile in producer_output_tiles)
        for position in range(rank)
    )
    all_origins = {tile.origin for tile in producer_output_tiles}
    for access in consumer_accesses:
        input_tile = next(
            input_tile
            for input_tile in access.inputs
            if input_tile.value_id == value_id
        )
        if (
            len(input_tile.origin) == rank
            and all(origin == 0 for origin in input_tile.origin)
            and all(size >= end for size, end in zip(input_tile.shape, extent, strict=True))
        ):
            overlaps, is_exact = _logical_tiles_overlap(producer_output_tiles[0], input_tile)
            if is_exact and overlaps:
                producer_origins |= all_origins
                consumer_tiles.add(access.output.origin)
                continue
        matched = False
        if len(input_tile.origin) != rank:
            # Mismatched ranks are never exact; the overlap check reports it.
            candidates = iter(producer_output_tiles)
        else:
            consumer_begin = input_tile.origin[axis]
            consumer_end = consumer_begin + input_tile.shape[axis]
            lower = bisect_left(origins, consumer_begin - maximum_extent + 1)
            upper = bisect_right(origins, consumer_end - 1)
            candidates = (
                output_tile
                for origin in origins[lower:upper]
                for output_tile in producer_by_origin[origin]
            )
        for output_tile in candidates:
            overlaps, is_exact = _logical_tiles_overlap(output_tile, input_tile)
            exact = exact and is_exact
            if not is_exact:
                break
            if overlaps:
                producer_origins.add(output_tile.origin)
                consumer_tiles.add(access.output.origin)
                matched = True
        if exact and not matched:
            raise InvalidSchedule(
                f"{consumer_region_id} reads no producer tile for {value_id}"
            )

    if not exact:
        return RegionDependency(
            producer_region_id,
            consumer_region_id,
            value_id,
            DependencyKind.PRODUCER_TO_CONSUMER,
            _output_tiles(producer_map),
            tuple(access.output.origin for access in consumer_accesses),
            True,
            "input axes do not permit exact tile-overlap analysis",
        )
    return RegionDependency(
        producer_region_id,
        consumer_region_id,
        value_id,
        DependencyKind.PRODUCER_TO_CONSUMER,
            tuple(sorted(producer_origins)),
        tuple(sorted(consumer_tiles)),
    )


def _logical_tiles_overlap(
    producer_tile: object, consumer_tile: object
) -> tuple[bool, bool]:
    if (
        len(producer_tile.origin) != len(consumer_tile.origin)
        or producer_tile.layout.shape != consumer_tile.layout.shape
    ):
        return False, False
    for producer_origin, consumer_origin, producer_extent, consumer_extent in zip(
        producer_tile.origin,
        consumer_tile.origin,
        producer_tile.shape,
        consumer_tile.shape,
        strict=True,
    ):
        producer_end = producer_origin + producer_extent
        consumer_end = consumer_origin + consumer_extent
        if max(producer_origin, consumer_origin) >= min(
            producer_end, consumer_end
        ):
            return False, True
    return True, True


def _output_tiles(tile_map: TileMap) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(access.output.origin for access in tile_map.accesses))


def _consumer_tiles(tile_map: TileMap, value_id: str) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            access.output.origin
            for access in tile_map.accesses
            if any(input_tile.value_id == value_id for input_tile in access.inputs)
        )
    )


def _storage_order_dependencies(
    selection: BodySelectionPlan, tile_maps: Mapping[str, TileMap]
) -> tuple[RegionDependency, ...]:
    graph = selection.graph
    assert graph is not None
    unknown_effects = any(
        graph.ops[operation_id].effects.get("unknown")
        for selected in selection.selections
        for operation_id in selected.source_op_ids
    )
    reads: dict[str, frozenset[str]] = {}
    writes: dict[str, frozenset[str]] = {}
    for selected in selection.selections:
        reads[selected.region_id] = _storage_ids(graph, selected.inputs)
        writes[selected.region_id] = _storage_ids(graph, selected.outputs)
        for operation_id in selected.source_op_ids:
            operation = graph.ops[operation_id]
            if operation.effects.get("mutates_inputs") or operation.effects.get(
                "may_mutate_inputs"
            ):
                operand_storage = _storage_ids(graph, operation.operands)
                reads[selected.region_id] = reads[selected.region_id] | operand_storage
                writes[selected.region_id] = writes[selected.region_id] | operand_storage
        if unknown_effects:
            reads[selected.region_id] = reads[selected.region_id] | {
                "unknown:global-storage"
            }
            writes[selected.region_id] = writes[selected.region_id] | {
                "unknown:global-storage"
            }

    dependencies: list[RegionDependency] = []
    selections = selection.selections
    for later_index, later in enumerate(selections[1:], start=1):
        for earlier in selections[:later_index]:
            write_after_write = writes[earlier.region_id] & writes[later.region_id]
            if write_after_write:
                dependencies.append(
                    _storage_dependency(
                        earlier,
                        later,
                        DependencyKind.WRITE_AFTER_WRITE,
                        write_after_write,
                        "regions write the same storage",
                        tile_maps[earlier.region_id],
                        tile_maps[later.region_id],
                    )
                )
            write_after_read = writes[later.region_id] & reads[earlier.region_id]
            if write_after_read:
                dependencies.append(
                    _storage_dependency(
                        earlier,
                        later,
                        DependencyKind.WRITE_AFTER_READ,
                        write_after_read,
                        "later write overlaps an earlier read",
                        tile_maps[earlier.region_id],
                        tile_maps[later.region_id],
                    )
                )
    return tuple(dependencies)


def _state_access_dependencies(
    selection: BodySelectionPlan, tile_maps: Mapping[str, TileMap]
) -> tuple[RegionDependency, ...]:
    dependencies: list[RegionDependency] = []
    selections = selection.selections
    for later_index, later in enumerate(selections[1:], start=1):
        for earlier in selections[:later_index]:
            earlier_states = earlier.program.state_accesses
            later_states = later.program.state_accesses
            shared = {
                access.storage_identity
                for access in earlier_states
                if any(
                    access.storage_identity == other.storage_identity
                    for other in later_states
                )
            }
            if not shared:
                continue
            dependencies.append(
                RegionDependency(
                    producer_region_id=earlier.region_id,
                    consumer_region_id=later.region_id,
                    value_id=",".join(
                        sorted(value_id for identity in shared for value_id in identity)
                    ),
                    kind=DependencyKind.STATE_ACCESS,
                    producer_tiles=_output_tiles(tile_maps[earlier.region_id]),
                    consumer_tiles=_output_tiles(tile_maps[later.region_id]),
                    conservative=True,
                    reason="regions read and write the same bounded KV state",
                )
            )
    return tuple(dependencies)


def _storage_dependency(
    earlier: SelectedBody,
    later: SelectedBody,
    kind: DependencyKind,
    storage_ids: frozenset[str],
    reason: str,
    producer_map: TileMap,
    consumer_map: TileMap,
) -> RegionDependency:
    return RegionDependency(
        earlier.region_id,
        later.region_id,
        ",".join(sorted(storage_ids)),
        kind,
        _output_tiles(producer_map),
        _output_tiles(consumer_map),
        True,
        reason,
    )


def _storage_ids(graph: ComputeGraph, value_ids: tuple[str, ...]) -> frozenset[str]:
    storage_ids: set[str] = set()
    for value_id in value_ids:
        if value_id not in graph.values:
            raise InvalidSchedule(f"body selection references unknown value {value_id}")
        relation = graph.values[value_id].storage_relation
        if relation.storage_id is not None:
            storage_ids.add(relation.storage_id)
        storage_ids.update(relation.aliases)
        if relation.unknown or relation.storage_id is None:
            storage_ids.add(f"unknown:{value_id}")
    return frozenset(storage_ids)


def _dependency_plan(
    dependencies: tuple[RegionDependency, ...],
    selected_by_region: Mapping[str, SelectedBody],
) -> DependencyPlan:
    prerequisites = {
        region_id: set() for region_id in selected_by_region
    }
    for dependency in dependencies:
        if dependency.producer_region_id == dependency.consumer_region_id:
            raise InvalidSchedule(f"region {dependency.producer_region_id} depends on itself")
        if dependency.producer_region_id not in selected_by_region:
            raise InvalidSchedule(
                f"dependency names unscheduled producer {dependency.producer_region_id}"
            )
        if dependency.consumer_region_id not in selected_by_region:
            raise InvalidSchedule(
                f"dependency names unscheduled consumer {dependency.consumer_region_id}"
            )
        prerequisites[dependency.consumer_region_id].add(dependency.producer_region_id)
    return DependencyPlan(
        dependencies,
        tuple(
            PrerequisiteSet(region_id, tuple(sorted(values)))
            for region_id, values in prerequisites.items()
        ),
    )


def _level_order(
    order: tuple[str, ...], dependencies: DependencyPlan
) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    """Group regions by dependency level and place one grid barrier per level.

    A region's level is one more than its deepest prerequisite (data, state,
    and storage-order dependencies alike), so regions in one level never
    depend on each other and can share a barrier interval. Within a level the
    topological order is kept. This needs the fewest barriers any phase
    schedule of this graph can have.
    """
    position = {region_id: index for index, region_id in enumerate(order)}
    prerequisites: dict[str, set[str]] = {region_id: set() for region_id in order}
    for dependency in dependencies.dependencies:
        prerequisites[dependency.consumer_region_id].add(dependency.producer_region_id)
    level: dict[str, int] = {}
    for region_id in order:
        level[region_id] = 1 + max(
            (level[producer] for producer in prerequisites[region_id]), default=-1
        )
    leveled = tuple(sorted(order, key=lambda region_id: (level[region_id], position[region_id])))
    barrier_after = tuple(
        index + 1 < len(leveled) and level[leveled[index]] != level[leveled[index + 1]]
        for index in range(len(leveled))
    )
    return leveled, barrier_after


def _topological_order(
    selections: tuple[SelectedBody, ...], dependencies: DependencyPlan
) -> tuple[str, ...]:
    positions = {
        selected.region_id: position for position, selected in enumerate(selections)
    }
    consumers = {region_id: [] for region_id in positions}
    indegree = {region_id: 0 for region_id in positions}
    for dependency in dependencies.dependencies:
        producer = dependency.producer_region_id
        consumer = dependency.consumer_region_id
        if producer not in positions or consumer not in positions:
            raise InvalidSchedule("dependency references an unscheduled region")
        consumers[producer].append(consumer)
        indegree[consumer] += 1

    ready = [positions[region_id] for region_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        position = heapq.heappop(ready)
        region_id = selections[position].region_id
        order.append(region_id)
        for consumer in consumers[region_id]:
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                heapq.heappush(ready, positions[consumer])
    if len(order) != len(selections):
        raise InvalidSchedule("body dependencies contain a cycle")
    return tuple(order)


def _worker_count(
    selection: BodySelectionPlan, tile_maps: Mapping[str, TileMap]
) -> int:
    maximum_work = max(
        (len(tile_map.accesses) for tile_map in tile_maps.values()), default=0
    )
    if maximum_work == 0:
        return 1
    if selection.target is None:
        return maximum_work
    active_blocks = min(
        selected.resources.maximum_active_blocks or 1
        for selected in selection.selections
    )
    return min(maximum_work, selection.target.multiprocessor_count * active_blocks)


# Limits on what folds into a later interval: a folded producer runs on every
# worker by itself, so it must be a few rounds of memory latency for one CTA.
_FOLD_FAMILIES = frozenset({"elementwise", "copy", "reduction", "gather", "concat"})
_FOLD_MAX_ROUNDS_PER_PHASE = 2
_FOLD_MAX_ROUNDS = 6
_FOLD_MAX_PHASES = 6


def _fold_small_intervals(
    order: tuple[str, ...],
    barrier_after: tuple[bool, ...],
    selection: BodySelectionPlan,
    selected_by_region: Mapping[str, SelectedBody],
) -> tuple[tuple[str, ...], tuple[bool, ...], frozenset[str]]:
    """Fold runs of barrier intervals that hold only small map and reduction
    bodies into the next interval, as recompute phases.

    Decode spends most of its barrier intervals on norms, residual adds,
    RoPE, and gated activations over a few thousand elements; each costs a
    grid barrier and a dispatch round for a fraction of a microsecond of
    work. Recomputed by every CTA at the start of the consumer interval they
    cost one CTA's latency instead (Mirage fuses the same operators into its
    linear and attention tasks). Every CTA writes the same values, and reads
    only what it has written itself. A run does not fold if the receiving
    interval writes storage the run reads or writes.
    """
    from .bodies.cuda.generic import single_cta_rounds

    graph = selection.graph
    threads = max(
        (selected.participants.participating_threads for selected in selection.selections),
        default=128,
    )
    converted = {
        region_id
        for conversion in selection.conversions
        for region_id in (conversion.source_region_id, conversion.target_region_id)
    }

    def rounds(region_id: str) -> int | None:
        selected = selected_by_region[region_id]
        program = selected.program
        if (
            selected.body.provider != "generic"
            or program.family not in _FOLD_FAMILIES
            or program.state_accesses
            or region_id in converted
        ):
            return None
        count = single_cta_rounds(program, threads)
        return count if count <= _FOLD_MAX_ROUNDS_PER_PHASE else None

    def storage(region_ids: Sequence[str], *, writes: bool) -> frozenset[str]:
        values: list[str] = []
        for region_id in region_ids:
            selected = selected_by_region[region_id]
            values.extend(selected.outputs)
            if not writes or selected.program.state_accesses:
                # Stateful bodies may write their inputs (KV appends).
                values.extend(selected.inputs)
        return _storage_ids(graph, tuple(values)) if graph is not None else frozenset()

    intervals: list[list[str]] = [[]]
    for region_id, barrier in zip(order, barrier_after, strict=True):
        intervals[-1].append(region_id)
        if barrier:
            intervals.append([])
    if not intervals[-1]:
        intervals.pop()

    result: list[list[str]] = []
    folded: set[str] = set()
    pending: list[list[str]] = []
    pending_rounds = 0

    def flush() -> None:
        nonlocal pending_rounds
        result.extend(pending)
        pending.clear()
        pending_rounds = 0

    for interval in intervals:
        costs = [rounds(region_id) for region_id in interval]
        if all(cost is not None for cost in costs):
            def fits() -> bool:
                phases = sum(len(group) for group in pending) + len(interval)
                return (
                    pending_rounds + sum(costs) <= _FOLD_MAX_ROUNDS
                    and phases <= _FOLD_MAX_PHASES
                )

            if not fits():
                flush()
            if fits():
                pending.append(interval)
                pending_rounds += sum(costs)
            else:
                result.append(interval)
            continue
        if pending:
            run = [region_id for group in pending for region_id in group]
            receiver_writes = storage(interval, writes=True)
            if receiver_writes & (storage(run, writes=False) | storage(run, writes=True)):
                flush()
            else:
                folded.update(run)
                interval = run + interval
                pending.clear()
                pending_rounds = 0
        result.append(interval)
    flush()

    new_order = tuple(region_id for interval in result for region_id in interval)
    new_barriers = tuple(
        position + 1 == len(interval) and index + 1 < len(result)
        for index, interval in enumerate(result)
        for position in range(len(interval))
    )
    return new_order, new_barriers, frozenset(folded)


def _recompute_assignments(
    region_id: str, work_items: tuple[WorkItem, ...], worker_count: int
) -> tuple[WorkerAssignment, ...]:
    return tuple(
        WorkerAssignment(worker_id, region_id, work_items) for worker_id in range(worker_count)
    )


def _worker_assignments(
    region_id: str, work_items: tuple[WorkItem, ...], worker_count: int
) -> tuple[WorkerAssignment, ...]:
    quotient, remainder = divmod(len(work_items), worker_count)
    start = 0
    assignments: list[WorkerAssignment] = []
    for worker_id in range(worker_count):
        count = quotient + (1 if worker_id < remainder else 0)
        assignments.append(
            WorkerAssignment(
                worker_id,
                region_id,
                work_items[start : start + count],
            )
        )
        start += count
    return tuple(assignments)
