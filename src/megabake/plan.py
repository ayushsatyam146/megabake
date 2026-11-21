from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from math import prod
from time import monotonic
from typing import TYPE_CHECKING, Any, Protocol

from .bodies.catalog import is_table_gemm
from .bodies.interface import (
    AxisExpression,
    BodyConfig,
    ParticipantContract,
    SelectionPolicy,
    TargetFacts,
    TensorLayout,
    WorkItem,
)
from .graph import ComputeGraph
from .ir import TensorProgram

if TYPE_CHECKING:
    from .memory import LaunchRequirements, StoragePlan
    from .schedule import DependencyPlan, ScheduledPhase


class InvalidTileMap(ValueError):
    pass


@dataclass(frozen=True)
class CuTeSwizzle:
    mask: int
    shift: int

    def apply(self, offset: int) -> int:
        return offset ^ (((offset >> self.shift) & self.mask) << self.shift)


@dataclass(frozen=True)
class CuTeLayout:
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    alignment_bytes: int
    swizzle: CuTeSwizzle | None = None

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.strides):
            raise InvalidTileMap("CuTe shape and stride ranks differ")
        if any(dimension <= 0 for dimension in self.shape):
            raise InvalidTileMap("CuTe layout dimensions must be positive")
        if self.alignment_bytes <= 0 or self.alignment_bytes & (self.alignment_bytes - 1):
            raise InvalidTileMap("CuTe layout alignment must be a power of two")
        if self.swizzle is not None and (
            self.swizzle.mask <= 0
            or self.swizzle.mask & (self.swizzle.mask - 1)
            or self.swizzle.shift < 0
        ):
            raise InvalidTileMap("CuTe XOR swizzle is invalid")

    @classmethod
    def from_contract(cls, layout: TensorLayout) -> CuTeLayout:
        shape = layout.shape or (1,) * len(layout.axes)
        if len(shape) != len(layout.axes) or len(shape) != len(layout.strides):
            raise InvalidTileMap(f"layout {layout.dtype} has mismatched axes, shape, and strides")
        return cls(layout.dtype, shape, layout.strides, layout.alignment_bytes)

    @property
    def cute_shape(self) -> tuple[int, ...]:
        return tuple(reversed(self.shape))

    @property
    def cute_stride(self) -> tuple[int, ...]:
        return tuple(reversed(self.strides))

    def offset(self, coordinates: Sequence[int]) -> int:
        if len(coordinates) != len(self.shape):
            raise InvalidTileMap("coordinates do not match the CuTe layout rank")
        for coordinate, size in zip(coordinates, self.shape, strict=True):
            if not 0 <= coordinate < size:
                raise InvalidTileMap("coordinates are outside the CuTe layout")
        offset = sum(
            coordinate * stride
            for coordinate, stride in zip(coordinates, self.strides, strict=True)
        )
        return self.swizzle.apply(offset) if self.swizzle is not None else offset

    def same_dense_memory(self, other: CuTeLayout) -> bool:
        """Whether both layouts are row-major and dense over the same number
        of elements, so one is a reshape of the other in memory."""

        def dense(layout: CuTeLayout) -> bool:
            stride = 1
            for size, actual in zip(reversed(layout.shape), reversed(layout.strides)):
                if size != 1 and actual != stride:
                    return False
                stride *= size
            return layout.swizzle is None

        return (
            self.dtype == other.dtype
            and prod(self.shape) == prod(other.shape)
            and dense(self)
            and dense(other)
        )

    def needs_conversion_to(self, other: CuTeLayout) -> bool:
        if self.shape != other.shape and self.same_dense_memory(other):
            # A GEMM with a fused epilogue stores its consumer's result as
            # the 2-D GEMM output; consumers read the same memory reshaped.
            return False
        if self.dtype != other.dtype or self.shape != other.shape:
            raise InvalidTileMap("conversion layouts must have the same value domain")

        def normalize(layout):
            return tuple(
                stride if size != 1 else 0
                for stride, size in zip(layout.strides, layout.shape, strict=True)
            )

        return (normalize(self), self.swizzle) != (normalize(other), other.swizzle)


@dataclass(frozen=True)
class LogicalTile:
    value_id: str
    axes: tuple[AxisExpression, ...]
    layout: CuTeLayout
    origin: tuple[int, ...]
    shape: tuple[int, ...]

    def element_offset(self, coordinates: Sequence[int]) -> int:
        if len(coordinates) != len(self.origin):
            raise InvalidTileMap("coordinates do not match the logical tile rank")
        for coordinate, begin, extent, size in zip(
            coordinates,
            self.origin,
            self.shape,
            self.layout.shape,
            strict=True,
        ):
            if not begin <= coordinate < begin + extent or not 0 <= coordinate < size:
                raise InvalidTileMap("coordinates are outside the logical tile")
        return self.layout.offset(coordinates)


@dataclass(frozen=True)
class TileAccess:
    output: LogicalTile
    inputs: tuple[LogicalTile, ...]
    reduction_range: tuple[int, int] | None

    @property
    def work_item(self) -> WorkItem:
        return WorkItem(self.output.origin, self.reduction_range)


@dataclass(frozen=True)
class TileGrid:
    shape: tuple[int, ...]
    tile_shape: tuple[int, ...]

    @property
    def counts(self) -> tuple[int, ...]:
        if len(self.shape) != len(self.tile_shape) or any(
            dimension <= 0 for dimension in self.tile_shape
        ):
            raise InvalidTileMap("tile shape does not match the output rank")
        return tuple(
            (dimension + tile - 1) // tile
            for dimension, tile in zip(self.shape, self.tile_shape, strict=True)
        )

    def origins(self):
        return product(
            *(
                range(0, dimension, tile)
                for dimension, tile in zip(self.shape, self.tile_shape, strict=True)
            )
        )


@dataclass(frozen=True)
class ResourceRequirements:
    shared_memory_bytes: int
    pipeline_stages: int
    barriers: tuple[str, ...]
    maximum_active_blocks: int | None = None


@dataclass(frozen=True)
class RejectionRecord:
    region_id: str
    provider: str
    reason: str


@dataclass(frozen=True)
class UnsupportedRegion:
    region_id: str
    reason: str


@dataclass(frozen=True)
class ConversionBody:
    value_id: str
    source_region_id: str
    target_region_id: str
    source_layout: CuTeLayout
    target_layout: CuTeLayout
    cost_bytes: int
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class SelectedBody:
    region_id: str
    source_op_ids: tuple[str, ...]
    body: BodyConfig
    program: TensorProgram
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    dependencies: tuple[str, ...]
    participants: ParticipantContract
    resources: ResourceRequirements


@dataclass(frozen=True)
class PlanReport:
    selected_region_ids: tuple[str, ...]
    rejections: tuple[RejectionRecord, ...]
    unsupported_regions: tuple[UnsupportedRegion, ...]
    conversions: tuple[ConversionBody, ...]

    @property
    def complete(self) -> bool:
        return not self.unsupported_regions


@dataclass(frozen=True)
class BodySelectionPlan:
    selections: tuple[SelectedBody, ...]
    rejections: tuple[RejectionRecord, ...]
    unsupported_regions: tuple[UnsupportedRegion, ...]
    conversions: tuple[ConversionBody, ...]
    graph: ComputeGraph | None = None
    target: TargetFacts | None = None

    @property
    def report(self) -> PlanReport:
        return PlanReport(
            tuple(selection.region_id for selection in self.selections),
            self.rejections,
            self.unsupported_regions,
            self.conversions,
        )


@dataclass(frozen=True)
class TileMap:
    region_id: str
    body: BodyConfig
    participants: ParticipantContract
    resources: ResourceRequirements
    output_grid: TileGrid
    output_layout: CuTeLayout
    input_layouts: tuple[tuple[str, CuTeLayout], ...]
    accesses: tuple[TileAccess, ...]
    conversion_bodies: tuple[ConversionBody, ...] = ()

    @property
    def work_items(self) -> tuple[WorkItem, ...]:
        return tuple(access.work_item for access in self.accesses)


@dataclass(frozen=True)
class KernelPlan:
    selection: BodySelectionPlan
    dependencies: DependencyPlan
    phases: tuple[ScheduledPhase, ...]
    tile_maps: tuple[TileMap, ...]
    storage: StoragePlan
    launch: LaunchRequirements
    # What the planner decided with the device tree (``mapping.KernelMapping``):
    # region mappings, the producer's staging, and the cost model's estimate.
    # None when the launch was planned without a tree.
    mapping: Any = None

    @property
    def bodies(self) -> tuple[BodyConfig, ...]:
        return tuple(phase.body for phase in self.phases)

    @property
    def target(self) -> TargetFacts | None:
        return self.selection.target

    @property
    def graph(self) -> ComputeGraph | None:
        return self.selection.graph


class PlanEvaluator(Protocol):
    def __call__(self, plan: KernelPlan, budget: SearchBudget) -> PlanMeasurement: ...


@dataclass(frozen=True)
class SearchBudget:
    max_plans: int = 8
    max_compilations: int = 8
    timing_samples: int = 8
    deadline_seconds: float | None = None
    evaluator: PlanEvaluator | None = None

    def __post_init__(self) -> None:
        if self.max_plans < 1 or self.max_compilations < 1 or self.timing_samples < 1:
            raise ValueError("search limits and timing sample count must be positive")
        if self.deadline_seconds is not None and self.deadline_seconds <= 0:
            raise ValueError("search deadline must be positive")


@dataclass(frozen=True)
class PlanMeasurement:
    compiled: Any | None = None
    resources: Any | None = None
    correct: bool = False
    complete_call_samples_us: tuple[float, ...] = ()
    device_event_span_samples_us: tuple[float, ...] = ()
    median_complete_call_us: float | None = None
    median_device_event_span_us: float | None = None
    isolated_launch_samples_us: tuple[float, ...] = ()
    median_isolated_launch_us: float | None = None
    verification_complete_call_us: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class PlanCandidate:
    plan: KernelPlan | None
    worker_count: int
    description: str
    status: str
    reason: str | None = None
    measurement: PlanMeasurement | None = None


@dataclass(frozen=True)
class PlanSearchResult:
    selected: KernelPlan | None
    selected_compiled: Any | None
    candidates: tuple[PlanCandidate, ...]
    rejections: tuple[RejectionRecord, ...]
    budget_exhausted: bool = False
    selected_description: str | None = None

    @property
    def measured_candidates(self) -> tuple[PlanCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.status == "measured")

    def report(self) -> dict[str, Any]:
        return {
            "selected": None if self.selected_description is None else self.selected_description,
            "budget_exhausted": self.budget_exhausted,
            "provider_rejections": [
                {
                    "region_id": rejection.region_id,
                    "provider": rejection.provider,
                    "reason": rejection.reason,
                }
                for rejection in self.rejections
            ],
            "candidates": [
                {
                    "description": candidate.description,
                    "worker_count": candidate.worker_count,
                    "status": candidate.status,
                    "reason": candidate.reason,
                    "median_complete_call_us": (
                        candidate.measurement.median_complete_call_us
                        if candidate.measurement is not None
                        else None
                    ),
                    "median_device_event_span_us": (
                        candidate.measurement.median_device_event_span_us
                        if candidate.measurement is not None
                        else None
                    ),
                    "median_isolated_launch_us": (
                        candidate.measurement.median_isolated_launch_us
                        if candidate.measurement is not None
                        else None
                    ),
                    "verification_complete_call_us": (
                        candidate.measurement.verification_complete_call_us
                        if candidate.measurement is not None
                        else None
                    ),
                    "complete_call_samples_us": (
                        list(candidate.measurement.complete_call_samples_us)
                        if candidate.measurement is not None
                        else []
                    ),
                    "resources": (
                        _resource_report(candidate.measurement.resources)
                        if candidate.measurement is not None
                        else None
                    ),
                }
                for candidate in self.candidates
            ],
        }


def _resource_report(resources: Any) -> Any:
    if resources is None:
        return None
    if isinstance(resources, dict):
        return resources
    return {
        name: getattr(resources, name)
        for name in (
            "registers_per_thread",
            "static_shared_memory_bytes",
            "dynamic_shared_memory_bytes",
            "local_memory_bytes_per_thread",
            "active_blocks_per_sm",
            "max_cooperative_workers",
            "workspace_bytes",
            "peak_live_workspace_bytes",
            "body_variant_count",
            "phase_runner_count",
            "compiled_code_bytes",
            "generated_code_bytes",
        )
        if hasattr(resources, name)
    }


def search_plans(
    graph: ComputeGraph,
    target: TargetFacts,
    policy: Any | None = None,
    budget: SearchBudget | int | None = None,
) -> PlanSearchResult:
    from .memory import assign_storage, verify_plan
    from .planner import enumerate_body_selections, make_tile_maps
    from .schedule import derive_dependencies, make_phase_schedule

    if budget is None:
        selected_budget = SearchBudget()
    elif isinstance(budget, int):
        if budget < 1:
            raise ValueError("search budget must be positive")
        selected_budget = SearchBudget(max_plans=budget)
    else:
        selected_budget = budget

    selected_policy = policy or SelectionPolicy()
    selections = enumerate_body_selections(
        graph, target, selected_policy, max_plans=selected_budget.max_plans
    )
    unique_selections: list[BodySelectionPlan] = []
    seen_selections: set[tuple[Any, ...]] = set()
    for selection in selections:
        identity = tuple(
            (selected.region_id, selected.body.name) for selected in selection.selections
        )
        if identity in seen_selections:
            continue
        seen_selections.add(identity)
        unique_selections.append(selection)
    selections = tuple(unique_selections)

    rejections = tuple(
        dict.fromkeys(rejection for selection in selections for rejection in selection.rejections)
    )
    candidates: list[PlanCandidate] = []
    seen_plans: set[tuple[Any, ...]] = set()
    legal_plan_count = 0
    started = monotonic()
    exhausted = False

    worker_variants: list[tuple[BodySelectionPlan, tuple[int, ...]]] = []
    for selection in selections:
        projected_shared_memory = _selection_projected_shared_memory(selection)
        if projected_shared_memory > target.max_shared_memory_per_cta:
            candidates.append(
                PlanCandidate(
                    None,
                    0,
                    _selection_description(selection, 0),
                    "rejected",
                    (
                        f"projected shared memory {projected_shared_memory} bytes "
                        f"exceeds {target.max_shared_memory_per_cta}"
                    ),
                )
            )
            continue
        try:
            tile_maps = make_tile_maps(selection)
            maximum_work = max((len(tile_map.accesses) for tile_map in tile_maps), default=0)
            active_blocks = min(
                (
                    selected.resources.maximum_active_blocks or 1
                    for selected in selection.selections
                ),
                default=1,
            )
            default_workers = min(
                maximum_work or 1,
                target.multiprocessor_count * active_blocks,
            )
            worker_counts = {default_workers}
            if default_workers >= 2:
                worker_counts.add(max(1, default_workers // 2))
            if default_workers >= 4:
                worker_counts.add(max(1, default_workers // 4))
        except Exception as error:  # noqa: BLE001  # Preserve the diagnostic/fallback boundary for arbitrary backend failures.
            candidates.append(
                PlanCandidate(
                    None,
                    0,
                    _selection_description(selection, 0),
                    "rejected",
                    f"tile enumeration failed: {error}",
                )
            )
            continue

        counts = tuple(sorted(worker_counts, reverse=selected_policy.mode != "decode"))
        if min(selected_budget.max_plans, selected_budget.max_compilations) < len(counts):
            # A budget too small to measure every count selects the first one
            # unmeasured; decode's ascending order then left three quarters of
            # the GPU idle. Try the full-machine default first.
            counts = (default_workers,) + tuple(
                count for count in counts if count != default_workers
            )
        worker_variants.append((selection, counts))

    worker_search_selections = max(1, selected_budget.max_compilations // 3)
    search_variants = []
    for selection, counts in worker_variants[:worker_search_selections]:
        search_variants.extend((selection, worker_count) for worker_count in counts)
    for selection, counts in worker_variants[worker_search_selections:]:
        search_variants.append((selection, counts[0]))
    for selection, worker_count in search_variants:
        if legal_plan_count >= selected_budget.max_plans:
            exhausted = True
            break
        identity = (
            _selection_description(selection, worker_count),
            tuple(selected.body.name for selected in selection.selections),
        )
        if identity in seen_plans:
            continue
        seen_plans.add(identity)
        description = _selection_description(selection, worker_count)
        try:
            dependencies = derive_dependencies(selection)
            schedule = make_phase_schedule(
                selection,
                dependencies,
                worker_count=worker_count,
                fold_small_intervals=selected_policy.table_launch,
            )
            plan = assign_storage(schedule)
            verify_plan(plan)
            if selected_policy.device is not None:
                from .planner import map_plan

                plan = map_plan(plan, selected_policy.device)
            projected_shared_memory = _projected_shared_memory(plan)
            if projected_shared_memory > target.max_shared_memory_per_cta:
                candidates.append(
                    PlanCandidate(
                        plan,
                        worker_count,
                        description,
                        "rejected",
                        (
                            f"projected shared memory {projected_shared_memory} bytes "
                            f"exceeds {target.max_shared_memory_per_cta}"
                        ),
                    )
                )
                continue
        except Exception as error:  # noqa: BLE001  # Preserve the diagnostic/fallback boundary for arbitrary backend failures.
            candidates.append(
                PlanCandidate(
                    None,
                    worker_count,
                    description,
                    "rejected",
                    f"invalid complete plan: {error}",
                )
            )
            continue
        candidates.append(PlanCandidate(plan, worker_count, description, "unmeasured"))
        legal_plan_count += 1

    if selected_policy.device is not None:
        candidates = _rank_by_estimate(candidates)

    compiled_count = 0
    for index, candidate in enumerate(candidates):
        if candidate.status != "unmeasured":
            continue
        if compiled_count >= selected_budget.max_compilations:
            candidates[index] = PlanCandidate(
                candidate.plan,
                candidate.worker_count,
                candidate.description,
                "unmeasured",
                "compilation budget exhausted",
            )
            exhausted = True
            continue
        if (
            selected_budget.deadline_seconds is not None
            and monotonic() - started >= selected_budget.deadline_seconds
        ):
            candidates[index] = PlanCandidate(
                candidate.plan,
                candidate.worker_count,
                candidate.description,
                "unmeasured",
                "search deadline reached",
            )
            exhausted = True
            continue
        if selected_budget.evaluator is None:
            candidates[index] = PlanCandidate(
                candidate.plan,
                candidate.worker_count,
                candidate.description,
                "unmeasured",
                "no plan evaluator was provided",
            )
            continue

        compiled_count += 1
        try:
            measurement = selected_budget.evaluator(candidate.plan, selected_budget)
        except Exception as error:  # noqa: BLE001  # Preserve the diagnostic/fallback boundary for arbitrary backend failures.
            measurement = PlanMeasurement(error=f"evaluator failed: {error}")
        status = (
            "measured"
            if measurement.correct and measurement.median_complete_call_us is not None
            else "rejected"
        )
        reason = None
        if status == "rejected":
            reason = measurement.error or "candidate failed correctness or timing"
        candidates[index] = PlanCandidate(
            candidate.plan,
            candidate.worker_count,
            candidate.description,
            status,
            reason,
            measurement,
        )

    measured = tuple(
        candidate
        for candidate in candidates
        if candidate.status == "measured" and candidate.measurement is not None
    )
    selected = None
    if measured:
        fastest = min(_measurement_rank(candidate.measurement) for candidate in measured)
        # Plans within measurement noise of the fastest are equivalent; prefer
        # the one with the fewest phases (fewer barriers and body transitions).
        close = [
            candidate
            for candidate in measured
            if _measurement_rank(candidate.measurement) <= fastest * (1 + _TIE_TOLERANCE)
        ]
        selected = min(
            close,
            key=lambda candidate: (
                len(candidate.plan.phases) if candidate.plan is not None else float("inf"),
                _measurement_rank(candidate.measurement),
            ),
        )
    return PlanSearchResult(
        None if selected is None else selected.plan,
        None if selected is None else selected.measurement.compiled,
        tuple(candidates),
        rejections,
        exhausted,
        None if selected is None else selected.description,
    )


def _rank_by_estimate(candidates: list[PlanCandidate]) -> list[PlanCandidate]:
    """Legal plans in the order the cost model would compile them: the
    default plan first, unless another is estimated faster by more than the
    model's margin; then by estimate. Compilation is the expensive step, so
    the budget goes to the plans the model ranks best."""
    from .search import MARGIN

    def seconds(candidate: PlanCandidate) -> float:
        mapping = candidate.plan.mapping if candidate.plan is not None else None
        if candidate.status != "unmeasured" or mapping is None or mapping.estimate is None:
            return float("inf")
        return mapping.estimate.total_s

    legal = [candidate for candidate in candidates if seconds(candidate) != float("inf")]
    if len(legal) < 2:
        return candidates
    default = legal[0]
    ranked = sorted(legal[1:], key=seconds)
    if seconds(ranked[0]) < seconds(default) * (1.0 - MARGIN):
        ranked = sorted(legal, key=seconds)
    else:
        ranked = [default, *ranked]
    chosen = {id(candidate) for candidate in legal}
    return ranked + [candidate for candidate in candidates if id(candidate) not in chosen]


def _selection_description(
    selection: BodySelectionPlan, worker_count: int
) -> str:
    bodies = ",".join(
        f"{selected.region_id}:{selected.body.provider}/{selected.body.name}"
        for selected in selection.selections
    )
    return f"workers={worker_count};bodies={bodies}"


# Relative timing difference treated as measurement noise when ranking plans.
_TIE_TOLERANCE = 0.05


def _measurement_rank(measurement: PlanMeasurement) -> float:
    if measurement.median_isolated_launch_us is not None:
        return measurement.median_isolated_launch_us
    if measurement.median_device_event_span_us is not None:
        return measurement.median_device_event_span_us
    if measurement.median_complete_call_us is not None:
        return measurement.median_complete_call_us
    return float("inf")


def _projected_shared_memory(plan: KernelPlan) -> int:
    g0_variants = {
        (
            phase.body.provider,
            phase.body.tiles.cta_shape[:2],
            phase.body.tiles.cluster_shape[:2],
            phase.body.scratch.size_bytes,
        )
        for phase in plan.phases
        if is_table_gemm(phase.body.provider)
    }
    g0_bytes = sum(variant[3] for variant in g0_variants)
    phase_sizes = {
        phase_layout.region_id: phase_layout.size_bytes
        for phase_layout in plan.storage.worker_scratch[0].phases
    }
    generic_bytes = max(
        (
            phase_sizes.get(phase.region_id, 0)
            for phase in plan.phases
            if not is_table_gemm(phase.body.provider)
        ),
        default=0,
    )
    return g0_bytes + generic_bytes if g0_variants else plan.launch.dynamic_shared_memory_bytes


def _selection_projected_shared_memory(selection: BodySelectionPlan) -> int:
    g0_variants = {
        (
            selected.body.provider,
            selected.body.tiles.cta_shape[:2],
            selected.body.tiles.cluster_shape[:2],
            selected.body.scratch.size_bytes,
        )
        for selected in selection.selections
        if is_table_gemm(selected.body.provider)
    }
    if not g0_variants:
        return 0
    g0_bytes = sum(variant[3] for variant in g0_variants)
    generic_bytes = max(
        (
            selected.resources.shared_memory_bytes
            for selected in selection.selections
            if not is_table_gemm(selected.body.provider)
        ),
        default=0,
    )
    return g0_bytes + generic_bytes
