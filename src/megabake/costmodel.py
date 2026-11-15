"""Estimated time of a kernel plan from a ``DeviceTree``.

The model works on a neutral description of a plan: a kernel is a list of
barrier ``Interval``s, and an interval holds the ``Work`` of its regions.
``plan_intervals`` builds that description from a ``KernelPlan``; tests and
the decision checks build it by hand.

One work item on one leaf, with ``k`` leaves active::

    t_item = fixed + max(bytes / min(share(k), rate_cap), flops / engine)

A leaf's time is a start cost per work it takes part in plus its items; an
interval takes its slowest leaf plus synchronization; the kernel is the sum
of its intervals::

    T_interval = max_leaf(Σ_work start + Σ_items t_item) + barrier + skew

Three things separate this from a roofline (``form="roofline"``), and each
decided a measured outcome (docs/memory-planner-research.md §2):

- ``share(k)`` comes from the measured curve, wave by wave, so a tail wave
  on few leaves streams faster and extra leaves add little once the edge
  saturates;
- ``fixed`` and ``start`` do not scale with bytes: pipeline fill, the first
  load's latency, statistics exchanges, dispatch;
- synchronization has a price, so recomputing a small producer on every
  leaf can beat a barrier.

The model ranks plans. It does not replace measuring the best few: whole
kernel register allocation alone moves decode by several percent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf

from .device import DeviceTree


@dataclass(frozen=True)
class Work:
    """One region's work in an interval, as a set of equal items."""

    name: str
    items: int
    item_bytes: float = 0.0  # bytes one item moves over the root edge
    item_flops: float = 0.0
    item_fixed_s: float = 0.0  # latency per item that bytes do not explain
    start_s: float = 0.0  # once per leaf that has items of this work
    rate_cap: float = inf  # bytes/s the body can issue on one leaf
    dtype: str = "bfloat16"
    # A recomputed (folded) producer: every leaf runs all the items.
    every_worker: bool = False
    # Capacity-aware lower bound on the bytes of one item (einsum bound).
    bound_bytes: float | None = None
    # The item's bytes arrive through the producer's staged pages.
    paged: bool = False


@dataclass(frozen=True)
class Interval:
    """The work between two synchronization points. ``sync`` is how the
    interval hands its results on: a grid barrier, a counter, or nothing."""

    work: tuple[Work, ...]
    workers: int
    sync: str = "grid"


@dataclass(frozen=True)
class IntervalEstimate:
    critical_s: float  # the slowest leaf's work
    sync_s: float
    active: int  # leaves that move bytes
    bytes: float
    flops: float

    @property
    def total_s(self) -> float:
        return self.critical_s + self.sync_s


def _queues(interval: Interval) -> list[list[Work]]:
    """Each leaf's items in order. Regions of one interval are dealt to the
    leaves end to end, each starting where the previous one stopped, as
    codegen rotates them."""
    queues: list[list[Work]] = [[] for _ in range(interval.workers)]
    offset = 0
    for work in interval.work:
        if work.every_worker:
            continue
        for item in range(work.items):
            queues[(offset + item) % interval.workers].append(work)
        offset += work.items
    return queues


def sync_time(interval: Interval, tree: DeviceTree) -> float:
    if interval.sync == "grid":
        return tree.barrier_s(interval.workers) + tree.sync.interval_skew_s
    if interval.sync == "counter":
        return tree.sync.counter_round_trip_s
    return 0.0


def interval_time(interval: Interval, tree: DeviceTree, form: str = "full") -> IntervalEstimate:
    """Estimated time of one interval.

    ``form`` selects the model: ``"full"`` (this module's), ``"x23"`` (the
    2026-11-07 starting form: 2.0 µs per tile plus bytes at the leaf's share,
    plus 1.45 µs) or ``"roofline"`` (bytes at the device's bandwidth, FLOPs
    at its peak). The simpler forms exist to be compared against."""
    total_bytes = sum(work.items * work.item_bytes * (interval.workers if work.every_worker else 1)
                      for work in interval.work)
    total_flops = sum(work.items * work.item_flops for work in interval.work)
    queues = _queues(interval)
    movers = sum(1 for queue in queues if any(work.item_bytes for work in queue))
    if any(work.every_worker and work.item_bytes for work in interval.work):
        movers = interval.workers

    if form == "roofline":
        peak = min((tree.flops_per_s(work.dtype) for work in interval.work), default=inf)
        time = max(
            total_bytes / tree.read_bandwidth(),
            total_flops / (peak * tree.workers) if total_flops else 0.0,
        )
        return IntervalEstimate(time, 0.0, movers, total_bytes, total_flops)

    if form == "x23":
        share = tree.read_share(max(1, movers))
        critical = max(
            (sum(2.0e-6 + work.item_bytes / share for work in queue) for queue in queues),
            default=0.0,
        )
        return IntervalEstimate(critical, 1.45e-6, movers, total_bytes, total_flops)

    def item_time(work: Work, active: int) -> float:
        rate = min(tree.read_share(active), work.rate_cap)
        stream = work.item_bytes / rate if work.item_bytes else 0.0
        compute = work.item_flops / tree.flops_per_s(work.dtype) if work.item_flops else 0.0
        return work.item_fixed_s + max(stream, compute)

    # Every leaf first runs the recomputed producers.
    shared = sum(
        work.start_s + work.items * item_time(work, interval.workers)
        for work in interval.work
        if work.every_worker
    )
    depth = max((len(queue) for queue in queues), default=0)
    # Leaves still streaming at each position of their queues: a tail wave
    # shares the edge among fewer leaves.
    active_at = [sum(1 for queue in queues if len(queue) > position) for position in range(depth)]
    critical = shared
    for queue in queues:
        started: set[str] = set()
        time = shared
        for position, work in enumerate(queue):
            if work.name not in started:
                started.add(work.name)
                time += work.start_s
            time += item_time(work, active_at[position])
        critical = max(critical, time)
    return IntervalEstimate(critical, sync_time(interval, tree), movers, total_bytes, total_flops)


@dataclass(frozen=True)
class Bounds:
    """Lower bounds on a plan's time, each adding one constraint.

    - ``b0``: bytes at the device's bandwidth, FLOPs at its peak.
    - ``b1``: with the bytes a leaf of this capacity must move (einsum bound).
    - ``b2``: with the bandwidth the plan's active leaves reach.
    - ``b3``: plus one hand-off between dependent intervals.
    """

    b0: float
    b1: float
    b2: float
    b3: float


@dataclass(frozen=True)
class PlanEstimate:
    intervals: tuple[IntervalEstimate, ...]
    bounds: Bounds

    @property
    def total_s(self) -> float:
        return sum(interval.total_s for interval in self.intervals)

    @property
    def sync_s(self) -> float:
        return sum(interval.sync_s for interval in self.intervals)

    @property
    def bytes(self) -> float:
        return sum(interval.bytes for interval in self.intervals)


def plan_bounds(intervals: Sequence[Interval], tree: DeviceTree) -> Bounds:
    b0 = b1 = b2 = b3 = 0.0
    for interval in intervals:
        timed = interval_time(interval, tree)
        peak = min((tree.flops_per_s(work.dtype) for work in interval.work), default=inf)
        # What the interval must move: every item once. A recomputed producer
        # is read by every leaf in the plan, but need not be.
        needed = sum(work.items * work.item_bytes for work in interval.work)
        floor = max(
            needed,
            sum(
                work.items * (work.item_bytes if work.bound_bytes is None else work.bound_bytes)
                for work in interval.work
            ),
        )
        compute = timed.flops / (peak * tree.workers) if timed.flops else 0.0
        active = max(1, timed.active)
        covered = max(
            floor / tree.read_bandwidth(active),
            timed.flops / (peak * active) if timed.flops else 0.0,
        )
        b0 += max(needed / tree.read_bandwidth(), compute)
        b1 += max(floor / tree.read_bandwidth(), compute)
        b2 += covered
        b3 += covered + (tree.sync.handoff_s if interval.sync != "none" else 0.0)
    return Bounds(b0, b1, b2, b3)


def estimate(intervals: Sequence[Interval], tree: DeviceTree, form: str = "full") -> PlanEstimate:
    return PlanEstimate(
        tuple(interval_time(interval, tree, form) for interval in intervals),
        plan_bounds(intervals, tree),
    )


# --------------------------------------------------------------------------
# Work of each body family. Rates come from ``tree.bodies``; the keys are
# documented in megabake/devices/h100.yaml.


def gemm_work(
    name: str,
    tree: DeviceTree,
    *,
    tiles: int,
    tile_bytes: float,
    tile_flops: float = 0.0,
    splits: int = 1,
    staged: bool = False,
    paged: bool = False,
    depth_bytes: float | None = None,
    bound_bytes: float | None = None,
) -> Work:
    """A persistent tensor-core GEMM: ``tiles`` output tiles, each split
    ``splits`` ways along K. A staged GEMM finds its first weight stage in
    shared memory (the producer loaded it), so it starts sooner.

    ``depth_bytes`` is what the body keeps in flight (its pipeline stages or
    its ring pages). A copy takes the mechanism's latency to arrive, so the
    body cannot stream faster than ``depth_bytes / latency``."""
    rates = tree.body("gemm")
    cap = rates.get("bytes_per_s", inf)
    if depth_bytes:
        cap = min(cap, depth_bytes / tree.mechanism().latency_s)
    start = rates.get("staged_start_s" if staged else "start_s", 0.0)
    fixed = rates.get("tile_s", 0.0)
    if splits > 1:
        # Each split stores FP32 partials and hands off through a counter.
        fixed += rates.get("split_exchange_s", tree.sync.counter_round_trip_s)
    return Work(
        name=name,
        items=tiles * splits,
        item_bytes=tile_bytes / splits,
        item_flops=tile_flops / splits,
        item_fixed_s=fixed,
        start_s=start,
        rate_cap=cap,
        bound_bytes=None if bound_bytes is None else bound_bytes / splits,
        paged=paged,
    )


def attention_key_seconds(tree: DeviceTree, heads: int, head_dim: int) -> float:
    """Seconds one decode-attention item spends per visible key."""
    rates = tree.body("attention_decode")
    measured = sorted(
        (int(key.removeprefix("key_s_")), value)
        for key, value in rates.items()
        if key.startswith("key_s_")
    )
    if not measured:
        return 0.0
    exact = dict(measured).get(head_dim)
    if exact is None:
        nearest, value = min(measured, key=lambda item: abs(item[0] - head_dim))
        exact = value * head_dim / nearest
    return exact + (heads - 1) * rates.get("head_key_s", 0.0)


def attention_item_seconds(
    tree: DeviceTree, *, keys: int, chunks: int, heads: int, head_dim: int
) -> float:
    """Time of one item of split-context decode attention: a fixed cost, its
    keys (at least one load round), and for a split row the statistics
    exchange and value merge."""
    rates = tree.body("attention_decode")
    work = max(
        rates.get("item_min_work_s", 0.0),
        attention_key_seconds(tree, heads, head_dim) * -(-keys // chunks),
    )
    exchange = 0.0
    if chunks > 1:
        exchange = rates.get("exchange_s", 0.0) + chunks * (
            rates.get("merge_head_s", 0.0) * heads + rates.get("merge_dim_s", 0.0) * head_dim
        )
    return rates.get("item_fixed_s", 0.0) + work + exchange


def attention_work(
    name: str,
    tree: DeviceTree,
    *,
    rows: int,
    chunks: int,
    keys: int,
    heads: int,
    head_dim: int,
    row_bytes: float = 0.0,
) -> Work:
    """Decode attention over ``keys`` cached keys: ``rows`` query rows of
    ``heads`` packed heads, each split into ``chunks`` items."""
    return Work(
        name=name,
        items=rows * chunks,
        item_bytes=row_bytes / chunks,
        item_fixed_s=attention_item_seconds(
            tree, keys=keys, chunks=chunks, heads=heads, head_dim=head_dim
        ),
        start_s=tree.body("attention_decode").get("pass_s", 0.0),
        # The body is issue-bound: its key work is already in ``fixed``.
        rate_cap=inf,
    )


def generic_work(
    name: str,
    tree: DeviceTree,
    *,
    items: int,
    rounds: int,
    item_bytes: float = 0.0,
    every_worker: bool = False,
) -> Work:
    """A generated map, reduction, gather or copy body: latency-bound, a few
    rounds of dependent memory accesses per item."""
    rates = tree.body("generic")
    return Work(
        name=name,
        items=items,
        item_bytes=item_bytes,
        item_fixed_s=rates.get("round_s", 0.0) * max(1, rounds) / max(1, items),
        start_s=rates.get("start_s", 0.0),
        every_worker=every_worker,
    )


# --------------------------------------------------------------------------
# From a KernelPlan


@dataclass(frozen=True)
class PlannedInterval:
    """An interval of a real plan with the phases it came from."""

    interval: Interval
    phase_indices: tuple[int, ...] = ()


def plan_intervals(
    plan, tree: DeviceTree, kv_valid_length: int | None = None, *, staged: bool | None = None
) -> list[PlannedInterval]:
    """The barrier intervals of a ``KernelPlan`` as model ``Interval``s.

    ``staged`` says whether a producer stages GEMM weights; by default the
    plan's own staging decision (``plan.mapping``), else no staging."""
    from .accounting import phase_traffic
    from .bodies.cuda.generic import single_cta_rounds
    from .einsum import einsum_view, traffic_lower_bound_bytes

    workers = plan.launch.grid_size
    mapping = getattr(plan, "mapping", None)
    if staged is None:
        staged = bool(mapping is not None and mapping.staging.producer)
    selections = {selected.region_id: selected for selected in plan.selection.selections}
    result: list[PlannedInterval] = []
    work: list[Work] = []
    indices: list[int] = []
    for phase in plan.phases:
        selected = selections[phase.region_id]
        program = selected.program
        traffic = phase_traffic(plan, phase, kv_valid_length)
        items = max(1, len(phase.work_items))
        moved = traffic.bytes_moved
        if phase.body.provider == "g0_range":
            from .bodies.cuda.sm90.gemm import split_k

            view = einsum_view(program)
            bound = traffic_lower_bound_bytes(view, tree.leaf.capacity_bytes) + view.bytes_written()
            paged = "swapab" in phase.body.name and split_k(phase.body) == 1
            # One stage or page holds a 64-wide K tile of 64 weight rows.
            stage_bytes = 64 * 64 * 2
            ring = mapping.staging.ring_pages if (mapping is not None and staged and paged) else 0
            depth = (ring or phase.body.pipeline.stage_count) * stage_bytes
            work.append(
                gemm_work(
                    phase.region_id,
                    tree,
                    tiles=items,
                    tile_bytes=moved / items,
                    tile_flops=traffic.flops / items,
                    splits=split_k(phase.body),
                    staged=staged and paged,
                    paged=paged,
                    depth_bytes=depth,
                    bound_bytes=bound / items,
                )
            )
        elif program.family == "attention" and program.state_accesses:
            from .bodies.cuda.sm90.attention import decode_row_split

            access = program.state_accesses[0]
            keys = access.capacity if kv_valid_length is None else min(kv_valid_length, access.capacity)
            split = decode_row_split(phase.body, workers, keys)
            work.append(
                attention_work(
                    phase.region_id,
                    tree,
                    rows=split.rows,
                    chunks=split.chunks,
                    keys=keys,
                    heads=split.heads,
                    head_dim=split.head_dim,
                    row_bytes=moved / max(1, split.rows),
                )
            )
        else:
            rounds = (
                single_cta_rounds(program, plan.launch.threads_per_block)
                if program.family in {"elementwise", "copy", "reduction", "gather", "concat"}
                else items
            )
            work.append(
                generic_work(
                    phase.region_id,
                    tree,
                    items=items,
                    rounds=rounds,
                    item_bytes=moved / items,
                    every_worker=phase.recompute,
                )
            )
        indices.append(phase.index)
        if phase.barrier is not None or phase is plan.phases[-1]:
            result.append(
                PlannedInterval(
                    Interval(
                        tuple(work),
                        workers,
                        "grid" if phase.barrier is not None else "none",
                    ),
                    tuple(indices),
                )
            )
            work, indices = [], []
    return result


def estimate_plan(plan, tree: DeviceTree, kv_valid_length: int | None = None) -> PlanEstimate:
    return estimate(
        [planned.interval for planned in plan_intervals(plan, tree, kv_valid_length)], tree
    )
