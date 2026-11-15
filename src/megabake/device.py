"""The hardware description the planner reads: a tree of storage levels.

A ``DeviceTree`` lists the storage levels of one device from the root (HBM)
to the leaf (an SM, a compute unit, a TensorCore), the edges data moves
along, and what synchronization costs. Compute lives only at the leaf.

    hbm ──(shared edge: bandwidth curve, mechanisms)──> sm × 132
     └ l2 (hardware-managed cache on the way; holds no planned tile)

Three things here go beyond a plain capacity-and-bandwidth tree, because
decode time depends on them (docs/memory-planner-research.md §2):

- a shared edge's throughput is a *curve* over the number of active leaves;
- an edge lists its transfer *mechanisms* with a latency each;
- the tree carries *synchronization costs* and per-body rate tables.

Every number is either measured on the device (``MachineModel``) or comes
from a spec file in ``megabake/devices/``; ``provenance`` says which.
Supporting a new chip of a known mechanism family is a new spec file plus a
calibration run. See ``device_tree``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any

SPEC_DIR = Path(__file__).parent / "devices"

Curve = Mapping[float, float]


def interpolate(curve: Curve, x: float) -> float:
    """Piecewise-linear value of ``curve`` at ``x``, held flat outside its
    range. A curve with one point is a constant."""
    points = sorted(curve.items())
    if x <= points[0][0]:
        return points[0][1]
    for (low_x, low_y), (high_x, high_y) in zip(points, points[1:]):
        if x <= high_x:
            return low_y + (high_y - low_y) * (x - low_x) / (high_x - low_x)
    return points[-1][1]


def saturating_curve(peak: float, per_child: float, children: int) -> dict[float, float]:
    """A shared edge's throughput by active children when only its peak and
    one child's rate are known: ``1 / sqrt((k·per_child)⁻² + peak⁻²)``.

    The form fits the measured H200 curve within 3% (X0.1: 1 CTA 80 GB/s,
    16 CTAs 1,141, 32 CTAs 1,865, 60 CTAs 2,247 GB/s)."""
    counts = sorted({1, 2, 4, 8, 16, 32, 48, 64, 96, 128, children} - {0})
    return {
        float(k): ((k * per_child) ** -2 + peak**-2) ** -0.5
        for k in counts
        if k <= children
    }


@dataclass(frozen=True)
class Level:
    """One storage level. ``fanout`` is its instance count per parent
    instance. Only ``managed == "software"`` levels can hold a planned tile;
    a hardware-managed cache (L2) cannot."""

    name: str
    capacity_bytes: int
    fanout: int = 1
    managed: str = "software"
    # Leaf only: compute engines, FLOP/s per instance by dtype.
    engines: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Mechanism:
    """One way to move data along an edge (TMA bulk copy, thread-issued
    loads, DMA). ``latency_s`` is the time to the first byte;
    ``bytes_per_s`` is what one issuer sustains (None: the edge's share)."""

    latency_s: float
    bytes_per_s: float | None = None
    occupies_compute: bool = True


@dataclass(frozen=True)
class Edge:
    """A data path into ``target`` from ``source``. ``read_bytes_per_s`` and
    ``write_bytes_per_s`` are total throughput by active ``target`` instances."""

    source: str
    target: str
    read_bytes_per_s: Curve
    write_bytes_per_s: Curve = field(default_factory=dict)
    mechanisms: Mapping[str, Mechanism] = field(default_factory=dict)
    # Measured slowdown of a dependent load by the edge's utilization
    # (0..1). Recorded for the device; the cost model does not use it: body
    # rates are fitted inside the kernel and already contain it.
    contention: Curve = field(default_factory=lambda: {0.0: 1.0})


@dataclass(frozen=True)
class Sync:
    """Synchronization costs in seconds (the multi-BSP ``L``)."""

    grid_barrier_s: Curve  # by participating workers
    # A counter released on one leaf and acquired on another, and back.
    counter_round_trip_s: float
    # Start skew and arrival spread of a barrier interval beyond the barrier.
    interval_skew_s: float = 0.0

    @property
    def handoff_s(self) -> float:
        """A value stored by one leaf, made visible and loaded by another:
        one leg of the counter round trip. No dependent hand-off is faster."""
        return self.counter_round_trip_s / 2


@dataclass(frozen=True, eq=False)
class DeviceTree:
    name: str
    family: str  # mechanism family: bodies are written against it
    levels: tuple[Level, ...]  # root first, leaf last
    edges: tuple[Edge, ...]
    sync: Sync
    # Rates of each body family on this device (seconds or bytes/s), read by
    # the cost model; see megabake/devices/*.yaml for the keys.
    bodies: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    # Where each group of numbers comes from: "measured", "spec", "assumed",
    # "fitted".
    provenance: Mapping[str, str] = field(default_factory=dict)

    # -- structure ---------------------------------------------------------

    @property
    def measured(self) -> bool:
        """Whether the edges were measured on the device. The planner leaves
        the tested defaults alone on a tree that is only a spec: it estimates
        and reports, but a kernel changes only on calibrated numbers."""
        return self.provenance.get("edges") == "measured"

    @property
    def root(self) -> Level:
        return self.levels[0]

    @property
    def leaf(self) -> Level:
        return self.levels[-1]

    @property
    def workers(self) -> int:
        """Leaf instances on the device."""
        count = 1
        for level in self.levels:
            count *= level.fanout
        return count

    def level(self, name: str) -> Level:
        return next(level for level in self.levels if level.name == name)

    def edge(self, source: str | None = None, target: str | None = None) -> Edge:
        """The edge between two levels; by default root to leaf."""
        source = source or self.root.name
        target = target or self.leaf.name
        return next(e for e in self.edges if e.source == source and e.target == target)

    # -- rates -------------------------------------------------------------

    def read_bandwidth(self, active: int | None = None) -> float:
        """Root-to-leaf bytes/s with ``active`` leaves reading (all if None)."""
        active = self.workers if active is None else max(1, min(active, self.workers))
        return interpolate(self.edge().read_bytes_per_s, active)

    def read_share(self, active: int) -> float:
        """Bytes/s one of ``active`` reading leaves gets."""
        active = max(1, min(active, self.workers))
        return self.read_bandwidth(active) / active

    def write_bandwidth(self, active: int | None = None) -> float:
        curve = self.edge().write_bytes_per_s or self.edge().read_bytes_per_s
        active = self.workers if active is None else max(1, min(active, self.workers))
        return interpolate(curve, active)

    def flops_per_s(self, dtype: str = "bfloat16", engine: str = "tensor") -> float:
        """Peak of one leaf's ``engine`` for ``dtype``."""
        rates = self.leaf.engines
        key = f"{engine}:{dtype}"
        if key in rates:
            return rates[key]
        same_engine = [value for name, value in rates.items() if name.startswith(engine + ":")]
        return max(same_engine or rates.values() or [float("inf")])

    def barrier_s(self, workers: int | None = None) -> float:
        return interpolate(self.sync.grid_barrier_s, workers or self.workers)

    def mechanism(self, name: str | None = None) -> Mechanism:
        """A root-to-leaf transfer mechanism; the first listed by default."""
        mechanisms = self.edge().mechanisms
        if name is None:
            return next(iter(mechanisms.values()))
        return mechanisms[name]

    def body(self, family: str) -> Mapping[str, float]:
        return self.bodies.get(family, {})

    def staging_depth_s(self, level: str | None = None, reserved_bytes: int = 0) -> float:
        """How long the root edge can run into the free capacity of ``level``
        (the leaf by default) across all its instances: the time of traffic
        that can be parked there ahead of need."""
        target = self.level(level) if level else self.leaf
        instances = 1
        for candidate in self.levels:
            instances *= candidate.fanout
            if candidate is target:
                break
        free = max(0, target.capacity_bytes - reserved_bytes) * instances
        return free / self.read_bandwidth()

    # -- views -------------------------------------------------------------

    def with_workers(self, workers: int) -> DeviceTree:
        """The same device with ``workers`` leaves (a MIG slice, a synthetic
        target). Curves keep their per-leaf rates up to the new count."""
        levels = (*self.levels[:-1], replace(self.leaf, fanout=workers // self._parents()))
        return replace(self, levels=levels)

    def _parents(self) -> int:
        count = 1
        for level in self.levels[:-1]:
            count *= level.fanout
        return count

    # -- files -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "levels": [
                {
                    "name": level.name,
                    "capacity_bytes": level.capacity_bytes,
                    "fanout": level.fanout,
                    "managed": level.managed,
                    **({"engines": dict(level.engines)} if level.engines else {}),
                }
                for level in self.levels
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "read_bytes_per_s": dict(edge.read_bytes_per_s),
                    "write_bytes_per_s": dict(edge.write_bytes_per_s),
                    "mechanisms": {
                        name: {
                            "latency_s": mechanism.latency_s,
                            "bytes_per_s": mechanism.bytes_per_s,
                            "occupies_compute": mechanism.occupies_compute,
                        }
                        for name, mechanism in edge.mechanisms.items()
                    },
                    "contention": dict(edge.contention),
                }
                for edge in self.edges
            ],
            "sync": {
                "grid_barrier_s": dict(self.sync.grid_barrier_s),
                "counter_round_trip_s": self.sync.counter_round_trip_s,
                "interval_skew_s": self.sync.interval_skew_s,
            },
            "bodies": {family: dict(rates) for family, rates in self.bodies.items()},
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeviceTree:
        levels = tuple(
            Level(
                name=level["name"],
                capacity_bytes=int(float(level["capacity_bytes"])),
                fanout=int(level.get("fanout", 1)),
                managed=level.get("managed", "software"),
                engines={k: float(v) for k, v in level.get("engines", {}).items()},
            )
            for level in data["levels"]
        )
        workers = 1
        for level in levels:
            workers *= level.fanout
        edges = tuple(
            Edge(
                source=edge["source"],
                target=edge["target"],
                read_bytes_per_s=_curve(edge["read_bytes_per_s"], workers),
                write_bytes_per_s=_curve(edge.get("write_bytes_per_s", {}), workers),
                mechanisms={
                    name: Mechanism(
                        latency_s=float(mechanism["latency_s"]),
                        bytes_per_s=(
                            None
                            if mechanism.get("bytes_per_s") is None
                            else float(mechanism["bytes_per_s"])
                        ),
                        occupies_compute=bool(mechanism.get("occupies_compute", True)),
                    )
                    for name, mechanism in edge.get("mechanisms", {}).items()
                },
                contention=_curve(edge.get("contention", {0.0: 1.0}), workers),
            )
            for edge in data["edges"]
        )
        sync = data["sync"]
        return cls(
            name=data["name"],
            family=data["family"],
            levels=levels,
            edges=edges,
            sync=Sync(
                grid_barrier_s=_curve(sync["grid_barrier_s"], workers),
                counter_round_trip_s=float(sync["counter_round_trip_s"]),
                interval_skew_s=float(sync.get("interval_skew_s", 0.0)),
            ),
            bodies={
                family: {key: float(value) for key, value in rates.items()}
                for family, rates in data.get("bodies", {}).items()
            },
            provenance=dict(data.get("provenance", {})),
        )

    @cached_property
    def digest(self) -> str:
        """Identifies the tree's numbers: its identity as a cache key and
        part of the plan cache key."""
        import hashlib
        import json

        text = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DeviceTree) and self.digest == other.digest

    def __hash__(self) -> int:
        return hash(self.digest)


def _curve(data: Mapping[Any, Any], workers: int) -> dict[float, float]:
    """A curve from a file: points, or ``{peak, per_child}`` for a
    saturating curve over ``workers`` children."""
    if "peak" in data:
        return saturating_curve(float(data["peak"]), float(data["per_child"]), workers)
    return {float(key): float(value) for key, value in data.items()}


def load_spec(name: str) -> DeviceTree:
    """The spec tree ``megabake/devices/<name>.yaml``."""
    import yaml

    return DeviceTree.from_dict(yaml.safe_load((SPEC_DIR / f"{name}.yaml").read_text()))


def spec_names() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in SPEC_DIR.glob("*.yaml")))


# Spec file of each CUDA compute capability's mechanism family.
_SPEC_BY_ARCHITECTURE = {
    "sm90": "h100",
    "sm90a": "h100",
    "sm100": "b200",
    "sm103": "b200",
    "sm120": "b200",
    "sm121": "b200",
}


def tree_for_target(target: Any, machine: Any | None = None) -> DeviceTree:
    """The tree of a ``TargetFacts``: the spec tree of its architecture's
    family with the target's SM count and shared memory, and with every
    number ``machine`` (a ``MachineModel``) measured on the device."""
    tree = load_spec(_SPEC_BY_ARCHITECTURE.get(target.architecture, "h100"))
    spec_workers = tree.workers
    workers = target.multiprocessor_count
    leaf = replace(
        tree.leaf,
        fanout=workers // tree._parents(),
        capacity_bytes=target.max_shared_memory_per_cta,
    )
    tree = replace(tree, levels=(*tree.levels[:-1], leaf))
    if workers != spec_workers:
        # A slice of the device: the spec curve's per-leaf rates up to the
        # slice's leaves, and the same share of the peak.
        edge = tree.edge()
        scale = workers / spec_workers
        clipped = {
            k: v for k, v in edge.read_bytes_per_s.items() if k <= workers
        } or {1.0: interpolate(edge.read_bytes_per_s, 1)}
        clipped[float(workers)] = min(
            interpolate(edge.read_bytes_per_s, workers),
            max(edge.read_bytes_per_s.values()) * max(scale, 0.5),
        )
        tree = _replace_edge(tree, replace(edge, read_bytes_per_s=clipped))
        tree = replace(tree, name=f"{tree.name} ({workers} of {spec_workers} leaves)")
    return tree if machine is None else with_measurements(tree, machine)


def with_measurements(tree: DeviceTree, machine: Any) -> DeviceTree:
    """``tree`` with the values a ``MachineModel`` measured in place of the
    spec's: capacities, the bandwidth curve, engines and sync costs."""
    fingerprint = machine.fingerprint
    workers = fingerprint.multiprocessor_count
    levels = []
    for level in tree.levels:
        if level is tree.root:
            level = replace(level, capacity_bytes=fingerprint.total_memory_bytes)
        elif level.managed == "hardware" and machine.l2_effective_bytes:
            level = replace(level, capacity_bytes=machine.l2_effective_bytes)
        elif level is tree.leaf:
            level = replace(
                level,
                fanout=workers // tree._parents(),
                engines={
                    **level.engines,
                    **{
                        f"tensor:{dtype}": peak / workers
                        for dtype, peak in machine.tensor_peak_flops.items()
                    },
                },
            )
        levels.append(level)
    edge = tree.edge()
    read = {float(k): v for k, v in machine.hbm_read_bytes_per_s_by_ctas.items()}
    read[float(workers)] = max(read.get(float(workers), 0.0), machine.hbm_read_bytes_per_s)
    mechanisms = dict(edge.mechanisms)
    if machine.thread_load_bytes_per_s and "thread_load" in mechanisms:
        mechanisms["thread_load"] = replace(
            mechanisms["thread_load"], bytes_per_s=machine.thread_load_bytes_per_s
        )
    if machine.dram_latency_s and "thread_load" in mechanisms:
        # One dependent load. A bulk copy's latency (issue, transfer, arrive)
        # is another quantity and keeps the spec file's value.
        mechanisms["thread_load"] = replace(
            mechanisms["thread_load"], latency_s=machine.dram_latency_s
        )
    edge = replace(
        edge,
        read_bytes_per_s=read,
        write_bytes_per_s=(
            {float(k): v for k, v in machine.hbm_write_bytes_per_s_by_ctas.items()}
            or edge.write_bytes_per_s
        ),
        mechanisms=mechanisms,
        contention=(
            {float(k): v for k, v in machine.contention_by_utilization.items()}
            or edge.contention
        ),
    )
    sync = replace(
        tree.sync,
        grid_barrier_s={float(k): v for k, v in machine.grid_barrier_s_by_ctas.items()},
        counter_round_trip_s=machine.counter_round_trip_s,
        **machine.body_rates.get("sync", {}),
    )
    measured = replace(
        _replace_edge(replace(tree, levels=tuple(levels)), edge),
        name=fingerprint.name,
    )
    for cache_edge in tree.edges:
        if cache_edge.source != tree.root.name and machine.l2_read_bytes_per_s:
            # The cache's own bandwidth: a footprint that fits it, all SMs reading.
            measured = _replace_edge(
                measured,
                replace(cache_edge, read_bytes_per_s={float(workers): machine.l2_read_bytes_per_s}),
            )
    if "edge" in machine.body_rates:
        # Fitted from a shallow-ring kernel (tools/plan_report.py).
        measured = with_copy_latency(measured, machine.body_rates["edge"]["latency_s"])
    return replace(
        measured,
        sync=sync,
        bodies={
            family: {**rates, **machine.body_rates.get(family, {})}
            for family, rates in tree.bodies.items()
        },
        provenance={
            **tree.provenance,
            "edges": "measured",
            "sync": "measured",
            "engines": "measured",
            "measured_at": machine.measured_at,
            **({"bodies": "measured"} if machine.body_rates else {}),
        },
    )


def with_copy_latency(tree: DeviceTree, latency_s: float) -> DeviceTree:
    """``tree`` with the latency of its root-to-leaf bulk copy (the first
    mechanism listed) set to ``latency_s``."""
    edge = tree.edge()
    name = next(iter(edge.mechanisms))
    mechanisms = {**edge.mechanisms, name: replace(edge.mechanisms[name], latency_s=latency_s)}
    return _replace_edge(tree, replace(edge, mechanisms=mechanisms))


def _replace_edge(tree: DeviceTree, edge: Edge) -> DeviceTree:
    return replace(
        tree,
        edges=tuple(
            edge if (e.source, e.target) == (edge.source, edge.target) else e
            for e in tree.edges
        ),
    )


def device_tree(target: Any) -> DeviceTree:
    """The tree the compiler plans ``target`` (a ``TargetFacts``) with: the
    cached calibration of the current CUDA device when it matches the target,
    else the spec tree. Never measures; run ``tools/calibrate.py`` for that."""
    machine = None
    try:
        from .machine import cached_machine_model

        cached = cached_machine_model()
        if (
            cached is not None
            and cached.fingerprint.multiprocessor_count == target.multiprocessor_count
        ):
            machine = cached
    except Exception:  # noqa: BLE001 - no CUDA device: plan from the spec
        machine = None
    return tree_for_target(target, machine)
