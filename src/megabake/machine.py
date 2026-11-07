"""Measured machine facts (experiment X0.1).

``TargetFacts`` says what a GPU can do (capabilities). ``MachineModel`` says
how fast it does it, measured on this device: HBM and L2 read bandwidth,
bandwidth as a function of the number of busy CTAs, grid-barrier cost,
counter round trip, and tensor-core peak. A speed-of-light time for any
piece of work is ``max(bytes / bandwidth, flops / peak)`` with these values,
never with spec-sheet numbers.

Calibration runs once per device fingerprint (name, SM count, MIG profile,
driver, clocks) and is cached on disk as JSON::

    .venv/bin/python tools/calibrate.py            # measure (or load) and print
    .venv/bin/python tools/calibrate.py --force    # measure again

The measurement kernels use Triton for streaming reads and MegaBake's own
``counting_grid_barrier`` (CuTe DSL) for the synchronization costs, so the
barrier numbers are the cost of the barrier the megakernel really runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

MACHINE_MODEL_VERSION = 1


def _cache_dir() -> Path:
    root = os.environ.get("MEGABAKE_MACHINE_CACHE")
    if root:
        return Path(root)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "megabake" / "machine"


@dataclass(frozen=True)
class DeviceFingerprint:
    name: str
    multiprocessor_count: int
    compute_capability: str
    total_memory_bytes: int
    l2_cache_bytes: int
    driver_version: int
    sm_clock_khz: int
    memory_clock_khz: int

    @property
    def key(self) -> str:
        text = json.dumps(asdict(self), sort_keys=True) + f"/v{MACHINE_MODEL_VERSION}"
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def device_fingerprint(device: torch.device | int | None = None) -> DeviceFingerprint:
    from cuda.bindings import driver as cuda

    index = torch.device("cuda" if device is None else device).index or 0
    properties = torch.cuda.get_device_properties(index)
    cuda.cuInit(0)
    _, handle = cuda.cuDeviceGet(index)
    _, driver = cuda.cuDriverGetVersion()

    def attribute(name):
        _, value = cuda.cuDeviceGetAttribute(getattr(cuda.CUdevice_attribute, name), handle)
        return int(value)

    return DeviceFingerprint(
        name=properties.name,
        multiprocessor_count=properties.multi_processor_count,
        compute_capability=f"{properties.major}.{properties.minor}",
        total_memory_bytes=properties.total_memory,
        l2_cache_bytes=properties.L2_cache_size,
        driver_version=int(driver),
        sm_clock_khz=attribute("CU_DEVICE_ATTRIBUTE_CLOCK_RATE"),
        memory_clock_khz=attribute("CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE"),
    )


@dataclass(frozen=True)
class MachineModel:
    fingerprint: DeviceFingerprint
    # Streaming read with every SM busy, footprint far above L2.
    hbm_read_bytes_per_s: float
    # The same read issued by only ``k`` CTAs (one per SM): {k: bytes/s}.
    # A phase with k busy CTAs cannot read faster than this.
    hbm_read_bytes_per_s_by_ctas: dict[int, float]
    # Repeated read of a footprint that fits L2 (all SMs, L1 bypassed).
    l2_read_bytes_per_s: float
    # Largest footprint whose read still runs at >= 1.5x HBM bandwidth.
    l2_effective_bytes: int
    # {footprint bytes: bytes/s}, the curve both values above come from.
    read_bytes_per_s_by_footprint: dict[int, float]
    # One counting_grid_barrier with all CTAs arriving together: {ctas: seconds}.
    grid_barrier_s_by_ctas: dict[int, float]
    # Release-store on one SM to acquire-poll seeing it on another, and back.
    counter_round_trip_s: float
    # Dense GEMM peak (cuBLAS, large square problem): {dtype: flop/s}.
    tensor_peak_flops: dict[str, float]
    measured_at: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def read_bandwidth(self, busy_ctas: int | None = None) -> float:
        """HBM read bandwidth reachable with ``busy_ctas`` CTAs (all if None).

        Linear interpolation over the measured curve; above the largest
        measured count the all-SM value holds.
        """
        if busy_ctas is None or busy_ctas >= self.fingerprint.multiprocessor_count:
            return self.hbm_read_bytes_per_s
        points = sorted(self.hbm_read_bytes_per_s_by_ctas.items())
        busy = max(1, busy_ctas)
        for (low_k, low_bw), (high_k, high_bw) in zip(points, points[1:]):
            if low_k <= busy <= high_k:
                return low_bw + (high_bw - low_bw) * (busy - low_k) / (high_k - low_k)
        return points[0][1] * busy / points[0][0] if busy < points[0][0] else self.hbm_read_bytes_per_s

    def grid_barrier_s(self, ctas: int | None = None) -> float:
        points = sorted(self.grid_barrier_s_by_ctas.items())
        ctas = self.fingerprint.multiprocessor_count if ctas is None else ctas
        best = min(points, key=lambda item: abs(item[0] - ctas))
        return best[1]

    def speed_of_light_s(
        self, bytes_moved: float, flops: float = 0.0, dtype: str = "bfloat16"
    ) -> float:
        peak = self.tensor_peak_flops.get(dtype) or max(self.tensor_peak_flops.values())
        return max(bytes_moved / self.hbm_read_bytes_per_s, flops / peak)

    def to_json(self) -> dict:
        data = asdict(self)
        data["version"] = MACHINE_MODEL_VERSION
        return data

    @classmethod
    def from_json(cls, data: dict) -> MachineModel:
        ints = lambda mapping: {int(key): float(value) for key, value in mapping.items()}  # noqa: E731
        return cls(
            fingerprint=DeviceFingerprint(**data["fingerprint"]),
            hbm_read_bytes_per_s=float(data["hbm_read_bytes_per_s"]),
            hbm_read_bytes_per_s_by_ctas=ints(data["hbm_read_bytes_per_s_by_ctas"]),
            l2_read_bytes_per_s=float(data["l2_read_bytes_per_s"]),
            l2_effective_bytes=int(data["l2_effective_bytes"]),
            read_bytes_per_s_by_footprint=ints(data["read_bytes_per_s_by_footprint"]),
            grid_barrier_s_by_ctas=ints(data["grid_barrier_s_by_ctas"]),
            counter_round_trip_s=float(data["counter_round_trip_s"]),
            tensor_peak_flops={key: float(value) for key, value in data["tensor_peak_flops"].items()},
            measured_at=data.get("measured_at", ""),
            notes=tuple(data.get("notes", ())),
        )


def cached_machine_model(device=None) -> MachineModel | None:
    """The cached calibration for this device, or None. Never measures."""
    path = _cache_dir() / f"{device_fingerprint(device).key}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("version") != MACHINE_MODEL_VERSION:
        return None
    return MachineModel.from_json(data)


def machine_model(device=None, *, force: bool = False, repeats: int = 3) -> MachineModel:
    """Load the cached calibration for this device, or measure and cache it."""
    if not force:
        cached = cached_machine_model(device)
        if cached is not None:
            return cached
    from . import calibration

    model = calibration.calibrate(device_fingerprint(device), repeats=repeats)
    path = _cache_dir() / f"{model.fingerprint.key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json(), indent=2) + "\n")
    return model


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _median(values) -> float:
    return statistics.median(values)
