"""Calibration microbenchmarks behind ``machine.machine_model`` (X0.1).

Every value is the median over ``repeats`` independent measurements; the
relative spread of each is returned in ``MachineModel.notes`` so callers can
check repeatability (the X0.1 pass rule asks for 2%).
"""

from __future__ import annotations

import statistics

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch
import triton
import triton.language as tl

from .machine import DeviceFingerprint, MachineModel, _now
from .runtime import counting_grid_barrier, make_ready, reset_ready

# One CTA per SM, as the megakernel runs: request more than half the SM's
# shared memory so no two calibration CTAs share an SM.
_ONE_CTA_PER_SM_SMEM = 120 * 1024


# --------------------------------------------------------------------------
# Streaming reads (Triton)


@triton.jit
def _read_kernel(x, out, chunk, reps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * chunk
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for _ in range(reps):
        for offset in range(0, chunk, BLOCK):
            index = base + offset + tl.arange(0, BLOCK)
            acc += tl.load(x + index, cache_modifier=".cg")
    tl.store(out + pid, tl.sum(acc))


def _time_ms(fn, iterations: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iterations):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _read_bandwidth(buffer: torch.Tensor, programs: int, footprint: int, reps: int,
                    block: int, warps: int) -> float:
    """bytes/s of ``programs`` CTAs each reading footprint/programs ``reps`` times."""
    elements = footprint // 4
    chunk = (elements // programs) // block * block
    if chunk == 0:
        raise ValueError("footprint too small for this many programs")
    out = torch.empty(programs, device=buffer.device, dtype=torch.float32)

    def run():
        _read_kernel[(programs,)](buffer, out, chunk, reps, BLOCK=block, num_warps=warps)

    ms = _time_ms(run)
    return programs * chunk * 4 * reps / (ms * 1e-3)


def _best_read(buffer, programs, footprint, reps, sms) -> float:
    # Few CTAs need many loads in flight each; many CTAs do better with less.
    configs = ((1024, 4), (2048, 8), (4096, 8)) + (((8192, 16), (16384, 16)) if programs <= 2 * sms else ())
    return max(
        _read_bandwidth(buffer, programs, footprint, reps, block, warps)
        for block, warps in configs
        if footprint // 4 // programs >= block
    )


# --------------------------------------------------------------------------
# Synchronization (CuTe DSL, MegaBake's own barrier)


@cute.kernel
def _barrier_loop_kernel(ready: cute.Tensor, workers: cutlass.Int32, count: cutlass.Int32):
    for ordinal in cutlass.range(count):
        counting_grid_barrier(ready, workers, ordinal)


@cute.jit
def _barrier_loop(ready: cute.Tensor, workers: cutlass.Int32, count: cutlass.Int32,
                  threads: cutlass.Constexpr, stream: cuda.CUstream):
    _barrier_loop_kernel(ready, workers, count).launch(
        grid=[1, 1, workers],
        block=[threads, 1, 1],
        cooperative=True,
        smem=_ONE_CTA_PER_SM_SMEM,
        stream=stream,
    )


@cute.kernel
def _ping_pong_kernel(flags: cute.Tensor, count: cutlass.Int32):
    _, _, block = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    if thread == 0:
        for index in cutlass.range(count):
            target = index + cutlass.Int32(1)
            if block == 0:
                cute.arch.red(flags.iterator, cutlass.Int32(1), op="add", dtype="s32",
                              sem="release", scope="gpu")
                seen = cute.arch.load(flags.iterator + 32, cutlass.Int32, sem="acquire", scope="gpu")
                while seen < target:
                    seen = cute.arch.load(flags.iterator + 32, cutlass.Int32, sem="acquire", scope="gpu")
            else:
                seen = cute.arch.load(flags.iterator, cutlass.Int32, sem="acquire", scope="gpu")
                while seen < target:
                    seen = cute.arch.load(flags.iterator, cutlass.Int32, sem="acquire", scope="gpu")
                cute.arch.red(flags.iterator + 32, cutlass.Int32(1), op="add", dtype="s32",
                              sem="release", scope="gpu")


@cute.jit
def _ping_pong(flags: cute.Tensor, count: cutlass.Int32, stream: cuda.CUstream):
    _ping_pong_kernel(flags, count).launch(
        grid=[1, 1, 2],
        block=[32, 1, 1],
        cooperative=True,
        smem=_ONE_CTA_PER_SM_SMEM,
        stream=stream,
    )


def _flat(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=0)


def _per_iteration_s(run, low: int, high: int) -> float:
    """Seconds per loop iteration: the time difference of two loop counts."""
    return (_time_ms(lambda: run(high)) - _time_ms(lambda: run(low))) * 1e-3 / (high - low)


def _grid_barrier_s(workers: int, threads: int = 384) -> float:
    stream = torch.cuda.current_stream()
    ready = make_ready(workers, "cuda")
    compiled = cute.compile(
        _barrier_loop, _flat(ready), cutlass.Int32(workers), cutlass.Int32(1), threads,
        cuda.CUstream(stream.cuda_stream),
    )

    def run(count):
        reset_ready(ready)
        compiled(_flat(ready), cutlass.Int32(workers), cutlass.Int32(count),
                 cuda.CUstream(stream.cuda_stream))

    return _per_iteration_s(run, 200, 4200)


def _counter_round_trip_s() -> float:
    stream = torch.cuda.current_stream()
    flags = torch.zeros(64, device="cuda", dtype=torch.int32)
    compiled = cute.compile(_ping_pong, _flat(flags), cutlass.Int32(1), cuda.CUstream(stream.cuda_stream))

    def run(count):
        flags.zero_()
        compiled(_flat(flags), cutlass.Int32(count), cuda.CUstream(stream.cuda_stream))

    return _per_iteration_s(run, 200, 10200)


# --------------------------------------------------------------------------
# Tensor cores (cuBLAS)


def _tensor_peak_flops(size: int = 8192) -> dict[str, float]:
    peaks = {}
    flops = 2 * size**3
    for name, dtype in (("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        a = torch.randn(size, size, device="cuda", dtype=dtype)
        b = torch.randn(size, size, device="cuda", dtype=dtype)
        peaks[name] = flops / (_time_ms(lambda: torch.mm(a, b), 10) * 1e-3)
    try:
        a = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn).t()
        one = torch.ones((), device="cuda")
        run = lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)  # noqa: E731
        peaks["float8_e4m3fn"] = flops / (_time_ms(run, 10) * 1e-3)
    except Exception:  # noqa: BLE001 - fp8 GEMM is optional on this device
        pass
    return peaks


# --------------------------------------------------------------------------


def _spread(values) -> float:
    middle = statistics.median(values)
    return (max(values) - min(values)) / middle if middle else 0.0


def calibrate(fingerprint: DeviceFingerprint, *, repeats: int = 3) -> MachineModel:
    sms = fingerprint.multiprocessor_count
    hbm_footprint = 2 << 30
    buffer = torch.ones(hbm_footprint // 4, device="cuda", dtype=torch.float32)
    notes = []

    def measure(name, fn):
        values = [fn() for _ in range(repeats)]
        notes.append(f"{name}: spread {_spread(values) * 100:.2f}% over {repeats}")
        return statistics.median(values)

    hbm = measure("hbm_read", lambda: max(
        _best_read(buffer, programs, hbm_footprint, 1, sms) for programs in (sms, 2 * sms, 4 * sms)
    ))
    cta_counts = sorted({1, 2, 4, 8, 16, sms // 4, 32, sms // 2, 48, (3 * sms) // 4, sms} - {0})
    cta_counts = [count for count in cta_counts if count <= sms]
    by_ctas = {
        count: measure(f"hbm_read[{count} ctas]",
                       lambda count=count: _best_read(buffer, count, hbm_footprint, 1, sms))
        for count in cta_counts
    }
    by_footprint = {}
    for megabytes in (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 128, 512):
        footprint = megabytes << 20
        if footprint > hbm_footprint:
            continue
        reps = max(1, (1 << 30) // footprint)
        programs = min(2 * sms, footprint // (4 * 2048))
        by_footprint[footprint] = measure(
            f"read[{megabytes} MiB]",
            lambda footprint=footprint, reps=reps, programs=programs: _best_read(
                buffer, programs, footprint, reps, sms
            ),
        )
    l2_bw = max(by_footprint.values())
    l2_bytes = max(
        (footprint for footprint, bandwidth in by_footprint.items() if bandwidth >= 1.5 * hbm),
        default=0,
    )
    del buffer
    torch.cuda.empty_cache()

    barrier_counts = sorted({1, max(1, sms // 4), max(1, sms // 2), sms})
    barriers = {
        count: measure(f"grid_barrier[{count} ctas]", lambda count=count: _grid_barrier_s(count))
        for count in barrier_counts
    }
    round_trip = measure("counter_round_trip", _counter_round_trip_s)
    peaks = _tensor_peak_flops()
    return MachineModel(
        fingerprint=fingerprint,
        hbm_read_bytes_per_s=hbm,
        hbm_read_bytes_per_s_by_ctas=by_ctas,
        l2_read_bytes_per_s=l2_bw,
        l2_effective_bytes=l2_bytes,
        read_bytes_per_s_by_footprint=by_footprint,
        grid_barrier_s_by_ctas=barriers,
        counter_round_trip_s=round_trip,
        tensor_peak_flops=peaks,
        measured_at=_now(),
        notes=tuple(notes),
    )
