from __future__ import annotations

import contextlib
from dataclasses import dataclass
import os
from typing import Protocol

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from .memory import verify_plan
from .plan import KernelPlan
from .schedule import EventSchedule, simulate_event_schedule


class InvalidRuntime(ValueError):
    pass


_MAX_INT32 = 2**31 - 1


# MEGABAKE_PHASE_TIMER=1 makes table launches stamp the global timer after
# every grid barrier (see tools/phase_profile.py). The stamps follow the
# barrier state in the ready buffer, so the kernel signature is unchanged.
# MEGABAKE_PHASE_TIMER=cta also stamps every CTA: once when it starts its
# first phase, then for every phase when its work is done and when it leaves
# the phase's barrier (or CTA synchronization). These follow the block-0
# stamps, CTA_TIMER_STRIDE slots per CTA.
_PHASE_TIMER_MODE = os.environ.get("MEGABAKE_PHASE_TIMER")
GRID_TIMER_SLOTS = 8192
CTA_TIMER_PHASES = 2048
CTA_TIMER_WORKERS = 160
CTA_TIMER_STRIDE = 1 + 2 * CTA_TIMER_PHASES
CTA_TIMER = _PHASE_TIMER_MODE == "cta"
PHASE_TIMER_SLOTS = (
    GRID_TIMER_SLOTS + CTA_TIMER_WORKERS * CTA_TIMER_STRIDE
    if CTA_TIMER
    else GRID_TIMER_SLOTS
    if _PHASE_TIMER_MODE == "1"
    else 0
)


def ready_slots(workers: int) -> int:
    """Barrier state: per-worker slots, the release flag, the arrival counter,
    any phase-timer stamps, and the call counter."""
    return workers + 3 + PHASE_TIMER_SLOTS


# Weight streaming into shared memory (see codegen._weight_stream_schedule):
# at the grid barrier before a GEMM interval, each CTA loads the weight
# operand of the first K tiles of its first tile in that interval into the
# GEMM's pipeline stages, so the loads overlap the barrier wait and the norms
# folded into the interval. At most this many stages per CTA (the GEMM's
# stage count bounds it); 0 disables streaming.
# Off by default: against the same kernel with the streaming code compiled in
# but never preloading, the preloads changed decode by -0.7% to +1.5% on a
# 60-SM H200 slice (artifacts/step3_2026-10-21). Issuing them blocks warp 0
# once the SM's TMA queue is full, so they overlap little beyond the barrier
# wait.
WEIGHT_STREAM_STAGES = int(os.environ.get("MEGABAKE_WEIGHT_STREAM_STAGES", 0))


# Host-mapped buffers holding table launches' per-call boundary addresses;
# launch ``n`` reads entry ``n % DYNAMIC_ADDRESS_RING``.
DYNAMIC_ADDRESS_RING = 64


def call_counter_offset() -> int:
    """Offset of the call counter from ``workers`` in the ready buffer.

    Table launches derive their barrier epochs from this counter, which the
    kernel advances itself, so no kernel argument changes between calls.
    """
    return 2 + PHASE_TIMER_SLOTS


def _arrival_counter(ready: torch.Tensor) -> int:
    return ready.numel() - 2 - PHASE_TIMER_SLOTS


def make_ready(workers: int, device: torch.device | str) -> torch.Tensor:
    ready = torch.full((ready_slots(workers),), -1, dtype=torch.int32, device=device)
    ready[_arrival_counter(ready)] = 0
    ready[-1] = 0
    return ready


def reset_ready(ready: torch.Tensor) -> None:
    ready.fill_(-1)
    ready[_arrival_counter(ready)] = 0
    ready[-1] = 0


@cute.jit
def grid_barrier(ready: cute.Tensor, workers: cutlass.Int32, epoch: cutlass.Int32):
    # One arrival atomic per CTA. The last arrival clears the counter before
    # releasing the epoch flag, so the counter is zero between barriers and no
    # CTA can arrive at the next barrier before the clear. The previous design
    # polled every worker's flag serially from one thread, which cost one
    # acquire round trip per worker (13 us per barrier at 60 workers).
    thread, _, _ = cute.arch.thread_idx()
    cute.arch.sync_threads()
    if thread == 0:
        arrived = cute.arch.atomic_add(
            ready.iterator + workers + 1,
            cutlass.Int32(1),
            sem="acq_rel",
            scope="gpu",
        )
        if arrived == workers - 1:
            cute.arch.atomic_exch(
                ready.iterator + workers + 1,
                cutlass.Int32(0),
                sem="relaxed",
                scope="gpu",
            )
            cute.arch.atomic_exch(
                ready.iterator + workers,
                epoch,
                sem="release",
                scope="gpu",
            )
        else:
            released = False
            while not released:
                seen = cute.arch.atomic_cas(
                    ready.iterator + workers,
                    cmp=epoch,
                    val=epoch,
                    sem="acquire",
                    scope="gpu",
                )
                released = seen == epoch
    cute.arch.sync_threads()


@cute.jit
def trap_if(condition):
    """Trap the kernel when ``condition`` holds on any lane of this warp.

    The warp votes before it branches, so the trap sits behind a warp-uniform
    branch. Behind a per-lane branch, lanes may leave a warp early as far as
    ptxas can tell, and it then serializes every wgmma of the megakernel
    (C7519, "WG.AR in divergent path"). Every lane of the warp must reach the
    call.
    """
    if cute.arch.vote_any_sync(condition):
        llvm.intr_trap()


# --------------------------------------------------------------------------
# Stall diagnostics (X0.3)
#
# Every spin wait of a table megakernel polls on all lanes of one warp and
# reads %globaltimer. Past the timeout, lane 0 writes one record for its CTA
# into a host-mapped region: {1, call, kind, phase, counter, target, observed,
# time}. The warp then waits a grace period, so that every other stuck CTA
# also reports, and traps. A trap leaves the context unusable, so the region
# lives in pinned host memory: the tail of the dynamic-address ring, which
# every table launch already receives. The host reads it with plain loads
# (``stall_records``) and prints a table on the way out (``install_stall_hook``).
#
# MEGABAKE_STALL_TIMEOUT_S (default 2) sets the timeout when the kernel is
# generated; 0 compiles the checks out.

STALL_WORDS = 8
STALL_GRACE_NS = 200_000_000
# Phase-counter scratch slots 0 and 1 hold the stall region's address (low,
# high), so a body that only receives its scratch can report too. Every
# phase's counters start after them.
STALL_COUNTER_SLOTS = 2
STALL_KINDS = {
    1: "grid barrier",
    2: "attention split exchange",
    3: "ring page full",
    4: "ring page empty",
}
STALL_GRID_BARRIER = 1
STALL_ATTENTION_EXCHANGE = 2
# Producer ring (X1.1): a GEMM waiting for a page's data (full) or for a
# page to be released (empty); the counter is the page, the target its parity.
STALL_RING_FULL = 3
STALL_RING_EMPTY = 4


def stall_timeout_ns() -> int:
    return int(float(os.environ.get("MEGABAKE_STALL_TIMEOUT_S", "2")) * 1e9)


# Test hook: MEGABAKE_STALL_INJECT=<k> makes the last CTA skip its arrival
# at the k-th phase grid barrier (read when the kernel is generated), so every
# CTA stalls there with the counter one arrival short.
def stall_inject_phase() -> int | None:
    value = os.environ.get("MEGABAKE_STALL_INJECT")
    return int(value) if value not in (None, "") else None


# --------------------------------------------------------------------------
# Producer warp (X1.1)
#
# Unless MEGABAKE_PRODUCER_WARP=0 (read when the kernel is generated), table
# kernels whose GEMMs fit the page ring launch with one extra warpgroup after
# the body threads; its first warp is the producer. It never takes part in
# CTA-wide synchronization: while such a kernel compiles, every CTA-wide
# barrier (``cute.arch.sync_threads``, CUTLASS's ``agent_sync``, ...) becomes
# ``bar.sync 0, <body threads>`` (``consumer_cta_sync``). The body threads
# release it once the kernel prologue is done through named barrier
# ``PRODUCER_START_BARRIER``.

PRODUCER_START_BARRIER = 7


def producer_warp_enabled() -> bool:
    """On by default (2026-11-09: 11 of 11 decode cells faster, outputs
    bit-identical); MEGABAKE_PRODUCER_WARP=0 turns it off."""
    return os.environ.get("MEGABAKE_PRODUCER_WARP", "1") == "1"


def producer_ring_pages() -> int:
    """Weight pages (8 KiB of A plus 2 KiB of B each) of the producer's
    ring at most; MEGABAKE_RING_PAGES (default 20), even. Codegen fits it
    to the shared memory left next to the other bodies."""
    pages = int(os.environ.get("MEGABAKE_RING_PAGES", "20"))
    return pages + pages % 2


def plan_with_tree() -> bool:
    """Whether launches are planned with the device tree (default). With
    MEGABAKE_PLAN_WITH_TREE=0 the planner uses its fixed rules, for A/B runs."""
    return os.environ.get("MEGABAKE_PLAN_WITH_TREE", "1") == "1"


def ring_pages_forced() -> bool:
    """Whether MEGABAKE_RING_PAGES fixes the ring size (an experiment knob);
    otherwise the staging planner sizes it from the plan and the device."""
    return "MEGABAKE_RING_PAGES" in os.environ


@contextlib.contextmanager
def consumer_cta_sync(threads: int):
    """While active, an argument-less barrier (a CTA-wide sync) is emitted as
    barrier 0 over the first ``threads`` threads only."""
    from cutlass.cute.arch import nvvm_wrappers

    nvvm = nvvm_wrappers.nvvm
    original = nvvm.barrier

    def barrier(*args, barrier_id=None, number_of_threads=None, loc=None, ip=None, **kwargs):
        if not args and barrier_id is None and number_of_threads is None:
            barrier_id = cutlass.Int32(0).ir_value(loc=loc, ip=ip)
            number_of_threads = cutlass.Int32(threads).ir_value(loc=loc, ip=ip)
        return original(
            *args, barrier_id=barrier_id, number_of_threads=number_of_threads, loc=loc, ip=ip, **kwargs
        )

    shadowed = "barrier" in vars(nvvm)
    nvvm.barrier = barrier
    try:
        yield
    finally:
        if shadowed:
            nvvm.barrier = original
        else:
            del nvvm.barrier


@cute.jit
def stall_base_from_counters(counters: cute.Tensor) -> cutlass.Int64:
    low = counters[0].to(cutlass.Int64) & cutlass.Int64(0xFFFFFFFF)
    high = counters[1].to(cutlass.Int64)
    return low | (high << 32)


@cute.jit
def publish_stall_base(counters: cute.Tensor, stall_base: cutlass.Int64):
    """Thread 0 of block 0, before the first barrier: put the stall region's
    address where bodies can find it."""
    counters[0] = (stall_base & cutlass.Int64(0xFFFFFFFF)).to(cutlass.Int32)
    counters[1] = (stall_base >> 32).to(cutlass.Int32)


@cute.jit
def stall_report(
    stall_base: cutlass.Int64,
    call: cutlass.Int32,
    kind: cutlass.Constexpr,
    phase: cutlass.Int32,
    counter: cutlass.Int64,
    target: cutlass.Int32,
    observed: cutlass.Int32,
):
    """Called by every lane of one warp of a stuck CTA: record, wait, trap."""
    _, _, block = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    now = cute.arch.globaltimer()
    if lane == 0:
        record = cute.make_ptr(
            cutlass.Int64,
            stall_base + block.to(cutlass.Int64) * (STALL_WORDS * 8),
            cute.AddressSpace.gmem,
            assumed_align=8,
        )
        cute.arch.store(record + 1, call.to(cutlass.Int64), scope="sys", sem="relaxed")
        cute.arch.store(record + 2, cutlass.Int64(kind), scope="sys", sem="relaxed")
        cute.arch.store(record + 3, phase.to(cutlass.Int64), scope="sys", sem="relaxed")
        cute.arch.store(record + 4, counter, scope="sys", sem="relaxed")
        cute.arch.store(record + 5, target.to(cutlass.Int64), scope="sys", sem="relaxed")
        cute.arch.store(record + 6, observed.to(cutlass.Int64), scope="sys", sem="relaxed")
        cute.arch.store(record + 7, now, scope="sys", sem="relaxed")
        cute.arch.store(record, cutlass.Int64(1), scope="sys", sem="release")
        cute.arch.fence_acq_rel_sys()
    # Grace period: the other stuck CTAs reach their own timeouts and report.
    waiting = cute.arch.vote_any_sync(cute.arch.globaltimer() - now < cutlass.Int64(STALL_GRACE_NS))
    while waiting:
        waiting = cute.arch.vote_any_sync(
            cute.arch.globaltimer() - now < cutlass.Int64(STALL_GRACE_NS)
        )
    trap_if(True)


@cute.jit
def poll_at_least(
    address,
    target: cutlass.Int32,
    stall_base: cutlass.Int64,
    call: cutlass.Int32,
    kind: cutlass.Constexpr,
    phase: cutlass.Int32,
    counter: cutlass.Int64,
    timeout_ns: cutlass.Constexpr,
):
    """Acquire-poll an Int32 counter until it reaches ``target``. Every lane of
    the calling warp polls, and the loop and the trap are warp-uniform (C7519:
    a spin or trap behind a per-lane branch serializes every wgmma)."""
    seen = cute.arch.load(address, cutlass.Int32, sem="acquire", scope="gpu")
    unmet = cute.arch.vote_any_sync(seen < target)
    if unmet:
        start = cute.arch.globaltimer()
        waiting = unmet
        while waiting:
            seen = cute.arch.load(address, cutlass.Int32, sem="acquire", scope="gpu")
            unmet = cute.arch.vote_any_sync(seen < target)
            in_time = cute.arch.vote_all_sync(
                cute.arch.globaltimer() - start <= cutlass.Int64(timeout_ns)
            )
            waiting = unmet & in_time
        # Still unmet after the loop: it ended on the timeout.
        if unmet:
            stall_report(stall_base, call, kind, phase, counter, target, seen)


@cute.jit
def checked_mbarrier_wait(
    bar,
    parity,
    stall_base: cutlass.Int64,
    kind: cutlass.Constexpr,
    page,
    timeout_ns: cutlass.Constexpr,
):
    """``cute.arch.mbarrier_wait`` for a whole warp, reporting a stall past
    ``timeout_ns`` (0: a plain wait). The fast path is one try_wait and one
    vote; the loop and the trap are warp-uniform."""
    if cutlass.const_expr(timeout_ns <= 0):
        cute.arch.mbarrier_wait(bar, parity)
    else:
        unmet = cute.arch.vote_any_sync(cute.arch.mbarrier_try_wait(bar, parity) == False)  # noqa: E712
        if unmet:
            start = cute.arch.globaltimer()
            waiting = unmet
            while waiting:
                unmet = cute.arch.vote_any_sync(
                    cute.arch.mbarrier_try_wait(bar, parity) == False  # noqa: E712
                )
                in_time = cute.arch.vote_all_sync(
                    cute.arch.globaltimer() - start <= cutlass.Int64(timeout_ns)
                )
                waiting = unmet & in_time
            if unmet:
                stall_report(
                    stall_base,
                    cutlass.Int32(-1),
                    kind,
                    cutlass.Int32(-1),
                    cutlass.Int32(page).to(cutlass.Int64),
                    cutlass.Int32(parity),
                    cutlass.Int32(-1),
                )


@cute.jit
def counting_grid_barrier(
    ready: cute.Tensor, workers: cutlass.Int32, ordinal: cutlass.Int32
):
    """Grid barrier on an arrival counter that only grows.

    Barrier ``ordinal`` (counted densely from the last ready reset) completes
    once the counter reaches ``(ordinal + 1) * workers``. Each CTA arrives
    with one release reduction and polls with acquire loads, so the last
    arrival is itself the release: one memory round trip on the critical
    path instead of an arrival, a counter reset, and a flag release. The
    caller keeps ``(ordinal + 1) * workers`` within int32 and resets the
    ready buffer before it would wrap.
    """
    counting_grid_arrive(ready, workers)
    counting_grid_wait(ready, workers, ordinal)


@cute.jit
def counting_grid_arrive(ready: cute.Tensor, workers: cutlass.Int32):
    """The arrival half of ``counting_grid_barrier``: once every thread of
    the CTA is done, thread 0 arrives. The CTA may issue work that does not
    depend on other CTAs (weight loads) before ``counting_grid_wait``."""
    thread, _, _ = cute.arch.thread_idx()
    cute.arch.sync_threads()
    if thread == 0:
        counter = ready.iterator + workers + 1
        cute.arch.red(counter, cutlass.Int32(1), op="add", dtype="s32", sem="release", scope="gpu")


@cute.jit
def counting_grid_wait(
    ready: cute.Tensor, workers: cutlass.Int32, ordinal: cutlass.Int32
):
    """The waiting half of ``counting_grid_barrier``."""
    thread, _, _ = cute.arch.thread_idx()
    if thread == 0:
        counter = ready.iterator + workers + 1
        target = (ordinal + cutlass.Int32(1)) * workers
        seen = cute.arch.load(counter, cutlass.Int32, sem="acquire", scope="gpu")
        while seen < target:
            seen = cute.arch.load(counter, cutlass.Int32, sem="acquire", scope="gpu")
    cute.arch.sync_threads()


@cute.jit
def checked_grid_barrier(
    ready: cute.Tensor,
    workers: cutlass.Int32,
    ordinal: cutlass.Int32,
    stall_base: cutlass.Int64,
    call: cutlass.Int32,
    phase: cutlass.Int32,
    timeout_ns: cutlass.Constexpr,
    arrive: cutlass.Boolean = True,
):
    """``counting_grid_barrier`` whose wait reports a stall (X0.3).
    ``arrive=False`` skips this CTA's arrival (the injection test only); it
    must be the same on every thread of the CTA."""
    if arrive:
        counting_grid_arrive(ready, workers)
    else:
        cute.arch.sync_threads()
    checked_grid_wait(ready, workers, ordinal, stall_base, call, phase, timeout_ns)


@cute.jit
def checked_grid_wait(
    ready: cute.Tensor,
    workers: cutlass.Int32,
    ordinal: cutlass.Int32,
    stall_base: cutlass.Int64,
    call: cutlass.Int32,
    phase: cutlass.Int32,
    timeout_ns: cutlass.Constexpr,
):
    """``counting_grid_wait`` with stall diagnostics: warp 0 polls on every
    lane. The report's counter is the barrier ordinal."""
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp == 0:
        poll_at_least(
            ready.iterator + workers + 1,
            (ordinal + cutlass.Int32(1)) * workers,
            stall_base,
            call,
            STALL_GRID_BARRIER,
            phase,
            ordinal.to(cutlass.Int64),
            timeout_ns,
        )
    cute.arch.sync_threads()


def stall_region_words(workers: int) -> int:
    return workers * STALL_WORDS


def stall_records(region) -> list[dict]:
    """Decode a stall region (an int64 numpy array or tensor, workers × 8)."""
    import numpy as np

    words = np.asarray(region).reshape(-1, STALL_WORDS)
    records = []
    for cta, row in enumerate(words):
        if row[0] != 1:
            continue
        call, kind, phase, counter, target, observed, time = (int(value) for value in row[1:])
        records.append({
            "cta": cta, "call": call, "kind": STALL_KINDS.get(kind, str(kind)), "phase": phase,
            "counter": counter, "target": target, "observed": observed, "time_ns": time,
        })
    return records


def format_stall_report(
    records: list[dict], workers: int | None = None, phase_names: dict[int, str] | None = None
) -> str:
    """One line per (kind, phase, counter, target, observed) with its CTAs.
    The counter is the barrier ordinal for grid barriers and the phase-counter
    index for attention exchanges (their phase is -1: bodies do not know it)."""
    if not records:
        return "MegaBake stall report: no CTA reported a stall."
    groups: dict[tuple, list[int]] = {}
    for record in records:
        key = (record["call"], record["kind"], record["phase"], record["counter"],
               record["target"], record["observed"])
        groups.setdefault(key, []).append(record["cta"])
    lines = [f"MegaBake stall report: {len(records)}"
             + (f" of {workers}" if workers is not None else "") + " CTAs timed out waiting"]
    lines.append(f"  {'call':>5} {'kind':<26} {'phase':>5} {'counter':>8} {'target':>10} {'observed':>10}  CTAs")
    for (call, kind, phase, counter, target, observed), ctas in sorted(groups.items()):
        name = (phase_names or {}).get(phase)
        lines.append(f"  {call:>5} {kind:<26} {phase:>5} {counter:>8} {target:>10} {observed:>10}  "
                     f"{_ranges(sorted(ctas))}" + (f"  (closes after {name})" if name else ""))
    if workers is not None and len(records) < workers:
        silent = sorted(set(range(workers)) - {record["cta"] for record in records})
        lines.append(f"  no report from CTAs {_ranges(silent)} (running, or stuck in a wait without a check)")
    return "\n".join(lines)


def _ranges(values: list[int]) -> str:
    parts, start = [], None
    for index, value in enumerate(values):
        if start is None:
            start = value
        if index + 1 == len(values) or values[index + 1] != value + 1:
            parts.append(str(start) if start == value else f"{start}-{value}")
            start = None
    return ",".join(parts)


# Live stall regions: (the ring view that owns the pinned allocation, region
# offset in int64 words, workers, phase names). Strong references: a failing
# call's runtime state is freed while its exception unwinds, before the hook
# runs. Each ring is a few KB of pinned memory; the newest 256 are kept (one
# per CUDA-graph capture adds up). The hook prints a report for every region
# with a record when the process dies on an error.
_STALL_REGIONS: list = []
_STALL_REGION_LIMIT = 256
_STALL_HOOK_INSTALLED = False


def register_stall_region(
    owner: torch.Tensor, offset: int, workers: int, phase_names: dict[int, str] | None = None
) -> None:
    _STALL_REGIONS.append((owner, offset, workers, phase_names))
    del _STALL_REGIONS[:-_STALL_REGION_LIMIT]
    install_stall_hook()


def pending_stall_reports() -> list[str]:
    reports = []
    for owner, offset, workers, phase_names in _STALL_REGIONS:
        region = torch.empty(0, dtype=torch.int64).set_(
            owner.untyped_storage(), offset, (stall_region_words(workers),)
        )
        records = stall_records(region.numpy())
        if records:
            reports.append(format_stall_report(records, workers, phase_names))
    return reports


def install_stall_hook() -> None:
    """Print pending stall reports before an uncaught exception's traceback."""
    global _STALL_HOOK_INSTALLED
    if _STALL_HOOK_INSTALLED:
        return
    import sys

    previous = sys.excepthook

    def hook(kind, value, traceback):
        for report in pending_stall_reports():
            print(report, file=sys.stderr)
        previous(kind, value, traceback)

    sys.excepthook = hook
    _STALL_HOOK_INSTALLED = True


@dsl_user_op
def prefetch_l1(pointer, *, loc=None, ip=None):
    """Prefetch the line holding a global ``pointer`` into L1."""
    llvm.inline_asm(
        res=None,
        operands_=[pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        asm_string="prefetch.global.L1 [$0];",
        constraints="l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prefetch_tensormap(address, *, loc=None, ip=None):
    """Prefetch the TMA descriptor at a global ``address`` (Int64), so a
    later TMA copy does not fetch it on its critical path."""
    llvm.inline_asm(
        res=None,
        operands_=[address.ir_value(loc=loc, ip=ip)],
        asm_string="prefetch.tensormap [$0];",
        constraints="l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def cooperative_worker_loop(
    ready: cute.Tensor,
    workers: cutlass.Int32,
    epoch_base: cutlass.Int32,
    phases: cutlass.Constexpr,
):
    for phase_index, run_phase, drain, publish in phases:
        run_phase(phase_index)
        drain()
        publish(epoch_base)


@cute.jit
def cooperative_event_worker_loop(
    events: cute.Tensor,
    workers: cutlass.Int32,
    epoch: cutlass.Int32,
    phases: cutlass.Constexpr,
):
    for phase_index, wait, run, publish in phases:
        wait(epoch)
        run(phase_index)
        publish()


class PhaseKernel(Protocol):
    def __call__(
        self,
        ready: torch.Tensor,
        epoch: int,
        stream: torch.cuda.Stream,
    ) -> None: ...


class EventKernel(Protocol):
    def __call__(
        self,
        events: torch.Tensor,
        epoch: int,
        stream: torch.cuda.Stream,
    ) -> None: ...


@dataclass(frozen=True)
class RuntimeAdmission:
    plan: KernelPlan
    barrier_count: int
    max_workers: int


@dataclass(frozen=True)
class RuntimeInvocation:
    epoch: int
    barrier_count: int


@dataclass(frozen=True)
class EventRuntimeAdmission:
    schedule: EventSchedule
    event_count: int
    max_invocations: int


@dataclass(frozen=True)
class EventRuntimeInvocation:
    epoch: int
    event_count: int


@cute.jit
def publish_event(
    events: cute.Tensor,
    event_id: cutlass.Int32,
):
    """Publish one producer CTA after the caller has drained async body work."""
    cute.arch.sync_threads()
    thread, _, _ = cute.arch.thread_idx()
    if thread == 0:
        cute.arch.atomic_add(
            events.iterator + event_id,
            1,
            sem="release",
            scope="gpu",
        )
    cute.arch.sync_threads()


@cute.jit
def wait_for_event(
    events: cute.Tensor,
    event_id: cutlass.Int32,
    trigger_count: cutlass.Int32,
    epoch: cutlass.Int32,
):
    thread, _, _ = cute.arch.thread_idx()
    cute.arch.sync_threads()
    if thread == 0:
        target = (epoch + 1) * trigger_count
        count = 0
        while count < target:
            count = cute.arch.atomic_cas(
                events.iterator + event_id,
                cmp=target,
                val=target,
                sem="acquire",
                scope="gpu",
            )
    cute.arch.sync_threads()


def admit_runtime(plan: KernelPlan) -> RuntimeAdmission:
    verify_plan(plan)
    if not plan.launch.cooperative:
        raise InvalidRuntime("phase plans require a cooperative launch")

    target = plan.target
    if target is None:
        raise InvalidRuntime("phase plan is missing target facts")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if _architecture_number(plan.launch.architecture) != _architecture_number(
        f"sm{properties.major}{properties.minor}"
    ):
        raise InvalidRuntime(
            f"plan architecture {plan.launch.architecture} does not match the current device"
        )
    if properties.multi_processor_count != target.multiprocessor_count:
        raise InvalidRuntime("plan multiprocessor count does not match the current device")
    if not _cooperative_launch_supported():
        raise InvalidRuntime("the current device does not support cooperative launch")

    if plan.launch.max_active_blocks_per_sm is None:
        raise InvalidRuntime("plan has no compiled active-block capacity")
    max_workers = (
        plan.launch.max_active_blocks_per_sm * properties.multi_processor_count
    )
    if plan.launch.grid_size > max_workers:
        raise InvalidRuntime(
            f"cooperative grid {plan.launch.grid_size} exceeds capacity {max_workers}"
        )

    barrier_count = sum(phase.barrier is not None for phase in plan.phases)
    if barrier_count > _MAX_INT32:
        raise InvalidRuntime("phase epoch values would overflow int32")
    return RuntimeAdmission(plan, barrier_count, max_workers)


def execute_phase_plan(
    plan: KernelPlan,
    kernel: PhaseKernel,
    stream: torch.cuda.Stream | None = None,
    *,
    admission: RuntimeAdmission | None = None,
    ready: torch.Tensor | None = None,
    epoch: int | None = None,
    synchronize: bool = True,
    capturable: bool = False,
) -> RuntimeInvocation:
    """Launch ``kernel`` on ``stream``. With ``synchronize=False`` the call
    returns once the launch is enqueued; its results are stream-ordered.
    ``capturable`` declares that ``kernel`` takes its epochs from the device,
    so a CUDA graph may capture it."""
    active_stream = stream or torch.cuda.current_stream()
    device = active_stream.device
    selected_admission = admission
    if selected_admission is None:
        with torch.cuda.device(device):
            selected_admission = admit_runtime(plan)

    if not capturable and _stream_is_capturing(active_stream):
        raise InvalidRuntime("CUDA Graph capture is not supported for phase plans")
    with torch.cuda.stream(active_stream):
        selected_ready = (
            ready if ready is not None else make_ready(plan.launch.grid_size, device)
        )
        selected_epoch = 0 if epoch is None else epoch
        kernel(selected_ready, selected_epoch, active_stream)
        if synchronize:
            active_stream.synchronize()

    return RuntimeInvocation(selected_epoch, selected_admission.barrier_count)


def admit_event_runtime(schedule: EventSchedule) -> EventRuntimeAdmission:
    if not isinstance(schedule, EventSchedule):
        raise InvalidRuntime("admit_event_runtime expects an EventSchedule")
    simulate_event_schedule(schedule)
    event_count = len(schedule.events)
    if event_count > _MAX_INT32:
        raise InvalidRuntime("event IDs would overflow int32")
    if any(event.trigger_count < 1 for event in schedule.events):
        raise InvalidRuntime("event plans require at least one producer per event")
    max_invocations = min(
        (
            _MAX_INT32 // event.trigger_count
            for event in schedule.events
        ),
        default=_MAX_INT32,
    )
    if max_invocations < 1:
        raise InvalidRuntime("event counters would overflow int32")
    return EventRuntimeAdmission(schedule, event_count, max_invocations)


def execute_event_plan(
    schedule: EventSchedule,
    kernel: EventKernel,
    stream: torch.cuda.Stream | None = None,
    *,
    admission: EventRuntimeAdmission | None = None,
    events: torch.Tensor | None = None,
    epoch: int | None = None,
    validate_counters: bool = True,
    synchronize: bool = True,
) -> EventRuntimeInvocation:
    active_stream = stream or torch.cuda.current_stream()
    device = active_stream.device
    selected_admission = admission
    if selected_admission is None:
        with torch.cuda.device(device):
            selected_admission = admit_event_runtime(schedule)

    if _stream_is_capturing(active_stream):
        raise InvalidRuntime("CUDA Graph capture is not supported for event plans")
    with torch.cuda.stream(active_stream):
        selected_events = events if events is not None else torch.zeros(
            selected_admission.event_count,
            dtype=torch.int32,
            device=device,
        )
        if tuple(selected_events.shape) != (selected_admission.event_count,):
            raise InvalidRuntime("event tensor shape does not match the event plan")
        if selected_events.dtype != torch.int32 or selected_events.device != device:
            raise InvalidRuntime("event tensor must be an int32 tensor on the stream device")
        if (
            epoch is not None
            and not 0 <= epoch < selected_admission.max_invocations
        ):
            raise InvalidRuntime("event epoch would overflow int32 counters")
        selected_epoch = 0 if epoch is None else epoch
        if events is not None and validate_counters:
            if selected_epoch == 0:
                if selected_events.count_nonzero():
                    raise InvalidRuntime(
                        "the first event invocation requires zeroed counters"
                    )
            else:
                for event in schedule.events:
                    if selected_events[event.id].item() != selected_epoch * event.trigger_count:
                        raise InvalidRuntime(
                            f"event {event.id} counters do not match epoch {selected_epoch}"
                        )
        kernel(selected_events, selected_epoch, active_stream)
        if synchronize:
            active_stream.synchronize()

    return EventRuntimeInvocation(
        selected_epoch,
        selected_admission.event_count,
    )


def _architecture_number(architecture: str) -> str:
    return architecture.removeprefix("sm").removesuffix("a")


def _cooperative_launch_supported() -> bool:
    device = cuda.cuDeviceGet(torch.cuda.current_device())[1]
    status, supported = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
        device,
    )
    return int(status) == 0 and int(supported) == 1


def _stream_is_capturing(stream: torch.cuda.Stream) -> bool:
    status, capture = cuda.cuStreamIsCapturing(cuda.CUstream(stream.cuda_stream))
    return int(status) == 0 and int(
        capture
    ) == int(cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE)
