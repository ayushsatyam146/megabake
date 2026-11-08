from __future__ import annotations

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
