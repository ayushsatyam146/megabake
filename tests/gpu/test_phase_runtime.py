from __future__ import annotations

from dataclasses import replace

import cutlass
import pytest
import torch
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from megabake import (
    admit_runtime,
    assign_storage,
    derive_dependencies,
    execute_phase_plan,
    make_phase_schedule,
    select_bodies,
)
from megabake.bodies import TargetFacts
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions
from megabake.runtime import (
    cooperative_worker_loop,
    grid_barrier,
    make_ready,
    ready_slots,
)
from megabake.schedule import WorkerAssignment

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@cute.jit
def _no_op():
    pass


@cute.jit
def _ignore_epoch(epoch: cutlass.Int32):
    pass


@cute.kernel
def _phase_kernel(
    source: cute.Tensor,
    produced: cute.Tensor,
    ready: cute.Tensor,
    sink: cute.Tensor,
    workers: cutlass.Int32,
    epoch: cutlass.Int32,
):
    _, _, block = cute.arch.block_idx()

    @cute.jit
    def produce(phase_index: cutlass.Int32):
        active = cutlass.Int32(block != workers - 1)
        total = cutlass.Float32(0.0)
        for index in cutlass.range(active * 100_000):
            total = total + source[index % 257]
        produced[block] = total

    @cute.jit
    def consume(phase_index: cutlass.Int32):
        total = cutlass.Float32(0.0)
        for worker in cutlass.range(workers - 1):
            total = total + produced[worker]
        sink[0] = total

    @cute.jit
    def publish(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch)

    cooperative_worker_loop(
        ready,
        workers,
        epoch,
        (
            (0, produce, _no_op, publish),
            (1, consume, _no_op, _ignore_epoch),
        ),
    )


@cute.jit
def _launch(
    source: cute.Tensor,
    produced: cute.Tensor,
    ready: cute.Tensor,
    sink: cute.Tensor,
    workers: cutlass.Int32,
    epoch: cutlass.Int32,
    stream: cuda.CUstream,
):
    _phase_kernel(
        source,
        produced,
        ready,
        sink,
        workers,
        epoch,
    ).launch(
        grid=[1, 1, workers],
        block=[1, 1, 1],
        cooperative=True,
        stream=stream,
    )


def _target() -> TargetFacts:
    from megabake.compiler import _target_facts

    return _target_facts(torch.device("cuda"))


def _plan(workers: int):
    class Model(torch.nn.Module):
        def forward(self, value):
            left = value.sum(dim=-1)
            right = value.amax(dim=-1)
            return left + right

    value = torch.randn(2, 3)
    exported = torch.export.export(Model(), (value,))
    prepared = prepare_fx_graph(exported.graph_module, (value,))
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    selection = select_bodies(graph, _target())
    schedule = make_phase_schedule(selection, derive_dependencies(selection))
    plan = assign_storage(schedule)
    phases = tuple(
        replace(
            phase,
            workers=(
                *phase.workers,
                *(
                    WorkerAssignment(worker_id, phase.region_id, ())
                    for worker_id in range(schedule.worker_count, workers)
                ),
            ),
            barrier=None
            if phase.barrier is None
            else replace(phase.barrier, participating_workers=tuple(range(workers))),
        )
        for phase in plan.phases
    )
    return replace(
        plan,
        phases=phases,
        storage=replace(
            plan.storage,
            worker_scratch=tuple(
                replace(scratch, worker_id=worker_id)
                for worker_id, scratch in enumerate(plan.storage.worker_scratch)
            )
            + tuple(
                replace(plan.storage.worker_scratch[0], worker_id=worker_id)
                for worker_id in range(schedule.worker_count, workers)
            ),
        ),
        launch=replace(
            plan.launch,
            grid_size=workers,
            max_active_blocks_per_sm=1,
            required_active_blocks=workers,
        ),
    )


def _flat(tensor):
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def test_phase_runtime_ports_delay_idle_rebinding_and_largest_grid():
    device = torch.device("cuda")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        source = torch.ones(257, device=device, dtype=torch.float32)
        produced = torch.empty(max(4, sms), device=device, dtype=torch.float32)
        ready = make_ready(4, device)
        sink = torch.empty(1, device=device, dtype=torch.float32)
        compiled = cute.compile(
            _launch,
            _flat(source),
            _flat(produced),
            _flat(ready),
            _flat(sink),
            cutlass.Int32(4),
            cutlass.Int32(0),
            cuda.CUstream(stream.cuda_stream),
        )

    current_source = source

    def kernel(ready_tensor, epoch, active_stream):
        compiled(
            _flat(current_source),
            _flat(produced),
            _flat(ready_tensor),
            _flat(sink),
            cutlass.Int32(ready_tensor.numel() - ready_slots(0)),
            cutlass.Int32(epoch),
            cuda.CUstream(active_stream.cuda_stream),
        )

    four_worker_plan = _plan(4)
    assert admit_runtime(four_worker_plan).max_workers >= 4
    invocation = execute_phase_plan(four_worker_plan, kernel, stream)
    assert invocation.epoch == 0
    # The independent sum and amax share one barrier before their addition.
    assert invocation.barrier_count == 1
    assert sink.item() == 300_000.0

    current_source = torch.full((257,), 2.0, device=device, dtype=torch.float32)
    execute_phase_plan(four_worker_plan, kernel, stream)
    assert sink.item() == 600_000.0

    largest_plan = _plan(sms)
    assert admit_runtime(largest_plan).max_workers == sms
    execute_phase_plan(largest_plan, kernel, stream)
    assert sink.item() == (sms - 1) * 200_000.0


def test_phase_runtime_rejects_cuda_graph_capture():
    plan = _plan(4)

    def kernel(ready, epoch, stream):
        raise AssertionError("kernel must not launch during rejected capture")

    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        graph = torch.cuda.CUDAGraph()
        with (
            pytest.raises(Exception, match="CUDA Graph capture is not supported"),
            torch.cuda.graph(graph, stream=capture_stream),
        ):
            execute_phase_plan(plan, kernel, capture_stream)
