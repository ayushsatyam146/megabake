from __future__ import annotations

import cutlass
import pytest
import torch
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from megabake import (
    derive_dependencies,
    execute_event_plan,
    make_event_schedule,
    select_bodies,
)
from megabake.bodies import TargetFacts
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions
from megabake.runtime import publish_event, wait_for_event

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@cute.kernel
def _event_kernel(
    events: cute.Tensor,
    event_id: cutlass.Int32,
    trigger_count: cutlass.Int32,
    epoch: cutlass.Int32,
):
    _, _, block = cute.arch.block_idx()
    if block == 0:
        publish_event(events, event_id)
        publish_event(events, cutlass.Int32(1))
    else:
        wait_for_event(events, event_id, trigger_count, epoch)


@cute.jit
def _event_launch(
    events: cute.Tensor,
    event_id: cutlass.Int32,
    trigger_count: cutlass.Int32,
    epoch: cutlass.Int32,
    stream: cuda.CUstream,
):
    _event_kernel(
        events,
        event_id,
        trigger_count,
        epoch,
    ).launch(
        grid=[1, 1, 2],
        block=[1, 1, 1],
        cooperative=True,
        stream=stream,
    )


def _target() -> TargetFacts:
    return TargetFacts(
        architecture="sm90a",
        threads_per_warp=32,
        max_threads_per_cta=1024,
        max_shared_memory_per_cta=232_448,
        max_registers_per_thread=255,
        multiprocessor_count=60,
        supports_tma=True,
        supports_wgmma=True,
    )


def _schedule():
    class Model(torch.nn.Module):
        def forward(self, value):
            left = value.sum(dim=-1)
            right = value.amax(dim=-1)
            return left + right

    value = torch.randn(1, 3)
    exported = torch.export.export(Model(), (value,))
    prepared = prepare_fx_graph(exported.graph_module, (value,))
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    selection = select_bodies(graph, _target())
    return make_event_schedule(selection, derive_dependencies(selection), worker_count=2)


def _flat(tensor):
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def test_event_runtime_publishes_and_waits_across_ctas():
    device = torch.device("cuda")
    stream = torch.cuda.Stream(device=device)
    schedule = _schedule()
    assert len(schedule.events) == 2
    assert all(event.trigger_count == 1 for event in schedule.events)
    events = torch.zeros(len(schedule.events), device=device, dtype=torch.int32)
    with torch.cuda.stream(stream):
        compiled = cute.compile(
            _event_launch,
            _flat(events),
            cutlass.Int32(0),
            cutlass.Int32(1),
            cutlass.Int32(0),
            cuda.CUstream(stream.cuda_stream),
        )

    def kernel(event_tensor, epoch, active_stream):
        compiled(
            _flat(event_tensor),
            cutlass.Int32(0),
            cutlass.Int32(1),
            cutlass.Int32(epoch),
            cuda.CUstream(active_stream.cuda_stream),
        )

    invocation = execute_event_plan(schedule, kernel, stream, events=events)
    assert invocation.event_count == len(schedule.events)
    execute_event_plan(schedule, kernel, stream, events=events, epoch=1)
    assert events.tolist() == [2, 2]
