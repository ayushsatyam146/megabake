from __future__ import annotations

from dataclasses import replace

import torch

from megabake import (
    admit_event_runtime,
    derive_dependencies,
    make_event_schedule,
    make_phase_schedule,
    select_bodies,
    simulate_event_schedule,
    simulate_phase_schedule,
)
from megabake.bodies import TargetFacts
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions
from megabake.schedule import DependencyKind, DependencyPlan, RegionDependency


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


def _graph(model, *inputs):
    exported = torch.export.export(model, inputs)
    prepared = prepare_fx_graph(exported.graph_module, inputs)
    return form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))


def _scheduled(model, *inputs):
    selection = select_bodies(_graph(model, *inputs), _target())
    dependencies = derive_dependencies(selection)
    return selection, dependencies, make_phase_schedule(selection, dependencies)


def test_phase_schedule_handles_fork_join_and_keeps_idle_workers_in_barriers():
    class Model(torch.nn.Module):
        def forward(self, left, right):
            left = left.sum(dim=-1)
            right = right.sum(dim=-1)
            return left + right

    left = torch.randn(1, 3)
    right = torch.randn(1, 3)
    selection, dependencies, schedule = _scheduled(Model(), left, right)
    join = next(
        selected
        for selected in selection.selections
        if selected.outputs[0] == selection.graph.outputs[0]
    )

    assert len(dependencies.prerequisites_for(join.region_id)) == 2
    assert len(schedule.phases) == 3
    # The two independent reductions share one barrier interval.
    assert len(schedule.barriers) == 1
    assert schedule.phases[0].barrier is None
    assert schedule.phases[1].barrier is not None
    assert all(
        barrier.participating_workers == tuple(range(schedule.worker_count))
        for barrier in schedule.barriers
    )

    simulation = simulate_phase_schedule(schedule)
    assert simulation.phase_order.index(join.region_id) == 2
    assert len(simulation.barrier_order) == 1


def test_phase_schedule_allows_aliasing_view_without_an_unscheduled_phase():
    class Model(torch.nn.Module):
        def forward(self, source, weight, bias):
            return torch.nn.functional.linear(source, weight, bias)

    source = torch.randn(2, 3)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)
    _, dependencies, schedule = _scheduled(Model(), source, weight, bias)

    assert dependencies.dependencies == ()
    assert len(schedule.phases) == 1
    assert schedule.barriers == ()
    simulation = simulate_phase_schedule(schedule)
    assert simulation.barrier_order == ()


def test_dependencies_include_all_reduction_input_tiles():
    class Model(torch.nn.Module):
        def forward(self, source, first_weight, second_weight):
            hidden = source @ first_weight.T
            return hidden @ second_weight.T

    source = torch.randn(128, 128, dtype=torch.bfloat16)
    first_weight = torch.randn(256, 128, dtype=torch.bfloat16)
    second_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    _, dependencies, schedule = _scheduled(Model(), source, first_weight, second_weight)
    dependency = next(
        dependency
        for dependency in dependencies.dependencies
        if dependency.kind == DependencyKind.PRODUCER_TO_CONSUMER
    )

    assert dependency.producer_tiles == (
        (0, 0),
        (0, 128),
        (64, 0),
        (64, 128),
    )
    assert dependency.consumer_tiles == ((0, 0), (64, 0))
    assert not dependency.conservative

    simulation = simulate_phase_schedule(schedule)
    assert simulation.work_item_counts[schedule.phases[0].region_id] == 4
    assert simulation.work_item_counts[schedule.phases[-1].region_id] == 2
    assert simulation.idle_worker_counts[schedule.phases[-1].region_id] == 2


def test_event_schedule_interleaves_fork_join_without_a_phase_barrier():
    class Model(torch.nn.Module):
        def forward(self, left, right):
            left = left.sum(dim=-1)
            right = right.sum(dim=-1)
            return left + right

    left = torch.randn(1, 3)
    right = torch.randn(1, 3)
    selection, dependencies, _ = _scheduled(Model(), left, right)
    schedule = make_event_schedule(selection, dependencies, worker_count=2)
    join = next(
        task for task in schedule.tasks if task.region_id == selection.selections[-1].region_id
    )

    assert [task.order for task in schedule.tasks] == [0, 1, 2]
    assert all(
        [schedule.task_for(task_id).order for task_id in worker.task_ids]
        == sorted(schedule.task_for(task_id).order for task_id in worker.task_ids)
        for worker in schedule.workers
    )
    assert len(join.wait_event_ids) == 2
    assert all(event.trigger_count == 1 for event in schedule.events)
    assert schedule.events

    simulation = simulate_event_schedule(schedule)
    assert simulation.finish_time == 2
    assert simulation.phase_finish_time == 3
    assert simulation.speedup_ticks == 1
    assert simulation.outputs_match_phase
    assert simulation.worker_idle_ticks == {0: 0, 1: 1}

    delayed_producer = next(
        task.id
        for task in schedule.tasks
        if task.region_id != join.region_id and task.worker_id == 1
    )
    delayed = simulate_event_schedule(schedule, {delayed_producer: 2})
    assert delayed.finish_time == 4
    assert delayed.phase_finish_time == 5
    assert delayed.speedup_ticks == 1
    assert delayed.worker_idle_ticks == {0: 2, 1: 1}
    assert delayed.outputs_match_phase


def test_event_schedule_groups_exact_reduction_prerequisites():
    class Model(torch.nn.Module):
        def forward(self, source, first_weight, second_weight):
            hidden = source @ first_weight.T
            return hidden @ second_weight.T

    source = torch.randn(128, 128, dtype=torch.bfloat16)
    first_weight = torch.randn(256, 128, dtype=torch.bfloat16)
    second_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    selection, dependencies, _ = _scheduled(Model(), source, first_weight, second_weight)
    schedule = make_event_schedule(selection, dependencies)

    assert len(schedule.events) == 2
    assert sorted(event.trigger_count for event in schedule.events) == [2, 2]
    assert all(
        not prerequisite.conservative
        for event in schedule.events
        for prerequisite in event.prerequisites
    )
    assert len(schedule.prerequisites) == 2

    simulation = simulate_event_schedule(schedule)
    assert len(simulation.completed_task_ids) == len(schedule.tasks)
    assert simulation.outputs_match_phase
    assert admit_event_runtime(schedule).event_count == 2


def test_event_schedule_keeps_conservative_state_effects_explicit():
    class Model(torch.nn.Module):
        def forward(self, source, first_weight, second_weight):
            hidden = source @ first_weight.T
            return hidden @ second_weight.T

    source = torch.randn(128, 128, dtype=torch.bfloat16)
    first_weight = torch.randn(256, 128, dtype=torch.bfloat16)
    second_weight = torch.randn(128, 256, dtype=torch.bfloat16)
    selection, dependencies, _ = _scheduled(Model(), source, first_weight, second_weight)
    producer = dependencies.dependencies[0].producer_region_id
    consumer = dependencies.dependencies[0].consumer_region_id
    state_effect = RegionDependency(
        producer_region_id=producer,
        consumer_region_id=consumer,
        value_id="state",
        kind=DependencyKind.WRITE_AFTER_READ,
        producer_tiles=(),
        consumer_tiles=(),
        conservative=True,
        reason="test state effect",
    )
    state_dependencies = replace(
        dependencies,
        dependencies=(*dependencies.dependencies, state_effect),
    )
    assert isinstance(state_dependencies, DependencyPlan)
    schedule = make_event_schedule(selection, state_dependencies)
    state_event = next(
        event
        for event in schedule.events
        if any(
            dependency is state_effect
            for prerequisite in event.prerequisites
            for dependency in prerequisite.dependencies
        )
    )

    assert state_event.trigger_count == 4
    assert state_event.consumer_task_ids == tuple(
        sorted(task.id for task in schedule.tasks if task.region_id == consumer)
    )
    assert any(prerequisite.conservative for prerequisite in state_event.prerequisites)
    simulation = simulate_event_schedule(schedule)
    assert simulation.outputs_match_phase
    assert len(simulation.completed_task_ids) == len(schedule.tasks)
