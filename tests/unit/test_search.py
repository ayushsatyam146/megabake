"""Pruning and joining candidates, the split planner, the staging planner,
and the mapping of a planned graph."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megabake.bodies import SelectionPolicy, TargetFacts
from megabake.bodies.cuda.sm90.gemm import plan_splits
from megabake.costmodel import Interval, gemm_work, generic_work
from megabake.device import load_spec, tree_for_target
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.plan import SearchBudget, search_plans
from megabake.regions import form_regions
from megabake.search import MARGIN, Candidate, ShapeClass, best_of, join, pareto
from megabake.staging import MIN_PAGES, plan_staging

TREE = load_spec("h200-mig-3g71gb")
PAGE, PAGE_WEIGHTS = 10240, 8192


def test_pareto_keeps_what_nothing_beats_on_time_and_reservations():
    fast_big = Candidate("fast", 1.0, {"smem": 200})
    slow_small = Candidate("small", 2.0, {"smem": 50})
    dominated = Candidate("worse", 2.5, {"smem": 200})
    assert pareto((fast_big, slow_small, dominated)) == (fast_big, slow_small)


def test_join_respects_a_shared_capacity():
    gemm = ShapeClass("gemm", 10, (Candidate("deep", 4.0, {"smem": 100}), Candidate("deeper", 3.0, {"smem": 180})))
    attention = ShapeClass("attention", 10, (Candidate("rows", 2.0, {"smem": 60}),))
    unlimited = join((gemm, attention))
    assert unlimited.choices == {"gemm": "deeper", "attention": "rows"}
    assert unlimited.time_s == pytest.approx(50.0)
    assert unlimited.reservations == {"smem": 180}
    # The faster GEMM does not fit: the kernel takes the other one.
    limited = join((gemm, attention), {"smem": 150})
    assert limited.choices["gemm"] == "deep"
    assert join((gemm, attention), {"smem": 50}) is None


def test_a_candidate_must_beat_the_default_by_the_margin():
    slightly = ShapeClass("a", 1, (Candidate("default", 1.0), Candidate("other", 1.0 - MARGIN / 2)))
    clearly = ShapeClass("a", 1, (Candidate("default", 1.0), Candidate("other", 1.0 - 2 * MARGIN)))
    assert join((slightly,)).choices == {"a": "default"}
    assert join((clearly,)).choices == {"a": "other"}
    kernels = {"default": join((slightly,)), "other": replace(join((slightly,)), time_s=0.99)}
    assert best_of(kernels) == "default"
    assert best_of({**kernels, "other": replace(kernels["other"], time_s=0.5)}) == "other"


def _program(rows: int, columns: int, k: int):
    """What plan_splits reads of a contraction program."""
    from megabake.ir import IndexedLoad, IterationDomain, TensorProgram, TensorStore

    return TensorProgram(
        family="contraction", source_target="mm", decomposition="",
        iteration_domains=(IterationDomain("o0", rows), IterationDomain("o1", columns), IterationDomain("r0", k)),
        reduction_axes=("r0",),
        loads=(
            IndexedLoad("load:0", "a", ("o0", "r0"), (k, 1), "torch.bfloat16", (rows, k)),
            IndexedLoad("load:1", "b", ("r0", "o1"), (1, k), "torch.bfloat16", (k, columns)),
        ),
        init=None, combine=None, post=None, accumulator_dtype="torch.float32",
        store=TensorStore("c", (rows, columns), "torch.bfloat16", (columns, 1), True),
        predicates=(),
    )


def test_split_planner_keeps_decode_gemms_whole_on_the_slice():
    # SmolLM-135M, Qwen2-0.5B and Llama-1B shapes: measured slower split.
    programs = [_program(1, 576, 576)] * 30 + [_program(1, 896, 896)] * 24 + [_program(1, 2048, 8192)] * 16
    assert plan_splits(programs, 16, TREE) == ()


def test_split_planner_splits_where_few_leaves_leave_the_edge_idle():
    # The same kernel on a device whose leaves each pull little of the edge.
    slow_leaves = replace(load_spec("mi300x"), bodies=TREE.bodies, sync=TREE.sync)
    splits = dict(plan_splits([_program(1, 576, 8192)] * 30, 16, slow_leaves))
    assert splits[(9, 128)] > 1


def _layer(tree, tile_bytes: float):
    gemm = lambda name, tiles: gemm_work(  # noqa: E731
        name, tree, tiles=tiles, tile_bytes=tile_bytes, staged=True, paged=True
    )
    idle = generic_work("attention", tree, items=9, rounds=5)
    return [Interval((idle,), 60), Interval((gemm("o", 9),), 60), Interval((gemm("down", 9),), 60)]


def test_staging_sizes_the_ring_from_in_flight_bytes_and_run_ahead():
    free = TREE.leaf.capacity_bytes
    small = plan_staging(_layer(TREE, 16 * 1024), TREE, free_bytes=free, page_bytes=PAGE, staged_bytes_per_page=PAGE_WEIGHTS)
    large = plan_staging(_layer(TREE, 512 * 1024), TREE, free_bytes=free, page_bytes=PAGE, staged_bytes_per_page=PAGE_WEIGHTS)
    in_flight = TREE.read_share(1) * TREE.mechanism().latency_s
    assert small.producer and large.producer
    # Small tiles: what stays in flight, plus one tile of run-ahead.
    assert small.ring_pages * PAGE_WEIGHTS >= in_flight + 16 * 1024
    assert small.ring_pages < large.ring_pages
    # Large tiles: as deep as the leaf's memory allows.
    assert large.ring_pages == free // PAGE // 2 * 2
    assert small.ring_pages % 2 == 0 and small.ring_pages >= MIN_PAGES


def test_staging_needs_room_and_paged_work():
    no_room = plan_staging(_layer(TREE, 512 * 1024), TREE, free_bytes=4 * PAGE, page_bytes=PAGE, staged_bytes_per_page=PAGE_WEIGHTS)
    assert not no_room.producer and "fit" in no_room.reason
    unpaged = [Interval((generic_work("norm", TREE, items=1, rounds=1),), 60)]
    assert not plan_staging(unpaged, TREE, free_bytes=1 << 20, page_bytes=PAGE, staged_bytes_per_page=PAGE_WEIGHTS).producer


def test_a_faster_edge_or_slower_copy_wants_a_deeper_ring():
    from megabake.device import with_copy_latency

    base = plan_staging(_layer(TREE, 16 * 1024), TREE, free_bytes=1 << 20, page_bytes=PAGE, staged_bytes_per_page=PAGE_WEIGHTS)
    slow_copy = with_copy_latency(TREE, 2 * TREE.mechanism().latency_s)
    deeper = plan_staging(_layer(slow_copy, 16 * 1024), slow_copy, free_bytes=1 << 20, page_bytes=PAGE, staged_bytes_per_page=PAGE_WEIGHTS)
    assert deeper.ring_pages > base.ring_pages


def _target(sms: int, architecture: str = "sm90a") -> TargetFacts:
    return TargetFacts(architecture, 32, 1024, 232_448, 255, sms, True, architecture == "sm90a")


def _graph(model, *inputs):
    prepared = prepare_fx_graph(torch.export.export(model, inputs).graph_module, inputs)
    return form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))


@pytest.mark.parametrize("sms", [60, 132])
def test_a_planned_graph_carries_its_mapping_on_any_hopper_tree(sms):
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return torch.relu(source @ weight.T)

    source = torch.randn(128, 256, dtype=torch.bfloat16)
    weight = torch.randn(512, 256, dtype=torch.bfloat16)
    target = _target(sms)
    tree = tree_for_target(target)
    result = search_plans(
        _graph(Model(), source, weight), target, SelectionPolicy(device=tree), SearchBudget(evaluator=None)
    )
    plans = [candidate.plan for candidate in result.candidates if candidate.plan is not None]
    assert plans
    mapping = plans[0].mapping
    assert mapping.device == tree.name
    gemm = next(region for region in mapping.regions if region.view.accumulators)
    assert gemm.items == len(next(p for p in plans[0].phases if p.region_id == gemm.region_id).work_items)
    assert gemm.leaf_bytes > 0 and gemm.traffic_bytes >= gemm.view.bytes_read()
    assert mapping.edges, "the GEMM hands its result to the activation"
    bounds = mapping.estimate.bounds
    assert 0 < bounds.b0 <= bounds.b2 <= bounds.b3 <= mapping.estimate.total_s
    assert mapping.report()["predicted_us"] == pytest.approx(mapping.estimate.total_s * 1e6)
    # Without a tree the plan is as before.
    plain = search_plans(_graph(Model(), source, weight), target, SelectionPolicy(), SearchBudget(evaluator=None))
    assert all(candidate.plan is None or candidate.plan.mapping is None for candidate in plain.candidates)


def test_only_a_calibrated_tree_changes_the_kernel():
    from megabake.bodies.cuda.sm90.gemm import launch_options

    # A shape the slow-leaf device splits (see above), as a launch's graph.
    graph = SimpleNamespace(
        ops={f"op:{index}": SimpleNamespace(tensor_program=_program(1, 576, 8192)) for index in range(30)}
    )
    slow_leaves = replace(load_spec("mi300x"), bodies=TREE.bodies, sync=TREE.sync)
    assert not slow_leaves.measured
    # A spec tree's rates are assumptions: the tested default stays.
    assert launch_options(graph, True, slow_leaves).splits == ()
    calibrated = replace(slow_leaves, provenance={**slow_leaves.provenance, "edges": "measured"})
    assert dict(launch_options(graph, True, calibrated).splits)[(9, 128)] > 1
    # No tree: as before the planner existed.
    assert launch_options(graph, True).splits == ()
