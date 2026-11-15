"""The cost model, and the decisions it must get right.

The decision checks compare two plans whose kernels were measured; the model
must prefer the one that measured faster. The trees are calibrated device
files, so the checks run without a GPU.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from megabake.costmodel import (
    Interval,
    Work,
    attention_work,
    estimate,
    gemm_work,
    generic_work,
    interval_time,
    plan_bounds,
)
from megabake.device import load_spec

TREE = load_spec("h200-mig-3g71gb")
WORKERS = 60
KIB = 1024


def _total(intervals, tree=TREE, form="full") -> float:
    return estimate(intervals, tree, form).total_s


def test_tail_wave_gets_a_larger_share():
    # 61 equal items on 60 leaves: the last runs alone at one leaf's rate.
    work = Work("stream", items=61, item_bytes=1 << 20)
    estimated = interval_time(Interval((work,), WORKERS, "none"), TREE)
    assert estimated.critical_s == pytest.approx(
        (1 << 20) / TREE.read_share(60) + (1 << 20) / TREE.read_share(1)
    )
    assert estimated.active == 60


def test_regions_of_an_interval_are_dealt_end_to_end():
    # 9 + 3 + 3 tiles land on 15 different leaves, as codegen rotates them.
    works = tuple(Work(name, items=count, item_fixed_s=1e-6) for name, count in (("q", 9), ("k", 3), ("v", 3)))
    estimated = interval_time(Interval(works, WORKERS, "none"), TREE)
    assert estimated.critical_s == pytest.approx(1e-6)


def test_recomputed_work_runs_on_every_leaf_before_its_items():
    norm = generic_work("norm", TREE, items=1, rounds=1, every_worker=True)
    gemm = Work("gemm", items=60, item_fixed_s=2e-6)
    alone = interval_time(Interval((gemm,), WORKERS, "none"), TREE).critical_s
    folded = interval_time(Interval((norm, gemm), WORKERS, "none"), TREE).critical_s
    rates = TREE.body("generic")
    assert folded - alone == pytest.approx(rates["start_s"] + rates["round_s"])


def test_sync_kinds_cost_what_the_tree_says():
    work = (Work("w", items=1, item_fixed_s=1e-6),)
    grid = interval_time(Interval(work, WORKERS, "grid"), TREE).sync_s
    counter = interval_time(Interval(work, WORKERS, "counter"), TREE).sync_s
    none = interval_time(Interval(work, WORKERS, "none"), TREE).sync_s
    assert grid == pytest.approx(TREE.barrier_s(60) + TREE.sync.interval_skew_s)
    assert counter == TREE.sync.counter_round_trip_s
    assert none == 0.0


def test_bounds_are_ordered_and_below_the_estimate():
    intervals = _smollm_layer()
    bounds = plan_bounds(intervals, TREE)
    assert bounds.b0 <= bounds.b1 <= bounds.b2 <= bounds.b3 <= _total(intervals)


# --------------------------------------------------------------------------
# SmolLM-135M decode at 100 cached keys: hidden 576, MLP 1536, 9 query heads
# over 3 KV heads of 64. One layer is five barrier intervals.

HIDDEN, MLP = 576, 1536


def _gemm(name, columns, k, *, paired=False, staged=True, ring=20, splits=1, tree=TREE):
    tiles = columns // 64
    tile_bytes = (2 if paired else 1) * 64 * k * 2
    depth = (ring if staged else 8) * 64 * 64 * 2
    return gemm_work(
        name, tree, tiles=tiles, tile_bytes=tile_bytes, splits=splits,
        staged=staged and splits == 1, paged=splits == 1, depth_bytes=depth,
    )


def _attention(keys=100, chunks=1, heads=1, tree=TREE):
    return attention_work(
        "attention", tree, rows=9 // heads, chunks=chunks, keys=keys, heads=heads, head_dim=64,
        row_bytes=2 * keys * 64 * 2,
    )


def _smollm_layer(*, staged=True, ring=20, splits=1, chunks=1, fold=True, tree=TREE):
    norm = generic_work("rms_norm", tree, items=1, rounds=1, every_worker=fold)
    gemm = lambda name, columns, k, **kw: _gemm(  # noqa: E731
        name, columns, k, staged=staged, ring=ring, tree=tree, **kw
    )
    intervals = [
        Interval((_attention(chunks=chunks, tree=tree),), WORKERS),
        Interval((gemm("o_proj", HIDDEN, HIDDEN, splits=splits),), WORKERS),
        Interval((norm, gemm("gate_up", MLP, HIDDEN, paired=True)), WORKERS),
        Interval((gemm("down", HIDDEN, MLP, splits=splits),), WORKERS),
        Interval(
            (
                norm,
                gemm("q", HIDDEN, HIDDEN, splits=splits),
                gemm("k", 192, HIDDEN),
                gemm("v", 192, HIDDEN),
            ),
            WORKERS,
        ),
    ]
    if not fold:
        # The norms as barrier intervals of their own, on one leaf.
        intervals[2:3] = [Interval((norm,), WORKERS), Interval(intervals[2].work[1:], WORKERS)]
        intervals[-1:] = [Interval((norm,), WORKERS), Interval(intervals[-1].work[1:], WORKERS)]
    return intervals


def _kernel(**kwargs):
    lm_head = Interval((_gemm("lm_head", 49152, HIDDEN, **{
        key: value for key, value in kwargs.items() if key in ("staged", "ring", "tree")
    }),), WORKERS, "none")
    return _smollm_layer(**kwargs) * 30 + [lm_head]


def test_estimate_of_the_measured_kernel_is_close():
    # Measured 2026-11-15: 842 µs (artifacts/newarch_2026-11-15/ab_smol135_decisions.json).
    assert _total(_kernel()) == pytest.approx(842e-6, rel=0.10)


# Measured on the 60-SM slice, SmolLM-135M decode, percent against the default
# kernel (same process, 4 interleaved rounds).
MEASURED = {
    "no producer": 4.25,
    "ring of 10 pages": 1.27,
    "split-K": 7.89,
}


@pytest.mark.parametrize(
    "name, alternative",
    [
        ("no producer", {"staged": False}),
        ("ring of 10 pages", {"ring": 10}),
        ("split-K", {"splits": 3, "staged": False}),
    ],
)
def test_measured_decisions_have_the_right_sign_and_size(name, alternative):
    default = _total(_kernel())
    other = _total(_kernel(**alternative))
    predicted = (other / default - 1) * 100
    # Never the slower kernel; a tie keeps the default. (The ring's depth
    # shows in the model only where it caps a leaf's stream rate, so a 1%
    # effect can come out as a tie.)
    assert predicted >= 0, f"{name}: the model prefers the slower kernel"
    assert predicted == pytest.approx(MEASURED[name], abs=3.5)


def test_roofline_cannot_see_any_of_them():
    # Bytes at the device's bandwidth: the three kernels move the same bytes.
    default = _total(_kernel(), form="roofline")
    for alternative in ({"staged": False}, {"ring": 10}, {"splits": 3, "staged": False}):
        assert _total(_kernel(**alternative), form="roofline") == pytest.approx(default)
    # And it puts the kernel at an eighth of its measured time.
    assert default < 842e-6 / 6


def test_short_contexts_do_not_split_attention_and_long_ones_do():
    # 2026-10-23: splitting 65-159 keys across CTAs cost ~330 µs per token on
    # Llama-1B; X2a: the split pays at thousands of keys.
    short = lambda chunks: _total([Interval((_attention(100, chunks),), WORKERS)])  # noqa: E731
    assert short(1) < short(2) < short(3)
    long = lambda chunks: _total([Interval((_attention(4000, chunks),), WORKERS)])  # noqa: E731
    assert long(4) < long(1)


def test_folding_a_small_producer_beats_a_barrier_interval():
    # Interval folding (2026-10): recompute on every CTA instead of a barrier.
    assert _total(_smollm_layer(fold=True)) < _total(_smollm_layer(fold=False))
    # A roofline sees no synchronization, so it sees no difference.
    assert _total(_smollm_layer(fold=True), form="roofline") == pytest.approx(
        _total(_smollm_layer(fold=False), form="roofline"), rel=0.02
    )


def test_split_pays_only_where_the_edge_is_far_from_saturated():
    def gemm_seconds(tree, columns, k, splits):
        work = _gemm("gemm", columns, k, splits=splits, staged=False, tree=tree)
        return interval_time(Interval((work,), tree.workers, "none"), tree).critical_s

    # Llama-1B's down projection (32 tiles of 1 MB) on the 60-SM slice: 32
    # CTAs already pull 80% of HBM, and 64 half tiles need a second wave.
    # Measured slower split (X2.2, 2026-11-07).
    assert gemm_seconds(TREE, 2048, 8192, 2) > gemm_seconds(TREE, 2048, 8192, 1)
    # Nine such tiles on 304 leaves that each pull little of a fast edge:
    # splitting K puts four times the leaves on it.
    slow_leaves = replace(load_spec("mi300x"), bodies=TREE.bodies, sync=TREE.sync)
    assert gemm_seconds(slow_leaves, 576, 8192, 4) < 0.7 * gemm_seconds(slow_leaves, 576, 8192, 1)
