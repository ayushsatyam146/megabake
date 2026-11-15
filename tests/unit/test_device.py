from __future__ import annotations

from dataclasses import replace

import pytest

from megabake.bodies import TargetFacts
from megabake.device import (
    DeviceTree,
    interpolate,
    load_spec,
    saturating_curve,
    spec_names,
    tree_for_target,
    with_measurements,
)
from megabake.machine import DeviceFingerprint, MachineModel


def _target(sms: int = 60, architecture: str = "sm90a") -> TargetFacts:
    return TargetFacts(
        architecture=architecture,
        threads_per_warp=32,
        max_threads_per_cta=1024,
        max_shared_memory_per_cta=232_448,
        max_registers_per_thread=255,
        multiprocessor_count=sms,
        supports_tma=True,
        supports_wgmma=True,
    )


def test_every_spec_file_loads_and_round_trips():
    assert {"h100", "b200", "mi300x", "tpu-v7", "h200-mig-3g71gb"} <= set(spec_names())
    for name in spec_names():
        tree = load_spec(name)
        assert tree.workers >= 1
        assert tree.read_bandwidth() > 0
        assert tree.leaf.managed == "software"
        again = DeviceTree.from_dict(tree.to_dict())
        assert again == tree and hash(again) == hash(tree)


def test_one_schema_covers_three_mechanism_families():
    workers = {name: load_spec(name).workers for name in ("h100", "mi300x", "tpu-v7")}
    assert workers == {"h100": 132, "mi300x": 8 * 38, "tpu-v7": 2}
    families = {load_spec(name).family for name in ("h100", "mi300x", "tpu-v7")}
    assert families == {"hopper", "cdna3", "tpu"}
    # AMD has no bulk copy engine; a TPU needs no barrier.
    assert list(load_spec("mi300x").edge().mechanisms) == ["thread_load"]
    assert load_spec("tpu-v7").barrier_s() == 0.0


def test_shared_edge_saturates_before_all_leaves_are_active():
    tree = load_spec("h200-mig-3g71gb")
    # Measured: 32 of 60 CTAs reach about 80% of the bandwidth.
    assert 0.75 < tree.read_bandwidth(32) / tree.read_bandwidth(60) < 0.85
    assert tree.read_share(1) > 2 * tree.read_share(60)
    assert tree.read_bandwidth(1000) == tree.read_bandwidth(60)


def test_saturating_curve_fits_the_measured_one():
    measured = load_spec("h200-mig-3g71gb").edge().read_bytes_per_s
    curve = saturating_curve(2.54e12, measured[1.0], 60)
    for active in (16, 32, 48, 60):
        assert interpolate(curve, active) == pytest.approx(measured[float(active)], rel=0.05)


def test_staging_depth_counts_only_free_capacity_of_all_instances():
    tree = load_spec("h100")
    leaf_bytes = tree.leaf.capacity_bytes * 132
    assert tree.staging_depth_s() == pytest.approx(leaf_bytes / tree.read_bandwidth())
    assert tree.staging_depth_s(reserved_bytes=tree.leaf.capacity_bytes) == 0.0
    # The Infinity Cache is one instance; the L2 below it has eight.
    amd = load_spec("mi300x")
    assert amd.staging_depth_s("xcd_l2") == pytest.approx(8 * 4e6 / amd.read_bandwidth())


def test_tree_for_target_takes_the_targets_leaf_count_and_family():
    tree = tree_for_target(_target(60))
    assert tree.workers == 60 and tree.family == "hopper"
    assert tree.read_bandwidth() < load_spec("h100").read_bandwidth()
    assert tree_for_target(_target(148, "sm100")).family == "blackwell"


def test_measurements_replace_spec_numbers():
    fingerprint = DeviceFingerprint("test", 60, "9.0", 70 << 30, 30 << 20, 1, 1, 1)
    machine = MachineModel(
        fingerprint=fingerprint,
        hbm_read_bytes_per_s=2.0e12,
        hbm_read_bytes_per_s_by_ctas={1: 8.0e10, 30: 1.6e12, 60: 2.0e12},
        l2_read_bytes_per_s=4.0e12,
        l2_effective_bytes=40 << 20,
        read_bytes_per_s_by_footprint={},
        grid_barrier_s_by_ctas={1: 5e-7, 60: 7e-7},
        counter_round_trip_s=9e-7,
        tensor_peak_flops={"bfloat16": 6.0e14},
        measured_at="now",
        body_rates={"gemm": {"tile_s": 9e-7}, "sync": {"interval_skew_s": 2e-7}, "edge": {"latency_s": 8e-7}},
    )
    tree = with_measurements(tree_for_target(_target(60)), machine)
    assert tree.name == "test"
    assert tree.read_bandwidth() == 2.0e12
    assert tree.read_share(30) == pytest.approx(1.6e12 / 30)
    assert tree.barrier_s(60) == 7e-7
    assert tree.sync.handoff_s == pytest.approx(4.5e-7)
    assert tree.sync.interval_skew_s == 2e-7
    assert tree.flops_per_s("bfloat16") == pytest.approx(6.0e14 / 60)
    assert tree.body("gemm")["tile_s"] == 9e-7
    assert tree.body("gemm")["start_s"] == load_spec("h100").body("gemm")["start_s"]
    assert tree.mechanism().latency_s == 8e-7
    assert tree.level("l2").capacity_bytes == 40 << 20
    assert tree.provenance["edges"] == "measured"
    assert tree != replace(tree, name="other")
