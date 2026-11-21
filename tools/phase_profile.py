"""Attribute MegaBake kernel time to phases with global-timer stamps.

Builds the timed kernel (MEGABAKE_PHASE_TIMER=1), runs warm calls, and reads
the stamp block 0 writes after every grid barrier. Phases between two
barriers run concurrently, so each barrier interval's time is split across
its phases in proportion to their work items; per-category totals are
therefore approximate, while interval totals are exact.

    .venv/bin/python tools/phase_profile.py --model HuggingFaceTB/SmolLM-1.7B --mode prefill
    .venv/bin/python tools/phase_profile.py --mode decode --skip-accuracy-gate
    .venv/bin/python tools/phase_profile.py --tune-model --model HuggingFaceTB/SmolLM-1.7B --out tuning.json

With --cta every CTA also stamps when its work in each phase is done and when
it leaves the phase's barrier, which splits each barrier interval into the
slowest CTA's work, the time other CTAs wait for it, and the barrier's own
release latency.
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import os
import re
import sys
from pathlib import Path
from statistics import median

os.environ["MEGABAKE_PHASE_TIMER"] = "cta" if "--cta" in sys.argv else "1"

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmarks.full_model import PROMPT, _load, decode_module
from megabake import (
    compiler,
    make_backend,
)
from megabake.bodies.catalog import is_table_gemm, table_gemm, tuner_for_device
from megabake.bodies.cuda.sm90.gemm import split_k
from megabake.runtime import (
    CTA_TIMER_STRIDE,
    GRID_TIMER_SLOTS,
    PHASE_TIMER_SLOTS,
)
from tests.fixtures import SMOLLM_MODEL_NAME, SMOLLM_REVISION


def _unwrap_relative(value: int, reference: int) -> int:
    """32-bit nanosecond stamp, unwrapped relative to a reference."""
    return ((value - reference + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


class _DecodeStepRunner:
    """Invokes compiled decode step while advancing device position IDs."""

    def __init__(self, model, token: torch.Tensor, state: list[torch.Tensor], position: int = 0):
        self.model = model
        self.token = token
        self.state = state
        self.position = position

    def __call__(self):
        position_ids = torch.tensor([[self.position]], device="cuda")
        self.position += 1
        return self.model(self.token, position_ids, *self.state)


def _category(phase) -> str:
    name = re.sub(r"_\d+x\d+$", "", phase.body.name)
    if phase.body.tiles.mma_family is not None:
        return f"{name} (tensor-core GEMM)"
    kind = phase.region_id.rsplit(":", 1)[-1]
    if kind.startswith(("mm", "bmm", "addmm")) or kind in name:
        return name
    return f"{name} [{kind}]"


def _stamps(callable_, compiled) -> list[int]:
    state = callable_.megabake_runtime_state
    workers = compiled.plan.launch.grid_size
    count = len(compiled.plan.phases)
    raw = state.ready[workers + 2 : workers + 3 + count].cpu().to(torch.int64)
    return [value & 0xFFFFFFFF for value in raw.tolist()]


def _cta_stamps(callable_, compiled) -> list[list[int]]:
    """Per CTA: its start stamp, then (work done, left barrier) per phase."""
    state = callable_.megabake_runtime_state
    workers = compiled.plan.launch.grid_size
    base = workers + 2 + GRID_TIMER_SLOTS
    width = 1 + 2 * len(compiled.plan.phases)
    raw = state.ready[base : base + workers * CTA_TIMER_STRIDE].view(workers, CTA_TIMER_STRIDE)
    return [
        [value & 0xFFFFFFFF for value in row]
        for row in raw[:, :width].cpu().to(torch.int64).tolist()
    ]


def cta_profile(callable_, compiled, run, calls: int) -> dict:
    """Split every barrier interval into work, straggler wait, and release.

    For each interval and CTA: ``busy`` runs from leaving the previous
    barrier to finishing this interval's work, ``wait`` from there until
    leaving this barrier. The interval's critical work is the slowest CTA's
    busy time; straggler wait is how long the average CTA waits for it; the
    release latency runs from the last CTA's arrival to the first CTA's exit.
    """
    phases = compiled.plan.phases
    programs = {
        selection.region_id: selection.program for selection in compiled.plan.selection.selections
    }
    barriers = [row[1] != 0 for row in compiled.generated.launch.phase_table]
    samples = []
    for _ in range(calls):
        run()
        torch.cuda.synchronize()
        samples.append(_cta_stamps(callable_, compiled))
    keys = ("span", "critical", "mean_busy", "straggler_wait", "release", "busy_ctas")
    intervals = []
    previous, members = None, []
    for index, phase in enumerate(phases):
        members.append(phase)
        if not barriers[index] and index + 1 < len(phases):
            continue
        rows = {key: [] for key in keys}
        busy_lists = []
        for stamps in samples:
            # 32-bit nanosecond stamps, unwrapped relative to CTA 0's start.
            reference = stamps[0][0 if previous is None else 2 + 2 * previous]
            prev_idx = 0 if previous is None else 2 + 2 * previous
            starts = [_unwrap_relative(row[prev_idx], reference) for row in stamps]
            dones = [_unwrap_relative(row[1 + 2 * index], reference) for row in stamps]
            exits = [_unwrap_relative(row[2 + 2 * index], reference) for row in stamps]
            busy = [done - start for done, start in zip(dones, starts, strict=True)]
            critical = max(busy)
            busy_lists.append(busy)
            rows["span"].append(max(exits) - min(starts))
            rows["critical"].append(critical)
            rows["mean_busy"].append(sum(busy) / len(busy))
            rows["straggler_wait"].append(sum(max(dones) - done for done in dones) / len(dones))
            rows["release"].append(max(0, min(exits) - max(dones)))
            rows["busy_ctas"].append(sum(value * 2 >= critical for value in busy))
        summary = {key: median(values) for key, values in rows.items()}
        # Each CTA's busy time in the sample with the median critical work.
        order = sorted(range(len(busy_lists)), key=lambda i: max(busy_lists[i]))
        summary["busy_per_cta"] = busy_lists[order[len(order) // 2]]
        intervals.append((summary, tuple(members)))
        previous, members = index, []
    # Per phase: the slowest CTA's time from the previous phase's exit (or the
    # interval start) to this phase's done stamp.
    phase_rows = []
    for index, phase in enumerate(phases):
        times, fastest, typical = [], [], []
        prev_idx = 0 if index == 0 else 2 * index
        for stamps in samples:
            reference = stamps[0][0]
            per_cta = sorted(
                _unwrap_relative(row[1 + 2 * index], reference)
                - _unwrap_relative(row[prev_idx], reference)
                for row in stamps
            )
            times.append(per_cta[-1])
            fastest.append(per_cta[0])
            typical.append(per_cta[len(per_cta) // 2])
        row = {
            "index": index,
            "us": median(times) / 1e3,
            # A CTA without tiles in the phase still walks it: these are
            # the fastest and the median CTA's time for the phase.
            "fastest_cta_us": median(fastest) / 1e3,
            "median_cta_us": median(typical) / 1e3,
            "phase": f"{_category(phase)} tiles={len(phase.work_items)}",
            "recompute": phase.recompute,
            "barrier": barriers[index],
        }
        program = programs.get(phase.region_id)
        if is_table_gemm(phase.body.provider) and program is not None:
            gemm_impl = table_gemm(phase.body.provider)
            reduction = next(
                domain.size
                for domain in program.iteration_domains
                if domain.axis in program.reduction_axes
            )
            row.update(
                gemm_shape=[*program.store.shape, reduction],
                feature_tile=phase.body.tiles.cta_shape[1],
                split_k=split_k(phase.body)
                if gemm_impl is None
                else gemm_impl.gemm.split_k(phase.body),
            )
        phase_rows.append(row)
    totals = {
        key: sum(row[key] for row, _ in intervals) / 1e3
        for key in ("span", "critical", "mean_busy", "straggler_wait", "release")
    }
    by_category: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {"critical_us": 0.0, "straggler_wait_us": 0.0, "intervals": 0}
    )
    for row, group in intervals:
        # Attribute an interval to its last (barrier-closing) phase's category.
        entry = by_category[_category(group[-1])]
        entry["critical_us"] += row["critical"] / 1e3
        entry["straggler_wait_us"] += row["straggler_wait"] / 1e3
        entry["intervals"] += 1
    return {
        "intervals": len(intervals),
        "interval_span_us": totals["span"],
        "critical_work_us": totals["critical"],
        "mean_cta_busy_us": totals["mean_busy"],
        "straggler_wait_us": totals["straggler_wait"],
        "barrier_release_us": totals["release"],
        "median_release_us": median([row["release"] for row, _ in intervals]) / 1e3,
        "by_closing_category": sorted(
            ({"category": name, **values} for name, values in by_category.items()),
            key=lambda row: -row["critical_us"],
        ),
        "phases": phase_rows,
        "all_intervals": [
            {
                **{
                    key: row[key] / 1e3
                    for key in ("span", "critical", "mean_busy", "straggler_wait", "release")
                },
                "busy_ctas": row["busy_ctas"],
                "busy_per_cta_us": [round(value / 1e3, 3) for value in row["busy_per_cta"]],
                "phase_indices": [phase.index for phase in group],
                "phases": [f"{_category(phase)} tiles={len(phase.work_items)}" for phase in group],
            }
            for row, group in intervals
        ],
        "largest_straggler_intervals": [
            {
                "span_us": row["span"] / 1e3,
                "critical_us": row["critical"] / 1e3,
                "straggler_wait_us": row["straggler_wait"] / 1e3,
                "busy_ctas": row["busy_ctas"],
                "phases": [f"{_category(phase)} tiles={len(phase.work_items)}" for phase in group],
            }
            for row, group in sorted(intervals, key=lambda item: -item[0]["straggler_wait"])[:12]
        ],
    }


def profile(callable_, compiled, run, calls: int) -> dict:
    phases = compiled.plan.phases
    barriers = [row[1] != 0 for row in compiled.generated.launch.phase_table]
    samples = []
    for _ in range(calls):
        run()
        torch.cuda.synchronize()
        samples.append(_stamps(callable_, compiled))
    intervals, members, previous = [], [], len(phases)
    for index, phase in enumerate(phases):
        members.append(phase)
        if barriers[index]:
            durations = [(stamp[index] - stamp[previous]) & 0xFFFFFFFF for stamp in samples]
            intervals.append((median(durations), tuple(members)))
            members, previous = [], index
    by_category: dict[str, float] = collections.defaultdict(float)
    counts: dict[str, int] = collections.Counter()
    for duration, group in intervals:
        weights = [max(1, len(phase.work_items)) for phase in group]
        for phase, weight in zip(group, weights, strict=True):
            by_category[_category(phase)] += duration * weight / sum(weights)
            counts[_category(phase)] += 1
    total = sum(duration for duration, _ in intervals)
    gemm_tiles = collections.Counter(
        len(phase.work_items) for phase in phases if phase.body.tiles.mma_family is not None
    )
    return {
        "workers": compiled.plan.launch.grid_size,
        "phases": len(phases),
        "barrier_intervals": len(intervals),
        "timed_total_us": total / 1e3,
        "trailing_phases_untimed": len(members),
        "gemm_phases_by_tile_count": dict(sorted(gemm_tiles.items())),
        "categories": sorted(
            (
                {"category": name, "phases": counts[name], "us": value / 1e3}
                for name, value in by_category.items()
            ),
            key=lambda row: -row["us"],
        ),
        "slowest_intervals": [
            {
                "us": duration / 1e3,
                "phases": [f"{_category(phase)} tiles={len(phase.work_items)}" for phase in group],
            }
            for duration, group in sorted(intervals, key=lambda item: -item[0])[:12]
        ],
    }


def _interval_kind(plan, indices: list[int], categories: list[str]) -> str:
    """A coarse op class for an interval, from its phases' categories. The
    interval that holds the plan's last phase and a tensor-core GEMM is the
    final projection (the LM head of a decoder)."""
    text = " ".join(categories)
    if "attention" in text:
        return "attention"
    if len(plan.phases) - 1 in indices and "tensor-core GEMM" in text:
        return "final_gemm"
    if "paired" in text:
        return "norm+gate_up"
    if "tensor-core GEMM" in text:
        return "gemm"
    return "other"


def sol_accounting(compiled, cta: dict, machine, kv_valid_length: int) -> dict:
    """Speed-of-light accounting per barrier interval (X0.1).

    For each interval: bytes the plan must move, the time those bytes take at
    the all-SM HBM bandwidth (``t_sol``) and at the bandwidth its busy CTAs
    can reach (``t_cov``), and the measured span split into

    - coverage: ``t_cov - t_sol``, lost to CTAs with no work in the interval;
    - body: ``critical - t_cov``, the slowest CTA's work above what its share
      of bandwidth allows (body inefficiency, pipeline fill, imbalance);
    - sync: ``span - critical``, barrier release and start skew.
    """
    from megabake.accounting import plan_traffic

    plan = compiled.plan
    traffic = plan_traffic(plan, kv_valid_length)
    rows = []
    for interval in cta["all_intervals"]:
        members = [traffic[index] for index in interval["phase_indices"]]
        moved = sum(item.bytes_moved for item in members)
        weights = sum(item.weight_bytes for item in members)
        flops = sum(item.flops for item in members)
        span = interval["span"] * 1e-6
        critical = interval["critical"] * 1e-6
        t_sol = machine.speed_of_light_s(moved, flops)
        t_cov = max(moved / machine.read_bandwidth(interval["busy_ctas"]), t_sol)
        rows.append({
            "kind": _interval_kind(plan, interval["phase_indices"], interval["phases"]),
            "span_us": interval["span"],
            "bytes": moved,
            "weight_bytes": weights,
            "busy_ctas": interval["busy_ctas"],
            "achieved_gb_s": moved / span / 1e9 if span else 0.0,
            "t_sol_us": t_sol * 1e6,
            "sol": t_sol / span if span else 0.0,
            "coverage_loss_us": (t_cov - t_sol) * 1e6,
            "body_loss_us": max(0.0, critical - t_cov) * 1e6,
            "sync_loss_us": (span - critical) * 1e6,
            "phases": interval["phases"],
        })
    by_kind: dict[str, dict] = {}
    for row in rows:
        entry = by_kind.setdefault(row["kind"], collections.Counter())
        entry["intervals"] += 1
        for key in ("span_us", "t_sol_us", "coverage_loss_us", "body_loss_us", "sync_loss_us", "bytes"):
            entry[key] += row[key]
    total = lambda key: sum(row[key] for row in rows)  # noqa: E731
    span = total("span_us")
    return {
        "kv_valid_length": kv_valid_length,
        "hbm_read_gb_s": machine.hbm_read_bytes_per_s / 1e9,
        "bytes": total("bytes"),
        "weight_bytes": total("weight_bytes"),
        "span_us": span,
        "t_sol_us": total("t_sol_us"),
        "sol": total("t_sol_us") / span if span else 0.0,
        "coverage_loss_us": total("coverage_loss_us"),
        "body_loss_us": total("body_loss_us"),
        "sync_loss_us": total("sync_loss_us"),
        "by_kind": {
            kind: {**dict(values), "sol": values["t_sol_us"] / values["span_us"] if values["span_us"] else 0.0}
            for kind, values in sorted(by_kind.items(), key=lambda item: -item[1]["span_us"])
        },
        "intervals": rows,
    }


def _kernel_device_us(run, calls: int) -> float:
    """Median device duration of the MegaBake kernel, from the CUDA profiler."""
    from torch.profiler import ProfilerActivity, profile as torch_profile

    with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            run()
        torch.cuda.synchronize()
    durations = sorted(
        event.device_time for event in prof.events()
        if "megabake" in event.name.lower() or "kernel_cutlass" in event.name.lower()
    )
    if not durations:
        return float("nan")
    longest = [value for value in durations if value >= 0.5 * durations[-1]]
    return longest[len(longest) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=SMOLLM_MODEL_NAME)
    parser.add_argument("--revision")
    parser.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--prompt-tokens", type=int, default=len(PROMPT))
    parser.add_argument("--attention", choices=("eager", "megabake"), default="eager")
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument(
        "--context", type=int, default=0, help="decode: tokens already cached before the first call"
    )
    parser.add_argument("--calls", type=int, default=9)
    parser.add_argument("--tune-model", action="store_true", help="sweep GEMM choices for decode")
    parser.add_argument("--profile-timeout", type=int, default=480)
    parser.add_argument("--tiles", nargs="+", choices=("auto", "64", "128"), default=("64", "128"))
    parser.add_argument(
        "--split-limits", nargs="+", choices=("1", "2", "4", "8"), default=("1", "2", "4", "8")
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--cta", action="store_true", help="also split intervals into work and wait per CTA"
    )
    parser.add_argument(
        "--sol", action="store_true",
        help="with --cta: speed-of-light accounting per interval from the cached MachineModel",
    )
    parser.add_argument(
        "--skip-accuracy-gate",
        action="store_true",
        help="profile a plan even if it fails the accuracy check (diagnostic only)",
    )
    args = parser.parse_args()
    if args.tune_model:
        tuner_for_device().tune_model(args, Path(__file__).resolve())
        return
    if args.prompt_tokens < 1:
        parser.error("--prompt-tokens must be positive")
    if args.skip_accuracy_gate:
        compiler._TABLE_PLAN_RELATIVE_L2_LIMIT = float("inf")
        compiler._PLAN_RELATIVE_L2_LIMIT = float("inf")
    revision = args.revision or (SMOLLM_REVISION if args.model == SMOLLM_MODEL_NAME else None)
    model = _load(args.model, revision, args.attention)
    backend = make_backend(mode=args.mode, compile_kernel=True, cache=False, scheduler="phase")
    with torch.no_grad():
        if args.mode == "prefill":
            prompt = (PROMPT * -(-args.prompt_tokens // len(PROMPT)))[: args.prompt_tokens]
            ids = torch.tensor([prompt], device="cuda")
            mask = torch.ones_like(ids)
            compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
            run = functools.partial(compiled_model, ids, attention_mask=mask)
        else:
            stateful, shapes = decode_module(model, args.capacity)
            state = [torch.zeros(s, device="cuda", dtype=d) for layer in shapes for s, d in layer]
            token = torch.tensor([[504]], device="cuda")
            compiled_model = torch.compile(stateful, backend=backend, fullgraph=True, dynamic=False)
            # As benchmarks/full_model.py --decode-contexts: setting every int32
            # valid length (and the position) to L puts the call at context L.
            for tensor in state:
                if tensor.dtype == torch.int32 and tensor.numel() == 1:
                    tensor.fill_(args.context)

            if args.context + 2 * args.calls + 4 > args.capacity:
                raise SystemExit("--capacity must exceed --context + 2 * --calls + 4")

            run = _DecodeStepRunner(compiled_model, token, state, position=args.context)

        run()
        callable_ = backend.megabake_last_callable
        compiled = backend.megabake_last_compiled
        if callable_ is None or compiled is None:
            raise SystemExit("MegaBake did not compile a kernel for this model")
        if compiled.generated.launch.launch_policy != "megabake_cooperative_table":
            raise SystemExit("phase timing needs a table launch")
        if len(compiled.plan.phases) >= min(PHASE_TIMER_SLOTS, GRID_TIMER_SLOTS):
            raise SystemExit("plan has more phases than timer slots")
        for _ in range(3):
            run()
        report = {
            "model": args.model,
            "mode": args.mode,
            "prompt_tokens": args.prompt_tokens if args.mode == "prefill" else None,
            **profile(callable_, compiled, run, args.calls),
        }
        resources = getattr(compiled, "resources", None)
        if resources is not None:
            report["resources"] = {
                "registers": resources.registers_per_thread,
                "local_bytes": resources.local_memory_bytes_per_thread,
                "code_bytes": resources.compiled_code_bytes,
            }
        if args.cta:
            report["cta"] = cta_profile(callable_, compiled, run, args.calls)
            if args.sol:
                from megabake.machine import machine_model

                # The profiled calls attend over context .. context + 3 + 2·calls keys.
                valid = args.context + 4 + args.calls if args.mode == "decode" else None
                report["sol"] = sol_accounting(compiled, report["cta"], machine_model(), valid)
                kernel_us = _kernel_device_us(run, args.calls)
                report["sol"]["kernel_device_us"] = kernel_us
                report["sol"]["interval_sum_vs_kernel"] = report["sol"]["span_us"] / kernel_us
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
