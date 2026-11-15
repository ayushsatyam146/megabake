"""Compare the cost model with a measured kernel, interval by interval.

Builds the kernel with the phase timer (one stamp per grid barrier, about 1%
overhead), runs it, and prints for the plan the compiler chose:

- the measured span of every barrier interval next to the model's estimate,
  by interval kind, with the rank correlation and the error of the totals;
- the ladder of lower bounds (B0..B3) and how far the kernel is from each;
- the planner's mapping: staging, shape classes, device.

    .venv/bin/python tools/plan_report.py --model HuggingFaceTB/SmolLM-135M --context 100
    .venv/bin/python tools/plan_report.py --fit        # fit body rates, save them

``--fit`` fits the body rates the model needs (GEMM start and tile costs,
generic round cost, interval skew) to this profile and stores them in the
device's calibration cache, where ``device_tree`` finds them. Fit on one
model; check the others without ``--fit``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from tools import phase_profile  # noqa: E402  (turns the phase timer on at import)
from megabake import make_backend  # noqa: E402
from megabake.costmodel import interval_time, plan_bounds, plan_intervals  # noqa: E402
from megabake.device import DeviceTree, device_tree, with_copy_latency  # noqa: E402
from megabake.machine import cached_machine_model, save_machine_model  # noqa: E402
from benchmarks.full_model import PROMPT, _load, decode_module  # noqa: E402
from tests.fixtures import SMOLLM_MODEL_NAME, SMOLLM_REVISION  # noqa: E402

# The rates --fit adjusts: (body family, rate). "sync" is the tree's own.
FITTED = (
    ("gemm", "start_s"),
    ("gemm", "staged_start_s"),
    ("gemm", "tile_s"),
    ("generic", "start_s"),
    ("generic", "round_s"),
    ("sync", "interval_skew_s"),
)
# Fitted only on request (--fit-only edge.latency_s), from a kernel with a
# shallow ring (MEGABAKE_RING_PAGES=10): the bulk copy's latency, which
# limits how fast a body streams with the bytes it keeps in flight.
ON_REQUEST = (("edge", "latency_s"),)


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(values):
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        for rank, index in enumerate(order):
            result[index] = rank
        return result

    ra, rb = ranks(a), ranks(b)
    mean = (len(a) - 1) / 2
    cov = sum((x - mean) * (y - mean) for x, y in zip(ra, rb))
    var = sum((x - mean) ** 2 for x in ra)
    return cov / var if var else 0.0


def measured_intervals(callable_, compiled, run, calls: int) -> dict[int, float]:
    """{index of the phase that closes the interval: median span in seconds}."""
    phases = compiled.plan.phases
    barriers = [row[1] != 0 for row in compiled.generated.launch.phase_table]
    samples = []
    for _ in range(calls):
        run()
        torch.cuda.synchronize()
        samples.append(phase_profile._stamps(callable_, compiled))
    spans, previous = {}, len(phases)
    for index in range(len(phases)):
        if barriers[index]:
            spans[index] = statistics.median(
                (stamp[index] - stamp[previous]) & 0xFFFFFFFF for stamp in samples
            ) * 1e-9
            previous = index
    return spans


def with_rates(tree: DeviceTree, rates: dict[tuple[str, str], float]) -> DeviceTree:
    bodies = {family: dict(values) for family, values in tree.bodies.items()}
    sync = tree.sync
    edges = tree.edges
    for (family, name), value in rates.items():
        if family == "sync":
            sync = replace(sync, **{name: value})
        elif family == "edge":
            edges = with_copy_latency(tree, value).edges
        else:
            bodies.setdefault(family, {})[name] = value
    return replace(tree, bodies=bodies, sync=sync, edges=edges)


def rows_for(plan, tree: DeviceTree, spans: dict[int, float], kv_valid_length):
    """(kind, measured s, predicted s, planned interval) per timed interval."""
    rows = []
    for planned in plan_intervals(plan, tree, kv_valid_length):
        closing = planned.phase_indices[-1]
        if closing not in spans:
            continue
        categories = [phase_profile._category(plan.phases[index]) for index in planned.phase_indices]
        kind = phase_profile._interval_kind(plan, list(planned.phase_indices), categories)
        rows.append((kind, spans[closing], interval_time(planned.interval, tree).total_s, planned))
    return rows


def error(rows) -> float:
    """Summed absolute error over the summed measured time."""
    return sum(abs(predicted - measured) for _, measured, predicted, _ in rows) / sum(
        measured for _, measured, _, _ in rows
    )


def fit(
    plan, tree: DeviceTree, spans, kv_valid_length, sweeps: int = 3, only=None
) -> dict[tuple[str, str], float]:
    """Coordinate descent on the fitted rates, each by golden-section search.
    A kernel exercises one of the two GEMM start costs: the staged one with a
    producer, the plain one without."""
    staged = plan.mapping is not None and plan.mapping.staging.producer
    names = [
        key for key in FITTED + ON_REQUEST
        if key != ("gemm", "start_s" if staged else "staged_start_s")
        and (".".join(key) in only if only else key in FITTED)
    ]
    rates = {
        key: (
            tree.sync.interval_skew_s if key[0] == "sync"
            else tree.mechanism().latency_s if key[0] == "edge"
            else tree.body(key[0]).get(key[1], 1e-6)
        )
        for key in names
    }

    def objective(candidate) -> float:
        return error(rows_for(plan, with_rates(tree, candidate), spans, kv_valid_length))

    golden = 0.618
    for _ in range(sweeps):
        for key in names:
            low, high = 0.0, max(4e-6, 4 * rates[key])
            for _ in range(18):
                a, b = high - golden * (high - low), low + golden * (high - low)
                if objective({**rates, key: a}) < objective({**rates, key: b}):
                    high = b
                else:
                    low = a
            rates[key] = (low + high) / 2
    return rates


def report(plan, tree: DeviceTree, spans, kv_valid_length) -> dict:
    rows = rows_for(plan, tree, spans, kv_valid_length)
    by_kind: dict[str, dict[str, float]] = {}
    for kind, measured, predicted, _ in rows:
        entry = by_kind.setdefault(kind, {"intervals": 0, "measured_us": 0.0, "predicted_us": 0.0})
        entry["intervals"] += 1
        entry["measured_us"] += measured * 1e6
        entry["predicted_us"] += predicted * 1e6
    for entry in by_kind.values():
        entry["predicted_over_measured"] = entry["predicted_us"] / entry["measured_us"]
    # Intervals of one shape repeat once per layer; their medians rank the
    # shapes without the noise among repeats.
    shapes: dict[tuple, list[tuple[float, float]]] = {}
    for kind, measured, predicted, planned in rows:
        signature = (kind, tuple((work.items, round(work.item_bytes)) for work in planned.interval.work))
        shapes.setdefault(signature, []).append((measured, predicted))
    shape_rows = [
        (
            signature[0],
            len(values),
            statistics.median(m for m, _ in values) * 1e6,
            statistics.median(p for _, p in values) * 1e6,
        )
        for signature, values in shapes.items()
    ]
    measured_total = sum(row[1] for row in rows)
    predicted_total = sum(row[2] for row in rows)
    bounds = plan_bounds([row[3].interval for row in rows], tree)
    return {
        "intervals": len(rows),
        "measured_us": measured_total * 1e6,
        "predicted_us": predicted_total * 1e6,
        "predicted_over_measured": predicted_total / measured_total,
        "absolute_error": error(rows),
        "interval_mape": statistics.mean(abs(p - m) / m for _, m, p, _ in rows),
        "spearman": spearman([row[1] for row in rows], [row[2] for row in rows]),
        "shape_spearman": spearman([row[2] for row in shape_rows], [row[3] for row in shape_rows]),
        "shapes": [
            {"kind": kind, "count": count, "measured_us": round(m, 2), "predicted_us": round(p, 2)}
            for kind, count, m, p in sorted(shape_rows, key=lambda row: -row[2])
        ],
        "by_kind": dict(sorted(by_kind.items(), key=lambda item: -item[1]["measured_us"])),
        "bounds_us": {name: getattr(bounds, name) * 1e6 for name in ("b0", "b1", "b2", "b3")},
        "measured_over_bound": {
            name: measured_total / getattr(bounds, name) for name in ("b0", "b1", "b2", "b3")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=SMOLLM_MODEL_NAME)
    parser.add_argument("--revision")
    parser.add_argument("--mode", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--capacity", type=int, default=0, help="decode KV capacity (default: context + 156)")
    parser.add_argument("--context", type=int, default=100, help="decode: tokens already cached")
    parser.add_argument("--calls", type=int, default=9)
    parser.add_argument("--fit", action="store_true", help="fit body rates to this profile and save them")
    parser.add_argument(
        "--fit-only", nargs="*", metavar="FAMILY.RATE",
        help="with --fit: the rates to fit (default: all the kernel exercises)",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    revision = args.revision or (SMOLLM_REVISION if args.model == SMOLLM_MODEL_NAME else None)
    model = _load(args.model, revision, "megabake")
    backend = make_backend(mode=args.mode, compile_kernel=True, cache=False, scheduler="phase")
    capacity = args.capacity or args.context + 156
    with torch.no_grad():
        if args.mode == "prefill":
            ids = torch.tensor([PROMPT], device="cuda")
            mask = torch.ones_like(ids)
            compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
            run = lambda: compiled_model(ids, attention_mask=mask)  # noqa: E731
            valid = None
        else:
            stateful, shapes = decode_module(model, capacity)
            state = [torch.zeros(s, device="cuda", dtype=d) for layer in shapes for s, d in layer]
            token = torch.tensor([[504]], device="cuda")
            position = torch.tensor([[args.context]], device="cuda")
            compiled_model = torch.compile(stateful, backend=backend, fullgraph=True, dynamic=False)
            valid = args.context

            def run():
                # The same position every call: the length stays at --context.
                for tensor in state:
                    if tensor.dtype == torch.int32 and tensor.numel() == 1:
                        tensor.fill_(args.context)
                return compiled_model(token, position, *state)

        run()
        callable_ = backend.megabake_last_callable
        compiled = backend.megabake_last_compiled
        if callable_ is None or compiled is None:
            raise SystemExit("MegaBake did not compile a kernel for this model")
        if compiled.generated.launch.launch_policy != "megabake_cooperative_table":
            raise SystemExit("phase timing needs a table launch")
        for _ in range(3):
            run()
        spans = measured_intervals(callable_, compiled, run, args.calls)
    plan = compiled.plan
    tree = plan.mapping.tree if plan.mapping is not None else device_tree(plan.target)
    result = {"model": args.model, "mode": args.mode, "context": args.context,
              "device": tree.name, "provenance": dict(tree.provenance)}
    if args.fit:
        rates = fit(plan, tree, spans, valid, only=args.fit_only)
        tree = with_rates(tree, rates)
        result["fitted"] = {f"{family}.{name}": value for (family, name), value in rates.items()}
        machine = cached_machine_model()
        if machine is None:
            raise SystemExit("run tools/calibrate.py first: the fit is stored with the device's calibration")
        merged = {family: dict(values) for family, values in machine.body_rates.items()}
        for (family, name), value in rates.items():
            merged.setdefault(family, {})[name] = value
        result["saved_to"] = str(save_machine_model(replace(machine, body_rates=merged)))
    result.update(report(plan, tree, spans, valid))
    if plan.mapping is not None:
        result["mapping"] = plan.mapping.report()
    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
