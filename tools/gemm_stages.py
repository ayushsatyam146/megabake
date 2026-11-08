"""Per-item GEMM stage profile (experiment X2.2 / GEMM fixed cost, 2026-11-07).

With ``MEGABAKE_GEMM_STAMPS=1`` every table GEMM variant writes globaltimer
stamps per (CTA, work item) (see ``bodies/cuda/sm90/cutlass_gemm.py``,
``_stamp``). The last phase of each variant leaves its stamps, so for a decoder
this is the last layer's GEMM of each shape. The phase timer (CTA mode) runs
too, so each CTA's first item is also placed against the moment the CTA left
the barrier that opens the GEMM's interval.

    .venv/bin/python tools/gemm_stages.py --model unsloth/Llama-3.2-1B-Instruct

Per variant, ``entry_breakdown_us`` gives medians over CTAs with work of:
pre_body (interval start to GEMM body entry: earlier phases of the interval,
dispatch), init (mbarrier init and the CTA sync), producer_setup (descriptor
acquire and proxy fence), first_issue (first producer_acquire and TMA issue),
first_ready (TMA latency of the first stage), and after the last item:
exit (store done to leaving the body), post_body (leaving the body to the
phase's done stamp), barrier (done stamp to leaving the closing barrier).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ["MEGABAKE_GEMM_STAMPS"] = "1"
os.environ["MEGABAKE_PHASE_TIMER"] = "cta"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

_argv, sys.argv = sys.argv, [*sys.argv, "--cta"]
import tools.phase_profile as pp  # noqa: E402

sys.argv = _argv
from megabake import make_backend  # noqa: E402
from megabake.bodies.cuda.sm90 import cutlass_gemm  # noqa: E402
from benchmarks.full_model import _load, decode_module  # noqa: E402

FIELDS = ("p_start", "p_end", "c_start", "c_ready", "c_main", "c_reduce", "c_store")


def _wrap(value: int, reference: int) -> float:
    return ((((value - reference) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)) / 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--context", type=int, default=100)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ctas", type=int, default=4, help="CTAs to print per variant")
    args = parser.parse_args()
    model = _load(args.model, None, "megabake")
    backend = make_backend(mode="decode", compile_kernel=True, cache=False, scheduler="phase")
    with torch.no_grad():
        stateful, shapes = decode_module(model, args.capacity)
        state = [torch.zeros(s, device="cuda", dtype=d) for layer in shapes for s, d in layer]
        token = torch.tensor([[504]], device="cuda")
        compiled_model = torch.compile(stateful, backend=backend, fullgraph=True, dynamic=False)
        for tensor in state:
            if tensor.dtype == torch.int32 and tensor.numel() == 1:
                tensor.fill_(args.context)
        position = [args.context]

        def run():
            position_ids = torch.tensor([[position[0]]], device="cuda")
            position[0] += 1
            return compiled_model(token, position_ids, *state)

        for _ in range(4):
            run()
        stamps = cutlass_gemm.gemm_stamps()
        stamps.zero_()
        run()
        torch.cuda.synchronize()
        raw = stamps.cpu().to(torch.int64).tolist()  # [tag][cta][item][field]
        callable_ = backend.megabake_last_callable
    compiled = backend.megabake_last_compiled
    # Whole timer rows: past the (done, left) pairs, the codegen stamps each
    # phase's loop top after its table row is loaded (slot 1 + 2 * phases + phase).
    from megabake.runtime import CTA_TIMER_STRIDE, GRID_TIMER_SLOTS

    workers = compiled.plan.launch.grid_size
    base = workers + 2 + GRID_TIMER_SLOTS
    ready = callable_.megabake_runtime_state.ready
    cta_rows = [
        [value & 0xFFFFFFFF for value in row]
        for row in ready[base : base + workers * CTA_TIMER_STRIDE]
        .view(workers, CTA_TIMER_STRIDE)[:, : 1 + 3 * len(compiled.plan.phases)]
        .cpu()
        .to(torch.int64)
        .tolist()
    ]
    launch = compiled.generated.launch
    table = launch.phase_table
    barriers = [row[1] != 0 for row in table]
    # The last phase of each GEMM variant, and the barrier phase before it.
    last_phase = {}
    for index, phase in enumerate(compiled.plan.phases):
        if phase.body.provider == "g0_range":
            variant = launch.gemm_table[table[index][6]][0]
            last_phase[variant + 1] = index
    report = {"model": args.model, "context": args.context, "variants": {}}
    for tag, index in sorted(last_phase.items()):
        phase = compiled.plan.phases[index]
        opening = max((q for q in range(index) if barriers[q]), default=None)
        closing = next(q for q in range(index, len(table)) if barriers[q] or q == len(table) - 1)
        per_cta, breakdown = {}, {key: [] for key in (
            "loop_to_row_loaded", "row_loaded_to_body", "pre_body", "init", "producer_setup", "first_issue", "first_ready",
            "mainloop_all", "exit", "post_body", "barrier", "interval")}
        mains, boundaries, stores = [], [], []
        for cta, items in enumerate(raw[tag % cutlass_gemm.GEMM_STAMP_VARIANTS]):
            if cta >= len(cta_rows) or items[0][7] != tag:
                continue
            row = cta_rows[cta]
            entry = row[0] if opening is None else row[2 + 2 * opening]
            done, left = row[1 + 2 * index], row[2 + 2 * closing]
            valid = [item for item in items if item[2]]
            if not valid:
                continue
            rel = lambda v: round(_wrap(v, entry), 3)  # noqa: E731
            first, last = valid[0], valid[-1]
            loaded = row[1 + 2 * len(compiled.plan.phases) + index]
            breakdown["loop_to_row_loaded"].append(rel(loaded) if loaded else None)
            breakdown["row_loaded_to_body"].append(rel(first[8]) - rel(loaded) if loaded else None)
            breakdown["pre_body"].append(rel(first[8]))
            breakdown["init"].append(rel(first[9]) - rel(first[8]))
            breakdown["producer_setup"].append(rel(first[0]) - rel(first[9]))
            breakdown["first_issue"].append(rel(first[10]) - rel(first[0]))
            breakdown["first_ready"].append(rel(first[3]) - rel(first[10]))
            breakdown["mainloop_all"].append(rel(last[6]) - rel(first[3]))
            breakdown["exit"].append(rel(first[11]) - rel(last[6]))
            breakdown["post_body"].append(rel(done) - rel(first[11]))
            breakdown["barrier"].append(rel(left) - rel(done))
            breakdown["interval"].append(rel(left))
            for k, item in enumerate(valid):
                mains.append(_wrap(item[4], item[3]))
                stores.append(_wrap(item[6], item[4]))
                if k:
                    boundaries.append(_wrap(item[3], valid[k - 1][4]))
            per_cta[cta] = [[rel(v) if v else None for v in item[:7]] for item in valid]
        med = lambda xs: round(statistics.median([x for x in xs if x is not None]), 3) if any(x is not None for x in xs) else None  # noqa: E731
        report["variants"][f"{phase.body.name} tiles={len(phase.work_items)} phase={index}"] = {
            "phase": index,
            "ctas_with_work": len(per_cta),
            "items_per_cta": sorted({len(v) for v in per_cta.values()}),
            "entry_breakdown_us": {key: med(values) for key, values in breakdown.items()},
            "per_item_us": {
                "mainloop": med(mains), "epilogue_store": med(stores), "boundary": med(boundaries),
            },
            "interval_phases": [
                compiled.plan.phases[q].body.name
                for q in range((opening + 1) if opening is not None else 0, closing + 1)
            ],
            "per_cta": {str(c): v for c, v in sorted(per_cta.items())[: args.ctas]},
        }
    print(json.dumps(report, indent=1))
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
