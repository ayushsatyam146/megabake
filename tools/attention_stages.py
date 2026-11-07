"""Decode-attention stage profile (experiment X2a, 2026-11-07).

Runs a decode step like ``tools/phase_profile.py --cta --sol`` and, with
``MEGABAKE_ATTENTION_STAMPS=1``, also reads the per-stage stamps that the
split-context decode body writes after each CTA-wide sync (see
``bodies/cuda/sm90/attention.py``, ``stamp``). Stamps survive from the last
attention layer of the last call.

    MEGABAKE_ATTENTION_STAMPS=1 .venv/bin/python tools/attention_stages.py \
        --model unsloth/Llama-3.2-1B-Instruct --context 100 --capacity 256

Stage k is the time from stamp k-1 (0 = item start) to stamp k:
1 stage query/new key (RoPE prologue), 2 scores + max, 3 chunk max,
4 exponentials + sum, 5 statistics exchange, 6 BF16 probabilities,
7 value sum, 8 value merge, 9 KV append.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ["MEGABAKE_PHASE_TIMER"] = "cta"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

# phase_profile picks its timer mode from sys.argv at import.
_argv, sys.argv = sys.argv, [*sys.argv, "--cta"]
import tools.phase_profile as pp  # noqa: E402
sys.argv = _argv
from megabake import make_backend  # noqa: E402
from megabake.bodies.cuda.sm90 import attention as attention_module  # noqa: E402
from benchmarks.full_model import _load, decode_module  # noqa: E402

STAGES = ("stage_qk", "scores_max", "chunk_max", "exp_sum", "exchange", "probabilities",
          "value_sum", "value_merge", "kv_append")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--context", type=int, default=100)
    parser.add_argument("--calls", type=int, default=9)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.context + 2 * args.calls + 4 > args.capacity:
        raise SystemExit("--capacity must exceed --context + 2 * --calls + 4")
    model = _load(args.model, None, "eager")
    backend = make_backend(mode="decode", compile_kernel=True, cache=False, scheduler="phase",
                           **({"dump_dir": os.environ["X_DUMP_DIR"]} if os.environ.get("X_DUMP_DIR") else {}))
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

        run()
        callable_ = backend.megabake_last_callable
        compiled = backend.megabake_last_compiled
        for _ in range(3):
            run()
        cta = pp.cta_profile(callable_, compiled, run, args.calls)
        from megabake.machine import machine_model

        valid = args.context + 4 + args.calls
        sol = pp.sol_accounting(compiled, cta, machine_model(), valid)
        kernel_us = pp._kernel_device_us(run, args.calls)
        attention = [
            (c, a) for c, a in zip(cta["all_intervals"], sol["intervals"], strict=True)
            if a["kind"] == "attention"
        ]
        med = lambda f: statistics.median(f(c) for c, _ in attention)  # noqa: E731
        report = {
            "model": args.model,
            "context": args.context,
            "kv_valid_length": valid,
            "kernel_us": kernel_us,
            "env": {k: v for k, v in os.environ.items() if k.startswith("MEGABAKE_ATTENTION_")},
            "attention_layers": len(attention),
            "attention_span_us": med(lambda c: c["span"]),
            "attention_critical_us": med(lambda c: c["critical"]),
            "attention_busy_ctas": med(lambda c: c["busy_ctas"]),
            "attention_sum_span_us": sum(c["span"] for c, _ in attention),
            "kernel_sol": sol["sol"],
        }
        if attention_module.DECODE_STAMPS:
            callable_.megabake_runtime_state._phase_scratch[0][0].zero_()
            run()
            torch.cuda.synchronize()
            state_ = callable_.megabake_runtime_state
            floats = state_._phase_scratch[0][0]
            table = compiled.generated.launch.phase_table
            attention_phases = [
                index for index, phase in enumerate(compiled.plan.phases)
                if "attention" in phase.body.name and table[index][4] is not None
            ]
            last = attention_phases[-1]
            base = table[last][4]
            stamp_base = attention_module._LAST_STAMP_BASE
            workers = compiled.plan.launch.grid_size
            stride = attention_module.DECODE_STAMP_STRIDE
            raw = floats[base + stamp_base: base + stamp_base + workers * stride].view(workers, stride).cpu()
            rows = [row[1:10].tolist() for row in raw if row[9] > 0]
            # Item start (absolute, low 32 bits) against this CTA's exit from
            # the previous phase's barrier, from the CTA timer row.
            cta_rows = pp._cta_stamps(callable_, compiled)
            pre = []
            for block, row in enumerate(raw.tolist()):
                if row[9] <= 0:
                    continue
                t0 = int(row[10]) | (int(row[11]) << 16)
                entered = cta_rows[block][2 + 2 * (last - 1)]
                pre.append((((t0 - entered) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31))
            report["per_cta"] = [
                {"block": block, "pre_us": pre_value / 1e3, "stamps_us": [value / 1e3 for value in row[1:10]]}
                for (block, row), pre_value in zip(
                    [(b, r) for b, r in enumerate(raw.tolist()) if r[9] > 0], pre, strict=True)
            ]
            report["pre_item_us"] = {"median": statistics.median(pre) / 1e3, "max": max(pre) / 1e3}
            stages = {}
            for k, name in enumerate(STAGES):
                deltas = [row[k] - (row[k - 1] if k else 0.0) for row in rows]
                stages[name] = {"median_us": statistics.median(deltas) / 1e3, "max_us": max(deltas) / 1e3}
            report["stage_ctas"] = len(rows)
            report["item_total_us"] = {"median": statistics.median(r[8] for r in rows) / 1e3,
                                       "max": max(r[8] for r in rows) / 1e3}
            report["stages"] = stages
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
