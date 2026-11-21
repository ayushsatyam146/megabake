"""Prefill at arbitrary prompt lengths through MegaBake's size buckets.

Each length runs through ``megabake.bucketed``: the first length of a bucket
compiles it (and verifies padding independence), later lengths reuse it.
Inductor with CUDA graphs is compiled for each exact length as the baseline.
Reports per length the bucket, the bucket's compile time when this length
triggered it, the median complete call, and the logits error against eager at
the real positions.

    .venv/bin/python benchmarks/bucketed_prefill.py --lengths 5 20 50 100 300
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.full_model import _load
from megabake import bucketed, make_backend
from tests.fixtures import SMOLLM_MODEL_NAME, SMOLLM_REVISION


def _median_call(function, samples: int) -> float:
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        torch.cuda.synchronize()
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1e6)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=SMOLLM_MODEL_NAME)
    parser.add_argument("--lengths", type=int, nargs="+", default=[5, 20, 50, 100, 300])
    parser.add_argument("--buckets", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--no-inductor", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("artifacts/bucketed"))
    args = parser.parse_args()
    revision = SMOLLM_REVISION if args.model == SMOLLM_MODEL_NAME else None
    model = _load(args.model, revision)
    prefill = bucketed(
        model,
        backend=make_backend(mode="prefill", compile_kernel=True, cache=False, scheduler="phase"),
        sequence_args={"input_ids": 1, "attention_mask": 1},
        pad_values={"input_ids": 0, "attention_mask": 0},
        buckets=args.buckets,
    )
    rows = []
    generator = torch.Generator().manual_seed(0)
    prompts = [
        torch.randint(0, 1000, (1, length), generator=generator).cuda() for length in args.lengths
    ]
    with torch.no_grad():
        # Every MegaBake length first: resetting Dynamo for the Inductor
        # baselines would drop the compiled buckets.
        for ids in prompts:
            mask = torch.ones_like(ids)
            length = ids.shape[1]
            bucket = prefill.bucket_for(length)
            first_bucket_call = bucket not in prefill.verified
            started = time.perf_counter()
            logits = prefill(input_ids=ids, attention_mask=mask).logits
            torch.cuda.synchronize()
            first_call_s = time.perf_counter() - started
            expected = model(input_ids=ids, attention_mask=mask).logits
            error = ((logits.float() - expected.float()).norm() / expected.float().norm()).item()
            row = {
                "length": length,
                "bucket": bucket,
                "bucket_compile_s": first_call_s if first_bucket_call else 0.0,
                "relative_l2_vs_eager": error,
                "megabake_call_us": _median_call(
                    lambda ids=ids, mask=mask: prefill(input_ids=ids, attention_mask=mask),
                    args.samples,
                ),
            }
            print(json.dumps(row), flush=True)
            rows.append(row)
        if not args.no_inductor:
            for ids, row in zip(prompts, rows, strict=True):
                mask = torch.ones_like(ids)
                torch._dynamo.reset()
                inductor = torch.compile(
                    model, mode="reduce-overhead", fullgraph=True, dynamic=False
                )
                started = time.perf_counter()
                inductor(input_ids=ids, attention_mask=mask)
                torch.cuda.synchronize()
                row["inductor_compile_s"] = time.perf_counter() - started
                row["inductor_cudagraphs_call_us"] = _median_call(
                    lambda inductor=inductor, ids=ids, mask=mask: inductor(
                        input_ids=ids, attention_mask=mask
                    ),
                    args.samples,
                )
                print(json.dumps(row), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.model.replace('/', '--')}.json"
    path.write_text(json.dumps({"model": args.model, "rows": rows}, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
