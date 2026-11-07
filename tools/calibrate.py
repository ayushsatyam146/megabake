"""Measure (or load the cached) MachineModel for this GPU and print it (X0.1).

    .venv/bin/python tools/calibrate.py [--force] [--repeats 3] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from megabake.machine import machine_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="measure even if a cached result exists")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    model = machine_model(force=args.force, repeats=args.repeats)
    data = model.to_json()
    print(json.dumps(data, indent=2))
    gb = lambda value: f"{value / 1e9:,.0f} GB/s"  # noqa: E731
    print(f"\n{model.fingerprint.name}, {model.fingerprint.multiprocessor_count} SMs")
    print(f"HBM read {gb(model.hbm_read_bytes_per_s)}; L2 read {gb(model.l2_read_bytes_per_s)}; "
          f"L2 effective {model.l2_effective_bytes >> 20} MiB")
    print("HBM read by busy CTAs: " + ", ".join(
        f"{k}: {gb(v)}" for k, v in sorted(model.hbm_read_bytes_per_s_by_ctas.items())))
    print("grid barrier: " + ", ".join(
        f"{k} CTAs {v * 1e6:.2f} us" for k, v in sorted(model.grid_barrier_s_by_ctas.items())))
    print(f"counter round trip {model.counter_round_trip_s * 1e6:.2f} us")
    print("tensor peak: " + ", ".join(f"{k} {v / 1e12:.0f} TFLOP/s" for k, v in model.tensor_peak_flops.items()))
    if args.out is not None:
        args.out.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
