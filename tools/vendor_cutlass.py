"""Show how MegaBake's CUTLASS GEMM bodies differ from pinned upstream files.

Upstream snapshots are stored beside the architecture-specific bodies.

    .venv/bin/python tools/vendor_cutlass.py          # diff upstream -> fork
    .venv/bin/python tools/vendor_cutlass.py --stat   # changed line counts only
    .venv/bin/python tools/vendor_cutlass.py --check  # verify the upstream copy's hash

Upgrading CUTLASS: replace the snapshot and its recorded hash, preserving any
documented prefix, then merge upstream changes into the fork by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BODIES = ROOT / "src" / "megabake" / "bodies" / "cuda"
UPSTREAM = BODIES / "sm90" / "cutlass_upstream"
FORKS = {"dense_gemm_persistent.py": BODIES / "sm90" / "cutlass_gemm.py"}
SM100_UPSTREAM = BODIES / "sm100" / "cutlass_upstream"
SM100_FORKS = {
    "fp16_gemm_0.py": (
        BODIES / "sm100" / "tcgen_gemm.py",
        "a15471df9a2a9c80a6a772416210ea256a58b142aa4b6358ca01851f6f02362a",
    ),
    "dense_gemm_persistent.py": (
        BODIES / "sm100" / "cutlass_gemm.py",
        "4f3d5fa8c6032c2190933cb43c71d358bb200225fe7c0ec0b54014a9a6235362",
    ),
}
SM100_HEADER = b"# ruff: noqa\n# fmt: off\n"


def check_upstream() -> None:
    """Fail unless every upstream copy has its recorded prefix and hash."""
    for line in (UPSTREAM / "UPSTREAM").read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() != "sha256":
            continue
        name, expected = value.split()
        actual = hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"cutlass_upstream/{name} was modified ({actual} != {expected})")
    for name, (_, expected) in SM100_FORKS.items():
        original = SM100_UPSTREAM / name
        source = original.read_bytes()
        if not source.startswith(SM100_HEADER):
            raise SystemExit(f"upstream {original} lint header was modified")
        actual = hashlib.sha256(source[len(SM100_HEADER) :]).hexdigest()
        if actual != expected:
            raise SystemExit(f"upstream {original} was modified ({actual} != {expected})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only verify the upstream copy")
    parser.add_argument("--stat", action="store_true", help="show changed line counts only")
    args = parser.parse_args()
    check_upstream()
    if args.check:
        return 0
    forks = [(UPSTREAM / name, fork) for name, fork in FORKS.items()]
    forks.extend((SM100_UPSTREAM / name, fork) for name, (fork, _) in SM100_FORKS.items())
    for original, fork in forks:
        subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                *(["--stat"] if args.stat else []),
                str(original.relative_to(ROOT)),
                str(fork.relative_to(ROOT)),
            ],
            check=False,
            cwd=ROOT,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
