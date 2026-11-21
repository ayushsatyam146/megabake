"""Report whether ptxas serialized the wgmma instructions of a model's kernel.

ptxas warning C7519 ("wgmma.mma_async instructions are serialized due to ...")
means every tensor-core instruction of the megakernel waits for the previous
one: the kernel then runs 15-25% slower with the same work (43% more SASS
instructions, mostly R2UR and WARPGROUP fences, measured on SmolLM-1.7B
prefill). Any trap behind a per-lane condition anywhere in the kernel fires it,
because ptxas then cannot assume whole warps reach a wgmma; bodies trap through
``runtime.trap_if``, which votes across the warp first. Whether ptxas sees such
a divergence varies with unrelated code, so this compile-only check is the
noise-free way to see it.

    .venv/bin/python tools/wgmma_serialization.py --model HuggingFaceTB/SmolLM-1.7B --mode prefill
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _compile(model_name: str, mode: str) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    from benchmarks.full_model import PROMPT, _load, decode_module
    from megabake import make_backend

    model = _load(model_name, None, "eager")
    backend = make_backend(mode=mode, compile_kernel=True, cache=False, scheduler="phase")
    with torch.no_grad():
        if mode == "prefill":
            ids = torch.tensor([PROMPT], device="cuda")
            compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
            compiled(ids, attention_mask=torch.ones_like(ids))
        else:
            step, shapes = decode_module(model, 128)
            state = [torch.zeros(s, device="cuda", dtype=d) for layer in shapes for s, d in layer]
            compiled = torch.compile(step, backend=backend, fullgraph=True, dynamic=False)
            compiled(
                torch.tensor([[504]], device="cuda"), torch.tensor([[0]], device="cuda"), *state
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--compile-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.compile_only:
        _compile(args.model, args.mode)
        return 0
    # A fresh process with ptxas remarks on and no cached kernel to reuse.
    environment = dict(
        os.environ, CUTE_DSL_DISABLE_FILE_CACHING="1", CUTE_DSL_COMPILER_OPT="remarks{ptx}"
    )
    result = subprocess.run(
        [sys.executable, __file__, "--model", args.model, "--mode", args.mode, "--compile-only"],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    if result.returncode:
        print(output[-4000:])
        return result.returncode
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if "C7519" in line:
            print(f"{args.model} {args.mode}: wgmma serialized")
            print("\n".join(lines[index : index + 3]))
            return 1
    print(f"{args.model} {args.mode}: wgmma not serialized")
    return 0


if __name__ == "__main__":
    sys.exit(main())
