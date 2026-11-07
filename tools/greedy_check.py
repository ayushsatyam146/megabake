"""Greedy decode check for a real checkpoint (experiment X2a, 2026-11-07).

Feeds a text prompt one token at a time through the decode step, then decodes
``--tokens`` greedy tokens, three ways: the MegaBake-compiled step, the same
step in eager bf16, and Hugging Face ``generate`` (its own KV cache). Reports
the first position where MegaBake's tokens leave each reference, and the
relative logit error against an fp32 copy of the step (MegaBake vs eager
bf16), fed the same tokens.

    .venv/bin/python tools/greedy_check.py --model unsloth/Llama-3.2-1B-Instruct --tokens 64
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from megabake import make_backend  # noqa: E402
from benchmarks.full_model import _load, decode_module  # noqa: E402

TEXT = "The history of the printing press begins in"


def _first_mismatch(a: list[int], b: list[int]) -> int | None:
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer(TEXT, return_tensors="pt").input_ids[0].tolist()
    model = _load(args.model, None, "eager")
    with torch.no_grad():
        hf_model = copy.deepcopy(model)
        hf_model.config.use_cache = True
        generation = getattr(hf_model, "generation_config", None)
        hf = hf_model.generate(
            torch.tensor([prompt], device="cuda"), max_new_tokens=args.tokens, do_sample=False,
            use_cache=True, attention_mask=torch.ones(1, len(prompt), device="cuda", dtype=torch.long),
            **({"top_p": None, "top_k": None, "temperature": None} if generation is not None else {}),
        )[0, len(prompt):].tolist()
        del hf_model
        stateful, shapes = decode_module(model, args.capacity)
        exact_step = copy.deepcopy(stateful).float()
        backend = make_backend(mode="decode", compile_kernel=True, cache=False, scheduler="phase")
        compiled = torch.compile(stateful, backend=backend, fullgraph=True, dynamic=False)

        def run(step, cast=None):
            state = [torch.zeros(s, device="cuda", dtype=cast if cast and d.is_floating_point else d)
                     for layer in shapes for s, d in layer]
            return state, (lambda token, position: step(
                torch.tensor([[token]], device="cuda"), torch.tensor([[position]], device="cuda"), *state))

        _, mega = run(compiled)
        _, eager = run(stateful)
        _, exact = run(exact_step, torch.float32)
        tokens = {"megabake": [], "eager": []}
        errors = {"megabake": [], "eager": []}
        feed = list(prompt)
        for position in range(len(prompt) + args.tokens - 1):
            token = feed[position]
            logits = {"megabake": mega(token, position).float(), "eager": eager(token, position).float()}
            reference = exact(token, position).float()
            for name, value in logits.items():
                errors[name].append(((value - reference).norm() / reference.norm()).item())
            if position >= len(prompt) - 1:
                for name, value in logits.items():
                    tokens[name].append(int(value.reshape(-1, value.shape[-1])[-1].argmax()))
                # Teacher-force eager's token, so both runs see the same history.
                feed.append(tokens["eager"][-1])
    report = {
        "model": args.model,
        "prompt_tokens": len(prompt),
        "generated": args.tokens,
        "megabake_vs_eager_first_mismatch": _first_mismatch(tokens["megabake"], tokens["eager"]),
        "megabake_vs_eager_matches": sum(a == b for a, b in zip(tokens["megabake"], tokens["eager"])),
        "eager_vs_hf_first_mismatch": _first_mismatch(tokens["eager"], hf),
        "megabake_vs_hf_first_mismatch": _first_mismatch(tokens["megabake"], hf),
        "max_rel_err_vs_fp32": {name: max(values) for name, values in errors.items()},
        "mean_rel_err_vs_fp32": {name: sum(values) / len(values) for name, values in errors.items()},
        "rel_err_vs_fp32_by_position": {name: [round(v, 5) for v in values] for name, values in errors.items()},
        "text_megabake": tokenizer.decode(tokens["megabake"]),
    }
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
