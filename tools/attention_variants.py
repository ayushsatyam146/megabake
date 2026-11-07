"""Decode-attention variant matrix on tiny random-weight models (X2a follow-up).

Builds small decoders from real ``transformers`` configs (Llama, Qwen3,
Mistral, Gemma 2, Gemma 3) that differ in head_dim, KV grouping, sliding
windows, soft caps and q/k norms, decodes them through ``DecodeStep`` with
a random token sequence one token at a time, and checks for each:

- the attention runs in MegaBake's split-context decode body (no fallback);
- at every position the logits stay near Hugging Face's own fp32 forward over
  the whole sequence (its masks, windows, soft caps and scaling): rel <= 1.5 *
  hf_bf16_rel + 1e-3 at each position, and the median ratio rel / hf_bf16_rel
  <= 1.1. Single bf16 outliers reach ~1.4x; real defects measured 50-280x
  (an empty split chunk, a dropped window or soft cap).

    .venv/bin/python tools/attention_variants.py            # all variants
    .venv/bin/python tools/attention_variants.py --only mistral_window
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

import megabake.compiler as compiler  # noqa: E402
from megabake import make_backend  # noqa: E402
from megabake.fx_handler.hf_attention import DecodeStep  # noqa: E402


def _config(kind: str, **overrides):
    from transformers import AutoConfig

    base = {
        "vocab_size": 8192, "num_hidden_layers": 2, "intermediate_size": 512,
        "max_position_embeddings": 4096, "tie_word_embeddings": False,
    }
    base.update(overrides)
    return AutoConfig.for_model(kind, **base)


VARIANTS = {
    # name: (model_type, overrides, contexts)
    "llama_gqa_hd64": ("llama", dict(hidden_size=512, num_attention_heads=8, num_key_value_heads=2), (0, 70, 300, 1000)),
    "llama_mha_hd64": ("llama", dict(hidden_size=512, num_attention_heads=8, num_key_value_heads=8), (0, 300, 1000)),
    "llama_mqa_hd64": ("llama", dict(hidden_size=512, num_attention_heads=8, num_key_value_heads=1), (0, 300, 1000)),
    "llama_hd80": ("llama", dict(hidden_size=640, num_attention_heads=8, num_key_value_heads=2, head_dim=80), (0, 70, 300, 1000)),
    "llama_hd96": ("llama", dict(hidden_size=768, num_attention_heads=8, num_key_value_heads=4, head_dim=96), (0, 300, 1000)),
    "llama_hd256": ("llama", dict(hidden_size=512, num_attention_heads=4, num_key_value_heads=1, head_dim=256), (0, 300, 1000)),
    "qwen3_qknorm_hd128": ("qwen3", dict(hidden_size=512, num_attention_heads=8, num_key_value_heads=4, head_dim=128), (0, 300, 1000)),
    "mistral_window": ("mistral", dict(hidden_size=512, num_attention_heads=8, num_key_value_heads=2, sliding_window=128), (0, 100, 300, 1000)),
    "gemma2_softcap_window": ("gemma2", dict(hidden_size=512, num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                                             sliding_window=128, attn_logit_softcapping=5.0, query_pre_attn_scalar=128), (0, 100, 300, 1000)),
    "gemma3_mixed_window": ("gemma3_text", dict(hidden_size=512, num_attention_heads=4, num_key_value_heads=1, head_dim=256,
                                                 sliding_window=128, layer_types=["sliding_attention", "full_attention"],
                                                 query_pre_attn_scalar=256), (0, 100, 300, 1000)),
}


def _relative(actual, expected):
    return (actual.float() - expected.float()).norm().item() / expected.float().norm().item()


CHECK_POSITIONS = (0, 63, 127, 128, 200, 400, 799)


def run_variant(name: str, capacity: int, length: int = 800, check_all: bool = True) -> dict:
    from transformers import AutoModelForCausalLM

    kind, overrides, _ = VARIANTS[name]
    torch.manual_seed(0)
    config = _config(kind, **overrides)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config).to("cuda", torch.bfloat16).eval()
    for parameter in model.parameters():
        # Larger weights than the default init, so attention is not uniform.
        parameter.data.normal_(0.0, 0.08)
    tokens = torch.randint(0, config.vocab_size, (1, length), device="cuda")
    positions = [p for p in CHECK_POSITIONS if p < length]
    with torch.no_grad():
        # References: HF's own attention over the whole sequence.
        hf_exact = copy.deepcopy(model).float()(tokens, use_cache=False).logits[0].float()
        hf_bf16 = copy.deepcopy(model)(tokens, use_cache=False).logits[0].float()
    step = DecodeStep(model, capacity).eval()
    shapes = step.state_shapes()
    backend = make_backend(mode="decode", compile_kernel=True, cache=False, scheduler="phase")
    compiled = torch.compile(step, backend=backend, fullgraph=True, dynamic=False)
    state = [torch.zeros(s, device="cuda", dtype=d) for layer in shapes for s, d in layer]
    report = {"variant": name, "model_type": kind, "head_dim": step.head_dim, "points": []}
    with torch.no_grad():
        for position in range(max(positions) + 1):
            logits = compiled(tokens[:, position : position + 1].clone(), torch.tensor([[position]], device="cuda"), *state)
            if position in positions or check_all:
                actual = logits.reshape(-1, logits.shape[-1])[-1].float()
                point = {
                    "position": position,
                    "rel_megabake": _relative(actual, hf_exact[position]),
                    "rel_hf_bf16": _relative(hf_bf16[position], hf_exact[position]),
                    "argmax_equal_hf_exact": int(actual.argmax()) == int(hf_exact[position].argmax()),
                }
                point["ok"] = point["rel_megabake"] <= 1.5 * point["rel_hf_bf16"] + 1e-3
                report["points"].append(point)
    compiled_plan = backend.megabake_last_compiled
    source = compiled_plan.generated.source_path.read_text() if compiled_plan else ""
    attention_phases = [p for p in compiled_plan.plan.phases if p.body.provider == "attention"] if compiled_plan else []
    report["megakernel"] = compiled_plan is not None
    report["attention_phases"] = len(attention_phases)
    report["split_body"] = "chunk_keys = (visible" in source
    report["window_in_body"] = "first = keys - cutlass.Int32(" in source
    report["softcap_in_body"] = "cute.math.tanh" in source
    ratios = sorted(point["rel_megabake"] / max(point["rel_hf_bf16"], 1e-9) for point in report["points"])
    report["median_ratio"] = ratios[len(ratios) // 2]
    report["ok"] = (
        report["megakernel"] and report["split_body"]
        and report["attention_phases"] == config.num_hidden_layers
        and all(point["ok"] for point in report["points"])
        and report["median_ratio"] <= 1.1
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--capacity", type=int, default=1100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    # Tiny models: use the table launch the split body needs, as the GPU tests do.
    compiler._TABLE_LAUNCH_OPS = 0
    reports = []
    for name in args.only or VARIANTS:
        # Every variant recompiles DecodeStep.forward; keep under Dynamo's limit.
        torch._dynamo.reset()
        try:
            report = run_variant(name, args.capacity)
        except Exception as error:  # report and continue
            report = {"variant": name, "ok": False, "error": f"{type(error).__name__}: {str(error)[:300]}"}
        reports.append(report)
        summary = {k: v for k, v in report.items() if k != "points"}
        worst = max((p["rel_megabake"] / max(p["rel_hf_bf16"], 1e-9) for p in report.get("points", [])), default=None)
        print(json.dumps({**summary, "worst_rel_ratio": worst}), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"{sum(r['ok'] for r in reports)}/{len(reports)} variants ok")


if __name__ == "__main__":
    main()
