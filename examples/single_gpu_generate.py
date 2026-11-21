"""Greedy generation with MegaBake's stateful decode on one CUDA GPU.

This simple example processes the prompt one token at a time, so it exercises
the same static KV cache as decoding; it is not an optimized prefill path.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from megabake import make_backend
from megabake.fx_handler.hf_attention import DecodeStep, register_transformers_attention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="Explain what a GPU does in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--raw-prompt", action="store_true", help="do not apply a chat template")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.raw_prompt:
        prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    else:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).input_ids
    tokens = prompt_ids[0].tolist()
    if not tokens:
        parser.error("the prompt must contain at least one token")
    capacity = len(tokens) + args.max_new_tokens
    if capacity > 8192:
        parser.error("prompt and generated tokens exceed the 8192-token cache limit")

    register_transformers_attention()
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation="megabake", dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    step = DecodeStep(model, capacity).eval()
    state = [
        torch.zeros(shape, dtype=dtype, device="cuda")
        for layer in step.state_shapes()
        for shape, dtype in layer
    ]
    backend = make_backend(mode="decode", compile_kernel=True, scheduler="phase")
    compiled_step = torch.compile(step, backend=backend, fullgraph=True, dynamic=False)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])

    def advance(token: int, position: int) -> torch.Tensor:
        token_ids = torch.tensor([[token]], device="cuda", dtype=torch.long)
        position_ids = torch.tensor([[position]], device="cuda", dtype=torch.long)
        return compiled_step(token_ids, position_ids, *state)

    with torch.no_grad():
        for position, token in enumerate(tokens):
            logits = advance(token, position)
        generated = []
        for _ in range(args.max_new_tokens):
            token = int(logits[0, -1].argmax().item())
            if token in eos_ids:
                break
            generated.append(token)
            if len(generated) < args.max_new_tokens:
                logits = advance(token, len(tokens) + len(generated) - 1)
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
