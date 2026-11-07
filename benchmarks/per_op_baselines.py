"""Per-op decode baselines at a model's shapes (experiment X0.4).

Times the best available standalone kernel for each decode op class of a
Hugging Face decoder config, so a MegaBake interval can be compared with
what one op alone can reach on this GPU:

- projections (fused q/k/v, o, fused gate·up, down, LM head): cuBLAS
  (``F.linear``) and a tuned Triton CUDA-core GEMV (one program per few
  output rows, as hand-written decode kernels do);
- decode attention over ``--keys`` cached keys: PyTorch SDPA and a Triton
  kernel with one program per query head that reads the context in one pass.

Weights and caches rotate through copies whose total is several times L2,
so no call reads its operands from L2. Each op runs ``--graph-calls`` times
inside one CUDA graph; the per-call time includes the launch gap a graph
leaves between kernels. Shapes come from the config only.

    .venv/bin/python benchmarks/per_op_baselines.py --model unsloth/Llama-3.2-1B-Instruct --keys 113
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from megabake.machine import machine_model  # noqa: E402


@triton.jit
def _gemv_kernel(x, w, y, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        cols = k + tl.arange(0, BLOCK_K)
        weights = tl.load(w + rows[:, None] * K + cols[None, :], mask=rows[:, None] < N, other=0.0)
        values = tl.load(x + cols)
        acc += tl.sum(weights.to(tl.float32) * values.to(tl.float32)[None, :], axis=1)
    tl.store(y + rows, acc.to(tl.bfloat16), mask=rows < N)


def triton_gemv(x, w, y, block_n, block_k, warps):
    n, k = w.shape
    _gemv_kernel[(triton.cdiv(n, block_n),)](x, w, y, n, k, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=warps)


@triton.jit
def _decode_attention_kernel(q, k_cache, v_cache, out, keys, scale,
                             KV_HEADS: tl.constexpr, GROUP: tl.constexpr,
                             HEAD_DIM: tl.constexpr, BLOCK_KEYS: tl.constexpr):
    head = tl.program_id(0)
    kv_head = head // GROUP
    dims = tl.arange(0, HEAD_DIM)
    query = tl.load(q + head * HEAD_DIM + dims).to(tl.float32)
    running_max = -float("inf")
    running_sum = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for start in range(0, keys, BLOCK_KEYS):
        rows = start + tl.arange(0, BLOCK_KEYS)
        offsets = rows[:, None] * (KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM + dims[None, :]
        valid = rows[:, None] < keys
        key = tl.load(k_cache + offsets, mask=valid, other=0.0).to(tl.float32)
        scores = tl.sum(key * query[None, :], axis=1) * scale
        scores = tl.where(rows < keys, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)
        value = tl.load(v_cache + offsets, mask=valid, other=0.0).to(tl.float32)
        acc = acc * correction + tl.sum(probabilities[:, None] * value, axis=0)
        running_sum = running_sum * correction + tl.sum(probabilities, axis=0)
        running_max = new_max
    tl.store(out + head * HEAD_DIM + dims, (acc / running_sum).to(tl.bfloat16))


def _graph_us(make_call, copies: int, calls: int) -> float:
    """Per-call µs of ``calls`` calls in one CUDA graph, rotating ``copies``."""
    operations = [make_call(index % copies) for index in range(calls)]
    for operation in operations[:copies]:
        operation()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for operation in operations[:copies]:
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for operation in operations:
            operation()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(7):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / calls)
    return sorted(samples)[len(samples) // 2]


def _copies(bytes_per_copy: int, l2_bytes: int) -> int:
    return max(2, math.ceil(4 * l2_bytes / bytes_per_copy))


def projection_baselines(name, n, k, machine, calls) -> dict:
    weight_bytes = n * k * 2
    copies = _copies(weight_bytes, machine.l2_effective_bytes or machine.fingerprint.l2_cache_bytes)
    weights = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(1, n, device="cuda", dtype=torch.bfloat16)
    reference = F.linear(x, weights[0]).float()
    results = {"op": name, "n": n, "k": k, "bytes": weight_bytes, "copies": copies}
    results["cublas_us"] = _graph_us(lambda i: (lambda: torch.mm(x, weights[i].t(), out=y)), copies, calls)
    best = None
    for block_n, block_k, warps in itertools.product((1, 2, 4, 8, 16, 32), (256, 512, 1024), (2, 4, 8)):
        if k % block_k or block_n * block_k > 32768:
            continue
        triton_gemv(x, weights[0], y, block_n, block_k, warps)
        if not torch.allclose(y.float(), reference, atol=2e-2, rtol=2e-2):
            continue
        us = _graph_us(
            lambda i, b=(block_n, block_k, warps): (lambda: triton_gemv(x, weights[i], y, *b)), copies, calls
        )
        if best is None or us < best[0]:
            best = (us, block_n, block_k, warps)
    if best is not None:
        results["triton_gemv_us"] = best[0]
        results["triton_gemv_config"] = {"block_n": best[1], "block_k": best[2], "warps": best[3],
                                         "ctas": math.ceil(n / best[1])}
    t_sol = weight_bytes / machine.hbm_read_bytes_per_s * 1e6
    results["t_sol_us"] = t_sol
    results["best_us"] = min(value for key, value in results.items() if key.endswith("_us") and key != "t_sol_us")
    results["best_sol"] = t_sol / results["best_us"]
    return results


def attention_baselines(heads, kv_heads, head_dim, keys, capacity, machine, calls) -> dict:
    cache_bytes = 2 * capacity * kv_heads * head_dim * 2
    copies = _copies(cache_bytes, machine.l2_effective_bytes or machine.fingerprint.l2_cache_bytes)
    caches = [
        (torch.randn(capacity, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16),
         torch.randn(capacity, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16))
        for _ in range(copies)
    ]
    q = torch.randn(heads, head_dim, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(heads, head_dim, device="cuda", dtype=torch.bfloat16)
    scale = head_dim**-0.5
    k0, v0 = caches[0][0][:keys], caches[0][1][:keys]
    group = heads // kv_heads
    key_rows = k0.float().repeat_interleave(group, 1).transpose(0, 1)  # heads, keys, dim
    value_rows = v0.float().repeat_interleave(group, 1).transpose(0, 1)
    weights = torch.softmax((key_rows @ q.float()[:, :, None])[..., 0] * scale, dim=-1)
    reference = (weights[:, None, :] @ value_rows)[:, 0]
    results = {"op": "decode_attention", "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim,
               "keys": keys, "bytes": 2 * keys * kv_heads * head_dim * 2, "copies": copies}

    def sdpa(i):
        key = caches[i][0][:keys].transpose(0, 1)[None]
        value = caches[i][1][:keys].transpose(0, 1)[None]
        def call():
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                return F.scaled_dot_product_attention(q[None, :, None], key, value, enable_gqa=True)
        return call

    try:
        results["sdpa_us"] = _graph_us(sdpa, copies, calls)
    except RuntimeError as error:
        results["sdpa_error"] = str(error).splitlines()[0][:200]
    block_keys = min(256, triton.next_power_of_2(keys))

    def triton_call(i):
        key, value = caches[i]
        return lambda: _decode_attention_kernel[(heads,)](
            q, key, value, out, keys, scale, KV_HEADS=kv_heads, GROUP=group, HEAD_DIM=head_dim,
            BLOCK_KEYS=block_keys, num_warps=4)

    triton_call(0)()
    if torch.allclose(out.float(), reference, atol=2e-2, rtol=2e-2):
        results["triton_one_pass_us"] = _graph_us(triton_call, copies, calls)
    else:
        results["triton_one_pass_error"] = float((out.float() - reference).abs().max())
    results["t_sol_us"] = results["bytes"] / machine.hbm_read_bytes_per_s * 1e6
    results["best_us"] = min(value for key, value in results.items() if key.endswith("_us") and key != "t_sol_us")
    return results


def main() -> None:
    from transformers import AutoConfig

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--keys", type=int, default=113)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--graph-calls", type=int, default=64)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model)
    machine = machine_model()
    hidden, inner = config.hidden_size, config.intermediate_size
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or hidden // heads
    torch.manual_seed(0)
    rows = [
        projection_baselines("q_k_v (fused)", (heads + 2 * kv_heads) * head_dim, hidden, machine, args.graph_calls),
        projection_baselines("o_proj", hidden, heads * head_dim, machine, args.graph_calls),
        projection_baselines("gate_up (fused)", 2 * inner, hidden, machine, args.graph_calls),
        projection_baselines("down_proj", hidden, inner, machine, args.graph_calls),
        projection_baselines("lm_head", config.vocab_size, hidden, machine, max(8, args.graph_calls // 8)),
        attention_baselines(heads, kv_heads, head_dim, args.keys, args.capacity, machine, args.graph_calls),
    ]
    report = {"model": args.model, "device": machine.fingerprint.name,
              "hbm_read_gb_s": machine.hbm_read_bytes_per_s / 1e9, "ops": rows}
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
