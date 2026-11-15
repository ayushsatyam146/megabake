Completed the progressive decode-context sweep (re-run 2026-11-15 with a fair Inductor baseline). Each cell is **pipelined / complete-call latency** in µs per one-token decode call; speedup compares the pipelined numbers. Runs used batch 1, BF16, 40 samples, and 5 warmups on an H200 MIG 3g.71gb.

This measures decode with a preset cached context length; it does not include prompt prefill. The tested contexts were 16, 32, 64, 128, 256, 512, 1k, 2k, 4k, and 8k tokens. At 8k, MegaBake was 2.11× faster on SmolLM-135M, 1.81× on Qwen2-0.5B, 1.83× on DeepSeek, and 1.06× on Gemma-2B. SmolLM-1.7B crossed over between 4k and 8k: 1.02× at 4k, and MegaBake was slower at 8k (0.93×).

**What the two columns are.**

- **Inductor** is the unmodified Hugging Face model with a Hugging Face `StaticCache`, compiled by `torch.compile(mode="reduce-overhead", fullgraph=True)`: Inductor kernels replayed as CUDA graphs (`benchmarks/full_model.py --variants inductor_hf`). Nothing from MegaBake is in its graph; Inductor compiles every op, attention included. Attention is `attn_implementation="eager"`, which Inductor fuses itself. Hugging Face attends over the whole static cache with a mask, so each context gets its own cache, just long enough for its timed calls (context + 88 keys), and its own compile: the strongest Inductor configuration for that context. Default SDPA attention was measured too and was slower everywhere it was run (16 and 8k keys on every model; at 8k it was 11.5–13.1 ms, because memory-efficient attention gives one CTA per head and does not split the keys).
- **MegaBake** is MegaBake's decode step (`megabake_graphed`): MegaBake captures the model through its own attention interface and compiles one megakernel, captured in a CUDA graph. It is compiled once at 8,088 keys of capacity and reads only the valid length.

**What changed from the previous version of this page.** The previous Inductor column compiled MegaBake's own decode wrapper. Its attention is the `megabake::kv_attention` custom op, which Inductor cannot see into: it calls it as an opaque kernel (120 times for SmolLM-135M's 30 layers). In the graphed baseline that op ran `_static_kv_attention`: eager PyTorch that repeats K and V per query head and computes FP32 scores and a softmax over the whole 8,088-key capacity at every context. That is why the old Inductor column was flat at about 3.8 ms (SmolLM-135M) to 6.5 ms (Gemma-2B) whatever the context. The old speedups (4.46× at 16 keys on SmolLM-135M, 2.15× on Gemma-2B) overstated MegaBake's lead; against the fair baseline they are 1.80× and 1.12×.

### SmolLM-135M

| Context | Inductor pipe / complete (µs) | MegaBake pipe / complete (µs) | Speedup |
|---:|---:|---:|---:|
| 16 | 1542.5 / 2106.4 | 859.1 / 892.1 | 1.80× |
| 32 | 1559.1 / 2105.6 | 861.1 / 893.4 | 1.81× |
| 64 | 1580.6 / 2153.1 | 878.8 / 898.0 | 1.80× |
| 128 | 1606.3 / 2158.2 | 904.8 / 937.8 | 1.78× |
| 256 | 1642.3 / 2198.8 | 953.6 / 991.8 | 1.72× |
| 512 | 1674.2 / 2217.9 | 999.7 / 1039.5 | 1.67× |
| 1k | 1753.1 / 2311.7 | 1000.9 / 1039.0 | 1.75× |
| 2k | 1977.3 / 2562.2 | 1038.5 / 1045.8 | 1.90× |
| 4k | 2127.5 / 2744.2 | 1067.0 / 1104.9 | 1.99× |
| 8k | 2498.8 / 3100.5 | 1182.7 / 1217.2 | 2.11× |

### Qwen2-0.5B

| Context | Inductor pipe / complete (µs) | MegaBake pipe / complete (µs) | Speedup |
|---:|---:|---:|---:|
| 16 | 1838.5 / 2415.7 | 1179.7 / 1206.2 | 1.56× |
| 32 | 1843.7 / 2421.3 | 1180.9 / 1208.1 | 1.56× |
| 64 | 1854.2 / 2373.4 | 1194.5 / 1209.5 | 1.55× |
| 128 | 1857.1 / 2414.7 | 1213.0 / 1240.0 | 1.53× |
| 256 | 1859.4 / 2417.3 | 1254.2 / 1283.8 | 1.48× |
| 512 | 1882.1 / 2432.7 | 1277.8 / 1305.2 | 1.47× |
| 1k | 2109.1 / 2648.8 | 1401.4 / 1432.9 | 1.50× |
| 2k | 2191.7 / 2775.8 | 1430.4 / 1429.0 | 1.53× |
| 4k | 2519.0 / 3129.9 | 1472.0 / 1502.8 | 1.71× |
| 8k | 2960.5 / 3559.2 | 1638.9 / 1674.6 | 1.81× |

### DeepSeek-V3-DRAFT-0.6B-v3.0

| Context | Inductor pipe / complete (µs) | MegaBake pipe / complete (µs) | Speedup |
|---:|---:|---:|---:|
| 16 | 1814.7 / 2377.3 | 1144.0 / 1170.1 | 1.59× |
| 32 | 1808.9 / 2374.9 | 1145.9 / 1172.4 | 1.58× |
| 64 | 1806.4 / 2386.4 | 1159.8 / 1176.2 | 1.56× |
| 128 | 1821.5 / 2430.0 | 1179.5 / 1209.3 | 1.54× |
| 256 | 1844.8 / 2399.8 | 1220.5 / 1250.9 | 1.51× |
| 512 | 1895.0 / 2452.4 | 1241.8 / 1270.6 | 1.53× |
| 1k | 2065.4 / 2648.0 | 1367.5 / 1397.0 | 1.51× |
| 2k | 2187.6 / 2768.0 | 1400.1 / 1393.3 | 1.56× |
| 4k | 2491.5 / 3110.3 | 1439.3 / 1472.2 | 1.73× |
| 8k | 2944.4 / 3583.9 | 1608.3 / 1640.2 | 1.83× |

### SmolLM-1.7B

| Context | Inductor pipe / complete (µs) | MegaBake pipe / complete (µs) | Speedup |
|---:|---:|---:|---:|
| 16 | 2717.8 / 3167.1 | 2188.2 / 2218.8 | 1.24× |
| 32 | 2741.3 / 3200.4 | 2189.1 / 2219.6 | 1.25× |
| 64 | 2731.7 / 3235.2 | 2204.8 / 2222.5 | 1.24× |
| 128 | 2838.7 / 3333.0 | 2224.8 / 2252.7 | 1.28× |
| 256 | 2771.2 / 3239.5 | 2262.7 / 2290.3 | 1.22× |
| 512 | 2880.7 / 3387.8 | 2331.9 / 2361.1 | 1.24× |
| 1k | 2950.3 / 3450.1 | 2471.7 / 2489.2 | 1.19× |
| 2k | 3058.0 / 3565.3 | 2794.6 / 2823.8 | 1.09× |
| 4k | 3226.4 / 3744.1 | 3165.9 / 3196.9 | 1.02× |
| 8k | 3618.3 / 4132.5 | 3900.4 / 3930.6 | 0.93× |

### Beam-1.0-501B-A23B-DFlash2

This checkpoint has a 4096-token sliding window, so its sweep ends at 4000 tokens. Every cache in the sweep is at most 4,088 keys, shorter than the window, so the window never drops a key; the Inductor baseline builds these layers as full attention. (Hugging Face's sliding static cache keeps a Python length that makes Dynamo recompile on every call, which stops `fullgraph=True` after eight calls.)

| Context | Inductor pipe / complete (µs) | MegaBake pipe / complete (µs) | Speedup |
|---:|---:|---:|---:|
| 16 | 2560.2 / 2763.5 | 2427.6 / 2441.7 | 1.05× |
| 32 | 2564.0 / 2785.5 | 2430.3 / 2443.5 | 1.06× |
| 64 | 2579.2 / 2793.5 | 2431.9 / 2447.6 | 1.06× |
| 128 | 2587.9 / 2798.6 | 2438.7 / 2454.7 | 1.06× |
| 256 | 2622.6 / 2830.9 | 2451.9 / 2467.9 | 1.07× |
| 512 | 2647.4 / 2862.6 | 2477.5 / 2494.0 | 1.07× |
| 1k | 2691.5 / 2909.4 | 2532.2 / 2555.9 | 1.06× |
| 2k | 2791.6 / 3018.0 | 2598.7 / 2617.7 | 1.07× |
| 4k | 2997.8 / 3226.7 | 2734.6 / 2750.2 | 1.10× |

### Gemma-2B

Gemma-2B is the ungated `unsloth/gemma-2b` copy of the checkpoint.

| Context | Inductor pipe / complete (µs) | MegaBake pipe / complete (µs) | Speedup |
|---:|---:|---:|---:|
| 16 | 3319.0 / 3745.1 | 2961.7 / 2986.5 | 1.12× |
| 32 | 3316.9 / 3719.1 | 2961.1 / 2992.2 | 1.12× |
| 64 | 3324.4 / 3746.8 | 2962.5 / 2990.8 | 1.12× |
| 128 | 3344.2 / 3754.7 | 2962.3 / 2989.0 | 1.13× |
| 256 | 3363.9 / 3739.9 | 2975.6 / 3004.7 | 1.13× |
| 512 | 3335.4 / 3778.8 | 3000.8 / 3029.7 | 1.11× |
| 1k | 3414.3 / 3821.1 | 3044.8 / 3072.3 | 1.12× |
| 2k | 3554.9 / 3984.8 | 3145.8 / 3167.2 | 1.13× |
| 4k | 3651.9 / 4078.0 | 3340.0 / 3352.6 | 1.09× |
| 8k | 3864.6 / 4262.1 | 3634.8 / 3662.8 | 1.06× |

Beam’s load also reported missing embedding and LM-head weights, so treat its numbers as a forward-path timing for that draft checkpoint.

**Complete calls.** Inductor's complete call is 200–600 µs above its pipelined call: Dynamo guards, the CUDA-graph tree's bookkeeping and the Hugging Face forward wrapper run on the host every call. MegaBake's graphed call adds at most about 35 µs.

**Correctness of the baseline.** Fed a prompt token by token, the Inductor baseline gives the same first 13 greedy tokens as Hugging Face `generate` on SmolLM-135M and then diverges by one token, the same with an SDPA or an eager-attention reference. Inductor's fused kernels round differently from eager PyTorch; MegaBake's own Llama-3.2-1B check diverges from eager at token 37 the same way.

Reproduce one model with:

    TORCHINDUCTOR_CACHE_DIR=... .venv/bin/python benchmarks/full_model.py --model HuggingFaceTB/SmolLM-135M --modes \
        --decode-contexts 16 32 64 128 256 512 1000 2000 4000 8000 \
        --variants inductor_hf megabake_graphed --hf-attention eager

Artifacts (per-model `report.json` and logs) are in `artifacts/fair_sweep_2026-11-15/`; the whole sweep is `experiments/newarch/fair_sweep.sh`.
