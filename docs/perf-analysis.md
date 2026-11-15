# Where MegaBake decode time goes (2026-11-15)

Six models, batch-1 decode, H200 MIG 3g.71gb slice (60 SMs, 2.31 TB/s measured). The speedups against Inductor are the fair ones from [progressive-context-sweep.md](progressive-context-sweep.md). This page answers two questions: why some models are only 1.05–1.12× faster, and what would move them.

## Method

| Source | What it gives | Overhead |
|---|---|---|
| `tools/plan_report.py` (phase timer: one stamp per grid barrier) | Measured time of every barrier interval, by interval kind | About 1% |
| `tools/phase_profile.py --cta --sol` (a stamp per CTA and phase) | Busy CTAs per interval; time split into bytes, idle SMs, body and sync | Inflates the kernel 0–45%. Used only for busy-CTA counts and for the idle-SM loss, which is computed from bytes and busy counts, not from stamps |
| `experiments/newarch/inductor_breakdown.py` | Inductor's kernel time per step by kernel class (cuBLAS GEMM, Triton fused, copies, softmax) | Profiler only |

"Bytes" below is the time a phase's bytes take at the slice's full HBM rate: the lower bound B0. Artifacts are in `artifacts/perf_analysis_2026-11-15/`.

One finding outside performance: with `--cta`, Beam and Gemma-2B crashed with a CUDA launch failure at a cache of 80 keys and ran cleanly at 256 keys. That is a possible bug in the per-CTA timer build at small capacities, not investigated.

## 1. How much speedup is possible at all

Every number is pipelined µs per token. "Bytes" is all weights plus the KV cache read at the full HBM rate. "Max possible" is Inductor's time over the bytes time: the speedup a perfect kernel would show.

| Model | Keys | MegaBake | Inductor | Speedup | Bytes | MegaBake % of bytes | Inductor % of bytes | Max possible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SmolLM-135M | 16 | 859 | 1,542 | 1.80× | 117 | 14% | 8% | 13.2× |
| SmolLM-135M | 8k | 1,183 | 2,499 | 2.11× | 196 | 17% | 8% | 12.7× |
| Qwen2-0.5B | 16 | 1,180 | 1,838 | 1.56× | 428 | 36% | 23% | 4.3× |
| Qwen2-0.5B | 8k | 1,639 | 2,961 | 1.81× | 470 | 29% | 16% | 6.3× |
| DeepSeek draft | 16 | 1,144 | 1,815 | 1.59× | 410 | 36% | 23% | 4.4× |
| DeepSeek draft | 8k | 1,608 | 2,944 | 1.83× | 453 | 28% | 15% | 6.5× |
| SmolLM-1.7B | 16 | 2,188 | 2,718 | 1.24× | 1,483 | 68% | 55% | 1.83× |
| SmolLM-1.7B | 8k | 3,900 | 3,618 | 0.93× | 2,161 | 55% | 60% | 1.67× |
| Beam DFlash2 | 16 | 2,428 | 2,560 | 1.05× | 2,101 | 87% | 82% | 1.22× |
| Beam DFlash2 | 4k | 2,735 | 2,998 | 1.10× | 2,137 | 78% | 71% | 1.40× |
| Gemma-2B | 16 | 2,962 | 3,319 | 1.12× | 2,169 | 73% | 65% | 1.53× |
| Gemma-2B | 8k | 3,635 | 3,865 | 1.06× | 2,233 | 61% | 58% | 1.73× |

**The thin margins are mostly the ceiling, not a failure.** Beam reads about 4.9 GB of weights per token (5 layers of 5,120 wide plus a 1.6 GB LM head), Gemma-2B about 5.0 GB (with a 1 GB LM head over a 256k vocabulary). At this bandwidth that alone takes about 2.1 ms. Inductor already runs them at 65–82% of that bound, so even a perfect kernel would be only 1.22× (Beam) and 1.53× (Gemma) faster. MegaBake is at 87% and 73%.

The small models are the opposite: they move little data and both systems are far from the bound (MegaBake at 14–36%). There the lead is larger, but so is the room left.

## 2. Where MegaBake loses time

Per interval kind, measured with the phase timer, against the time its bytes take. A layer has five barrier intervals: attention, o_proj, norm + gate·up, down, and norm + q/k/v. "gemm" groups the three single-projection intervals.

| Model @ keys | Kind | Intervals | Measured µs | Bytes µs | Above bytes µs | Per interval |
|---|---|---:|---:|---:|---:|---:|
| SmolLM-135M @16 | gemm | 90 | 463 | 46 | 416 | 4.6 |
| | norm + gate·up | 30 | 176 | 46 | 130 | 4.3 |
| | attention | 30 | 147 | 0 | 147 | 4.9 |
| SmolLM-135M @8k | attention | 30 | 477 | 80 | 397 | 13.2 |
| Qwen2-0.5B @16 | gemm | 72 | 540 | 129 | 411 | 5.7 |
| | norm + gate·up | 24 | 329 | 181 | 148 | 6.2 |
| | attention | 24 | 122 | 0 | 122 | 5.1 |
| Qwen2-0.5B @8k | attention | 24 | 579 | 43 | 536 | 22.3 |
| SmolLM-1.7B @16 | gemm | 72 | 1,030 | 697 | 333 | 4.6 |
| | norm + gate·up | 24 | 892 | 697 | 195 | 8.1 |
| SmolLM-1.7B @8k | attention | 24 | 1,821 | 681 | 1,140 | 47.5 |
| Beam @16 | gemm | 15 | 831 | 658 | 173 | 11.6 |
| | norm + gate·up | 5 | 839 | 771 | 68 | 13.7 |
| Beam @4k | attention | 5 | 347 | 36 | 312 | 62.3 |
| Gemma-2B @16 | gemm | 54 | 1,066 | 670 | 396 | 7.3 |
| | norm + gate·up | 18 | 1,221 | 1,045 | 176 | 9.8 |
| | other (parameter-only elementwise) | 4 | 43 | 0 | 43 | 10.7 |
| Gemma-2B @8k | attention | 18 | 854 | 64 | 790 | 43.9 |

The DeepSeek draft matches Qwen2-0.5B to within 3%: same hidden size, layers and heads.

The LM head is not on the list: it runs at 92–99% of bandwidth on every model, including at 8k keys.

From these, five separate causes.

### 2.1 A fixed cost of about 4.5–6 µs per barrier interval

Every GEMM interval pays a cost that does not depend on its bytes. The fitted rates (`devices/h200-mig-3g71gb.yaml`) put it at:

- about 0.8 µs for the grid barrier and start skew;
- about 1.5 µs for the GEMM start: dispatch, loading the activation vector after the barrier, and filling the first stage;
- about 0.6 µs per tile;
- about 1.3 µs where a folded RMSNorm is recomputed on every CTA before the GEMM (the gate·up and q/k/v intervals).

Attention at short contexts costs about 5 µs per layer for almost no bytes.

On SmolLM-135M this is the whole story: 26 µs per layer against 3 µs of bytes, 151 intervals per token. The same cost is about 30% of Qwen2's kernel and 20% of SmolLM-1.7B's.

Decode is a dependency chain: every projection needs the whole vector the previous one produced. Five dependent stages per layer cannot be overlapped away, but each stage's fixed cost can shrink, and some stages can merge.

### 2.2 Narrow projections leave most SMs idle

A swapAB tile is 64 output rows (WGMMA's minimum M), so a projection with N outputs has N/64 work items:

| Model | Hidden | Items in q/o/down | Busy SMs of 60 | What 60 SMs would read |
|---|---:|---:|---:|---|
| SmolLM-135M | 576 | 9 | 9 | about 3.3× faster |
| Qwen2-0.5B, DeepSeek | 896 | 14 | 14 | about 2.3× faster |
| SmolLM-1.7B, Gemma-2B | 2,048 | 32 | 32 | about 1.2× faster |

14 SMs reach about 1.0 TB/s of the slice's 2.31. Qwen2's down projection (8.7 MB) takes 10.8 µs on 14 SMs; at the full rate it is 3.8 µs plus the fixed cost.

Computed from bytes and busy-CTA counts, idle SMs (this cause and the tail waves of 2.3 together) cost about 320 µs per token on Qwen2 and DeepSeek (27% of the kernel), about 140 µs on Gemma-2B and about 100 µs on SmolLM-1.7B. On SmolLM-135M the GEMMs are so small that the fixed cost of 2.1 dominates even with idle SMs (about 75 µs).

On a full 132-SM GPU the same projections would use 7–24% of the SMs, so this cause gets worse there.

### 2.3 Tail waves

When the item count is just above the SM count, a second, mostly empty round runs:

- Qwen2's and DeepSeek's gate·up: 76 paired tiles on 60 SMs, so 16 in a second round. The interval reads at 55% of bandwidth.
- Beam's o_proj: 80 tiles, 20 in the second round. Its q/k/v interval: 80 + 16 + 16 = 112, so 52 in the second.
- Gemma-2B's q/k/v: 32 + 4 + 4 = 40 items, one partly filled round.

Splitting K doesn't fix these: Llama-1B's and SmolLM-1.7B's 32-tile GEMMs split two ways give 64 items, a second round of 4. The model predicts split-2 slower for SmolLM-1.7B's and Gemma-2B's down projections (28.5 against 21.0 µs, 50.6 against 39.0 µs). Balancing K iterations exactly over the SMs (Stream-K) is the general fix.

### 2.4 Long-context attention is the slowest body

At 8k keys the attention body reads K and V at a fraction of bandwidth:

| Model | Attention @8k µs | Its bytes µs | % of bandwidth | What limits it |
|---|---:|---:|---:|---|
| SmolLM-1.7B (32 × 64, MHA) | 1,821 | 681 | 37% | About 14 GB/s per CTA against a ~38 GB/s share. The body issues too few loads (about 220 registers per thread, no staging) |
| Gemma-2B (8 query heads over 1 KV head, head_dim 256) | 854 | 64 | 7% | One head per row at head_dim 256 (`DECODE_PACKED_HEAD_DIM`), so each of 8 heads reads every key; 170 ns per key at head_dim 256 |
| Qwen2 / DeepSeek (14 over 2, head_dim 64) | 579 | 43 | 7% | Fixed per-item and exchange cost; few items |
| SmolLM-135M (9 over 3) | 477 | 80 | 17% | As above |

Inductor's attention is cuBLAS batched matmuls plus a fused softmax. On MHA (SmolLM-1.7B) that adds about 850 µs from 16 to 8k keys, against our 1,700 µs. This is the whole SmolLM-1.7B loss at 8k. On grouped-head models Inductor copies K and V once per query head (`clone_expand` kernels, about 390 µs per token at 8k), which is why we keep our lead there.

### 2.5 Work that depends only on parameters

Gemma-2B spends 43 µs per token in four intervals of elementwise phases whose inputs are weights only: the `1 + weight` of its 37 RMSNorm weights (cast and add), and the embedding scale. They give the same result on every call.

### 2.6 What is not the problem

- **The LM head:** 92–99% of bandwidth.
- **GEMM inner loops:** in intervals with all 60 SMs busy and one round (SmolLM-1.7B's and Gemma's gate·up, Beam's gate·up), MegaBake reads at 78–92% of bandwidth (phase timer: SmolLM-1.7B 78%, Gemma-2B 86%, Beam 92%).
- **Against cuBLAS:** our weight-streaming intervals together take 2,747 µs on Gemma-2B at 16 keys against 2,764 µs for Inductor's cuBLAS GEMMs, and 2,011 against 2,143 µs on SmolLM-1.7B. Our intervals include the folded norms, RoPE and residuals, so the GEMMs themselves are slightly faster than cuBLAS, not slower. That is also why GEMM-heavy models gain little: what MegaBake removes is Inductor's other kernels and launch gaps (about 10–20% of Inductor's step), not GEMM time.

## 3. What to do, by expected gain

Gains are estimates on the 60-SM slice, from the measured interval times and the cost model. None is built.

| # | Change | Fixes | Expected gain (µs per token) | Models it moves most |
|---:|---|---|---|---|
| 1 | **Attention K/V through the producer ring** (TMA bulk loads into shared memory, as GEMM weights already are), then a lighter consumer with fewer registers. Target 70% of bandwidth at long contexts | 2.4 | SmolLM-1.7B @8k: −800 to −1,000 (0.93× → about 1.2×) | SmolLM-1.7B, every model at long contexts |
| 2 | **Pack query heads at head_dim 256**, reading each key once for the KV group. Needs (1), or per-role register budgets (roadmap R1) | 2.4 | Gemma-2B @8k: −600 to −700 (1.06× → about 1.3×) | Gemma, any MQA/GQA model with wide heads |
| 3 | **Fill the SMs on narrow projections.** Either a CUDA-core GEMV body with 8–16 output rows per item (roadmap X2.1), or split-K that works with the producer ring. Today one split GEMM turns the producer off for the whole kernel, which is why the planner never splits; the model gives Qwen2's down projection 11.5 → 8.2 µs at split 4 | 2.2 | Qwen2/DeepSeek: −200 to −300 (1.56× → about 1.9×); SmolLM-135M: −50 to −100 | Hidden sizes ≤ 1,024; all models on 132 SMs |
| 4 | **Stream-K for tail waves**: split the K iterations evenly over the SMs instead of whole tiles | 2.3 | Qwen2/DeepSeek gate·up: −80 to −100; Beam: −60 to −100; Gemma q/k/v: −30 to −60 | Qwen2, DeepSeek, Beam, Gemma |
| 5 | **Cut the per-interval fixed cost.** (a) RMSNorm across the GEMM boundary: the producing GEMM's epilogue writes sum-of-squares partials, so the consumer does not recompute the norm on every CTA (X3.1; about 1.3 µs × 2 per layer). (b) o_proj consumes attention per head through counters, one interval fewer per layer (X3.3). (c) Load the next interval's activation vector as soon as it is published instead of after the barrier | 2.1 | SmolLM-135M: −150 to −250; Qwen2/DeepSeek: −80 to −120; all models some | Small models |
| 6 | **Evaluate parameter-only subgraphs once at bake time** (`transform_params`, architecture §9 item 4) | 2.5 | Gemma-2B: −40 | Gemma family, any model with `1 + weight` norms |

What these add up to, roughly: SmolLM-1.7B at 8k from 0.93× to about 1.2×; Gemma-2B from 1.06–1.12× to about 1.3×; Qwen2 and DeepSeek from about 1.6× to about 2×; SmolLM-135M from 1.8× to about 2.3×. Beam stays near 1.1–1.15×, because its ceiling is 1.22×.

## 4. Suggested order

1. **(1) and (2) together.** They are one piece of work, attention through the producer, and they remove the only loss to Inductor (SmolLM-1.7B at 8k) and most of Gemma's gap.
2. **(3).** It is the largest item for the small and mid models, and it becomes the largest for every model on a full GPU.
3. **(5a) and (6).** Small, independent and low-risk.
4. **(4) and (5b).** More invasive; measure (3) first, because it changes the tile counts these depend on.

Each change should be A/B tested in one process against the inert-code control (the roadmap's rule), with greedy-text checks on two real checkpoints.

## 5. Caveats

- **One GPU, a 60-SM MIG slice.** On 132 SMs the narrow-projection loss (2.2) is larger and the per-CTA share of bandwidth differs.
- **Inductor runs with default GEMM autotuning off.** Inductor disables max-autotune GEMMs below 68 SMs, so cuBLAS picks its own kernels. On a full GPU Inductor may be somewhat faster.
- **The bytes-based split of losses is a model.** It agrees with the measured intervals to about 7% in total (new-arch.md §6.4), not exactly.
