# MegaBake roadmap

**Updated 2026-11-07 (decode attention generalized and sped up, more of step 11; see the section below and `agent_space/research_notes/landscape_2026-11/landscape_and_experiments.md` §9); earlier: 2026-10-27 (flash-decoding attention); steps 1-2b measured in `artifacts/step2b_2026-10-21` (steps 1-2 in `artifacts/step12b_2026-10-21`), step 3 in `artifacts/step3_2026-10-21`. This file says what to do next. The analysis behind it is the 2026-10-18 megakernel gap analysis; the 2026-10-20/21 measurements below supersede its estimates where they disagree. Steps R0–R3 come from the 2026-10-22 HazyResearch comparison. The external Llama-3.2-1B comparison (2026-10-23) follows step 3.**

Where we are, in three lines:

- Against a real CUDA-graphed Inductor decode, MegaBake wins on 6 of 8 models when calls are pipelined (5–55%), and roughly ties when each call synchronizes. The old "1.2–5x" wins came from host syncs in the ungraphed baseline.
- Decode is bound by fixed per-phase latency, not by grid barriers: barrier release is ~11% of the 135M kernel. Gemma 3 (1.3x) and Qwen3.5 (1.9x) decode still lose to graphed Inductor. Prefill calls lose on every model except DeepSeek draft (a tie).
- Cross-GEMM fusion is built (step 1): residual adds and SiLU·mul run in GEMM epilogues, gate/up as one paired GEMM, RoPE inside decode attention.
- Decode attention is one body for every softmax decode variant (2026-11-07): MHA/GQA/MQA, head_dim any multiple of 8 up to 256, a score scale, sliding windows and soft caps are parameters. Coalesced score loads and an exchange/split rework made decode 2–37% faster on 7 checkpoints (attention 8–64%), held-out models included. Before that (2026-10-27): flash decoding with one statistics exchange per chunk and KV-group rows at long contexts.
- Weight streaming into shared memory across phases is built and off (step 3): the preloads themselves change decode by −0.7% to +1.5%. Weights still stream only inside GEMM phases by default.
- HazyResearch's megakernels suggest a different reading of what step 3, L2 prefetch and the events estimate each measured. Cross-op overlap needs three things together: fixed warp roles with a producer warp that never joins consumer syncs, shared-memory pages handed from op to op, and fine-grained counters. Megabake has tried each piece alone. The next runtime experiment is the role split (R1), not another streaming variant.
- On an external H100 benchmark (Llama-3.2-1B, 2026-10-23), MegaBake runs 1.8x slower than both hand-written CUDA graphs + PDL and Hazy's megakernel. The causes are per-call host time, decode attention that splits short contexts across CTAs (~330 µs of a ~1,850 µs kernel at position 100), GEMMs whose tile counts do not fill the SMs, and barrier intervals that carry only RoPE. The LM head runs near peak. Graphed Inductor on the 60-SM slice hid all of it, and a wrong RoPE in the test reference hid that the Llama reference itself produced gibberish.

## Done: more of step 11, decode attention generalized (2026-11-07)

Measured on the 60-SM H200 slice; artifacts in `artifacts/x2a_2026-11-07/`; method and every number in `agent_space/research_notes/landscape_2026-11/landscape_and_experiments.md` §9 (X2a).

- [x] **Coalesced scores.** One thread per key loaded its row as head_dim scalar 2-byte loads with lanes a KV row apart (5.25 of 7.4 µs per item on Llama-1B, found with per-stage `globaltimer` stamps, `tools/attention_stages.py`). Now `head_dim/8` lanes per key take one 16-byte load each and sum with butterfly shuffles; 4 keys per lane in flight; value unroll 16.
- [x] **Merge loads unrolled to the compile-time chunk bound.** A loop to the run-time chunk count issued the coherent L2 loads one at a time (~2 µs per exchange). The new key's score is computed by warp 0, not thread 0.
- [x] **Split cost model** (`decode_split_steps`) replaces `DECODE_MIN_CHUNK_KEYS` (64), `DECODE_SECOND_WAVE_KEYS` (512) and the 0.15 pass factor. It never leaves a chunk empty: it picks c chunks only past (c−1)² keys (an empty trailing chunk counted the new key twice; found by a 600-token Gemma-3 decode, fixed before merge; a device-side recount instead cost ~0.8 µs per layer). Rates are measured on this slice (`shortcut:` in the code) and need a per-body calibration.
- [x] **One body, more variants.** `kv_attention` takes `scale`, `window` and `softcap`; `DecodeStep` passes HF's scaling (no longer folded into q), the interface's `sliding_window` argument and the soft cap. Head dims 80, 96 and 112-style multiples of 8 now take the split body instead of the one-row-per-warp body. Sliding windows no longer cap the KV capacity (Gemma-3 decodes past 512 keys). `tools/attention_variants.py` checks 10 configurations (Llama GQA/MHA/MQA, head_dim 64/80/96/128/256, Qwen3 q/k-norm, Mistral window, Gemma-2 soft cap + window, Gemma-3 mixed layers) at every position of an 800-token decode against HF's own fp32 forward: all pass, and dropping the window, the soft cap or the empty-chunk fix fails it (50–280× HF bf16's error).
- [x] **Bug fixed: Mistral-style sliding windows were ignored.** transformers 5.x passes the window as an attention-interface argument, not a module attribute; `DecodeStep` ran full attention past the window.
- [x] **Results.** Kernel faster in all 20 (model, context) cells, −2.4% to −37.4%; attention −8% to −64%; Llama-1B at 113 keys 1,685 → 1,581 µs, SmolLM-135M 1,074 → 920 µs, SmolLM-1.7B at 4,013 keys 5,274 → ~3,300 µs. Greedy tokens on 7 checkpoints are unchanged against HEAD.
- [ ] SmolLM-1.7B at 4,013 keys runs +1.8% against the old fitted split with the same 3 chunks: a codegen effect in the value sum, open.
- [ ] **q/k-norm models lose the RoPE fusion** (table below): the norm is a reduction, and the prologue takes elementwise producers only. Top next item for model breadth.
- [ ] Still open from step 11: the 8,192-key cap (FP32 score row in scratch), the packing thresholds, sinks, K/V of different widths (MLA, via weight absorption), per-body calibration of the split rates. Long contexts stay at 20–30% of HBM bandwidth: the body is load-issue and register bound (220 registers per thread), so the next gain needs a producer warp staging K/V (R1/R2).

## Done: part of step 11, flash-decoding attention (2026-10-27)

The split-context decode body (`bodies/attention.py`) keeps the captured rounding points (BF16 scores, FP32 statistics, BF16 normalized probabilities, FP32 value sum) and changes how work is split:

- [x] **One statistics exchange per chunk instead of two.** Each chunk publishes its max and its exponential sum against that max; the row denominator follows from the chunk statistics. Probabilities are normalized once per key into scratch instead of in every lane of the value sum.
- [x] **Rows that pack a KV group.** A row is one query head at short contexts. From `DECODE_PACKED_KEYS_PER_HEAD` (128) visible keys per packed head it is the query heads of a KV group that fit `DECODE_PACKED_FLOATS` (512) query elements, so each cached key and value is read once for all of them. Both variants are in one body and every CTA takes the same one, because all read the same valid length. Packing a whole group at short contexts was measured slower (Qwen2-0.5B +11%, gemma-2b +24%): too few items to fill the SMs.
- [x] **Single-head rows keep their query in registers** across the key loop (as before). Grouped rows reload theirs from shared scratch per key; a key pass per head measured 3x slower on Qwen2-0.5B.
- [x] **Correctness.** Bit-exact with one chunk (existing table-launch test at capacity 4); new `test_split_context_decode_matches_eager_at_long_contexts` runs both variants at 0–1,000 keys against fp32. On all 7 checkpoints at 0–4,000 keys with random caches, error against fp32 equals the old body's at 33 of 40 points, is within 9% at the others, and argmax never changes. The phase-scratch fallback for scores was run forced.
- [ ] **head_dim 256 keeps one head per row** (`DECODE_PACKED_HEAD_DIM`). Packing gemma-2b took 4,096-key decode from −0.6% to −17.6%, but a second variant pushed Qwen3.5-0.8B's register-saturated kernel into more spills (+1.4% at every context). Revisit with per-role register budgets (R1).
- [ ] Still open from step 11: the online softmax (removes the FP32 score row and the 8,192-key cap), and a cost model for the 64-key chunks, the one-wave cap and the packing thresholds. *2026-11-07: the cost model is built (section above); the online softmax was not needed for speed.*

Pipelined decode, 60-SM H200 slice, MegaBake old → new (graphed calls move the same way; generated prefill kernels are byte-identical):

| Model (KV group × head_dim) | 128-key cache | 512 keys | 2,048 keys | 4,096 keys |
| --- | ---: | ---: | ---: | ---: |
| SmolLM-135M (3 × 64) | 1,137 → 1,066 (−6.3%) | 1,299 → 1,200 (−7.6%) | 1,662 → 1,304 (−21.5%) | 2,110 → 1,492 (−29.3%) |
| SmolLM-1.7B (1 × 64) | 2,419 → 2,378 (−1.7%) | 2,982 → 2,903 (−2.6%) | 4,149 → 3,945 (−4.9%) | 5,686 → 5,323 (−6.4%) |
| Qwen2-0.5B (7 × 64) | 1,358 → 1,338 (−1.5%) | 1,523 → 1,478 (−2.9%) | 2,002 → 1,706 (−14.8%) | 2,620 → 1,883 (−28.1%) |
| DeepSeek-V3 draft 0.6B (7 × 64) | 1,383 → 1,369 (−1.1%) | 1,555 → 1,499 (−3.6%) | 2,049 → 1,759 (−14.2%) | 2,672 → 1,946 (−27.2%) |
| gemma-2b (8 × 256, one head per row) | 3,441 → 3,318 (−3.6%) | 3,514 → 3,518 (+0.1%) | 4,132 → 4,126 (−0.1%) | 4,886 → 4,857 (−0.6%) |
| Gemma-3-270M (4 × 256, one head per row) | 1,801 → 1,765 (−2.0%) | 1,883 → 1,848 (−1.8%) at 400 keys | — (512-key window) | — |
| Qwen3.5-0.8B (4 × 256, one head per row; 2 interleaved rounds) | 4,622 → 4,595 (−0.6%) | 4,672 → 4,635 (−0.8%) | — | 5,125 → 5,086 (−0.8%) |

The gemma rows are one round and Gemma 3 cannot cache past its sliding window. gemma-2b's +0.1% at 512 keys is within run-to-run spread (its graphed call there is −2.9%).

On the reproduction workload of PR #8 (SmolLM-135M random checkpoint, prefill 128, continuous greedy decode, SGLang 0.5.22 with CUDA graphs on the same GPU), MegaBake goes from 1.195 / 1.269 / 1.630 to 1.093 / 1.142 / 1.257 ms per token at 128 / 1,024 / 4,096 output tokens; SGLang runs 0.992 / 1.028 / 1.053. The remaining gap is mostly attention, which grows with context faster than SGLang's FlashAttention-3 decode, and two more GEMM phases per layer (separate Q/K/V).

## Done: step 0, honest baselines (2026-10-20)

- [x] **Graphed decode baseline.** `benchmarks/full_model.py --variants inductor_graphed` captures one Inductor decode step in a CUDA graph by hand; during capture `kv_attention` runs a static-capacity version that is bit-identical to the reference.
- [x] **Per-CTA stamps.** `tools/phase_profile.py --cta` splits every barrier interval into the slowest CTA's work, straggler wait and barrier release.
- [x] **Per-phase fixed latency cut.** CTAs skip phases they have no tiles in, slot addresses resolve in one load, swapAB GEMMs store from registers, and single-chunk decode attention skips its statistics exchange: decode kernel −1–15% across models.
- [x] Re-ran all 8 models (`artifacts/full_model_2026-10-21`).
- [ ] Inductor `max-autotune` variant (on the 60-SM slice Inductor disables GEMM autotuning; `is_big_gpu` needs 68 SMs).
- [ ] Graphed baseline for `--decode-contexts` and bucketed prefill.
- [ ] External baseline: run the [megakernels-vs-cuda-graphs](https://github.com/msaroufim/megakernels-vs-cuda-graphs) Llama protocol on a full-SM H100/H200 (the Hazy arm rejects partitioned GPUs, so the 60-SM slice cannot run it). The checkpoint is now available; the protocol still needs KV capacity 16,384 (step 11). Report percent of read bandwidth alongside speedups.

## Done: step 1, cross-GEMM fusion (fusion built; 0.9 ms target moved to step 3, then step 10)

A SmolLM-135M decoder layer ran 15 phases in 6 barrier intervals; it now runs 9 phases in 5 intervals (7 serial steps, q/k/v run in parallel), and the kernel 278 phases instead of 603:

```text
before: [rope q, rope k] | attention | o_proj | [add(R), rmsnorm(R), gate, up] | [silu·mul(R), down] | [add(R), rmsnorm(R), q, k, v]
now:    attention(+rope) | o_proj(+residual) | [rmsnorm(R), gate·up(+silu·mul)] | down(+residual) | [rmsnorm(R), q, k, v]
```

- [x] **GEMM epilogue expressions.** A GEMM fuses with the elementwise chain that consumes its result (`regions._gemm_epilogue_candidates`); the chain becomes the contraction's `post`, generated into a function the swapAB direct-store epilogue evaluates per element (`bodies/gemm.py` `epilogue_source`). GEMMs of one shape share one kernel body and pick their epilogue at run time (`gemm_table` epilogue-kind column), with the epilogue's tensors built after the mainloop.
- [x] **Paired GEMM.** Two projections of one activation share a 128-row tile: TMA loads 64 rows of each weight into the halves of the A stage from two descriptors, so the chain (SiLU·mul, GELU·mul) reads both accumulators of a row in one thread. No derived weight copies.
- [x] **RoPE in decode attention.** The split-context decode body evaluates the query's and new key's elementwise producer for its row into shared scratch and appends that key to the cache (`regions._attention_prologue_candidates`).
- [x] **Common-subexpression elimination** on the captured graph (`frontend.eliminate_common_subexpressions`): per-layer recomputations (the fixture's RoPE tables and causal mask) run once.
- [x] Bit-exact: the 2-layer table-launch tests compare logits and KV state with eager at zero tolerance.
- [ ] **135M decode kernel ≤ 0.9 ms: not met** (1,282 → 1,059–1,137 µs across runs). What remains is latency in four small GEMM intervals per layer (~3–4 µs each, mostly pipeline fill: an o_proj tile moves ~90 KB) and attention (~6 µs on 9 CTAs). Split-K was re-measured and is still slower (1,141 vs 1,075 µs). The target moved to step 3, which attacked the GEMM fill latency and did not meet it either (below).

## Done: step 2 and 2b, prefill regressions and stable GEMM code generation

Both regressions came from one compiler behaviour: ptxas warning **C7519**, "wgmma.mma_async instructions are serialized due to program dependence on compiler-inserted WG.AR in divergent path". When it fires, every tensor-core instruction of the megakernel waits for the previous one (SmolLM-1.7B prefill: +43% SASS instructions, mostly R2UR and WARPGROUP fences, +20% time). It fires when any trap in the kernel sits behind a per-lane condition (2b below). Check a kernel with `tools/wgmma_serialization.py`.

- [x] The 1.7B regression from the fused o_proj epilogue was not the fusion: one more inlined GEMM variant alone caused it (2,455 → 3,051 µs with the o_proj GEMM unfused but in its own variant). Fixed by sharing one body per GEMM shape (2,460 µs).
- [x] The direct register store's per-element branch triggered C7519 on gemma-2b; plain GEMMs now store with one predicated instruction (gemma-2b prefill 3,755 → 3,164 µs, below 2026-10-18).
- [x] **Root cause (2b): traps behind per-lane conditions.** The gather bodies' index bounds checks ran `llvm.intr_trap()` under each lane's own condition. ptxas then cannot assume that whole warps reach a later wgmma, so it serializes every wgmma of the kernel. It isn't register allocation: what toggled with unrelated changes was only whether ptxas could see the divergence. Every trap now goes through `runtime.trap_if`, which votes across the warp first (same semantics: the kernel traps if any lane is out of range). Found by compiling dumped PTX directly with a CUDA 13.4 ptxas (~1 s per try; `CUTE_DSL_KEEP=ptx`) and narrowing to the embedding gather's bounds check.
- [x] `tools/wgmma_serialization.py` clean on all 15 compilable kernels (8 models decode, 7 prefill; Qwen3.5 prefill needs step 5). Prefill kernel (measured): SmolLM-1.7B 3,043 → 2,455 µs (−19%), Qwen2-0.5B 1,710 → 1,366 (−20%), DeepSeek draft 1,709 → 1,345 (−21%), each now below 2026-10-18. Other models flat.
- [x] **Tried and reverted: one GEMM body per configuration** (problem shape read from the GEMM row at run time instead of one body per shape). It did nothing for C7519 or prefill (1.7B prefill span identical) and made decode 2–4% slower in a same-session A/B (135M 1,031 → 1,066 µs): the in-kernel shape rebuild moves the whole kernel's register allocation (174/244 → 128) plus ~0.1–0.25 µs per GEMM phase. The old kernel capped at 128 registers reproduces the 1.7B loss exactly, and makes 135M *faster* (1,008 µs), so check step 3 gains against a capped baseline too. The measured table below was taken with it in, so its decode numbers are 2–4% pessimistic.

## Done: step 3, weight streaming into shared memory (built, off: no runtime gain)

A GEMM's weights do not depend on the phases before it. At the grid barrier that closes an interval, each CTA arrives, then warp 0 loads the weight (CUTLASS A of a swapAB GEMM) of its first tile in the next interval into the GEMM's pipeline stages, as many leading K tiles as the stages hold. The GEMM phase finds a tag in shared memory, skips its mbarrier init and loads only the activations for those K tiles. ARCHITECTURE §5 has the mechanism. `MEGABAKE_WEIGHT_STREAM_STAGES` caps the streamed stages; it is 0 (off) by default.

- [x] Built for every swapAB table GEMM whose weight no region writes, paired GEMMs included. Bit-exact: with streaming on, the 2-layer table-launch decode matches eager at zero tolerance (`test_table_launch_streams_weights_in_stateful_decode`). GPU tests and `tools/wgmma_serialization.py` (all 15 kernels) are unchanged with it on. Off, the kernel's PTX is identical to the kernel before step 3 (checked on the 2-layer fixture and DeepSeek-draft decode), so the measured table below still holds.
- [x] **Measured: the preloads do not pay.** The A/B that decides is against the same kernel with the streaming code compiled in but a zero stream table, so no CTA preloads (`dead_stream*/`, 2 interleaved rounds, decode kernel):

  | Model | Off | Streaming on | Code in, no preloads | Preloads alone |
  | --- | ---: | ---: | ---: | ---: |
  | SmolLM-135M | 1,163 | 1,078 | 1,084 | −0.5% |
  | SmolLM-1.7B | 2,650 | 2,606 | 2,624 | −0.7% |
  | Qwen2-0.5B | 1,352 | 1,374 | 1,360 | +0.9% |
  | DeepSeek-V3 draft | 1,373 | 1,419 | 1,398 | +1.5% |

  Against "off" alone, streaming looked like −7% on 135M (`ab*/`, `sweep/`). That is code shape: the kernel's registers went 174 → 239, and that alone makes 135M 6.8% faster. Gemma-2b, Gemma-3-270M, Beam and Qwen3.5 are flat with streaming on (`ab_models/`).
- [x] **Why: issuing blocks.** Once the SM's TMA queue is full, each further load blocks warp 0 until one drains. Issuing 13 stages takes a median of ~1,000 and a p90 of ~4,000 cycles (clock64). So a CTA waits for most of its own preload before it leaves the barrier, and the loads overlap only barrier wait it would have spent anyway. The per-CTA profile shows it: slowest-CTA work falls (135M 809 → 667 µs per call), but each preload-issuing interval's span grows ~1.5 µs (135M) to ~2 µs (1.7B), because the CTA that arrives last pays the issue.
- [x] **Tried, no gain:** any stage count (2, 4, 8, all); L1-prefetching the stream-table entry; `prefetch.tensormap` of the target descriptor (the first TMA costs only ~160 cycles); a two-interval lookahead for CTAs without tiles in the next interval; adaptive issue (two stages per barrier poll, so the last CTA issues at most two).
- [ ] **135M ≤ 0.9 ms and 1.7B ≤ 2.2 ms: not met** (now ~1,160 and ~2,650 µs off). At peak bandwidth 1.7B decode needs ~1.55 ms. gate·up already runs at ~90% of peak and the LM head at peak. The loss is in the 32-tile projections (q/k/v/o/down on 32 of 60 SMs, 70–72% of peak), attention (~4.7 µs × 24) and barrier release (~1.3 µs × 121). The targets move to steps 7 and 10.
- [ ] Streaming can pay only with a producer warp that the CTA's other work does not wait for. Today every body synchronizes all 256 threads. That needs generic and attention bodies on 7 warps, with named barriers. Now step R1.

## Done: external baseline, Llama-3.2-1B (2026-10-23)

A beta tester ran MegaBake (`DecodeStep`) against [megakernels-vs-cuda-graphs](https://github.com/msaroufim/megakernels-vs-cuda-graphs) on a full H100 80GB:

| Arm | µs/token |
|---|---:|
| Hand-written CUDA GEMVs + Triton attention, CUDA graphs + PDL | 980 |
| HazyResearch Llama-1B megakernel | 966 |
| MegaBake | 1,784 |

Their protocol uses a 32-token prompt, 127 decode steps captured in one CUDA graph with on-device argmax, and KV capacity 16,384, so the timed steps attend over 32–159 keys. We reran it on the 60-SM slice with the real checkpoint (2026-10-23); there was no H100 run, so the H100 attribution below is inferred. On the slice, the `DecodeStep` kernel takes 1,602 µs at KV position 10, 1,632 at 45, 1,853 at 95 and 1,893 at 205; pipelined calls take ~1,790 µs/token against 2,255 for graphed Inductor.

- [x] **The Llama reference wrapper was wrong.** `tests/fixtures.py:rotate_half` had its second half as `x2·sin + x1·cos` instead of `x2·cos + x1·sin`. `StatefulSmolLM` (and every fixture wrapper) produced gibberish on the real checkpoint, and MegaBake matched it because tests compare against the same wrapper. Fixed; MegaBake now matches HF `generate` for 64 greedy tokens (max logit difference 0.28). Random weights could not show this: check real-checkpoint greedy text against HF `generate`, not only against our own reference.
- [x] **Capacity 16,384 does not compile.** Decode attention keeps one FP32 score row in ≤ 32 KB of scratch, which caps KV capacity at 8,192 (`bodies/cuda/sm90/attention.py`, `ATTENTION_SCRATCH_BYTES`). `fallback=inductor` runs 5.8 ms/token, so the tester measured the real kernel at a smaller capacity. Capacity does not change kernel time, because attention reads only the valid length; context length does (below).
- [x] **Host time: ~630 µs per call** with the GPU busy, vs 63 µs for a graph replay. Dynamo guards take ~200–300 µs. MegaBake's own call path takes ~300 µs: the `compiler.py` forward input checks, table rebind, argument packing and the CuTe DSL launch. About 430 µs of it lands before the launch, so it is exposed whenever calls synchronize. Both baselines pay ~0 per token.
- [x] **Host time removed for graphed decode** (2026-10-24). `megabake.graphed(compiled, *args)` captures a table-launch call in a CUDA graph: the kernel accepts a ring stamped −1 whose addresses fit every replay, and each replay advances the host's call-counter mirror. SmolLM-135M decode on the 60-SM slice: synced call 1,777 → 1,096 µs (kernel ~1,140 µs), pipelined 1,142 → 1,107 µs; graphed Inductor 2,087 µs synced. Outputs are the captured tensors, overwritten by the next replay.
- [x] **Kernel ~1,850 µs at position 100 vs a ~1,070 µs read roofline** (2.47 GB of weights at the ~2.3 TB/s the LM head reaches on this slice): ~58% of peak. Per-CTA profile (`phase_profile.py --cta` logic, real checkpoint, `DecodeStep`), one layer:

  | Interval | µs | At peak | Busy CTAs | Loss × 16 layers |
  |---|---:|---:|---:|---:|
  | decode attention | 21.2 (8.4 at position 40) | ~0.1 | 56 of 60 | ~330 µs |
  | norm + gate·up (128 tiles) | 38.2 | 29 | 60 | ~150 µs |
  | down (32 tiles) | 21.1 | 14.6 | 32 | ~105 µs |
  | norm + q/k/v (32+8+8 tiles) | 11.7 | 5.5 | 48 | ~100 µs |
  | elementwise (2 tiles) + RoPE (20 tiles), two intervals | 3.3 + 4.0 | 0 | 2–3 | ~115 µs |
  | o_proj (32 tiles) | 7.9 | 3.7 | 32 | ~70 µs |
  | LM head (2,004 tiles) | 232 | ~228 | 60 | ~0 |

  Straggler wait is 460 µs of the 1,851 µs span and barrier release 105 µs. `StatefulSmolLM` has one small interval instead of two (RoPE stays outside attention for Llama with both wrappers, although step 1 fused it on SmolLM).
- [x] **The LM head is not the problem.** It runs at ~94% of the slice's bandwidth. The earlier "~470 µs at ~1.1 TB/s" subtracted a profile taken near position 10 from a benchmark kernel measured near position 90; the difference was attention, not the head.
- [x] **Decode attention grows with context through its split, not its bytes.** Each CTA takes 64 keys, so from 65 keys a row splits over 2 CTAs and from 129 over 3. Each split row runs three GPU-scope exchanges (max, sum, output) with spinning `atomic_cas` waits, and 32 heads × 2–3 chunks is 64–96 items on 60 CTAs, so some CTAs take two items serially. All of it moves ~200 KB of KV per layer. The baseline's Triton kernel runs one program per (query head, half of head_dim), reads the whole context in one pass and needs no cross-CTA reduction. On 132 SMs the items fit one wave, so the H100 pays the exchanges but not the second wave.
- [x] **Short contexts no longer split into a second wave** (2026-10-23). Below 512 keys, decode attention keeps its chunks in use to one wave of items (`DECODE_SECOND_WAVE_KEYS`); the old rule is unchanged above that and for models whose chunks already fit one wave. Pipelined decode, 3 interleaved rounds: Llama-1B −11.3% at 100 keys, −4.7% at 400; SmolLM-1.7B −9.0% and −3.4%; within ±1.5% (all ≤ 0) at 16, 500, 600 and 2,000 keys, and the default decode kernel unchanged. The one-wave crossover is ~700 keys on both. `wgmma_serialization.py` clean on both decodes. Step 11's split along KV heads and the online softmax are still open.
- [x] **Why the gap grows on H100.** Tile counts follow the weight shapes, not the GPU, so q/k/v, o and down keep at most 48 of 132 SMs busy. Hazy reaches ~77% of H100 bandwidth; MegaBake's 1,784 µs is ~41%. Split-K (`MEGABAKE_SPLIT_K=1`) is also slower on the slice (1,954 vs 1,850 µs).
- [x] **How the baseline gets there** (`llama/kernels/`):
  - One CUDA-core GEMV body serves every projection, with norm, RoPE, the KV write, the residual add and SiLU·mul fused per mode.
  - It launches one CTA per 2 output rows, so o_proj gets 1,024 CTAs.
  - Weights are repacked once for 16-byte loads.
  - Each kernel loads its weights into registers before `cudaGridDependencySynchronize()` (PDL), so the loads overlap the previous kernel.
  - Decode attention never splits the context at these lengths.
  - The whole 127-step loop is one graph replay.

**Lessons, as compiler rules rather than Llama code** (ARCHITECTURE §9):

1. A phase's work items should cover the SMs. Choose the row and K split from the shape and the SM count, not a fixed 64-row tile (step 7).
2. Select the body by shape class: a CUDA-core streaming matvec when M is a handful of rows, WGMMA otherwise (step 7).
3. Per-token host time is part of the product. MegaBake is one kernel, so it should be the easiest thing to capture in a graph (steps 6, 14).
4. Load weights before waiting on the producer. The baseline gets cross-op overlap from PDL with that one ordering rule; R1's producer warp is the same rule inside one kernel.
5. Measure against hand-tuned baselines on full-SM hardware and report percent of bandwidth (step 0). Graphs + PDL land within 1–5% of Hazy, so one launch alone is not the advantage. MegaBake's case is reaching that efficiency on any model without hand-writing it.
6. Split decode attention only when the bytes pay for the exchange. Split first along dimensions that need no reduction (KV heads, value columns); split keys only past thousands of keys, with the chunk count chosen so items ≤ workers. Use an online softmax (one pass, no score row in scratch), and let one CTA serve all query heads of a KV head so K and V are read once (step 11).
7. Every grid barrier must carry work. An interval with 2–20 tiles of elementwise work costs a full barrier: fold RoPE and the KV write into the QKV epilogue or the attention prologue for every wrapper, as the baseline's QKV GEMV does (steps 4, 11).
8. Profile at the protocol's operating point. Kernel time depends on context length (1,602 → 1,893 µs from position 10 to 205), so attribute time from one run at the benchmark's positions, never by subtracting runs taken at different positions.
9. Check real checkpoints end to end. A reference that is wrong in the same way as the compiled model passes every test; compare greedy text with HF `generate`.

## Next: runtime structure (R0–R3, from the 2026-10-22 HazyResearch comparison)

Gains are qualitative, not measured. The only measured bounds are: barrier release ~11% of the 135M span; four GEMM intervals per layer at ~3–4 µs, mostly pipeline fill; and Hazy's 10.2 → 3.4 µs inter-instruction gap from cross-op pipelining. Every A/B runs against the inert-code control.

| # | Step | Addresses | Effort | Done when |
|---:|---|---|---|---|
| R0 | Plumbing and oracle. Replace the positional phase-table columns with a typed `PhaseRecord` (one dataclass generates the host packer and device accessors, with a layout hash). Add per-phase, per-role timing slots behind a constexpr flag and an expected Gantt chart per model. Write the memory-model contract per sync site and type every constant | Positional-column brittleness, A/B noise, fence correctness | S–M | Timing slots compiled out leave the PTX identical; R1's profile can be checked against an expected chart |
| R1 | Fixed role layout for the table launch. One producer warp owns all TMA traffic (GEMM stages and weight preloads). Two consumer warpgroups run generic and attention bodies with named barriers instead of `sync_threads`. `setmaxregister` per role at kernel entry (re-enable `cutlass_gemm.py:268-275`), recorded in `ParticipantContract`. Every trap stays warp-uniform | Streaming blocked on the TMA queue; whole-kernel register coupling | L | Bit-exact; `wgmma_serialization.py` clean; weight preloads issued by the producer warp beat the inert-code control |
| R2 | Persistent GEMM pipeline state across phases: ring index and phase bits live for the whole kernel, with no per-phase mbarrier init (retire `_without_mbarrier_init` and `STREAM_TAG_SLOT`). `ResourcePlan` page pool with compiler-derived acquire/release order per CTA; per-GEMM tile configs follow (feeds step 7) | GEMM fill and drain every phase; one layout per launch | M–L | GEMM intervals on 135M decode drop measurably below ~3–4 µs; or, if a producer warp with persistent state does not hide the fill, record that as evidence against the whole family (streaming, events) |
| R3 | Bodies as `@cute.jit` role functions with hook slots, called from generated dispatch; retire `_nested_body_source` string surgery. Check inlining with `CUTE_DSL_KEEP=ptx` first | Dropped helpers, codegen-shape artifacts | M | No body source is rewritten as text |

R3 can start alongside R1 if converting bodies to roles through text replacement becomes painful. Step 10 (events) waits for R2's profile to show the producer warp stalling on a dependency that a finer counter would remove.

## Then: the ordered list

Effort: S small, M medium, L large. Gains are estimates unless marked measured.

| # | Step | Est. gain | Effort | Done when |
|---:|---|---|---|---|
| 4 | Qwen3.5 and Gemma 3 decode: profile with `phase_profile.py --cta`, remove generic-body fallbacks; semantic matching by scalar-expression normal form (norms, RoPE incl. GPT-J and partial rotary, activations); retire `StatefulSmolLM` | Closes the 1.3x / 2.2x gaps | M | Both within 1.1x of graphed Inductor in decode |
| 5 | Declared gated delta rule (chunked prefill: `cumsum`, triangular solve) | Coverage | M | Qwen3.5 prefill compiles |
| 6 | Host work per call: Dynamo guards ~200–300 µs and MegaBake binding and launch ~300 µs (~630 µs measured on Llama-1B). Make the launch CUDA-graph capturable: positions read on the device, no per-call rebind | Wins the synced comparison; ~430 µs per synced token | M | Synced decode call ≤ kernel + 150 µs; a decode step captures in a CUDA graph |
| 7 | `TargetFacts` + per-GEMM configs + cost model: query L2, bandwidth, SM count, capabilities and MIG. Each GEMM chooses tile, stages, swapAB and split-K, with work items ≥ SM count, and a CUDA-core matvec body when M is a few rows. Measure the top 3–5 candidates and cache them by device fingerprint; re-enable search for table launches (`compile_plan` forces `max_plans=1`). The 32-tile projections of 1.7B run at 70–72% of peak bandwidth on 32 of 60 SMs. Llama-1B's q/k/v/o/down use ≤ 48 of 132 H100 SMs (on the slice: down waits 9.2 µs per layer on stragglers); its LM head already runs at ~94% of peak | 1.7B −0.2–0.45 ms; Llama-1B toward the ~980 µs external baselines; removes the >64-row prefill cliff | M–L | No GEMM constant fitted to a model or the 60-SM slice remains; 1.7B decode kernel ≤ 2.2 ms; Llama-1B decode kernel ≥ 70% of read bandwidth |
| 8 | CI generality guard: a tiny-model zoo (LayerNorm, GELU, MQA, head_dim 256, sliding window, MoE, gated delta) against synthetic targets (60/108/132/148 SMs; sm80/90/100/120); fail on any silent generic-body fallback; keep held-out models | Prevents regressions | M | Runs in CI |
| 9 | Split-K for projections with fewer tiles than workers (built, off: no gain on the 60-SM slice) | Narrow projections | S | Re-measured after 1 and 7 |
| 10 | Event counters in the table launch, only on edges a running-ahead producer can use (GEMM→GEMM activation subsets, attention chunks, KV append with last-block-only wait); targets derived from tile overlap with RAW/WAR lists; `>=` waits on epoch targets | ≤ barrier release share (~11% measured) plus overlap R1–R2 unlock | M | Only after R1–R2, if the profile shows producer stalls a finer counter removes; 135M decode kernel ≤ 0.9 ms |
| 11 | Derive fold limits, attention chunking and tolerances from cost, dtype and depth. Decode attention: split along KV heads and value columns before keys; split keys only when the bytes pay for the exchange, with items ≤ workers; one CTA serves all query heads of a KV head; an online softmax over key tiles removes the FP32 score row and with it the 8,192-key cap. Fold RoPE and the KV write into the QKV epilogue or attention prologue for every wrapper, so no barrier interval carries only elementwise work. *Built 2026-10-27: one exchange per chunk and KV-group rows at long contexts (section above)* | Llama-1B: ~330 µs per token at position 100 from attention, ~115 µs from the RoPE intervals (measured on the slice); runs the external protocol | M | No fitted constant left in `schedule.py`/`attention.py`; KV capacity 16,384 compiles; Llama-1B attention interval ≤ 5 µs at 159 keys; no interval with < 8 tiles stands alone |
| 12 | Per-sequence KV lengths → paged KV → ragged mixed prefill/decode step | ~12–15x tokens/s at batch 16 | S → L | Batch 16 with different lengths |
| 13 | Prefill: tensor-core attention, >64-row GEMM path; optional Inductor fallback above a measured crossover | 3–21x at 128–300 tokens | L | Prefill ≤ Inductor + graphs at 128 tokens |
| 14 | Multi-step decode loop with on-device argmax (both external baselines time 127 steps as one graph replay); then speculative verify | −10–25% per token; 2–4x with speculation | L | — |
| 15 | Typed edge formats; opt-in fp8 weights | 1.7B about −0.9 ms, different numerics | M–L | — |
| 16 | In-cluster DSMEM split-K with a searched split factor | Narrow projections | M | — |
| 17 | Real SM100 provider (tcgen05, TMEM, CLC) and SM80 provider | Portability | L | — |

**Measured and rejected:**

- **Shared-memory weight streaming at the barrier before each GEMM interval** (step 3): −0.7% to +1.5% from the preloads themselves; issuing blocks warp 0 on the TMA queue. Built and off (`MEGABAKE_WEIGHT_STREAM_STAGES=0`).

- **L2 weight prefetch** of the next GEMM interval's weights (`cp.async.bulk.prefetch.L2`): 16–24% slower issued by every CTA, 8–10% slower issued only by idle CTAs (`artifacts/chunk3_2026-10-20`, `chunk4_2026-10-20`). Built and off (`MEGABAKE_L2_PREFETCH_BYTES=0`).
- **Events as the first runtime step.** The report estimated −17–30% on 135M; the per-CTA profile bounds the gain by barrier release, ~11%.

**Do not build:** a dynamic work queue for batch-1 decode (measured slower than static scheduling elsewhere; Hazy found it pays only for ~100 µs instructions), or a port of the Mirage superoptimizer (report §2). From the Hazy comparison:
- a generic on-GPU opcode interpreter (deleted from ThunderKittens main; it forces every op into one union register budget);
- hand-written page-release tables or literal barrier counts inside bodies;
- host-side counter resets per call (epoch ordinals already avoid them);
- exact-equality counter waits;
- sixteen thin consumer warps;
- green contexts for SM partitioning;
- agent-written bespoke megakernels in place of the compiler.

## What exists and what is missing

| Mechanism | State | Step |
|---|---|---|
| FX capture, declared attention ops, generic `DecodeStep`, size buckets, `fallback=inductor` | Done | — |
| Primitive families (views, pointwise, scatters, padding, 1-D conv, input writes) | Done; scans, triangular solve and multi-axis reductions missing | 5 |
| swapAB small-M GEMM (16/32/64), register direct store, bias epilogue | Done, one configuration per launch | 7 |
| Interval folding (recompute instead of barrier), idle-phase skipping | Done | 11 (limits) |
| Counting grid barrier (~0.7 µs), constant kernel arguments | Done | — |
| Split-context decode attention: coalesced scores, one statistics exchange, KV-group rows (head_dim ≤ 128), split cost model; any head_dim multiple of 8 ≤ 256, scale, sliding window, soft cap | Done (2026-10-27, 2026-11-07) | 11 (calibrated rates, 8,192-key cap, sinks, MLA) |
| Graphed decode baseline, per-CTA profile | Done | — |
| GEMM epilogue expressions, paired gate/up GEMM, one body per GEMM shape | Done (swapAB launches) | 1 |
| Common-subexpression elimination on the captured graph | Done | 1 |
| RoPE in decode attention | Done (split-context decode body) | 1 |
| RMSNorm as GEMM prologue, q/k-norm in attention, RoPE in prefill attention | Missing. Decode with q/k-norm is correct, but the norm and the RoPE after it run as two separate intervals per layer (Qwen3-0.6B: 56 intervals, 179 µs, 10.5% of the kernel; gemma-3-270m 55 µs; OLMo-2-1B 44 µs; Llama-style models 0). Needs a per-head reduction prologue in decode attention | 1, 4 |
| Tile/phase events | Built, not wired for full models | 10 |
| Deterministic split-K | Built, off | 9, 16 |
| L2 weight prefetch | Built, off (measured slower) | — |
| Shared-memory weight streaming across phases | Built, off (preloads measured −0.7% to +1.5%) | — |
| Device cost model, per-GEMM tuning, plan search for full models | Missing | 7 |
| Paged KV, per-sequence lengths, on-device token count | Missing | 12 |
| Fast prefill attention and >64-row GEMM | Missing | 13 |
| Multi-step loop, on-device argmax, speculative verify | Missing | 14 |
| fp16, fp8, non-SM90 performance providers | Missing | 15, 17 |
| MoE task family | Missing | not scheduled |

## Measured state (2026-10-21, steps 1-2b)

H200 MIG `3g.71gb` slice (60 SMs, ~2.2 TB/s), shared with other tenants (10–15% noise between runs; compare within one run). bf16, 40 calls, KV capacity 128 (context < 100 tokens), 4-token prefill. Kernel is profiled GPU time of one call. Decode before is `artifacts/full_model_2026-10-21`; prefill is shown against 2026-10-18 and steps 1-2 (`artifacts/step12b_2026-10-21`); now is `artifacts/step2b_2026-10-21/full_model`. Reproduce with `benchmarks/full_model.py --variants inductor_cudagraphs inductor_graphed megabake`. Medians in µs.

| Model | Decode kernel, before → now | Decode pipelined: MegaBake / graphed Inductor | Decode synced call: MegaBake / graphed Inductor | Prefill kernel, 10-18 / steps 1-2 → now | Prefill call: MegaBake / Inductor + graphs |
| --- | --- | ---: | ---: | --- | ---: |
| SmolLM-135M | 1,282 → **1,186** | **1,208** / 2,019 | **1,818** / 2,297 | 1,381 / 1,129 → **1,145** | 1,942 / **1,823** |
| SmolLM-1.7B | 2,816 → **2,665** | **2,576** / 3,015 | **3,051** / 3,105 | 2,639 / 3,043 → **2,455** | 3,174 / **2,886** |
| Qwen2-0.5B | 1,395 → **1,364** | **1,406** / 2,165 | **2,147** / 2,382 | 1,605 / 1,710 → **1,366** | 2,154 / **2,088** |
| DeepSeek-V3 draft 0.6B | 1,416 → **1,387** | **1,417** / 2,143 | **2,132** / 2,377 | 1,590 / 1,709 → **1,345** | **2,071** / 2,086 |
| Beam DFlash2 draft | 2,553 → **2,530** | **2,535** / 2,664 | 2,959 / **2,727** | 2,626 / 2,540 → **2,579** | 2,975 / **2,779** |
| gemma-2b | 3,464 → **3,318** | **3,329** / 3,614 | 3,948 / **3,692** | 3,341 / 3,164 → **3,152** | 3,804 / **3,518** |
| Gemma-3-270M | 1,872 → **1,793** | 1,818 / **1,416** | 2,481 / **1,598** | — / 1,564 → **1,573** | 2,278 / **1,513** |
| Qwen3.5-0.8B | 5,057 → **4,445** | 4,490 / **2,319** | 5,366 / **2,502** | fails (`cumsum`) | — / 4,114 |

Before 2b, runs toggled C7519 per model (`artifacts/step12_2026-10-21` measured 1.7B prefill 2,491 but gemma-2b prefill 3,758). GPU tests: 40 passed, 1 known failure (below).

Per-CTA profiles (`artifacts/step12b_2026-10-21/phases_*.json`): 135M decode 278 phases, 151 intervals, ~1.0 ms span (was 603, 184, 1.21 ms before step 1); 1.7B decode 224 phases, 121 intervals, 2.31 ms span.

`benchmarks/full_model.py` collects garbage before the pipelined window: one full collection over the compilers' objects takes ~0.6 s and had landed inside the 40-call window (16–19 ms per call on two models).

What the per-CTA profile showed before fusion (SmolLM-135M decode, 603 phases, 184 intervals, `artifacts/chunk4_2026-10-20/phases_135m.json`):

- The critical path is per-phase work: barrier release is ~136 µs (~0.74 µs each, ~11%) of a ~1.2 ms span.
- Per layer: RoPE 2.7 µs, attention 8.5 µs (9 of 60 CTAs busy), o_proj 3.8 µs, add+norm+gate+up 7.4 µs, silu·mul+down 7.0 µs, add+norm+qkv 6.9 µs.
- On 1.7B the 32-tile projections run on 32 of 60 SMs; split-K never applies to them (two splits need 64 CTAs).

Bandwidth utilization in decode: ~55% on 1.7B and ~10% on 135M.

Correctness: 135M and Qwen2-0.5B decode are bit-exact with eager over advancing calls; 1.7B GEMMs differ from cuBLAS by accumulation order only (≤ 2.6e-4 per region).

## Open questions

- Why does 135M decode run 6.8% faster with 239 registers than with 174, and can the compiler get that without unrelated code? (Step 3's inert-code control; 1.7B and the Qwen2 shape do not gain.) Per-role register budgets (R1) may answer it.
- Does a dedicated producer warp with persistent pipeline state (R1–R2) hide the GEMM fill? If not, stop investing in cross-op overlap.
- Would a CUDA-core matvec decode body with 16- or 32-row output tiles fill the idle SMs (32 of 60 on the 1.7B projections)? Hazy found CUDA cores better than tensor cores for batch-1 matvec on Hopper. Partly answered: the external baseline's CUDA-core GEMV, at one CTA per 2 output rows, reaches ~75% of H100 bandwidth (2026-10-23). Still open: whether a generated body matches it inside the megakernel.
- Is mk-v2's relaxed spin plus one `fence.acquire.gpu` cheaper than our `ld.acquire` polling?
- Why does RoPE stay outside decode attention on Llama-1B (one 2-tile interval with `StatefulSmolLM`, a 2-tile and a 20-tile interval with `DecodeStep`) when step 1 fused it on SmolLM? Llama 3's `rope_type="llama3"` tables or the `DecodeStep` graph shape may miss the matcher.

- What L2 share does the MIG slice give, and why did idle-CTA L2 prefetch still slow decode?
- Does the RMSNorm GEMM prologue stay bit-exact against HF's bf16 rounding?
- Does split-K help on a full 132-SM H100/H200? (On the 60-SM slice it slows Llama-1B decode: 1,954 vs 1,850 µs.)
- Does the plan cache key cover matcher and body source? `make_plan_key` takes `body_revisions` only when the caller passes them.

## Known smaller gaps

- Compile time grows on large buckets of large models (SmolLM-1.7B, 512 bucket: 41 minutes).
- Long chains of elementwise ops exceed the planner's recursion limit.
- `x + mean * c + bias` with a broadcast is not fused into one region.
- `test_phase_runtime_ports_delay_idle_rebinding_and_largest_grid` fails; `test_task24_compiles_exact_rope_with_supported_stride` is flaky.
- A new body must not call `llvm.intr_trap()` directly: use `runtime.trap_if`, or C7519 returns (check with `tools/wgmma_serialization.py`).
- Whole-kernel register allocation moves decode by up to ±7% with unrelated code. A/B a feature against "off" **and** against the same code compiled in but inert (step 3).
- The environment is logged in to Hugging Face with access to `meta-llama/Llama-3.2-1B-Instruct` (2026-10-23); Gemma 3 access is unchecked.
- `tools/phase_profile.py` sets `MEGABAKE_PHASE_TIMER` at import from its own `sys.argv`, so a script that imports it without `--cta` gets a 1-element CTA stamp buffer and crashes reading it. The command-line `--cta` run works on Llama-1B decode.
- `phase_profile.py` times decode only at the first ~15 positions; decode attention cost depends on context length, so pass the benchmark's positions when attributing time.
