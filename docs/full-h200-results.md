# Full H200 results (2026-11-19 to 2026-11-20)

The first measurements on a whole H200 (132 SMs, no MIG, no other tenant). Every earlier result in `docs/` came from a 60-SM MIG 3g.71gb slice shared with other processes.

**Setup.** Batch-1 BF16 decode. "MegaBake" is the graphed decode step (`megabake_graphed`); "Inductor" is the unmodified Hugging Face model with a static cache and eager attention, compiled by `torch.compile(mode="reduce-overhead")` (`inductor_hf`). Times are pipelined (back-to-back) calls in µs per token. Calibrated HBM read: 4.39 TB/s with all SMs busy. Raw data: `artifacts/FULLGPU_H200-132SM_2026-11-19/`.

**Status.** The end-to-end sweep with the current code is complete for all nine models.

## End-to-end sweep, current code (2026-11-20)

Speedup is Inductor over MegaBake, by number of cached tokens.

| Model | 16 | 512 | 2k | 4k | 8k |
|---|---:|---:|---:|---:|---:|
| SmolLM-135M | **2.03×** (786 / 1,597) | 1.78× (933 / 1,665) | 1.95× (979 / 1,913) | 2.07× (1,002 / 2,075) | **2.21×** (1,124 / 2,488) |
| Qwen2-0.5B | **1.76×** (906 / 1,596) | 1.71× (981 / 1,675) | 1.70× (1,140 / 1,935) | 1.68× (1,170 / 1,966) | **1.76×** (1,310 / 2,308) |
| Llama-3.2-1B | **1.41×** (1,077 / 1,522) | 1.38× (1,173 / 1,614) | 1.51× (1,186 / 1,796) | 1.67× (1,205 / 2,014) | **1.84×** (1,284 / 2,361) |
| DeepSeek-V3 draft 0.6B | **1.76×** (898 / 1,579) | 1.72× (971 / 1,666) | 1.67× (1,133 / 1,887) | 1.66× (1,165 / 1,939) | **1.73×** (1,309 / 2,271) |
| SmolLM-1.7B | **1.33×** (1,534 / 2,039) | 1.32× (1,658 / 2,188) | 1.30× (1,808 / 2,343) | 1.23× (1,987 / 2,443) | 1.13× (2,381 / 2,699) |
| Gemma-3-270m | **1.27×** (842 / 1,067) | — (858 / failed) | — (874 / failed) | — (893 / failed) | — (940 / failed) |
| gemma-2b | **1.20×** (1,792 / 2,147) | 1.20× (1,817 / 2,175) | 1.19× (1,920 / 2,288) | 1.15× (2,051 / 2,352) | 1.07× (2,343 / 2,506) |
| Beam DFlash2 | 1.00× (1,459 / 1,461) | 1.01× (1,490 / 1,504) | 1.03× (1,547 / 1,599) | 1.05× (1,625 / 1,711) | — (max context 4k) |
| Qwen3.5-0.8B | 0.86× (2,579 / 2,218) | 0.91× (2,583 / 2,358) | 0.91× (2,600 / 2,374) | 0.92× (2,626 / 2,425) | 0.95× (2,690 / 2,557) |

Cells are MegaBake / Inductor µs. The Llama row predates prologue fusion, which leaves Llama unchanged (1,054 → 1,058 µs kernel); every other row includes it.

- **Gemma-3:** Inductor fails from 512 cached tokens on (`FailOnRecompileLimitHit` under `fullgraph=True`: its sliding-window cache recompiles), so there is nothing to compare there. MegaBake runs to 8k (940 µs).
- **SmolLM-1.7B and gemma-2b:** the lead shrinks with context, so MegaBake's attention scales worse than Inductor's at long context on these models.
- **Beam:** the kernel-only figure is 1.08×; the graphed step is even with Inductor.
- **Qwen3.5:** prologue fusion brought the step to 2,580 µs at 256 keys (kernel 2,822 µs before). It is still slower than Inductor, by 5–14%.

## All models: decode kernel time, current code

MegaBake kernel at a 256-key cache, against Inductor's step at 16 cached tokens (from the first full-H200 sweep). The speedup is that step over MegaBake's kernel; a graphed MegaBake step adds about 5–25 µs to its kernel.

| Model | Bytes floor | MegaBake kernel | Inductor kernels (sum) | Inductor step | Speedup now | First full-H200 sweep (11-19) | 60-SM slice (11-15) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM-135M | 61 | 769 | 1,335 | 1,584 | **2.06×** | 1.79× | 1.80× |
| Qwen2-0.5B | 225 | 891 | 1,353 | 1,606 | **1.80×** | 1.57× | 1.56× |
| DeepSeek-V3 draft 0.6B | 216 | 888 | 1,355 | 1,575 | **1.77×** | 1.55× | 1.59× |
| Llama-3.2-1B | 563 | 1,058 | 1,348 | 1,512 | **1.43×** | 1.12× | — |
| Gemma-3-270m | 122 | 815 | 905 | 1,062 | **1.30×** | 0.80× | — |
| SmolLM-1.7B | 780 | 1,585 | 1,854 | 2,035 | **1.28×** | 1.11× | 1.24× |
| gemma-2b | 1,143 | 1,783 | 1,979 | 2,147 | **1.20×** | 0.92× | 1.12× |
| Beam DFlash2 | 1,108 | 1,364 | 1,382 | 1,468 | **1.08×** | 1.03× | 1.05× |
| Qwen3.5-0.8B | 343 | 2,822 | 1,985 | 2,242 | **0.79×** | 0.54× | — |

The bytes floor is the weights read per token at 4.39 TB/s. Qwen3.5's kernel time predates prologue fusion; its graphed step with prologue fusion is 2,580 µs (0.86× at 16 tokens, table above).

## What changed, by step

Decode kernel time in µs (256-key cache).

| Change | Llama-3.2-1B | Others |
|---|---|---|
| Baseline on the full GPU | 1,330 | SmolLM-1.7B 1,853; gemma-2b 2,349 |
| Keep the producer ring: no split-K when it would turn the ring off; counted-GEMM fix; cost-model refit | 1,164 | SmolLM-1.7B 1,618; gemma-2b 2,206 |
| CUDA-core GEMV body for narrow projections | 1,053 | SmolLM-135M 881 → 763; Qwen2 1,018 → 892; DeepSeek 1,012 → 878; gemma-2b 2,200 → 1,968; Gemma-3 1,259 → 1,206; Qwen3.5 4,177 → 3,444; Beam 1,348 → 1,363 |
| Local elementwise chains and parameter hoisting | 1,054 | Gemma-3 1,206 → 815; gemma-2b 1,968 → 1,783; Qwen3.5 3,444 → 2,822 |
| Reduction prologue fusion | 1,058 | Qwen3.5 2,822 kernel → 2,580 graphed step |

Greedy decoding was checked against eager after every step. Each model's relative error against an FP32 reference stayed equal to eager's own.

## Why the speedups differ

- **Small models** (SmolLM-135M, Qwen2, DeepSeek): Inductor runs 530–630 kernels per token at 6–22× the bytes floor. MegaBake removes most of that per-kernel cost.
- **Large models** (gemma-2b, Beam): Inductor already runs within 1.2–1.7× of the bytes floor, so even a perfect kernel can be at most ~1.3–1.9× faster.
- **Gemma-3 and Qwen3.5** lost because they had nearly as many MegaBake steps as Inductor kernels: chains that didn't fuse, and parameter-only ops. Gemma-3 went from 309 to 111 barrier intervals.

## Open

- Qwen3.5 is still 5–14% slower than Inductor, even with prologue fusion. Its linear attention is still many separate steps.
- Long-context attention on SmolLM-1.7B and gemma-2b: the lead drops from 1.33× / 1.20× at 16 tokens to 1.13× / 1.07× at 8k.
- The Gemma-3 Inductor baseline needs a fix in the harness (recompile limit) before contexts past 256 can be compared.
- Look-ahead loading for GEMV steps is not started.
- `tests/gpu/test_phase_runtime.py::test_phase_runtime_ports_delay_idle_rebinding_and_largest_grid` fails, also before these changes.
