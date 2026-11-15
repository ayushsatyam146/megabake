# Hardware-agnostic memory and tiling planner: research and plan

**Written 2026-11-15.** This report answers the research brief "hardware-agnostic memory/tiling planner for Megabake". It covers the six research areas, checks the proposed model against Megabake's own measurements, recommends a minimal architecture, and ends with a plan of action. [architecture.md](architecture.md) and [roadmap.md](roadmap.md) describe the compiler as built; the measurement log is `agent_space/research_notes/landscape_2026-11/landscape_and_experiments.md` (called "the log" below).

How to read the evidence marks:

- **(paper)**: taken from the primary source, which I read for this report.
- **(recalled)**: from my knowledge of the source; I did not re-read it. Check before building on it.
- **(inferred)**: my conclusion from the evidence.
- **(verify)**: sources disagree or I could not confirm it.

All Megabake numbers come from the H200 MIG slice (60 SMs) unless stated. No full-GPU measurement exists yet.

---

## 1. Verdict

**The planner is feasible and worth building. Build it as an extension of `MachineModel`, `accounting.py` and `TensorProgram`, not as a port of Timeloop.** The tree-of-storage model with einsum workloads is sound and has direct precedent. The cost function in the brief, `max(compute time, bytes / bandwidth per level)`, is not enough: it would get the wrong sign on most of the decisions Megabake measured in the last month.

Six findings carry the report:

1. **In batch-1 decode, tile size does not change the bytes.** Every weight is read once per token whatever the tiling, so the data-movement lower bound equals the weight size and the classic "tile for reuse" search is degenerate. What the planner really decides in decode is: how work is spread over SMs (coverage against a saturating bandwidth curve), when bytes move (staging into free capacity), where intermediates live (register, shared memory, HBM), and where synchronization goes. Reuse-driven tiling matters for prefill, batch > 1, and attention over grouped KV heads.
2. **The brief's model needs four more terms.** (a) Shared-edge bandwidth as a *curve over active children*, not a scalar. (b) Per-mechanism *latency* and *issue rate* on each edge. The brief calls latency secondary because pipelining hides it; in decode it is first-order (§2). (c) A *synchronization cost* per level. (d) An *overlap rule* per level: transfer times add when they cannot overlap and only take a max when they can.
3. **The closest prior art is TileSight (arXiv, July 2026).** It is a tile-centric analytical GPU model calibrated by microbenchmarks: 12.35% MAPE on GEMM across A100, H200, B200 and RTX PRO 6000, against 33.85% for a plain roofline. It confirms the approach. It also states that latency-bound small-batch decode, warp-level issue and compiler register allocation are outside its model, and it is not open source yet. Those exclusions are Megabake's operating regime.
4. **The MIT line (Timeloop → LoopTree → TCM/FFM → AccelForge) gives the right representations and search algorithm, not a usable cost model.** Borrow the LoopTree mapping structure, FFM's compatibility keys and Pareto pruning over shared-resource reservations, and Orojenesis-style bounds. Do not depend on their latency model: it assumes perfect overlap, steady state and explicit buffers, and has no latency, synchronization or occupancy.
5. **"A new chip is a new tree file" holds only inside one mechanism family.** A tree file retargets the planner. Bodies encode transfer and compute mechanisms (TMA + WGMMA; thread-issued loads + MFMA; DMA + MXU), and code generation is per vendor. H100 → B200 is close to a tree file plus a body variant. AMD and TPU need mechanism adapters and a backend.
6. **Nothing in the literature models cross-op scheduling inside one persistent kernel.** MPK and HazyResearch set it by hand-written rules, and so does Megabake's producer ring (X1.1). A planner that sizes and schedules run-ahead from a hardware description would be new.

**Recommendation:** start with a one-week retro-validation gate (§8, P0) that tests model forms against the decisions already measured. Then build the tree schema and calibration, the einsum view, and the search, in that order. Total effort is about the same as roadmap step 7 plus X2.4, which this plan replaces.

---

## 2. What Megabake's measurements say about the proposed model

The log holds a set of decisions with measured outcomes. They are the cheapest test of any cost model. The right column is my expectation of what the brief's model predicts (inferred; P0 confirms it).

| # | Decision (source in the log) | Measured | Brief's model: `max(compute, bytes/bandwidth)` |
|---|---|---|---|
| 1 | Split-K / even split for decode GEMMs (X2.2) | Slower. gate·up already moves ~91% of HBM | No gain (right only if bandwidth is a hard cap); cannot predict "slower" |
| 2 | L2 prefetch of the next interval's weights (roadmap; X1.0) | 8–24% slower by CTAs; ~1.8× slower from the producer | **Wrong.** L2 has capacity and 4.0 TB/s, so it predicts a gain |
| 3 | Shared-memory weight preload issued by warp 0 at the barrier (step 3) | −0.7% to +1.5% | **Wrong.** Predicts a gain: latency is assumed hidden |
| 4 | Producer warpgroup + persistent ring (X1.1) | −0.2% to −7.0%; Llama-1B −1.8% where X1.0 predicted −7% | Right sign, large overestimate (no contention term) |
| 5 | Pack a KV group's query heads in one row (2026-10-27) | Faster from ~128 keys per packed head; +11% to +24% slower at short contexts | Right at long contexts (reuse). **Wrong** at short ones (too few items for the SMs) |
| 6 | Split attention keys across CTAs (2026-10-23; X2a) | Splitting short contexts lost ~330 µs per Llama-1B token near 100 keys; the split pays only at long contexts | **Wrong.** No exchange cost |
| 7 | Interval folding: recompute small producers instead of a barrier | Faster | **Wrong.** No synchronization cost, so recompute looks like pure waste |
| 8 | Pre-tiled 1-D weight streams (X1.6) | No better than 2-D TMA | Neutral (right) |

Four measured facts explain the misses:

- **Bandwidth saturates early.** HBM read with k busy CTAs: 32 of 60 reach 82%, 48 reach 94%. Straggler wait was 367 µs on Llama-1B, but the coverage loss was only 71 µs.
- **Bodies run at their own rate, not the HBM rate.** Decode attention reads K and V at ~16 GB/s per CTA against a ~59 GB/s HBM share. `bytes / HBM share` under-predicts attention about 4× (X2.3 status).
- **Fixed costs dominate small intervals.** Each GEMM interval carries a few microseconds that do not scale with bytes: first-stage DRAM latency 1–1.5 µs, dispatch ~0.5 µs, and a barrier of ~1.5 µs. On SmolLM-135M the kernel is at 0.11 of speed of light and no standalone kernel sequence reaches it either.
- **Background streaming slows latency-bound accesses.** With the producer on, epilogues grew from 0.1–0.6 to 1.0–2.0 µs.

HazyResearch report the same shape on B200: of a 600 µs Llama-1B step, ~250 µs is activation store, consistency wait and load, ~40 µs is warp-level synchronization, ~80 µs is setup, and only ~30 µs is waiting for weights (paper: their blog post). The dominant term is the dependence chain's latency, which a bandwidth roofline does not contain.

One decision cannot be modelled by anyone: whole-kernel register allocation moves decode by ±7% with unrelated code. TileSight lists "compiler register allocation" as out of scope too. **The model's job is ranking and pruning; measurement of the top few plans stays in the loop.**

---

## 3. Research findings by area

### 3.1 Timeloop, Accelergy, LoopTree, TCM, FFM, AccelForge (MIT, Emer and Sze)

**Core idea.** Separate three things: architecture, workload, mapping. An analytical model counts tile accesses per storage level from the mapping; a mapper searches the mapping space.

**Data model.**

| Part | Representation | Source |
|---|---|---|
| Architecture | A hierarchy of storage levels down to compute. Each level has capacity, instance count (`meshX`, `meshY`), and `read_bandwidth` / `write_bandwidth` in words per cycle; v4 adds `shared_bandwidth`. Each level fans out to an integral number of children | Timeloop docs |
| Energy and area | Accelergy plug-ins give energy per action | Not relevant to Megabake |
| Workload | Extended einsums: named ranks, tensors indexed by ranks or affine expressions of them (`p+r`). Reuse follows from which ranks index which tensor | LoopTree (paper) |
| Mapping, one einsum | Per level: temporal and spatial loop bounds (tile shape), loop order, and which tensors bypass the level | Timeloop |
| Mapping, fused | A LoopTree: loop nodes, storage nodes, compute nodes. A storage node above a loop means the tile is reused across that loop's iterations. A split means the einsums below it share a stored tile. Per-tensor choice of retain, refetch or **recompute** | LoopTree, FFM (paper) |
| "Dataplacement" | Which tensor tiles are held at which level, in which order. Search order is dataplacement, then loop order, then tile shapes | TCM (paper) |

**Model.** Tile size at a storage node is the product of the loop bounds below it; tiles fetched is the product of the bounds above it. Latency is `max(max over memories of accesses / bandwidth, computes / compute units)`: a fully overlapped roofline (TCM, paper). LoopTree computes tile shapes and recomputation with ISL sets and relations, working backward from the last layer, and analyses tiles of identical shape once.

**Search.**

- Timeloop's mapper walks index factorization × loop permutation × bypass with linear-pruned, random-pruned or hybrid search. It gives no optimality guarantee.
- **TCM** (Turbo-Charged Mapper, arXiv 2602.15172) prunes without losing the optimum: loop orders that differ only between the same two storage nodes are equivalent; a loop that does not index the tensor stored below it is useless; partial tile shapes are pruned by dominance on criteria derived from the symbolic objective. It partially evaluates the model once per (dataplacement, loop order) into a tile-shape-only expression of sums, products and maxes. The mapspace falls from 10^23–10^37 to about 10^7 or fewer; a GPT-3 attention einsum takes ~17 s single-threaded.
- **FFM** (Fast and Fusiest Mapper, arXiv 2602.15166) joins per-einsum partial mappings. For each shared tensor a partial mapping has a **compatibility key**: (tile shape, backing storage level, tile exchange order). Two partial mappings join if and only if their keys agree for every shared tensor. Partial mappings with the same key are interchangeable, so Pareto pruning inside a key group, over objectives *and* buffer reservations with their lifetimes, keeps the optimum. Generation is linear in the number of einsums. GPT-3 6.7B maps in ~1.5 CPU hours; SET needs ~15,000 for the same quality.
- **AccelForge** (arXiv 2609.11906) packages LoopTree, TCM and FFM in Python: `pip install accelforge`, MIT licence, https://github.com/Accelergy-Project/accelforge. The README marks parts as experimental.
- **Orojenesis** (ISCA 2024) sweeps the mapspace of a one-buffer proxy architecture and reports, for each buffer size, the least data movement any mapping reaches (the "ski-slope" curve). It covers fused chains.

**Validation.** LoopTree: worst-case 4% against five published accelerators. TCM and FFM match LoopTree, Timeloop and CiMLoop exactly and report within 10% of published chip data. All targets are fixed-function accelerators; the datacenter example is a "TPUv4i-like" design (128 MiB global buffer, four cores with 4 MiB local buffers, 128×128 multiply array, 614 GB/s DRAM). No GPU validation exists in this line.

**What transfers to a GPU.**

- Reuse analysis from einsum index sets.
- The LoopTree structure (storage nodes between loops) as the mapping representation.
- Capacity checks per level and spatial fan-out.
- FFM's join by compatibility key. This is the same thing as Megabake's planned typed edge formats (architecture §9 item 3) with the tile shape and backing level added.
- Pareto pruning over shared reservations. In one megakernel every body shares one shared-memory layout and one register allocation, so this is the right place for a Pareto front (§5.5).
- Retain versus recompute. Megabake's interval folding is this choice, made by hand today.

**What breaks.**

| Assumption in the Timeloop line | GPU megakernel reality |
|---|---|
| Latency is a max over components (perfect overlap) | Latency-bound terms add (§2) |
| Steady state. LoopTree states it omits weight loading and systolic start-up and assumes stalls are negligible | Fixed costs are 2–3 µs per GEMM interval; small intervals are mostly fixed cost |
| Buffers are explicitly managed | L2 is a hardware-managed cache. TileSight needs a stack-distance model for it |
| A parent's bandwidth is a hard cap shared by children | Sub-linear ramp: 16 CTAs reach 50%, 32 reach 82% |
| No synchronization, no occupancy, no register limit, no warp roles | These decide Megabake plans |
| Static shapes | KV valid length is run-time data |
| Objective is energy or energy-delay product | Objective is time |

Independent evidence: GPU-Tile-Sim (MICRO 2026) measured Timeloop-derived TileFlow at 24–34% MAPE on GPU GEMMs and often above 100% on attention kernels (paper).

**Borrow:** the representations and the search algorithm. Use Orojenesis or AccelForge offline to produce bounds for prefill shapes. **Build:** the cost model. **Do not:** put Timeloop or AccelForge in the compile path.

### 3.2 TileSight and GPU-Tile-Sim

**TileSight** (arXiv 2607.22432; Imperial, Peking, Tile-AI, Microsoft Research, Edinburgh).

- **Hardware abstraction (paper).** Per GPU: SM count; vector, tensor-core and special-function throughput; L2 and DRAM bandwidth, each as spec and measured. Levels: registers, shared memory, "L1.5" (H200, B200), L2, DRAM, plus Blackwell TMEM. Occupancy is limited by shared memory, registers, warps and blocks per SM. Above one GPU: a network hierarchy with per-link latency and inverse bandwidth.
- **Workload.** A tile-action DAG. Each tile has an operation type, bytes, FLOPs, and source and destination placement levels.
- **Schedule knobs modelled.** Tile shape, launch order, swizzle (row-panel, block swizzle, Z-order, persistent-block), loop and reduction depth, software pipeline stages, resident tiles per SM, which tiles issue together, distributed partition. Not mentioned: warp specialization, TMA.
- **Latency model.** Each action has a resource-use vector over ⟨tensor core, CUDA core, SFU, TMEM, SMEM, L1.5, L2, DRAM, network⟩. Steady-state time per tile is the max over resources of the summed use, minimized over legal action orders. Kernel time is a pipeline envelope:

  ```text
  T = T_prologue + max(N − d, 0) · T_steady + T_epilogue
  d = stages × resident_tiles_per_SM − 1
  ```

  Tail waves run on fewer SMs, so each gets a larger bandwidth share and the envelope is recomputed.
- **L2 model.** Reuse distance in tile-sized blocks feeds a stack-distance cache model with a Gaussian approximation; L1.5 → L2 → DRAM is a cascade. Predicted L2 hit rates are within 0.8–1.5 points of measurement on four GPUs.
- **Calibration.** One-time microbenchmarks per architecture, minutes: bandwidth and throughput sweeps and short matmul probes. Cache capacity comes from the bandwidth sweep. No per-kernel profiling or training.
- **Accuracy.** GEMM, 703 shapes: 12.35% pooled MAPE (PipeWeave 21.97%, NeuSight 32.95%, roofline 33.85%). MI210: 23.4%. FlashAttention-3 on H100: 2.7%. vLLM decode end to end: 13.5% weighted MAPE. As a pruner for TileLang, keeping its top 5% of candidates reaches 99.66% of the exhaustive best.
- **Stated limits.** Not modelled: data-dependent control flow, instruction-level compiler decisions, undocumented warp and CTA scheduling, warp-level issue, register allocation. "Latency-bound cases (for example small-batch decode attention) and B200 SM-to-HBM affinity need finer parameters." SM desynchronization made one deep-K L2 prediction 82% against 43% measured. AMD support goes through Composable Kernel, which has no swizzle control.
- **Code.** About 6K lines of Python; "will be open-sourced upon publication".

**GPU-Tile-Sim** (arXiv 2607.11262, MICRO 2026) is the simulator counterpart. A kernel is a warp-level tile graph: nodes carry an executor (one warp, a producer or consumer group), an operation, and a tile; data edges and order edges express pipelines, barriers and buffer reuse. It models warp specialization directly and reaches 1.2–8.7% MAPE on A100 and H100. Its frontend takes TileLang IR. Porting it to Blackwell changed 11.3% of the simulator core. It is too slow for a compile path and is not released.

**Borrow from TileSight:** the pipeline-envelope form (prologue + steady + epilogue with an effective depth), tail-wave recomputation, the spec-versus-measured parameter table, and the stack-distance L2 model if prefill needs it. Megabake's `MachineModel` is the same idea at a smaller scale. **Build:** everything TileSight excludes: per-mechanism latency and issue rate, synchronization, cross-op scheduling, contention.

### 3.3 Polyhedral optimisation

**How tiling and loop order are formulated.**

- **Pluto** (PLDI 2008). Find affine hyperplanes φ per statement such that every dependence has a non-negative distance (legality) and the distance is bounded by `u·n + w`; minimize (u, w) lexicographically with an ILP. Bands of permutable hyperplanes are tilable. Pluto finds *that* loops can be tiled and in which order; tile *sizes* are a separate problem (default 32). (recalled)
- **PPCG** (TACO 2013). The isl scheduler finds the bands, the outer parallel band maps to blocks and threads, and array tiles can be copied to shared memory. Tile, grid and block sizes are user options; the paper lists tile-size selection as future work.
- **isl schedule trees.** Band, sequence, set, mark and extension nodes. This is nearly the LoopTree structure, which is why LoopTree could use ISL.
- **MLIR.** `linalg.generic` is an einsum: indexing maps (affine) plus iterator types. The transform dialect applies tiling and fusion; the affine dialect carries integer sets. (recalled)
- **Tiramisu** separates algorithm, loop schedule, data layout and communication into four layers. **Tensor Comprehensions** pairs an einsum notation with isl scheduling and an evolutionary autotuner for tile sizes and GPU mapping. (recalled)

**Why tile size is not an ILP.** A capacity constraint is a product: tile footprint = Π tile sizes × item size ≤ capacity. Products are not affine. Three known answers:

1. **Geometric programming.** Footprints and access counts are posynomials in the tile sizes, so the problem is convex in log space (Renganarayana and Rajopadhye, SC 2004; the accelerator co-design work of Vaidya, Sukumaran-Rajam, Rountev and Sadayappan uses disciplined geometric programming with CVXPY).
2. **Log-transformed mixed-integer programming.** CoSA (ISCA 2021) assigns prime factors of each dimension to levels; it is 90× faster to a solution than search and 2.5× better on its benchmarks.
3. **Analytic reduction, then a small non-linear solve.** MOpt (ASPLOS 2021) reduces permutations and tile sizes for CNNs on CPUs to a few non-linear problems.

TCM's result weakens the case for all three: with good pruning, exhaustive enumeration of integer tile shapes is exact and takes seconds.

**Could a polyhedral backend implement the mapping analysis?** Yes, and LoopTree is the proof. For Megabake it is needed in only two places:

1. **Tile-to-tile dependence.** "Which producer tiles does this consumer tile read" is a relation between two tile spaces. Architecture §9 item 7 already asks for counter targets "derived from tile-region overlap, with separate RAW and WAR lists". That is polyhedral dependence analysis.
2. **Affine windows.** Sliding-window attention, strided convolution, and the `AxisAffine` terms in `ir.py`.

For dense einsums with rectangular tiles, counts are products of loop bounds and closed forms are faster and simpler. `planner._affine_bounds` already does interval analysis for monotonic affine axes.

**Borrow:** the schedule-tree shape; dependence relations for counters; `islpy` as an optional analysis backend. **Do not build:** a polyhedral scheduler. Megabake's loop structure is fixed inside bodies, and its ops are not general affine loop nests.

### 3.4 Theory: bounds and the optimality gap

**The results.**

| Result | Statement | Use |
|---|---|---|
| Red-blue pebble game (Hong and Kung, 1981) | With S red pebbles (fast memory), I/O for classical matrix multiply is Ω(n³/√S) | Data-movement bound for one level |
| Tight constants (Kwasniewski et al., SC 2019, COSMA) | Leading term 2mnk/√S; their schedule is within 0.04% for 10 MB (verify the exact form in Theorem 1) | A usable constant, not only an order |
| Memory hierarchy game (Savage, 1993–95) | A multi-level pebble game; the Hong-Kung bound applies at each level boundary with that level's capacity | **One bound per tree edge** |
| HBL bounds (Christ, Demmel, Knight, Scanlan, Yelick, 2013) | For loops whose arrays are indexed by subsets of the loop indices, a linear program gives the exponent s: a segment with M words does at most M^s operations. The dual LP gives the optimal block sizes M^(x_i) | **Bound and tile aspect ratios for any einsum from a small LP** |
| IOLB, IOOpt (Olivry et al., PLDI 2020, 2021) | Automatic symbolic lower and upper bounds for affine programs; tight on 11 of 30 PolyBench codes | Automation exists for affine code |
| Multiprocessor red-blue pebbling (Böhnlein, Papp, Yzelman, 2024) | p processors, each with its own fast memory, one shared slow memory. Optimal pebbling is NP-hard and cannot be approximated to any finite factor | Do not try to solve it; use closed-form bounds |
| Orojenesis (ISCA 2024) | Attainable bound by exhaustive mapping search, per buffer size, fusion included | Tighter than asymptotic bounds; needs no proof |
| SOLAR, SOL-ExecBench (NVIDIA, 2026) | Speed-of-light bounds for GPU kernels from FLOPs, bytes and peak rates; the cache-aware variant uses Orojenesis and tightens the plain roofline by up to 10× | The same "sol" metric as Megabake's scoreboard |

**What they say for Megabake (inferred).**

- **Batch-1 decode GEMV: the bound is trivial.** The weights have no reuse within a token, so the bound is |W| bytes at any capacity. `MachineModel.speed_of_light_s` already computes it. No tiling can reduce HBM bytes.
- **Where the bounds bite.**
  - Prefill and batched GEMM: activations (M × K) and weights compete for shared memory, and traffic across the L2 → SM edge grows as 1/√S.
  - Decode attention with grouped heads: K and V are not indexed by the query head inside a KV group, so a tile that keeps a group's queries resident reads each key once. The einsum index sets derive the packing rule that Megabake found by experiment on 2026-10-27.
  - Fused chains: whether an intermediate goes through registers, shared memory or HBM is an Orojenesis-style retention choice.
- **The gap that matters in decode is in time, not bytes.** The pebble game counts transfers. It has no clock, no bandwidth and no latency. Two additions give a time bound:
  - **Dependence depth.** A decode layer is a chain of about five dependent intervals. Each hand-off costs at least one activation round trip (store, visibility, load). Time ≥ depth × round trip, whatever the bandwidth. HazyResearch's 250 µs of 600 µs is this term.
  - **Multi-BSP** (Valiant, 2008) is the model that has the missing parameters: each tree level i has (p_i, g_i, L_i, m_i) = fan-out, bandwidth cost, synchronization cost, memory. A superstep at level i costs work + g_i · traffic + L_i. This is the brief's tree with a synchronization cost per level. The related Parallel Memory Hierarchy model (Alpern, Carter, Ferrante, 1993) is literally "a tree of memory modules with processors at the leaves", and Sequoia (SC 2006) built a language on it.

**A ladder of bounds gives the optimality gap.** Each rung is computable from the tree and the plan:

```text
B0  algorithmic bytes / peak bandwidth, FLOPs / peak rate      (exists: speed_of_light_s)
B1  capacity-aware bytes per tree edge (HBL / Orojenesis)      (= B0 for batch-1 decode)
B2  + bandwidth curve at the plan's coverage, wave by wave     (data exists: hbm_read_bytes_per_s_by_ctas)
B3  + dependence depth × calibrated round trip + barriers      (new)
B4  best measured structure (the X1.0 streaming-ceiling kernel) (exists for Llama-1B)
```

Report `measured / B_k` for each k. Each ratio names a different owner: B1/B0 is the tiling, B2/B1 the work split, B3/B2 the dependence structure and fusion, measured/B3 the bodies. The log's current decomposition (sync 116 µs + bodies 242 µs + standalone over speed of light 180 µs) is an empirical version of the same ladder.

### 3.5 Prior art for a GPU-wide, cross-op planner

| System | Scheduling unit | Cost model | Cross-op inside one kernel | Take |
|---|---|---|---|---|
| **Welder** (OSDI 2023, Microsoft) | Tile-graph: each node processes one tile; the edge between nodes is placed in a memory layer (register, shared memory, global) | Tile traffic per memory layer; layers are optimized independently; candidates are aligned to hardware tile shapes and the best few are profiled (recalled for the last part) | Fuses operators into kernels by connecting tiles in a faster layer. Many kernels; no persistent kernel | The idea that fusion is "which layer holds the edge", and that layers decouple. Closest ancestor of the brief |
| **Roller** (OSDI 2022) | rTile: tile shapes aligned to hardware features | A micro-performance model in place of on-device search; compiles in seconds | No | Hardware-aligned enumeration shrinks the tile space |
| **Chimera** (HPCA 2023) | Fused compute-intensive operators | Analytical data-movement model chooses block order and sizes (recalled) | Within a fused kernel | Analytical tile ordering for GEMM chains |
| **Mirage / MPK** (OSDI 2026) | SM-level task graph: tasks and events alternate. Tasks per operator are proportional to the SM count | **None.** The partition minimizes device-to-shared-memory loads; no tile-size tuning (paper) | Yes. 32 KB shared-memory pages (7 per SM on H100). A task has a pre-load phase and a compute phase; task T2's pre-load overlaps T1's compute once pages are free. Hybrid ahead-of-time and just-in-time dispatch | Page and pre-load structure (Megabake's ring is the same idea). Evidence that no published megakernel compiler has a cost model |
| **HazyResearch Llama megakernel** (2025) | Instruction per SM, scheduled ahead of time in Python | None; hand layout | Yes. 13 pages of 16 KiB; freed pages pass to the next instruction so weight loads start early; counters in global memory, the MLP split into four chunks | The time breakdown in §2: the target is the dependence chain, not weights |
| **Cohere megakernel** (2026-09) | Sixteen op types in one persistent kernel, global counters | None; one model | Yes | Reports 62% of H100 speed of light on a larger model (secondary source) |
| **Inferact TPU megakernels** (2026) | One Pallas call for all layers | None; hand layout | Yes. `make_async_copy` starts layer N+1's weights during layer N's MoE and communication phases; activations stay resident in VMEM | Shows the single-kernel constraint holds on TPU; hand-scheduled |
| **ThunderKittens** | Register and shared tiles; load-compute-store templates | None | No | Body mechanics (architecture §10) |
| **TileLang** | Tile operators; scheduling (thread binding, layout, pipeline) separated from dataflow | TileSight is evaluated as its cost model | No | Dataflow and schedule separation |
| **Hexcute** | Tile language; layouts and thread-to-tile mapping by type inference | Constraint solving, not cost | No | Layout synthesis for edge formats |
| **Triton autotune; Inductor max-autotune** | One kernel | None. Times a list of configurations on the device | No | — |
| **Helion** | One kernel | Search (pattern search, differential evolution), accuracy gate, per-architecture cache | No | Already taken in X2.4 |
| **Ansor, MetaSchedule (TVM)** | One subgraph | Learned (gradient-boosted trees over loop features), evolutionary search, thousands of device measurements | No | Fallback only (X2.5). Megabake cannot afford thousands of megakernel compiles: one SmolLM-1.7B bucket took 41 minutes |

**On "idle SMs prefetch the next layer's weights".** No published system plans this from a model. Megabake's own results bound it:

- Bandwidth is saturated by about 48 of 60 CTAs, so an idle SM adds little bandwidth. It can only move bytes *earlier*, into time when HBM would idle.
- Preloads by CTAs with no tiles competed for HBM and lost (step 3). L2 prefetch lost twice.
- The producer ring wins because it parks bytes next to their consumer and never joins consumer synchronization.

So the planning question is not "which SM is idle" but "which bytes can move in which idle window, into which free capacity, without delaying latency-bound work". §5.6 makes this a planner output.

### 3.6 Hardware-agnostic descriptions

**Existing descriptions.**

| Tool | Schema | Covers |
|---|---|---|
| Timeloop / AccelForge | Storage hierarchy with instances and bandwidths | Any fixed-function accelerator; a TPUv4i-like design |
| TileSight | Per-GPU parameter table plus a level list | A100, H200, B200, RTX PRO 6000, MI210 |
| LLMCompass (ISCA 2024) | One template: system → device → core → lane, with optional systolic arrays | **Validated on NVIDIA A100, AMD MI210 and Google TPUv3** with one schema: 10.4% operator error, 4.1% end to end |
| Kerncraft (ECM model) | A YAML machine file per CPU: cache sizes, per-level bandwidths measured with `likwid-bench`, and whether transfers between levels overlap | CPUs. The model for "tree file + microbenchmarks" |
| Sequoia machine files | A tree of memory modules, separate from the algorithm | Clusters, Cell, CPUs |

The ECM work adds one lesson (paper: Hofmann et al., 2020): whether transfer times across levels **overlap** (take the max) or **do not** (add) is a property of the machine, differs between Intel, AMD, IBM and Arm cores, and must be measured. Its saturation predictions were too optimistic until a latency penalty that grows with bus utilization was added. That is the contention effect X1.1 measured.

**The three targets as trees.**

| | NVIDIA H100 (sm90) | NVIDIA B200 (sm100) | AMD MI300X (CDNA 3) | Google TPU v7 |
|---|---|---|---|---|
| Root | HBM3 80 GB, 3.35 TB/s | HBM3e, 8 TB/s | HBM3 192 GB, 5.3 TB/s | HBM 206 GB, 7.38 TB/s |
| Shared cache levels | L2 50 MB in two halves; crossing 25 MB costs | L2 in four partitions (secondary source); size (verify: sources give 64 MB to 126 MB) | Infinity Cache 256 MB, memory-side, ~17 TB/s → L2 4 MB per XCD (8 XCDs), not shared across XCDs | none |
| Leaf fan-out | 132 SMs | 148 SMs | 8 XCDs × 38 CUs = 304 CUs | 2 TensorCores per chip |
| Leaf storage | 228 KB L1/shared memory (227 KB per block); 64K registers | 256 KB L1/shared memory; TMEM 256 KB per SM (secondary source) | 32 KB L1 per CU; LDS (verify size on CDNA 3) | VMEM 64 MiB per TensorCore, software-managed |
| Leaf compute | Tensor cores (WGMMA), CUDA cores | Tensor cores (tcgen05, 2-SM MMA), CUDA cores | Matrix cores (MFMA; half an H100 SM's FP16 rate per CU), vector | MXU (systolic), VPU |
| Transfer mechanism | TMA: asynchronous bulk copy, no compute threads | TMA; TMEM load and store | Loads issued by compute threads; no bulk engine | DMA: asynchronous copies between HBM and VMEM |
| Synchronization | mbarrier, global atomics, grid barrier | same (mbarrier ~60 ns per HazyResearch) + cluster launch control | atomics | none needed in one TensorCore; DMA waits |
| Kernel language | CuTe DSL / CUDA | CuTe DSL / CUDA | HIP; Composable Kernel | Pallas (`make_async_copy`, `emit_pipeline`, `run_scoped`) |

Sources: NVIDIA tuning guides; Chips and Cheese on CDNA 3; the Inferact post for TPU v7; Pallas docs. TPU v4 VMEM is 16 MiB per the Pallas docs.

**Can one tree schema cover all three? Yes for the planner, with five extensions to the brief's schema.**

1. **Edges carry mechanisms.** Each edge lists its transfer mechanisms with latency, per-issuer throughput, in-flight limit, and whether issuing occupies compute threads. This one attribute separates TMA, AMD's thread-issued loads and TPU DMA, and it is what makes attention "load-issue bound" on Hopper.
2. **Shared edges carry a curve**: throughput against the number of active children.
3. **Levels carry a synchronization table** (the multi-BSP L_i) and a management flag: software-managed (shared memory, VMEM, TMEM) or hardware-managed (L2, Infinity Cache). Only software-managed levels can hold a planned tile.
4. **Leaves carry several engines** (tensor and vector; MXU and VPU) and a register budget.
5. **A tree with exceptions.** Three real structures are not trees: distributed shared memory inside a CTA cluster (sibling edges), L2 partitions with SM affinity, and links between GPUs. Model the first as an optional "cluster" level whose capacity is the sum of its children's; treat the second as a calibration effect; leave the third out of scope.

**What a tree file does not port.** Bodies. A body is written against mechanisms. Within a family (sm90 → sm100) the body changes its MMA and uses TMEM, and the planner retargets by file. Across families the body, the code generator and the launch contract are new. One check for AMD: a persistent kernel needs every workgroup resident at once; HIP has no documented guarantee like CUDA's cooperative launch (verify).

**One number the tree gives for free: staging depth**, the free software-visible capacity at a level divided by the root bandwidth. It is how many microseconds of HBM traffic can be parked at that level (inferred; rough arithmetic).

| Level | Capacity | Staging depth |
|---|---|---|
| H100 shared memory, 132 SMs | ~30 MB | ~9 µs |
| H200 slice, 192 KB ring × 60 CTAs (X1.2 as built) | ~11.5 MB | ~5 µs |
| H100 L2 | 50 MB | ~15 µs, but hardware-managed, and prefetch into it measured slower |
| MI300X Infinity Cache | 256 MB | ~48 µs, hardware-managed; whether software can direct it is unknown (verify) |
| TPU v7 VMEM, one TensorCore | 64 MiB | ~9 µs, in one pool that any op can use |

On the slice, a Llama-1B layer has roughly 13–15 µs of non-GEMM work and barriers against ~5 µs of staging depth. That fits X1.0's result: 0.90 of HBM with loads only, 0.78 with the real non-GEMM work in the path, and no gain from a ring deeper than 24 stages.

---

## 4. Corrections to the brief's model

| Brief | Keep or change | Why |
|---|---|---|
| Hardware as a tree, compute at the leaves | Keep | PMH, Sequoia, multi-BSP, Timeloop and LLMCompass all use it |
| Edges weighted by bandwidth; latency secondary because pipelining hides it | **Change.** Latency and issue rate are first-class per mechanism | §2: fill latency, activation round trips and issue-bound bodies dominate decode |
| Shared edges marked as shared | **Change** to a throughput curve over active children | Measured saturation curve |
| Workloads as einsums | Keep. Derive the view from `TensorProgram` | §5.2 |
| Mapping = per-level tiles + loop order + spatial partition | **Extend** with interval placement, role, sync edge type and staging | The megakernel's degrees of freedom are in time |
| Time = max(compute, bytes/bandwidth per level) | **Change** to an envelope: fixed terms + max over overlapping terms + sum over non-overlapping terms + sync | ECM; TileSight Eq. 2; §2 |
| A Pareto front of coupled objectives | **Narrow.** One objective (time) under resource constraints for the final plan. Pareto fronts only for partial mappings, over shared reservations | FFM; §5.5 |
| New chip = new tree file, not new bodies | **Narrow** to one mechanism family | §3.6 |
| Calibrated by microbenchmarks | Keep and extend to mechanisms and bodies | TileSight, Kerncraft, X0.1 |

---

## 5. Recommended minimal architecture

Design rule: every piece replaces or extends something that exists. Nothing here needs a new IR.

### 5.1 Hardware tree (`DeviceTree`)

One file per device fingerprint. `TargetFacts` (capabilities) and `MachineModel` (speeds) become views of it. Every number is tagged `spec` or `measured`; the planner uses `spec` values only for CPU-side planning in CI (X0.2, X5.4).

```yaml
device: h200-mig-3g71gb            # fingerprint key, as machine.py builds it
levels:                            # root first
  - name: hbm
    capacity_bytes: 71e9
    managed: software
  - name: l2
    capacity_bytes: {measured: 40e6, spec: 30.8e6}
    managed: hardware              # cannot hold a planned tile
    partitions: 1
  - name: sm
    fanout: 60
    managed: software
    stores:
      smem_bytes: 227e3
      registers_per_thread_max: 255
      threads_max: 1024
    engines:
      tensor: {bfloat16: 6.9e12}   # FLOP/s per SM, measured
      vector: {loads_16B_per_s: ...}
    sync:                          # multi-BSP L at this level
      mbarrier_s: ...
edges:
  - from: hbm
    to: sm
    via: l2
    shared_throughput_bytes_per_s:   # curve over active children
      {1: 80e9, 8: 635e9, 16: 1151e9, 32: 1883e9, 48: 2179e9, 60: 2307e9}
    mechanisms:
      tma_bulk:
        occupies_compute: false
        first_byte_latency_s: ...    # GEMM stamps suggest 1–1.5e-6; no probe yet
        in_flight_max: ...
        per_issuer_bytes_per_s: ...
      thread_load:
        occupies_compute: true
        per_warp_bytes_per_s: ...
    write: {...}                     # same fields for stores
    overlap_with_compute: 0.0..1.0   # ECM overlap coefficient, measured
    contention:                      # slowdown of latency-bound access
      latency_factor_at_utilization: {0.5: ..., 0.9: ...}
device_sync:
  grid_barrier_s: {1: 4.7e-7, 60: 7.0e-7}
  counter_round_trip_s: 9.5e-7
  activation_round_trip_s: ...       # store → visible → load on another SM
```

The numbers shown are X0.1's measurements on the slice (the tensor rate is the cuBLAS peak divided by 60 SMs); `...` marks what calibration must add (§5.7).

### 5.2 Workload: an einsum view of `TensorProgram`

`ir.TensorProgram` already holds what an extended einsum needs: `iteration_domains` are ranks with sizes, `reduction_axes` are the contracted ranks, each `IndexedLoad.axes` is the rank tuple of a tensor (with `AxisAffine` for windows), and `store` is the output. Add one derived, read-only structure:

```python
@dataclass(frozen=True)
class EinsumView:
    ranks: dict[str, int]                    # rank → size (symbolic for KV length)
    tensors: dict[str, tuple[RankExpr, ...]] # value id → ranks that index it
    reduced: frozenset[str]
    output: str
    opaque: bool                             # gather / DataIndex: no reuse analysis
```

- Reuse is derived, not declared: tensor T is reused along rank r exactly when r does not index T.
- Declared ops expand to a cascade. Attention is `S[h,t] = Q[h,d]·K[g(h),t,d]`, a row map (softmax), then `O[h,d] = P[h,t]·V[g(h),t,d]`, plus the KV state access. TCM and FFM model GPT-3 the same way (Q, K, V, QK, AV, FFA, FFB).
- Data-dependent access (`DataIndex`) is opaque and keeps today's byte accounting.
- Run-time sizes (KV valid length) stay symbolic. The plan is evaluated at a few lengths, as `decode_split_steps` does today.

This also fixes a weak point: `accounting.phase_traffic` infers FLOPs from the shapes of 2-D inputs. The view gives exact counts.

### 5.3 Mapping representation

A mapping has two layers.

**Per region: a LoopTree restricted to what a body can do.** `BodyConfig` already holds a point of it: `TileContract` (tile shape, reduction tile, cluster shape), `PipelineContract` (stages), `ParticipantContract` (threads, registers, roles), `ScratchContract` (shared memory). The change: a provider declares a **mapspace**, the legal ranges of its knobs, instead of a list of configs.

```python
@dataclass(frozen=True)
class RegionMapping:
    body: str
    placement: dict[str, str]        # tensor → innermost level where a tile is retained
    tile: dict[str, int]             # rank → tile size at the leaf
    order: tuple[str, ...]           # loop order above the leaf tile
    spatial: tuple[str, ...]         # ranks split across leaves → work items
    stages: int                      # pipeline depth
    intermediates: Retain | Recompute | Refetch   # per fused input
```

**Per kernel: what only a megakernel has.**

```python
@dataclass(frozen=True)
class KernelMapping:
    intervals: tuple[tuple[RegionId, ...], ...]  # regions sharing a barrier interval
    edge_sync: dict[Edge, Barrier | Counter | SameCta]   # how each dependence is enforced
    edge_format: dict[Edge, CompatKey]           # (tile shape, backing level, exchange order)
    staging: tuple[StagePlan, ...]               # which bytes the producer moves early, into which level
    reservations: ResourcePlan                   # shared memory pages and registers per role
```

`CompatKey` is FFM's compatibility key and the typed `EdgeFormat` of architecture §9 item 3 in one type. `reservations` is the `ResourcePlan` of §9 item 6.

### 5.4 Cost model

Per interval, per CTA, then per kernel. Every coefficient comes from the tree or from a body probe.

```text
# one tile of one region on one CTA, with k CTAs active in the interval
t_stream = max( bytes / share(k),                 # shared edge, from the curve
                bytes / body_rate,                # issue-bound bodies (attention)
                flops / engine_rate )
t_tile   = t_fill(mechanism, staged?) + t_stream + (1 − overlap) · t_nonoverlap + t_epilogue · contention(u)

# one interval
t_cta(c) = Σ tiles of c  t_tile                   # waves follow from items and workers
T_int    = max_c t_cta(c) + t_sync(k, kind) + t_skew

# the kernel
T        = Σ intervals T_int
subject to, in every window:   Σ bytes moved ≤ ∫ shared_throughput(k) dt
           at every software level: Σ live reservations ≤ capacity
```

Notes:

- `t_fill` is zero for stages the producer staged before the interval. That is how staging enters the model: it removes fill, and it is limited by free capacity and by the window constraint.
- `contention(u)` raises the cost of latency-bound accesses with the HBM utilization of the background stream. This is the term X1.1 showed to be missing (−1.8% measured against −7% predicted).
- `t_sync` depends on the kind of edge: grid barrier, counter, or none (same CTA).
- `body_rate` and the fixed terms are per body, measured by probes. The log's lesson: costs must come from measured body rates, not from HBM numbers.
- The X2.3 starting form (≈2.0 µs fill + tile bytes / per-CTA share + ≈1.45 µs sync) is this model with constants. It already predicts o_proj and down within about 5%.

Precision target: rank correlation ≥ 0.8 and the right sign on the §2 decisions. Absolute error of 10–15% is acceptable (TileSight reaches 12%), because the top few plans are measured.

### 5.5 Search

The space is small, so exact methods fit.

1. **Deduplicate by shape class.** Layers repeat. A 16-layer model has about ten distinct (einsum, shape) classes. LoopTree and Welder do the same.
2. **Enumerate per shape class.** Body × tile × spatial split × stages: hundreds of points. Use hardware-aligned sizes only (Roller's rule): multiples of the MMA tile, of the TMA box, of the page size.
3. **Prune per class.** Drop points that break a capacity. Keep the Pareto set over (time contribution, shared-memory reservation, register demand) within each compatibility key. This is FFM's rule, and it is where a Pareto front is needed: shared memory and registers are shared by every body in the kernel, so the fastest GEMM alone can be the wrong choice.
4. **Join along the graph.** A decoder is nearly a chain, so dynamic programming over intervals with the compatibility key as state is exact. Interval folding and fusion are joins with a different backing level for the shared tensor.
5. **Plan staging** on the joined plan (§5.6).
6. **Measure the top 2–3** whole kernels, verify them against the FX reference, and cache by device fingerprint and shape class (X2.4). Keep every plan within ~10% of the best for measurement, because register-allocation noise is ±7%.

No ILP, geometric program, genetic search or learned model is needed at this size. Keep X2.5 (a learned residual) as the fallback if P0 or P3 fail their gates.

### 5.6 Staging planner (the megakernel-only part)

Input: the static schedule (which CTA reads which bytes in which interval), the tree, and the cost model. Output: for each CTA, an ordered list of loads the producer issues ahead of need.

- **Budget:** free software-managed capacity per level over time, from `reservations`.
- **Window:** HBM idle time, from the window constraint in §5.4.
- **Price:** the contention term. Run-ahead that delays an epilogue or an attention load has a cost; the planner can pause the producer in latency-critical phases (X1.1 open item 1).
- **Scope:** weights first (they depend on nothing), then K and V for attention (X1.1 open item 3).

Today the ring size and the issue order are fixed rules in `codegen.py` and `cutlass_gemm.py`. This step derives them, which is what makes the producer portable to a device with a different staging depth.

### 5.7 Calibration

`calibration.py` measures HBM bandwidth, the bandwidth curve over CTAs, the L2 curve, the grid barrier, the counter round trip and tensor-core peaks. Add:

| Probe | Gives | Why |
|---|---|---|
| TMA first-byte latency and steady rate against stages in flight | `mechanisms.tma_bulk` | The ~1–1.5 µs per GEMM interval; listed as not measured in X0.1 |
| Thread-issued load rate per warp, by load width | `mechanisms.thread_load` | Issue-bound bodies; the only mechanism on AMD |
| Store bandwidth and latency; activation round trip across SMs | `write`, `activation_round_trip_s` | HazyResearch's largest term; bound B3 |
| Latency of a small access under a background stream at 0%, 50%, 90% utilization | `contention` | X1.1 finding 3 |
| Compute-and-load overlap on one SM | `overlap_with_compute` | ECM lesson: do not assume a max |
| Per-body probes: two or three shapes per body in a minimal persistent harness; fit fixed + per-byte + per-item | `body_rate`, fixed terms | Replaces the `DECODE_*` rates fitted on the slice (X0.5) |

Rules: every parameter is measured on the device or read from a capability query. None is fitted on a model. Each has a repeatability check, as X0.1 does. Probes run inside a persistent cooperative kernel, because rates in a standalone kernel differ (X0.4).

### 5.8 Bounds report

Add the ladder of §3.4 to `tools/phase_profile.py --sol` and to the scoreboard: B0 to B4 per interval and per kernel. A small HBL module (one linear program per einsum, `scipy.optimize.linprog`) gives B1 and tile aspect ratios for any einsum. For prefill shapes, run Orojenesis or AccelForge offline and store the curves as test data.

---

## 6. Known gaps: what none of this work handles

| Gap | State of the literature | What Megabake needs |
|---|---|---|
| **Inter-SM contention on shared edges** | Timeloop: a hard cap. TileSight: recomputes the share for tail waves; no contention formula; notes B200 SM-to-HBM affinity as unmodelled. ECM: an empirical latency penalty near saturation, for CPUs. GPU-Tile-Sim: queueing models, simulation only | The measured curve plus a contention coefficient (§5.4, §5.7) |
| **Synchronization cost** | Absent from the Timeloop line and from TileSight. Multi-BSP has the parameter but no GPU instantiation. MPK lists globally coordinated scheduling as future work | Barrier, counter and round-trip costs per level; sync kind as a mapping choice |
| **TMA / asynchronous-copy pipelining** | TileSight models stage count as an effective depth; not TMA queue limits or issue blocking. GPU-Tile-Sim models TMA at ~100 B/cycle in 128 B transactions | Issue blocking is why step 3 failed (a p90 of ~4,000 cycles to issue 13 stages). Needs in-flight limits per mechanism |
| **Warp specialization and roles** | Only GPU-Tile-Sim represents it (executor groups, order edges). No analytical model | Roles as part of `reservations`; per-role register budgets |
| **Register pressure and code shape** | Nobody. TileSight excludes it | Not modelled. Handle by measuring the top plans and by the inert-code control |
| **Fill and drain across op boundaries** | TileSight has a prologue and epilogue per kernel. No model of state that persists across ops | The staging planner; persistent ring state is already built |
| **Dependence-chain latency** | The pebble game has no clock. HazyResearch measured it; no model | Bound B3; fusion and counters are its levers |
| **Run-time sizes** | All tools are static | Evaluate plans at several KV lengths; plan switches on the device (as `decode_split_steps`) |
| **Hardware-managed caches in a megakernel** | TileSight's L2 model assumes a uniform tile rate; desynchronized SMs broke it (82% predicted, 43% measured) | For decode, treat L2 as a calibrated edge property, not a level that can hold a tile. Revisit for prefill |
| **MIG slices and partial GPUs** | Nobody | Already handled: calibration is per fingerprint, including the MIG profile |
| **Numerics as a constraint on mappings** | Nobody. FFM treats mappings as equivalent if the einsum is | Megabake's rounding points restrict fusion and retention; the compatibility key must carry dtype and rounding |

---

## 7. Is it feasible?

Yes, with three conditions.

1. **It must be judged on ranking, not absolute time.** The decision set of §2 is the acceptance test. If no analytical form gets those signs right, the fallback is a learned residual over the same features (X2.5).
2. **A full GPU is needed before the coverage terms can be trusted.** The slice saturates HBM with 48 CTAs. On 132 SMs the curve, the wave structure and the value of splitting all change. Roadmap step 0 already asks for this.
3. **Portability claims must wait for a second mechanism family.** Until a non-Hopper provider plans and runs from a tree file, the schema is a hypothesis. B200 is the cheap test; AMD or TPU is the real one.

Cost: the plan below is one small step, three medium, one medium-to-large and one large. It replaces roadmap step 7 and X2.3–X2.4 and feeds steps 9, 11 and 17, so most of it is work the roadmap already owes.

Risks:

| Risk | Sign | Response |
|---|---|---|
| Latency-bound regimes resist analytical modelling (TileSight says so) | P0 sign test fails on decisions 3, 6, 7 | Per-body probes; then X2.5 |
| Model drift between isolated probes and the megakernel | Probe rates differ from in-kernel stamps by > 15% | Calibrate from in-kernel stamps (`MEGABAKE_GEMM_STAMPS`, `MEGABAKE_ATTENTION_STAMPS`) |
| Schema overfits Hopper | The B200 file needs schema changes | Fix the schema before adding bodies; this is the tier 5 rule |
| Compile time grows | Search adds minutes | Shape-class deduplication; cache by fingerprint; measure at most three kernels |

---

## 8. Plan of action

Format follows the log: objective, method, pass, kill. Effort: S, M, L.

### P0. Retro-validation gate (S, about one week). Do this first

- **Objective:** know which model form explains what is already measured, before building anything.
- **Method:**
  1. Write the eight decisions of §2 as a table of (plan A, plan B, measured sign, measured size), with shapes and CTA counts from the artifacts.
  2. Implement the cost model of §5.4 as a pure function over `KernelPlan`, `MachineModel` and a small dict of body rates taken from the log. No compiler changes.
  3. Score three forms: (a) the brief's `max(compute, bytes/bandwidth)`; (b) the X2.3 starting form; (c) the §5.4 form.
  4. Compute per-interval error on the Llama-1B and SmolLM-135M profiles in `artifacts/x01_2026-11-07`.
- **Pass:** form (c) gets the sign right on at least 7 of 8 decisions and per-interval rank correlation ≥ 0.8.
- **Kill:** no form reaches 6 of 8. Then build the feature extraction only and go to X2.5.
- **Output:** `tests/unit/test_cost_model_decisions.py` (the decision set as a regression test) and a table in the log.

### P1. `DeviceTree` and calibration (M)

- **Objective:** one device file holds every number the planner reads.
- **Method:** add the schema of §5.1 and a loader; make `MachineModel` and `TargetFacts` views of it; add the probes of §5.7; write spec-only tree files for H100, B200, MI300X and TPU v7 for CPU-side planning tests.
- **Pass:** calibration repeats within 2% (X0.1's rule); the planner reads no constant outside the tree; P0's test still passes with calibrated values in place of log values.
- **Kill:** none (infrastructure).
- **Replaces:** the `TargetFacts` part of roadmap step 7.

### P2. Einsum view, traffic and bounds (M)

- **Objective:** bytes, FLOPs, reuse and bounds come from index sets.
- **Method:** `EinsumView` from `TensorProgram`; per-level traffic and footprint for a `RegionMapping`; rewrite `accounting.phase_traffic` on it; the HBL module; the bound ladder in `phase_profile.py --sol`.
- **Pass:** traffic equals today's `phase_traffic` on the full zoo (or each difference is explained); the view derives the KV-group packing rule from the attention einsum with no attention-specific code; B0–B3 are reported per interval.
- **Kill:** none.

### P3. Mapspaces and search (M–L)

- **Objective:** the planner picks tile, split, stages and body from the model, then measures.
- **Method:** providers declare mapspaces; implement §5.5 steps 1–4 and 6; remove the `max_plans=1` overrides in `compiler.py`; replace `_body_cost` as the ranking key.
- **Pass:** fitted-constant debt D falls (the `DECODE_*` rates, the fixed 64-row tile); P_heldout does not fall; cold search ≤ 10 minutes per model; on a synthetic 132-SM tree the planner chooses work items ≥ SM count for q/k/v/o/down without a rule that says so.
- **Kill:** rank correlation between predicted and measured plans < 0.6 on held-out cells after two iterations (X2.3's rule).
- **Replaces:** roadmap step 7, X2.3, X2.4. Feeds steps 9 and 11.

### P4. Staging planner (M)

- **Objective:** ring size, issue order and producer pauses come from the tree.
- **Method:** §5.6. Start with GEMM weights, then attention K and V.
- **Pass:** equal or better than the fixed rules on all 11 decode cells of X1.1; the Llama-1B epilogue growth (0.1–0.6 → 1.0–2.0 µs) is reduced; no ring constant remains in `codegen.py`.
- **Kill:** no cell improves beyond noise against the inert-code control. Then keep the fixed rules and record the tree's staging depth as a diagnostic only.

### P5. Second target (L)

- **Objective:** test the claim "new chip in the same family = tree file".
- **Method:** B200 first: a calibrated tree, the sm100 provider (roadmap step 17), no planner changes allowed. Then one design spike for a different family: write the MI300X and TPU v7 trees and list what the bodies and the backend need. Do not build that backend yet.
- **Pass:** the planner and the cost model change by zero lines for B200; P0's decision test, re-measured on B200, passes.
- **Kill:** the schema needs a structural change for B200. Fix the schema first.

### Order and dependencies

```text
P0 ──> P1 ──> P3 ──> P4
  └──> P2 ──┘         └──> P5
```

P1 and P2 can run in parallel after P0. A full-GPU calibration run (roadmap step 0) should happen before P3's pass criteria are judged.

---

## 9. Reading list, in priority order

**Read first (they change the design).**

1. TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model from Cores to Clusters. https://arxiv.org/abs/2607.22432. Closest prior art; read §3–4 for the envelope and cache models and the limitations section.
2. Fast and Fusiest: An Optimal Fusion-Aware Mapper. https://arxiv.org/abs/2602.15166. Compatibility keys, Pareto pruning over reservations.
3. The Turbo-Charged Mapper. https://arxiv.org/abs/2602.15172. Dataplacement, pruning rules, the curried symbolic model.
4. LoopTree: Exploring the Fused-layer Dataflow Accelerator Design Space. https://arxiv.org/abs/2409.13625. Mapping representation; retain, refetch, recompute.
5. HazyResearch, Look Ma, No Bubbles! https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles. The B200 time breakdown.
6. MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs (OSDI 2026). https://arxiv.org/abs/2512.22219. Task graph, pages, cross-task pipelining.
7. Welder: Scheduling Deep Learning Memory Access via Tile-graph (OSDI 2023). https://www.usenix.org/conference/osdi23/presentation/shi. Code: https://github.com/nox-410/Welder.

**Bounds and theory.**

8. Orojenesis (ISCA 2024). https://people.csail.mit.edu/emer/media/papers/2024.06.isca.orojenesis.pdf; tool: https://timeloop.csail.mit.edu/orojenesis.
9. Christ, Demmel, Knight, Scanlan, Yelick. Communication lower bounds and optimal algorithms for programs that reference arrays. https://arxiv.org/abs/1308.0068. Theorem 7.1: tile sizes from the dual LP.
10. Kwasniewski et al. Red-blue pebbling revisited (SC 2019). https://arxiv.org/abs/1908.09606.
11. Böhnlein, Papp, Yzelman. Red-Blue Pebbling with Multiple Processors. https://arxiv.org/abs/2409.03898.
12. Olivry et al. IOLB: https://arxiv.org/abs/1911.06664; IOOpt (PLDI 2021): https://pldi21.sigplan.org/details/pldi-2021-papers/78/.
13. Savage. Space-time tradeoffs in memory hierarchies (Brown CS-93-08). https://cs.brown.edu/research/pubs/techreports/reports/CS-93-08.html. Hong and Kung, I/O complexity: the red-blue pebble game (STOC 1981).
14. Valiant. A bridging model for multi-core computing (multi-BSP, 2008). Alpern, Carter, Ferrante. Modeling parallel computers as memory hierarchies (1993). Sequoia (SC 2006): https://lph.ece.utexas.edu/merez/uploads/MattanErez/sc06_sequoia.pdf.
15. SOLAR: https://arxiv.org/abs/2606.26383. SOL-ExecBench: https://arxiv.org/abs/2603.19173.

**Performance models and machine descriptions.**

16. Hofmann, Alappat, Hager, Fey, Wellein. Bridging the architecture gap (ECM machine models). https://arxiv.org/abs/1907.00048. Kerncraft: https://arxiv.org/abs/1702.04653.
17. GPU-Tile-Sim (MICRO 2026). https://arxiv.org/abs/2607.11262.
18. LLMCompass (ISCA 2024). https://arxiv.org/abs/2312.03134.
19. Timeloop documentation, architecture and mapper. https://timeloop.csail.mit.edu/v4/input-formats/design/architecture. AccelForge: https://arxiv.org/abs/2609.11906, https://github.com/Accelergy-Project/accelforge.
20. TileFlow (MICRO 2023). https://github.com/pku-liang/TileFlow.

**Polyhedral and tile-size optimization.**

21. Bondhugula et al. A practical automatic polyhedral parallelizer and locality optimizer (Pluto, PLDI 2008). https://pluto-compiler.sourceforge.net/.
22. Verdoolaege et al. Polyhedral parallel code generation for CUDA (PPCG, TACO 2013). https://dblp.org/rec/journals/taco/VerdoolaegeJCGTC13.html.
23. CoSA (ISCA 2021). https://arxiv.org/abs/2105.01898. MOpt (ASPLOS 2021). https://arxiv.org/abs/2101.09808.
24. Renganarayana, Rajopadhye. A geometric programming framework for optimal multi-level tiling (SC 2004).
25. Roller (OSDI 2022). https://www.usenix.org/conference/osdi22/presentation/zhu. Chimera (HPCA 2023), DOI 10.1109/HPCA56546.2023.10071018.

**Hardware.**

26. NVIDIA Hopper and Blackwell tuning guides. https://docs.nvidia.com/cuda/hopper-tuning-guide/ and https://docs.nvidia.com/cuda/archive/13.4.2/blackwell-tuning-guide/.
27. Chips and Cheese. AMD's CDNA 3 compute architecture. https://old.chipsandcheese.com/2023/12/17/amds-cdna-3-compute-architecture/.
28. Pallas TPU pipelining and kernel docs. https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html. Inferact, TPU megakernels. https://inferact.ai/blog/tpu-megakernels. Jouppi et al., Ten lessons from three generations shaped Google's TPUv4i (ISCA 2021).
29. TileLang. https://arxiv.org/abs/2504.17577. Hexcute. https://arxiv.org/abs/2504.16214. Cohere megakernels. https://cohere.com/blog/megakernels.

---

## 10. Not verified

Check these before relying on them.

- Welder's internals beyond its abstract (the search algorithm, profiling of top candidates). The paper PDF could not be parsed in this environment.
- Chimera's method; the Pluto cost function and the MLIR, Tiramisu and Tensor Comprehensions summaries. These are recalled, not re-read.
- The exact form of COSMA's bound (2mnk/√S plus lower-order terms).
- B200 L2 size (sources give 64 MB to 126 MB) and TMEM size and latency (secondary sources).
- AMD LDS size on CDNA 3 and whether HIP guarantees co-resident workgroups for a persistent kernel.
- Whether software can direct the MI300X Infinity Cache.
- TPU v7 numbers come from the Inferact post, not from Google documentation.
- TileSight reports 14.83% in its text and 16.18% in its abstract for fused distributed kernels. TCM's pruning figures also differ between its abstract and its tables.
- The "expected prediction" column in §2 is my reasoning about each model form. P0 replaces it with computed results.
- The staging-depth table in §3.6 is rough arithmetic from capacities and bandwidths.
