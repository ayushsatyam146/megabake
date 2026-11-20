"""Batch-1 decode GEMV on CUDA cores: out[0, n] = post(sum_k x[k] * W[n, k]).

A swapAB WGMMA tile is 64 weight rows, so a decode projection with N outputs
is N / 64 work items: 32 for a 2,048-wide projection, a quarter of a 132-SM
H200. This body gives each CTA a tile of output columns (``gemv_rows``: about
one round over the SMs, so every row's loads are in flight at once) and splits their
K range over all of its threads, each streaming 16-byte vectors of the
weight rows (K contiguous BF16 in ``nn.Linear``'s own [N, K] layout); warps
reduce by shuffles, the CTA through shared memory, and one thread per column
evaluates the program's post expression (rounding, bias, residual add).

Measured standalone on a full H200 (experiments/newarch/gemv_probe,
2026-11-20), against cuBLAS: 2048x2048 5.3 vs 7.0 us, 512x2048 3.6 vs 6.6
us, 2048x8192 13.1 vs 15.4 us. The loop shape follows mvm's ``gemv_bf16``
(github.com/Bobliuuu/mvm, feat/serve-decode-lowering).

On by default (``MEGABAKE_GEMV=0`` turns it off); it is offered only for
decode GEMMs whose 64-row tiles would fill at most half the SMs. Decode
kernel on a full H200 (2026-11-20, off -> on): SmolLM-135M 881 -> 763 us,
Qwen2-0.5B 1,018 -> 892, DeepSeek draft 1,012 -> 878, SmolLM-1.7B 1,593 ->
1,578, gemma-2b 2,200 -> 1,968, Gemma-3-270m 1,259 -> 1,206, Qwen3.5-0.8B
4,177 -> 3,444, Llama-3.2-1B 1,161 -> 1,050-1,057, Beam DFlash2 1,348 ->
1,363.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os

from ...interface import (
    BodyConfig,
    CandidateSet,
    CompletionContract,
    ParticipantContract,
    PipelineContract,
    ProviderRejection,
    ResourceEvidence,
    ScratchContract,
    SelectionPolicy,
    SemanticContract,
    TargetFacts,
    TileContract,
    access_effects,
    widened_participants,
)
from ....graph import Region
from ....ir import TensorProgram

# 16-byte vectors per thread and row in flight before the first is used, and
# the bounds of a tile's output columns (a multiple of 4).
GEMV_UNROLL = 2
GEMV_MIN_ROWS = 8
GEMV_MAX_ROWS = 24
GEMV_THREADS = 256


def gemv_enabled() -> bool:
    return os.environ.get("MEGABAKE_GEMV", "1") in {"1", "epilogue"}


@dataclass(frozen=True)
class GeneratedGemvBody:
    source: str
    function_name: str = "run"
    threads_per_cta: int = GEMV_THREADS
    tile_count: int = 1
    work_step: int = 1
    scratch_elements: int = (GEMV_THREADS // 32) * GEMV_MIN_ROWS
    work_iterator_dtype: str = "torch.int64"
    work_iterator_encoding: str = "one output-column tile index per CTA"
    arguments: tuple[str, ...] = ("work_iterator", "tensor_bindings", "scratch", "epilogue")


def _reduction_size(program: TensorProgram) -> int:
    return next(
        domain.size for domain in program.iteration_domains if domain.axis in program.reduction_axes
    )


def gemv_rows(columns: int, workers: int) -> int:
    """Output columns per tile: one round of tiles over the SMs, so each
    CTA has all its rows' loads in flight at once instead of finishing a
    small tile per latency round trip (4 rows per tile measured 12.7 us for
    Llama-1B's q/k/v in the kernel, against 10.8 us for the WGMMA body)."""
    low = int(os.environ.get("MEGABAKE_GEMV_MIN_ROWS", GEMV_MIN_ROWS))
    high = int(os.environ.get("MEGABAKE_GEMV_MAX_ROWS", GEMV_MAX_ROWS))
    rows = -(-columns // workers)
    rows = -(-rows // 4) * 4
    return max(low, min(high, rows))


def gemv_rejection(program: TensorProgram, target: TargetFacts, policy: SelectionPolicy) -> str | None:
    """Why the GEMV body does not take ``program``, or None."""
    from .gemm import _epilogue_rejection, paired_contraction

    if program.family != "contraction":
        return "provider supports contraction only"
    if policy.mode != "decode" or not policy.table_launch:
        return "GEMV runs decode table launches only"
    if paired_contraction(program):
        return "paired GEMMs stay on the tensor-core body"
    if len(program.store.shape) != 2 or program.store.shape[0] != 1:
        return "GEMV takes one output row"
    columns = program.store.shape[1]
    if program.store.strides != (columns, 1):
        return "GEMV stores a contiguous row"
    # Where 64-row tiles fill over half the SMs the ring-fed WGMMA body wins:
    # Beam DFlash2's 5,120-wide projections (80 tiles on 132 SMs) measured
    # 1,348 us on it and 1,495 us as GEMVs (2026-11-20).
    if columns > 32 * target.multiprocessor_count:
        return "64-row tensor-core tiles already fill half the SMs"
    if len(program.loads) < 2 or any(load.dtype != "torch.bfloat16" for load in program.loads[:2]):
        return "GEMV reads BF16 operands"
    if program.store.dtype != "torch.bfloat16" or program.accumulator_dtype != "torch.float32":
        return "GEMV stores BF16 from an FP32 accumulator"
    reduction = _reduction_size(program)
    if reduction % 8:
        return "K must be a multiple of 8 (16-byte vectors)"
    left, right = program.loads[:2]
    left_strides = dict(zip(left.axes, left.strides, strict=True))
    right_strides = dict(zip(right.axes, right.strides, strict=True))
    if left_strides.get("r0") != 1 or set(left_strides) - {"o0", "r0"}:
        return "activation must be K-contiguous"
    if right_strides.get("r0") != 1 or "o1" not in right_strides or set(right_strides) - {"o1", "r0"}:
        return "weights must be [N, K] with K contiguous"
    if right_strides["o1"] % 8:
        return "weight rows must be 16-byte aligned"
    if os.environ.get("MEGABAKE_GEMV") == "epilogue" and len(program.loads) <= 2:
        return "MEGABAKE_GEMV=epilogue: plain GEMMs stay on the tensor-core body"
    if len(program.loads) > 2:
        reason = _epilogue_rejection(program, policy)
        if reason is not None:
            return reason
    try:
        _post_code(program)
    except Exception as error:  # noqa: BLE001 - any unsupported expression rejects the body
        return f"post expression is not supported: {error}"
    return None


def _binding_indices(program: TensorProgram) -> dict[str, int]:
    """Load id -> index into the body's tensor bindings, which hold each input
    value once in first-load order, then the output (codegen's slots)."""
    values = list(dict.fromkeys(load.value_id for load in program.loads))
    return {load.id: values.index(load.value_id) for load in program.loads}


def _post_code(program: TensorProgram) -> str:
    from ..generic import _cast_expression, _emit_expression

    assert program.post is not None
    value = _emit_expression(
        program.post, _binding_indices(program), {"o0": "o0", "o1": "o1"}, tuple(program.loads)
    )
    return _cast_expression(value, program.store.dtype).code


def emit_gemv_body(program: TensorProgram, config: BodyConfig) -> GeneratedGemvBody:
    threads = config.participants.participating_threads
    rows = config.tiles.cta_shape[1]
    warps = threads // 32
    columns = program.store.shape[1]
    reduction = _reduction_size(program)
    vectors = reduction // 8
    # At most 16 weight vectors in flight per thread (128 FP32 registers).
    unroll = GEMV_UNROLL if vectors >= GEMV_UNROLL * threads and rows * GEMV_UNROLL <= 16 else 1
    indices = _binding_indices(program)
    left, right = program.loads[:2]
    x_index, w_index = indices[left.id], indices[right.id]
    row_stride = dict(zip(right.axes, right.strides, strict=True))["o1"]
    output_index = len(set(indices.values()))
    tiles = -(-columns // rows)
    binding_type = ", ".join("cute.Tensor" for _ in range(output_index + 1))

    def vector(base: str, offset: str) -> str:
        return (
            "cute.make_tensor(cute.make_ptr(cutlass.BFloat16, "
            f"({base} + {offset}).toint(), cute.AddressSpace.gmem, assumed_align=16), "
            "cute.make_layout(8)).load().to(cutlass.Float32)"
        )

    lines = []
    lines += [f"        row_{r} = n0 + {r}" for r in range(rows)]
    lines += [
        f"        if row_{r} >= {columns}:\n            row_{r} = cutlass.Int32({columns - 1})"
        for r in range(rows)
    ]
    lines += [f"        acc_{r} = cutlass.Float32(0.0)" for r in range(rows)]
    lines.append(f"        for base in cutlass.range(0, {vectors}, {threads * unroll}):")
    for u in range(unroll):
        lines.append(f"            j_{u} = base + {u * threads} + thread")
        lines.append(f"            if j_{u} < {vectors}:")
        lines.append(f"                x_{u} = {vector('x_base', f'j_{u} * 8')}")
        for r in range(rows):
            lines.append(
                f"                w_{u}_{r} = "
                + vector("w_base", f"cutlass.Int64(row_{r}) * {row_stride} + cutlass.Int64(j_{u}) * 8")
            )
        for r in range(rows):
            lines.append(
                f"                acc_{r} = acc_{r} + "
                f"(x_{u} * w_{u}_{r}).reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0)"
            )
    for r in range(rows):
        lines.append(f"        acc_{r} = cute.arch.warp_reduction_sum(acc_{r})")
    lines.append("        if lane == 0:")
    lines += [f"            scratch[warp * {rows} + {r}] = acc_{r}" for r in range(rows)]
    lines.append("        cute.arch.sync_threads()")
    lines.append(f"        if thread < {rows}:")
    lines.append("            o1 = n0 + thread")
    lines.append(f"            if o1 < {columns}:")
    lines.append(
        "                accumulator = "
        + " + ".join(f"scratch[{w * rows} + thread]" for w in range(warps))
    )
    lines.append("                o0 = cutlass.Int32(0)")
    # The output slot's layout need not have the program's rank: store
    # through the program's own (contiguous) row.
    lines.append(f"                stored = {_post_code(program)}")
    lines.append("                cute.make_tensor(output.iterator + o1, cute.make_layout(1))[0] = stored")
    lines.append("        cute.arch.sync_threads()")
    body = "\n".join(lines)

    source = f'''import cutlass
import cutlass.cute as cute


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    output = tensor_bindings[{output_index}]
    x_base = tensor_bindings[{x_index}].iterator
    w_base = tensor_bindings[{w_index}].iterator
    lane = thread % 32
    warp = thread // 32
    for tile in cutlass.range(block, {tiles}, workers):
        n0 = cutlass.Int32(tile) * {rows}
{body}
'''
    return GeneratedGemvBody(
        source=source,
        threads_per_cta=threads,
        tile_count=tiles,
        scratch_elements=warps * rows,
    )


def gemv_config(
    region: Region, program: TensorProgram, workers: int, threads: int = GEMV_THREADS
) -> BodyConfig:
    from ..generic import _layouts

    warps = threads // 32
    rows = gemv_rows(program.store.shape[1], workers)
    epilogue = len(program.loads) > 2
    return BodyConfig(
        provider=GemvBodyProvider.provider_name,
        name=f"gemv_r{rows}" + ("_epilogue" if epilogue else ""),
        semantic=SemanticContract(
            operation_family="contraction",
            source_op_ids=region.source_op_ids,
            numerical_policy="source_semantics",
            expression_hooks=(),
            access_effects=access_effects(program),
        ),
        layouts=_layouts(program, alignment_bytes=16),
        tiles=TileContract(
            cta_shape=(1, rows),
            cluster_shape=(1, 1),
            work_partition=f"{rows} output columns per CTA, K split over its threads",
            mma_family=None,
            reduction_tile=(_reduction_size(program),),
        ),
        participants=ParticipantContract(
            participating_threads=threads,
            waiting_threads=threads,
            warp_roles=("worker",) * warps,
            registers_per_thread=64,
        ),
        pipeline=PipelineContract(
            stage_count=1,
            barriers=(),
            async_operations=(),
            drain_rule="no asynchronous pipeline to drain",
        ),
        scratch=ScratchContract(owner="cta", size_bytes=warps * rows * 4, alignment_bytes=16),
        completion=CompletionContract(
            outputs_ready="assigned output columns are stored when run returns",
            scratch_reusable="scratch is reusable after the closing CTA barrier",
            register_fragments="none",
        ),
        evidence=ResourceEvidence(
            measured=False,
            source="CUDA-core batch-1 GEMV, K split over the CTA (mvm gemv_bf16 loop shape)",
            revision="2026-11-20",
            resources={"threads": threads, "shared_bytes": warps * rows * 4},
            measurements={},
        ),
    )


class GemvBodyProvider:
    provider_name = "gemv"

    def get_candidates(
        self, region: Region, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        def reject(reason: str) -> CandidateSet:
            return CandidateSet((), (ProviderRejection(self.provider_name, reason),))

        if not gemv_enabled():
            return reject("GEMV bodies are off (MEGABAKE_GEMV=0)")
        if not policy.allow_tuned:
            return reject("tuned providers disabled")
        if not region.tensor_programs or len(region.tensor_programs) != 1:
            return reject("region has no single tensor program")
        program = region.tensor_programs[0]
        reason = gemv_rejection(program, target, policy)
        if reason is not None:
            return reject(reason)
        return CandidateSet((gemv_config(region, program, target.multiprocessor_count),), ())

    def emit(
        self,
        program: TensorProgram,
        config: BodyConfig,
        *,
        defer_length_publish: bool = False,
        workers: int | None = None,
    ) -> GeneratedGemvBody:
        del defer_length_publish, workers
        return emit_gemv_body(program, config)

    def widen(self, program: TensorProgram, config: BodyConfig, threads: int) -> BodyConfig:
        warps = threads // 32
        return replace(
            config,
            participants=widened_participants(config, threads),
            scratch=replace(config.scratch, size_bytes=warps * config.tiles.cta_shape[1] * 4),
        )
