from __future__ import annotations

from dataclasses import dataclass, replace
import os
import textwrap
from functools import cache
from typing import Any, Mapping

from ....graph import Region
from ....ops import HF_EAGER_NUMERICS, Attention, HeadGrouping, Prologue
from ...interface import (
    BodyConfig,
    CandidateSet,
    CompletionContract,
    LayoutContract,
    ParticipantContract,
    PipelineContract,
    ProviderRejection,
    ResourceEvidence,
    ScratchContract,
    SelectionPolicy,
    SemanticContract,
    TargetFacts,
    TensorLayout,
    TileContract,
    access_effects,
    widened_participants,
)


def split_decode_head_dim(head_dim: int) -> bool:
    """Head dims the split-context decode body serves: whole 16-byte vectors,
    at most one per lane of a warp."""
    return head_dim % 8 == 0 and head_dim <= 256


def decode_window(attention: Attention) -> int:
    """Keys a stateful decode query sees (the new key included), 0 for all."""
    offsets = attention.mask.window_offsets
    return 1 - offsets[0] if offsets is not None else 0


class UnsupportedAttentionBody(ValueError):
    pass


def _attention_sequence_length(load: Any) -> int | None:
    if load.source_shape is None:
        return None
    for index, axis in enumerate(load.axes):
        if axis in {"s", "k"}:
            return load.source_shape[index]
    return None


def _attention_shape_is_valid(
    load: Any, batch: int, length: int, heads: int, head_dim: int
) -> bool:
    return load.source_shape in {
        (batch, length, heads, head_dim),
        (batch, heads, length, head_dim),
    }


def _attention_access(
    load: Any, head_variable: str, binding_index: int, dimension_variable: str = "d"
) -> str:
    indices = []
    for axis in load.axes:
        if isinstance(axis, str):
            if axis == "k":
                indices.append("key")
            elif axis == "d":
                indices.append(dimension_variable)
            else:
                indices.append(axis)
        elif isinstance(axis, int):
            indices.append(str(axis))
        elif axis.terms[0].modulus == 0:
            indices.append("b")
        else:
            indices.append(head_variable)
    return f"tensor_bindings[{binding_index}][{', '.join(indices)}]"


@dataclass(frozen=True)
class GeneratedAttentionBody:
    source: str
    function_name: str = "run"
    threads_per_cta: int = 128
    tile_count: int = 1
    # Work-iterator stride between tiles (``GeneratedBody.work_step``).
    work_step: int = 1
    work_iterator_dtype: str = "torch.int64"
    work_iterator_encoding: str = "one row-block base per CTA tile"
    # Phase scratch (FP32 elements, Int32 counters) the body takes as its
    # epilogue argument: (floats, counters, float offset, counter offset).
    phase_floats: int = 0
    phase_counters: int = 0


class AttentionBodyProvider:
    provider_name = "attention"

    def emit(
        self,
        program: Any,
        config: BodyConfig,
        *,
        defer_length_publish: bool = False,
        workers: int | None = None,
    ) -> GeneratedAttentionBody:
        return emit_attention_body(
            program, config, defer_length_publish=defer_length_publish, workers=workers
        )

    def widen(self, program: Any, config: BodyConfig, threads: int) -> BodyConfig:
        # Decode attention keeps a statistic and a head-dim row of value
        # partials per warp in its shared scratch.
        head_dim = config.tiles.cta_shape[-1]
        added_warps = (threads - config.participants.participating_threads) // 32
        return replace(
            config,
            participants=widened_participants(config, threads),
            scratch=replace(
                config.scratch,
                size_bytes=config.scratch.size_bytes + added_warps * (1 + head_dim) * 4,
            ),
        )

    def get_candidates(
        self, region: Region, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        def reject(reason: str) -> CandidateSet:
            return CandidateSet((), (ProviderRejection(self.provider_name, reason),))

        if not policy.allow_tuned:
            return reject("tuned providers disabled")
        if not region.tensor_programs or len(region.tensor_programs) != 1:
            return reject("region has no single tensor program")

        program = region.tensor_programs[0]
        if program.family != "attention":
            return reject("provider supports attention only")
        attention = region.semantics
        if not isinstance(attention, Attention):
            return reject("region semantics are not attention")

        if attention.numerics != HF_EAGER_NUMERICS:
            return reject("provider preserves only the captured BF16/float32 policy")
        if program.accumulator_dtype != "torch.float32":
            return reject("attention accumulator must be float32")
        if program.store.dtype != "torch.bfloat16":
            return reject("attention output must be BF16")
        allowed_outputs = {
            (program.store.value_id,),
        }
        if attention.boundary_output is not None:
            allowed_outputs.update(
                (
                    (attention.boundary_output,),
                    (program.store.value_id, attention.boundary_output),
                )
            )
        if attention.outputs not in allowed_outputs:
            return reject("materialized scores and extra attention outputs are unsupported")

        loads = {load.id: load for load in program.loads}
        if not {"attention:q", "attention:k", "attention:v"} <= set(loads):
            return reject("attention program lacks Q/K/V accesses")
        if any(
            loads[load_id].dtype != "torch.bfloat16"
            for load_id in ("attention:q", "attention:k", "attention:v")
        ):
            return reject("provider supports BF16 Q/K/V only")

        grouping = attention.grouping
        batch, query_heads, kv_heads, head_dim = (
            grouping.batch,
            grouping.query_heads,
            grouping.kv_heads,
            grouping.head_dim,
        )
        if min(batch, query_heads, kv_heads, head_dim) <= 0 or query_heads % kv_heads:
            return reject("attention head grouping is invalid")

        q_load = loads["attention:q"]
        k_load = loads["attention:k"]
        v_load = loads["attention:v"]
        query_length = _attention_sequence_length(q_load)
        kv_length = _attention_sequence_length(k_load)
        if (
            query_length is None
            or kv_length is None
            or not _attention_shape_is_valid(
                q_load, batch, query_length, query_heads, head_dim
            )
            or not _attention_shape_is_valid(
                k_load, batch, kv_length, kv_heads, head_dim
            )
            or not _attention_shape_is_valid(
                v_load, batch, kv_length, kv_heads, head_dim
            )
            or program.store.shape
            != (batch * query_heads, query_length, head_dim)
        ):
            return reject("attention shapes do not match the captured access maps")

        mask = attention.mask
        if mask.kind == "additive_tensor":
            if "attention:mask" not in loads:
                return reject("additive mask is not bound")
        elif mask.kind == "stateful_causal":
            if not program.state_accesses:
                return reject("stateful causal mask has no KV state contract")
            if (mask.window_offsets is not None or attention.softcap) and not (
                policy.table_launch and split_decode_head_dim(head_dim)
            ):
                return reject("sliding windows and soft caps need the split-context decode body")
        elif mask.kind == "typed_where":
            if mask.window_offsets is not None or not isinstance(
                mask.false_value, (int, float)
            ):
                return reject("provider supports additive tensors and causal masks only")
            if mask.causal_offset is None and (
                "attention:mask" not in loads or mask.true_value not in (0, 0.0)
            ):
                return reject("typed mask lacks a bound boolean condition and zero true value")
        elif mask.kind != "none":
            return reject("unsupported attention mask hook")

        cache = attention.cache
        stateful = bool(program.state_accesses)
        if (
            cache.policy != ("stateful" if stateful else "read_only")
            or (stateful and cache.updates == ())
            or (not stateful and cache.appends != ())
            or (not stateful and cache.updates != ())
        ):
            return reject("provider supports read-only or bounded stateful KV caches only")
        if stateful and query_length != 1:
            return reject("stateful KV attention supports one-token decode only")
        if attention.softcap and not stateful:
            return reject("soft caps are declared for stateful decode only")
        if attention.prologues and not (
            stateful and policy.table_launch and split_decode_head_dim(head_dim)
        ):
            # Only the table launch's split-context decode body computes them.
            return reject("query and key prologues need the split-context decode body")

        if target.architecture != "sm90a" or not target.supports_wgmma:
            return reject("provider targets WGMMA-capable sm90a only")
        if attention_scratch_rows(kv_length) < 1:
            return reject(
                f"a {kv_length}-key FP32 score row exceeds {ATTENTION_SCRATCH_BYTES} scratch bytes"
            )
        body_name = (
            "attention_stateful_decode_rows"
            if stateful
            else "attention_decode_rows"
            if query_length == 1
            else "attention_prefill_rows"
        )
        return CandidateSet(
            (
                _attention_config(
                    region,
                    program,
                    body_name,
                    target.multiprocessor_count,
                    stateful and policy.table_launch and split_decode_head_dim(head_dim),
                ),
            ),
            (),
        )


# Each warp owns one query row at a time and keeps that row's FP32 scores in
# shared scratch; the scratch holds at most this many rows and this many bytes.
ATTENTION_SCRATCH_ROWS = 16
ATTENTION_SCRATCH_BYTES = 32 * 1024


# Value-sum keys loaded together before accumulation in the decode body.
_VALUE_BLOCK = 8
# Keys per warp iteration of the split-context decode value sum.
_VALUE_UNROLL = 16


def attention_scratch_rows(kv_length: int) -> int:
    return min(ATTENTION_SCRATCH_ROWS, ATTENTION_SCRATCH_BYTES // (kv_length * 4))


def _attention_config(
    region: Region, program: Any, name: str, workers: int, split: bool
) -> BodyConfig:
    """``split`` sizes the scratch for the table launch's split-context decode
    body, which ``workers`` CTAs run."""
    attention = region.semantics
    stateful = attention.kv_state is not None
    threads = 128
    warp_roles = ("one query row at a time: lanes split keys, then head dimensions",) * 4
    kv_capacity = next(
        length
        for load in program.loads
        if load.id == "attention:k"
        for length in (_attention_sequence_length(load),)
        if length is not None
    )
    scratch_bytes = attention_scratch_rows(kv_capacity) * kv_capacity * 4
    if split:
        # The split-context decode body's statistics, query rows and a
        # chunk's scores when ``workers`` CTAs run it (see _DecodeScratch).
        scratch_bytes = max(
            scratch_bytes,
            decode_scratch_bytes(
                attention.grouping,
                min(kv_capacity, decode_window(attention) or kv_capacity),
                threads,
                workers,
            ),
        )
    elif stateful:
        head_dim = attention.grouping.head_dim
        warps = threads // 32
        rows = 2 if attention.prologues else 0
        scratch_bytes = max(
            scratch_bytes, (warps + warps * head_dim + kv_capacity + rows * head_dim) * 4
        )
    layouts = {
        load.value_id: TensorLayout(
            load.dtype,
            load.axes,
            load.strides,
            16,
            load.source_shape,
        )
        for load in program.loads
    }
    layouts[program.store.value_id] = TensorLayout(
        program.store.dtype,
        ("bh", "s", "d"),
        program.store.strides,
        16,
        program.store.shape,
    )
    return BodyConfig(
        provider="attention",
        name=name,
        semantic=SemanticContract(
            operation_family="attention",
            source_op_ids=region.source_op_ids,
            numerical_policy="source_semantics",
            expression_hooks=("score", "mask"),
            access_effects=access_effects(program),
            semantics=attention,
        ),
        layouts=LayoutContract(layouts, {program.store.value_id: layouts[program.store.value_id]}),
        tiles=TileContract(
            cta_shape=(1, 1, next(load.source_shape[3] for load in program.loads if load.id == "attention:q")),
            cluster_shape=(1, 1, 1),
            work_partition=(
                "one query row per warp in one CTA" if stateful else "one query row per warp"
            ),
            mma_family=None,
            reduction_tile=(
                next(
                    length
                    for load in program.loads
                    if load.id == "attention:k"
                    for length in (_attention_sequence_length(load),)
                    if length is not None
                ),
            ),
        ),
        participants=ParticipantContract(
            participating_threads=threads,
            waiting_threads=0,
            warp_roles=warp_roles,
            registers_per_thread=48,
        ),
        pipeline=PipelineContract(
            stage_count=3,
            barriers=(),
            async_operations=(),
            drain_rule=(
                "the score-maximum, softmax-denominator, and value-accumulation "
                "passes complete before return"
            ),
        ),
        scratch=ScratchContract(
            owner="cta",
            size_bytes=scratch_bytes,
            alignment_bytes=16,
        ),
        completion=CompletionContract(
            outputs_ready=(
                "assigned rows and all KV state writes are stored when run returns"
                if stateful
                else "assigned rows are stored when run returns"
            ),
            scratch_reusable="scratch is reusable when run returns",
            register_fragments="row score, softmax maximum and total, and value accumulators",
        ),
        evidence=ResourceEvidence(
            measured=not stateful,
            source=(
                "stateful decode extension of the CuTe row pipeline adapted from "
                "Mirage multitoken Hopper attention"
                if stateful
                else "CuTe row pipeline adapted from Mirage multitoken Hopper attention"
            ),
            revision=(
                "mirage@f9eb70c254acefc9f3667b2a973d0dcf25471fce; task-28"
                if stateful
                else "mirage@f9eb70c254acefc9f3667b2a973d0dcf25471fce; task-26"
            ),
            resources={
                "threads": threads,
                "shared_bytes": scratch_bytes,
                "registers_per_thread": 48,
                "active_blocks_per_sm": 10,
            },
            measurements={
                "target": "NVIDIA H200 MIG 3g.71GB",
                "workload": "batch=2, query=5, kv=7, query_heads=4, kv_heads=2, head_dim=8",
                "samples": 8,
                "median_isolated_launch_us": 16.794,
                "median_device_event_span_us": 137.040,
                "median_complete_call_us": 150.397,
            },
        ),
    )


# Split-context decode: a query row's visible keys split into up to this many
# chunks, each one CTA work item; ``decode_split_steps`` picks how many.
DECODE_MAX_CHUNKS = 16
# Packed rows: the query heads of an item stay within this many FP32 query
# elements, and an item packs its heads from this many visible keys per head.
# shortcut: fitted on SmolLM-135M, Qwen2-0.5B, gemma-2b and Qwen3.5-0.8B
# decode, 60-SM H200 slice (2026-10-27); replace with step 11's cost model.
DECODE_PACKED_FLOATS = 512
DECODE_PACKED_KEYS_PER_HEAD = 128
DECODE_PACKED_HEAD_DIM = 128

# Keys each lane of the coalesced score loop has in flight.
DECODE_SCORE_UNROLL = 4

# Diagnostics, off unless set (tools/attention_stages.py): stamp each stage
# of the decode body, or force the chunks in use for split-cost sweeps.
DECODE_STAMPS = os.environ.get("MEGABAKE_ATTENTION_STAMPS") == "1"
_FORCE_CHUNKS = int(os.environ.get("MEGABAKE_ATTENTION_FORCE_CHUNKS", "0"))
DECODE_STAMP_STRIDE = 16
# Split cost model (X2a, 2026-11-07). An item costs a fixed time plus its
# key work, which is at least one load round; a split row also pays one
# statistics exchange and a value merge that grow with chunks, heads and
# head_dim, and items past one wave of CTAs run in another pass. Fitted to
# forced-chunk sweeps on the 60-SM H200 slice (Llama-3.2-1B, SmolLM-135M,
# Qwen2-0.5B, Qwen3-0.6B, gemma-3-270m at 113-4,013 keys).
# shortcut: these are one body's rates on one GPU; they belong in a per-body
# calibration (X0.1-style), not here. They are not HBM rates: at head_dim 64
# a CTA reads K and V at ~16 GB/s, against a ~59 GB/s calibrated HBM share.
DECODE_ITEM_FIXED_S = 1.2e-6
DECODE_ITEM_MIN_WORK_S = 1.8e-6
DECODE_EXCHANGE_S = 2.2e-6
DECODE_MERGE_HEAD_S = 2.0e-8
DECODE_MERGE_DIM_S = 5.0e-10
# Seconds per visible key for one head, by head_dim, and per extra head.
DECODE_KEY_S = {64: 16e-9, 128: 34e-9, 256: 170e-9}
DECODE_HEAD_KEY_S = 2.0e-9
# Each extra pass of items re-enters the item loop late (measured ~1.4 µs
# from phase entry to the first item).
DECODE_PASS_S = 1.4e-6
_LAST_STAMP_BASE = 0


def emit_attention_body(
    program: Any,
    config: BodyConfig,
    *,
    defer_length_publish: bool = False,
    workers: int | None = None,
) -> GeneratedAttentionBody:
    """Emit an attention device body.

    With ``defer_length_publish`` the stateful decode body leaves the KV
    valid-length increment to the enclosing kernel, which must perform it after
    the grid barrier that ends this phase's barrier interval. Its query rows can
    then spread across CTAs, because no CTA can observe the new length while
    another is still reading the old one.
    """
    if program.family != "attention":
        raise UnsupportedAttentionBody("program is not attention")
    attention = config.semantic.semantics
    if not isinstance(attention, Attention):
        raise UnsupportedAttentionBody("attention body lacks its attention semantics")
    grouping = attention.grouping
    batch, query_heads, group_size, head_dim = (
        grouping.batch,
        grouping.query_heads,
        grouping.group_size,
        grouping.head_dim,
    )
    scale = attention.scale

    loads = {load.id: index for index, load in enumerate(program.loads)}
    required_loads = ("attention:q", "attention:k", "attention:v")
    if not set(required_loads) <= set(loads):
        raise UnsupportedAttentionBody("attention body lacks Q/K/V bindings")
    query_load = next(load for load in program.loads if load.id == "attention:q")
    key_load = next(load for load in program.loads if load.id == "attention:k")
    query_length = _attention_sequence_length(query_load)
    kv_length = _attention_sequence_length(key_load)
    if not isinstance(query_length, int) or not isinstance(kv_length, int):
        raise UnsupportedAttentionBody("attention sequence lengths must be static")

    if program.state_accesses:
        state = program.state_accesses[0]
        required_state_loads = (
            "attention:new_k",
            "attention:new_v",
            "attention:valid_length",
        )
        if not set(required_state_loads) <= set(loads):
            raise UnsupportedAttentionBody("stateful attention lacks K/V and length bindings")
        if query_length != 1 or state.query_length != 1:
            raise UnsupportedAttentionBody("stateful attention supports one-token decode only")
        threads = config.participants.participating_threads
        row_count = batch * query_heads
        if not isinstance(scale, (int, float)):
            raise UnsupportedAttentionBody("attention scale must be static")
        warps = min(threads // 32, config.scratch.size_bytes // (kv_length * 4))
        if warps < 1:
            raise UnsupportedAttentionBody(
                "stateful attention scratch must hold one score row per warp"
            )
        prologues = attention.prologues
        if defer_length_publish and workers is not None and split_decode_head_dim(head_dim):
            return _split_decode_attention_body(
                program, loads, batch, query_heads // group_size, group_size, head_dim,
                kv_length, scale, threads, workers, config.scratch.size_bytes,
                prologues, decode_window(attention), attention.softcap,
            )
        if attention.mask.window_offsets is not None or attention.softcap:
            raise UnsupportedAttentionBody(
                "sliding windows and soft caps need the split-context decode body"
            )
        if prologues:
            raise UnsupportedAttentionBody(
                "query and key prologues need the split-context decode body"
            )
        if defer_length_publish:
            # One warp-sized group of rows per CTA tile.
            tile_count = (row_count + warps - 1) // warps
            # A single iteration: this warp's row of the tile, if it exists.
            row_loop = (
                f"for bh in cutlass.range(cutlass.Int32(work_iterator[tile]) * {warps} + warp, "
                f"row_count, {warps * row_count}):"
            )
            length_publish = ""
        else:
            # One CTA owns every row so the length publish can follow a CTA barrier.
            tile_count = (row_count + threads - 1) // threads
            if tile_count != 1:
                raise UnsupportedAttentionBody(
                    "stateful attention requires every query row in one CTA"
                )
            row_loop = f"for bh in cutlass.range(warp, row_count, {warps}):"
            length_publish = (
                "        cute.arch.sync_threads()\n"
                "        if thread == 0:\n"
                f"            tensor_bindings[{loads['attention:valid_length']}][0] = current_length + cutlass.Int32(1)\n"
            )
        binding_type = ", ".join("cute.Tensor" for _ in range(len(program.loads) + 1))
        VALUE_BLOCK = _VALUE_BLOCK
        value_loads = "\n                        ".join(
            f"value_{index} = tensor_bindings[{loads['attention:v']}][b, block_start + {index}, kv_head, d]"
            for index in range(VALUE_BLOCK)
        )
        value_sums = "\n                        ".join(
            f"value = value + scratch[base + block_start + {index}] * value_{index}.to(cutlass.Float32)"
            for index in range(VALUE_BLOCK)
        )
        source = f'''import cutlass
import cutlass.cute as cute
from megabake.runtime import trap_if


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch: cute.Tensor,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    row_count = {row_count}
    current_length = tensor_bindings[{loads['attention:valid_length']}][0].to(cutlass.Int32)
    trap_if(current_length < 0 or current_length + cutlass.Int32(1) > cutlass.Int32({kv_length}))
    # Each of the first {warps} warps takes whole query rows: lanes split the
    # visible keys for scores and the head dimensions for the value sum, and
    # the row's scores and probabilities stay in that warp's scratch slice.
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        warp = thread // 32
        lane = thread % 32
        if warp < {warps}:
            {row_loop}
                s = 0
                b = bh // {query_heads}
                h = bh % {query_heads}
                kv_head = h // {group_size}
                base = warp * {kv_length}
                maximum = cutlass.Float32(-1.0e30)
                for key in cutlass.range(lane, current_length + cutlass.Int32(1), 32):
                    score = cutlass.Float32(0.0)
                    if key < current_length:
                        for d in cutlass.range_constexpr({head_dim}):
                            score = score + tensor_bindings[{loads['attention:q']}][b, s, h, d].to(cutlass.Float32) * tensor_bindings[{loads['attention:k']}][b, key, kv_head, d].to(cutlass.Float32)
                    else:
                        for d in cutlass.range_constexpr({head_dim}):
                            score = score + tensor_bindings[{loads['attention:q']}][b, s, h, d].to(cutlass.Float32) * tensor_bindings[{loads['attention:new_k']}][b, 0, kv_head, d].to(cutlass.Float32)
                    score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                    score = score * cutlass.Float32({scale!r})
                    scratch[base + key] = score
                    maximum = cute.arch.fmax(maximum, score, nan=True)
                maximum = cute.arch.warp_reduction_max(maximum)
                total = cutlass.Float32(0.0)
                for key in cutlass.range(lane, current_length + cutlass.Int32(1), 32):
                    probability = cute.exp(scratch[base + key] - maximum)
                    scratch[base + key] = probability
                    total = total + probability
                total = cute.arch.warp_reduction_sum(total)
                denominator = total
                if total <= cutlass.Float32(0.0):
                    denominator = cutlass.Float32(1.0)
                for key in cutlass.range(lane, current_length + cutlass.Int32(1), 32):
                    scratch[base + key] = (scratch[base + key] / denominator).to(cutlass.BFloat16).to(cutlass.Float32)
                cute.arch.sync_warp()
                for d in cutlass.range(lane, {head_dim}, 32):
                    value = cutlass.Float32(0.0)
                    # Cached keys first, then the new key: the source key order.
                    # Eight loads are issued before they are accumulated, so
                    # their latencies overlap; the sum order is unchanged.
                    blocked = current_length - current_length % {VALUE_BLOCK}
                    for block_start in cutlass.range(0, blocked, {VALUE_BLOCK}):
                        {value_loads}
                        {value_sums}
                    for key in cutlass.range(blocked, current_length):
                        value = value + scratch[base + key] * tensor_bindings[{loads['attention:v']}][b, key, kv_head, d].to(cutlass.Float32)
                    value = value + scratch[base + current_length] * tensor_bindings[{loads['attention:new_v']}][b, 0, kv_head, d].to(cutlass.Float32)
                    tensor_bindings[{len(program.loads)}][bh, s, d] = value.to(cutlass.BFloat16)
                # Keys before current_length come from the cache and the new
                # key from new_k, so appending here cannot race with readers.
                if h % {group_size} == 0:
                    for d in cutlass.range(lane, {head_dim}, 32):
                        tensor_bindings[{loads['attention:k']}][b, current_length, kv_head, d] = tensor_bindings[{loads['attention:new_k']}][b, 0, kv_head, d]
                        tensor_bindings[{loads['attention:v']}][b, current_length, kv_head, d] = tensor_bindings[{loads['attention:new_v']}][b, 0, kv_head, d]
                cute.arch.sync_warp()
{length_publish}'''
        return GeneratedAttentionBody(
            source=source,
            threads_per_cta=threads,
            tile_count=tile_count,
            work_iterator_encoding="one stateful query-row block base per CTA tile",
        )

    mask = attention.mask
    if mask.kind == "additive_tensor":
        if "attention:mask" not in loads:
            raise UnsupportedAttentionBody("additive mask is not bound")
        mask_load = next(load for load in program.loads if load.id == "attention:mask")
        if mask_load.source_shape is None or len(mask_load.source_shape) != 4:
            raise UnsupportedAttentionBody("additive mask must be four-dimensional")
        mask_head_is_broadcast = mask_load.source_shape[1] == 1
        mask_access = (
            f"tensor_bindings[{loads['attention:mask']}][b, 0, s, key]"
            if mask_head_is_broadcast
            else f"tensor_bindings[{loads['attention:mask']}][b, h, s, key]"
        )
        mask_code = f"score = (score.to(cutlass.Float32) + ({mask_access}).to(cutlass.Float32)).to(cutlass.BFloat16)"
    elif mask.kind == "typed_where":
        causal_offset = mask.causal_offset
        masked_value = mask.false_value
        if not isinstance(masked_value, (int, float)):
            raise UnsupportedAttentionBody("attention mask lacks a fill value")
        if isinstance(causal_offset, int):
            mask_code = (
                f"score = (score.to(cutlass.Float32) + cutlass.Float32({masked_value!r})).to(cutlass.BFloat16) if key > s + {causal_offset} else score"
            )
        else:
            if "attention:mask" not in loads:
                raise UnsupportedAttentionBody("typed attention mask is not bound")
            mask_load = next(
                load for load in program.loads if load.id == "attention:mask"
            )
            mask_access = _attention_access(
                mask_load, "h", loads["attention:mask"]
            )
            mask_code = (
                f"score = (score.to(cutlass.Float32) + cutlass.Float32({masked_value!r})).to(cutlass.BFloat16) if not {mask_access} else score"
            )
    elif mask.kind == "none":
        mask_code = "pass"
    else:
        raise UnsupportedAttentionBody("unsupported attention mask hook")

    threads = config.participants.participating_threads
    row_count = batch * query_heads * query_length
    if not isinstance(scale, (int, float)):
        raise UnsupportedAttentionBody("attention scale must be static")
    output_type = "cutlass.BFloat16"
    binding_type = ", ".join("cute.Tensor" for _ in range(len(program.loads) + 1))
    q_access = _attention_access(query_load, "h", loads["attention:q"])
    k_access = _attention_access(key_load, "kv_head", loads["attention:k"])
    value_load = next(load for load in program.loads if load.id == "attention:v")
    v_access = _attention_access(value_load, "kv_head", loads["attention:v"])
    warps = min(threads // 32, config.scratch.size_bytes // (kv_length * 4))
    if warps < 1:
        raise UnsupportedAttentionBody("attention scratch must hold one score row per warp")
    tile_count = (row_count + warps - 1) // warps
    source = f'''import cutlass
import cutlass.cute as cute


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch: cute.Tensor,
    epilogue,
):
    # Each CTA tile covers {warps} query rows, one per warp: lanes split the keys
    # for scores and the head dimensions for the value sum, with the row's
    # scores and probabilities in that warp's scratch slice. Rounding points
    # match the captured program; the value sum runs in key order.
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    warp = thread // 32
    lane = thread % 32
    row_count = {row_count}
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        row = work_iterator[tile] * {warps} + warp
        if warp < {warps}:
            if row < row_count:
                bh = row // {query_length}
                s = row % {query_length}
                b = bh // {query_heads}
                h = bh % {query_heads}
                kv_head = h // {group_size}
                base = warp * {kv_length}
                maximum = cutlass.Float32(-1.0e30)
                for key in cutlass.range(lane, {kv_length}, 32):
                    score = cutlass.Float32(0.0)
                    for d in cutlass.range_constexpr({head_dim}):
                        score = score + {q_access}.to(cutlass.Float32) * {k_access}.to(cutlass.Float32)
                    score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                    score = (score * cutlass.Float32({scale!r})).to(cutlass.BFloat16)
                    {mask_code}
                    score = score.to(cutlass.Float32)
                    scratch[base + key] = score
                    maximum = cute.arch.fmax(maximum, score, nan=True)
                maximum = cute.arch.warp_reduction_max(maximum)
                total = cutlass.Float32(0.0)
                for key in cutlass.range(lane, {kv_length}, 32):
                    probability = cute.exp(scratch[base + key] - maximum)
                    scratch[base + key] = probability
                    total = total + probability
                total = cute.arch.warp_reduction_sum(total)
                denominator = total
                if total <= cutlass.Float32(0.0):
                    denominator = cutlass.Float32(1.0)
                for key in cutlass.range(lane, {kv_length}, 32):
                    scratch[base + key] = (scratch[base + key] / denominator).to(cutlass.BFloat16).to(cutlass.Float32)
                cute.arch.sync_warp()
                for d in cutlass.range(lane, {head_dim}, 32):
                    value = cutlass.Float32(0.0)
                    for key in cutlass.range({kv_length}, unroll=8):
                        value = value + scratch[base + key] * {v_access}.to(cutlass.Float32)
                    tensor_bindings[{len(program.loads)}][bh, s, d] = value.to({output_type})
                cute.arch.sync_warp()
'''
    return GeneratedAttentionBody(
        source=source,
        threads_per_cta=threads,
        tile_count=tile_count,
        work_iterator_encoding="one query-row block base per CTA tile",
    )


def _prologue_lines(
    prologue: Prologue, program: Any, head: str, target: str, name: str
) -> str:
    """Lines storing to ``target`` (as FP32) an elementwise producer of a
    (batch, sequence, head, dim) attention operand evaluated at (b,
    sequence, ``head``, d), rounded as the producer stores it, from the
    attention's own bindings."""
    from ..generic import _HOISTED_LOADS, _cast_expression, _emit_expression, _load_access

    bindings = {load.value_id: index for index, load in enumerate(program.loads)}
    loads = prologue.loads
    indices = {load.id: bindings[load.value_id] for load in loads}
    coordinates = {"o0": "b", "o1": "sequence", "o2": head, "o3": "d"}
    indent = "                    "
    # Each load is bound once, which keeps the expression within the DSL
    # preprocessor's recursion limit.
    hoisted = {load.id: f"{name}_load_{index}" for index, load in enumerate(loads)}
    lines = "".join(
        f"{indent}{hoisted[load.id]} = {_load_access(load, indices[load.id], loads, coordinates).code}\n"
        for load in loads
    )
    token = _HOISTED_LOADS.set(hoisted)
    try:
        value = _cast_expression(
            _emit_expression(prologue.combine, indices, coordinates, loads),
            prologue.dtype,
        )
    finally:
        _HOISTED_LOADS.reset(token)
    return lines + f"{indent}{target} = ({value.code}).to(cutlass.Float32)\n"


def _decode_key_seconds(heads: int, head_dim: int) -> float:
    known = sorted(DECODE_KEY_S)
    base = DECODE_KEY_S.get(head_dim)
    if base is None:
        # Nearest measured head_dim, scaled by bytes.
        nearest = min(known, key=lambda dim: abs(dim - head_dim))
        base = DECODE_KEY_S[nearest] * head_dim / nearest
    return base + (heads - 1) * DECODE_HEAD_KEY_S


@cache
def decode_split_steps(
    rows: int, workers: int, heads: int, head_dim: int, kv_length: int
) -> tuple[tuple[int, int], ...]:
    """(first visible key count, chunks in use) steps of the split cost model:
    for each key count the chunk count with the least modeled interval time,
    never decreasing as keys grow. Empty means one chunk at every length.

    c chunks of ceil(K / c) keys leave none empty when (c - 1)**2 < K, and K
    only grows within a step, so a step to c starts past (c - 1)**2 keys. An
    empty last chunk would take the new key a second time (X2a: Gemma-3 at
    143 keys in 14 chunks); recomputing the chunk count on the device instead
    cost ~0.8 µs per layer."""
    limit = min(DECODE_MAX_CHUNKS, workers)
    if _FORCE_CHUNKS:
        forced = min(_FORCE_CHUNKS, limit)
        return tuple(((count - 1) ** 2 + 1, count) for count in range(2, forced + 1))
    per_key = _decode_key_seconds(heads, head_dim)

    def cost(keys: int, chunks: int) -> float:
        passes = -(-rows * chunks // workers)
        work = max(DECODE_ITEM_MIN_WORK_S, per_key * -(-keys // chunks))
        exchange = (
            DECODE_EXCHANGE_S + chunks * (DECODE_MERGE_HEAD_S * heads + DECODE_MERGE_DIM_S * head_dim)
            if chunks > 1
            else 0.0
        )
        return passes * (DECODE_ITEM_FIXED_S + work + exchange) + (passes - 1) * DECODE_PASS_S

    steps: list[tuple[int, int]] = []
    current = 1
    for keys in range(1, kv_length + 1):
        allowed = [chunks for chunks in range(1, limit + 1) if chunks == 1 or (chunks - 1) ** 2 < keys]
        best = min(allowed, key=lambda chunks: (cost(keys, chunks), chunks))
        if best > current:
            steps.append((keys, best))
            current = best
    return tuple(steps)


@cache
def _modeled_capacity(steps: tuple[tuple[int, int], ...], kv_length: int) -> int:
    capacity, in_use, index = 1, 1, 0
    for keys in range(1, kv_length + 1):
        while index < len(steps) and keys >= steps[index][0]:
            in_use = steps[index][1]
            index += 1
        capacity = max(capacity, -(-keys // in_use))
    return capacity


@dataclass(frozen=True)
class _DecodeScratch:
    """Shared-scratch layout of the split-context decode body, in FP32
    elements: a flag slot per warp, one head's warp value partials (which
    hold every head's per-warp statistics before the value sum), the group's
    query rows, the computed new-key row, each head's row maximum and
    denominator, then the chunk's scores. Per warp it takes one statistic and
    a head-dim row, as codegen assumes when it widens the body to the CTA."""

    warps: int
    groups: int
    head_dim: int

    @property
    def values(self) -> int:
        return self.warps

    @property
    def query_rows(self) -> int:
        return self.values + self.warps * self.head_dim

    @property
    def key_row(self) -> int:
        return self.query_rows + self.groups * self.head_dim

    @property
    def row_stats(self) -> int:
        return self.key_row + self.head_dim

    @property
    def scores(self) -> int:
        return self.row_stats + 2 * self.groups

    def bytes(self, chunk_keys: int = 0) -> int:
        return (self.scores + self.groups * chunk_keys) * 4


def decode_packed_heads(group_size: int, head_dim: int) -> int:
    """Query heads one item serves at long contexts: the largest divisor of the
    KV group whose query rows fit in DECODE_PACKED_FLOATS. Wider items keep
    more FP32 accumulators live than registers hold.

    Rows wider than DECODE_PACKED_HEAD_DIM already keep a whole FP32 key row
    per thread, and a second variant pushes the whole kernel into spills:
    Qwen3.5-0.8B decode ran +1.4% at short contexts with it, so those models
    keep one query head per row until per-role register budgets (step R1)."""
    if head_dim > DECODE_PACKED_HEAD_DIM:
        return 1
    return max(
        heads
        for heads in range(1, group_size + 1)
        if group_size % heads == 0 and heads * head_dim <= DECODE_PACKED_FLOATS
    )


def decode_scratch_bytes(
    head_grouping: HeadGrouping, kv_length: int, threads: int, workers: int
) -> int:
    """Shared scratch the split-context decode body needs to keep a chunk's
    scores in shared memory when ``workers`` CTAs run it."""
    packed = decode_packed_heads(head_grouping.group_size, head_grouping.head_dim)
    sizes = []
    for heads in dict.fromkeys((1, packed)):
        rows = head_grouping.batch * head_grouping.query_heads // heads
        layout = _DecodeScratch(threads // 32, heads, head_grouping.head_dim)
        steps = decode_split_steps(rows, workers, heads, head_grouping.head_dim, kv_length)
        sizes.append(layout.bytes(_modeled_capacity(steps, kv_length)))
    return max(sizes)


@dataclass(frozen=True)
class _DecodeVariant:
    block: str
    tile_count: int
    phase_floats: int
    phase_counters: int


def _split_decode_attention_body(
    program: Any,
    loads: Mapping[str, int],
    batch: int,
    kv_heads: int,
    group_size: int,
    head_dim: int,
    kv_length: int,
    scale: float,
    threads: int,
    workers: int,
    scratch_bytes: int = 0,
    prologues: Mapping[str, Prologue] | None = None,
    window: int = 0,
    softcap: float = 0.0,
) -> GeneratedAttentionBody:
    """One-token decode with each row's visible keys split into chunks that
    run on different CTAs (flash decoding), keeping the captured rounding
    points: BF16 scores, FP32 softmax statistics, BF16 normalized
    probabilities, and an FP32 value sum.

    A row is one query head at short contexts, where spreading heads over
    CTAs pays more than sharing their key and value reads. From
    DECODE_PACKED_KEYS_PER_HEAD visible keys per packed head, a row is the
    query heads of a KV group that ``decode_packed_heads`` packs together, so
    each cached key and value is read once for all of them (see
    _decode_variant). Every CTA reads the same valid length, so all take the
    same variant; the variants share scratch and counters.

    The visible keys are the last ``window`` keys when ``window`` is set
    (sliding-window attention over a linear cache), else all of them; a
    ``softcap`` maps each scaled score s to softcap * tanh(s / softcap).
    """
    length_index = loads["attention:valid_length"]
    binding_type = ", ".join("cute.Tensor" for _ in range(len(program.loads) + 1))
    packed = decode_packed_heads(group_size, head_dim)
    def make_variants(stamp_base: int | None) -> list[_DecodeVariant]:
        return [
            _decode_variant(
                program, loads, batch, kv_heads, group_size, head_dim, kv_length, scale,
                threads, workers, scratch_bytes, prologues or {}, heads, stamp_base,
                window, softcap,
            )
            for heads in dict.fromkeys((1, packed))
        ]

    variants = make_variants(None)
    stamp_floats = 0
    if DECODE_STAMPS:
        global _LAST_STAMP_BASE
        stamp_base = max(variant.phase_floats for variant in variants)
        _LAST_STAMP_BASE = stamp_base
        variants = make_variants(stamp_base)
        stamp_floats = workers * DECODE_STAMP_STRIDE
    window_lines = (
        f"    if keys > cutlass.Int32({window}):\n        first = keys - cutlass.Int32({window})\n"
        if window
        else ""
    )
    blocks = variants[0].block
    if len(variants) == 2:
        blocks = (
            f"    if visible < cutlass.Int32({DECODE_PACKED_KEYS_PER_HEAD * packed}):\n"
            + textwrap.indent(variants[0].block, "    ")
            + "    else:\n"
            + textwrap.indent(variants[1].block, "    ")
        )
    source = f'''import cutlass
import cutlass.cute as cute
from megabake.runtime import trap_if


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch: cute.Tensor,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    floats, counters, float_base, counter_base = epilogue
    current_length = tensor_bindings[{length_index}][0].to(cutlass.Int32)
    trap_if(current_length < 0 or current_length + cutlass.Int32(1) > cutlass.Int32({kv_length}))
    keys = current_length + cutlass.Int32(1)
    first = cutlass.Int32(0)
{window_lines}    visible = keys - first
    warp = thread // 32
    lane = thread % 32
{blocks}'''
    return GeneratedAttentionBody(
        source=source,
        threads_per_cta=threads,
        tile_count=max(variant.tile_count for variant in variants),
        work_iterator_encoding="one (row, key chunk) item per CTA tile",
        phase_floats=max(variant.phase_floats for variant in variants) + stamp_floats,
        phase_counters=max(variant.phase_counters for variant in variants),
    )


def _decode_variant(
    program: Any,
    loads: Mapping[str, int],
    batch: int,
    kv_heads: int,
    group_size: int,
    head_dim: int,
    kv_length: int,
    scale: float,
    threads: int,
    workers: int,
    scratch_bytes: int,
    prologues: Mapping[str, Prologue],
    heads_per_item: int,
    stamp_base: int | None = None,
    window: int = 0,
    softcap: float = 0.0,
) -> _DecodeVariant:
    """The item loop of split-context decode with ``heads_per_item`` query
    heads of one KV group per row, so each cached key and value row is read
    once per chunk for all of them. Work item ``row * CHUNKS + chunk`` covers
    one chunk of one row. Warps take keys; in the value sum lanes take head
    dimensions.

    Each chunk publishes its per-head maxima and exponential sums (against its
    own maximum) to phase scratch and waits until every chunk in use has; the
    row maximum and denominator then follow from the chunk statistics. The
    wait cannot deadlock because a row's items are consecutive and there are
    at most as many chunks as CTAs, so a waiting CTA only waits on items that
    are some CTA's current or next item. The last chunk to finish sums the
    chunks' FP32 value partials in chunk order, so the result does not depend
    on which CTA finishes last, and resets the row's counters. A single chunk
    in use skips the exchange and its fences.

    A chunk's scores, then its probabilities, are read only by the CTA that
    computes them, so they stay in shared scratch whenever they fit there;
    otherwise they go through phase scratch.

    ``prologues`` computes the query or new key from its elementwise producer
    (RoPE) into shared scratch, rounded as the producer stores it; the new key
    appended to the cache is that row.
    """
    warps = threads // 32
    # Each item serves ``groups`` query heads of one KV head: a tile of its group.
    groups = heads_per_item
    tiles = group_size // groups
    query_heads = kv_heads * group_size
    rows = batch * kv_heads * tiles
    # At most this many keys are visible to a row.
    visible_limit = min(kv_length, window) if window else kv_length
    steps = decode_split_steps(rows, workers, groups, head_dim, visible_limit)
    chunks = steps[-1][1] if steps else 1
    capacity = _modeled_capacity(steps, visible_limit)
    choose_in_use = "    in_use = cutlass.Int32(1)\n" + "".join(
        f"    if visible >= cutlass.Int32({first}):\n        in_use = cutlass.Int32({count})\n"
        for first, count in steps
    )
    layout = _DecodeScratch(warps, groups, head_dim)
    if layout.bytes() > scratch_bytes:
        raise UnsupportedAttentionBody("attention scratch cannot hold the decode statistics")
    scores_in_shared = layout.bytes(capacity) <= scratch_bytes
    q_index = loads["attention:q"]
    k_index = loads["attention:k"]
    v_index = loads["attention:v"]
    new_k_index = loads["attention:new_k"]
    new_v_index = loads["attention:new_v"]
    output_index = len(program.loads)
    # Phase scratch: scores (rows x chunks x heads x chunk keys) when they do
    # not fit in shared scratch, per-chunk statistics and value partials for
    # every head; two counters per row.
    score_floats = 0 if scores_in_shared else rows * chunks * groups * capacity
    stats_base = score_floats
    partial_base = stats_base + rows * chunks * groups * 2
    floats = partial_base + rows * chunks * groups * head_dim
    dims_per_lane = -(-head_dim // 32)
    dim_tail = head_dim % 32
    # Value accumulators of one pass stay in registers: heads share a pass in
    # tiles, and fewer keys are unrolled when a tile is wide.
    head_tile = max(t for t in range(1, groups + 1) if groups % t == 0 and t * dims_per_lane <= 64)
    unroll = max(1, min(_VALUE_UNROLL, 72 // dims_per_lane - head_tile))

    def lines(template: str, indent: str, *, store: bool = False, **values: object) -> str:
        """One line per head dimension a lane holds. A partial last dimension
        (head_dim not a multiple of 32) reads a wrapped, in-row index, so its
        loads stay in bounds and its sums are never used, and guards stores."""
        out = []
        for j in range(dims_per_lane):
            partial = dim_tail and j == dims_per_lane - 1
            d = f"(lane + {32 * j}) % {head_dim}" if partial and not store else f"lane + {32 * j}"
            line = indent + template.format(j=j, d=d, **values)
            if partial and store:
                line = f"{indent}if lane < {dim_tail}:\n    {line}"
            out.append(line)
        return "\n".join(out)

    def score_load(g: int, key: str) -> str:
        if scores_in_shared:
            return f"scratch[{layout.scores + g * capacity} + {key} - start]"
        return (
            f"cute.arch.load(floats.iterator + scores + {g * capacity} + {key} - start, "
            "cutlass.Float32, cop=\"cg\")"
        )

    def score_store(g: int, key: str, value: str) -> str:
        if scores_in_shared:
            return f"scratch[{layout.scores + g * capacity} + {key} - start] = {value}"
        return (
            f"cute.arch.store(floats.iterator + scores + {g * capacity} + {key} - start, "
            f"{value}, cop=\"cg\")"
        )

    def shared_row(offset: int) -> str:
        return f"cute.make_tensor(scratch.iterator + {offset}, cute.make_layout({head_dim})).load()"

    def per_head(template: str, indent: str) -> str:
        return "\n".join(indent + template.format(g=g) for g in range(groups))

    # The group's query rows, and the computed new key, in shared scratch.
    staging = "".join(
        f"            head_{g} = head_base + {g}\n" for g in range(groups)
    )
    for g in range(groups):
        target = f"scratch[{layout.query_rows + g * head_dim} + d]"
        staging += f"            for d in cutlass.range(thread, {head_dim}, {threads}):\n"
        if "attention:q" in prologues:
            staging += _prologue_lines(
                prologues["attention:q"], program, f"head_{g}", target, f"query_{g}"
            )
        else:
            staging += (
                f"                {target} = "
                f"tensor_bindings[{q_index}][b, 0, head_{g}, d].to(cutlass.Float32)\n"
            )
    if "attention:new_k" in prologues:
        staging += f"            for d in cutlass.range(thread, {head_dim}, {threads}):\n"
        staging += _prologue_lines(
            prologues["attention:new_k"], program, "kv_head", f"scratch[{layout.key_row} + d]", "key"
        )
    if prologues:
        staging = "            sequence = cutlass.Int32(0)\n" + staging
    new_key_row = (
        shared_row(layout.key_row)
        if "attention:new_k" in prologues
        else f"tensor_bindings[{new_k_index}][(b, 0, kv_head, None)].load().to(cutlass.Float32)"
    )
    new_key_value = (
        f"scratch[{layout.key_row} + d].to(cutlass.BFloat16)"
        if "attention:new_k" in prologues
        else f"tensor_bindings[{new_k_index}][b, 0, kv_head, d]"
    )

    # One query row stays in registers across the key loop; a group's rows
    # would spill, so each key reloads them from shared scratch (measured
    # faster than a key pass per head).
    hoisted_query = (
        f"            query_0 = {shared_row(layout.query_rows)}\n" if groups == 1 else ""
    )

    def query(g: int) -> str:
        return "query_0" if groups == 1 else shared_row(layout.query_rows + g * head_dim)

    def capped(indent: str) -> str:
        if not softcap:
            return ""
        return (
            f"{indent}score = cutlass.Float32({softcap!r}) * cute.math.tanh(score * cutlass.Float32({1.0 / softcap!r}))\n"
        )

    def scores_for(row: str, key: str, indent: str) -> str:
        return "\n".join(
            f"{indent}score = cutlass.Float32(({query(g)} * {row})"
            ".reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0))\n"
            f"{indent}score = score.to(cutlass.BFloat16).to(cutlass.Float32)\n"
            f"{indent}score = score * cutlass.Float32({scale!r})\n"
            + capped(indent)
            + f"{indent}{score_store(g, key, 'score')}\n"
            f"{indent}maximum_{g} = cute.arch.fmax(maximum_{g}, score, nan=True)"
            for g in range(groups)
        )

    # Coalesced scores: a key row is read by ``key_lanes`` lanes with one
    # 16-byte load each, then summed with butterfly shuffles. One thread per
    # key with a whole-row load issues head_dim scalar 2-byte loads whose
    # lanes are a KV row apart (X2a: 5.2 of 7.4 µs per item on Llama-1B).
    key_load = next(load for load in program.loads if load.id == "attention:k")
    key_strides = dict(zip(key_load.axes, key_load.strides, strict=True)) if key_load.strides else {}
    # Lanes per key: the 16-byte vectors of a row, rounded up to a power of
    # two for the shuffles; lanes past the row read its last vector and count
    # zero.
    key_vectors = head_dim // 8
    key_lanes = 1 << max(0, (key_vectors - 1).bit_length())
    vector_keys = (
        head_dim % 8 == 0
        and key_lanes <= 32
        and key_load.strides is not None
        and key_strides.get("d") == 1
        and all(stride % 8 == 0 for axis, stride in key_strides.items() if axis != "d")
    )
    key_slots = threads // key_lanes
    score_unroll = max(1, DECODE_SCORE_UNROLL)

    weight = "" if key_lanes == key_vectors else " * score_weight"

    def lane_clamp(indent: str) -> str:
        if key_lanes == key_vectors:
            return f"{indent}score_vector = score_lane\n"
        return (
            f"{indent}score_vector = score_lane\n"
            f"{indent}score_weight = cutlass.Float32(1.0)\n"
            f"{indent}if score_lane >= {key_vectors}:\n"
            f"{indent}    score_vector = cutlass.Int32({key_vectors - 1})\n"
            f"{indent}    score_weight = cutlass.Float32(0.0)\n"
        )

    def vector_scores(indent: str) -> str:
        shuffles = []
        offset = key_lanes // 2
        while offset:
            shuffles.append(offset)
            offset //= 2
        head_lines = "".join(
            f"{indent}    score_key = score_key_{u}\n"
            f"{indent}    score_valid = score_key < cached_stop and score_lane == 0\n"
            f"{indent}    partial = cutlass.Float32((query_part_{g} * key_part_{u}).reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0)){weight}\n"
            + "".join(
                f"{indent}    partial = partial + cute.arch.shuffle_sync_bfly(partial, offset={o})\n" for o in shuffles
            )
            + f"{indent}    if score_valid:\n"
            f"{indent}        score = partial.to(cutlass.BFloat16).to(cutlass.Float32)\n"
            f"{indent}        score = score * cutlass.Float32({scale!r})\n"
            + capped(f"{indent}        ")
            + f"{indent}        {score_store(g, 'score_key', 'score')}\n"
            f"{indent}        maximum_{g} = cute.arch.fmax(maximum_{g}, score, nan=True)\n"
            for u in range(score_unroll)
            for g in range(groups)
        )
        return (
            f"{indent}score_lane = thread % {key_lanes}\n"
            f"{indent}score_slot = thread // {key_lanes}\n"
            + lane_clamp(indent)
            + "".join(
                f"{indent}query_part_{g} = cute.make_tensor(scratch.iterator + {layout.query_rows + g * head_dim} + score_vector * 8, cute.make_layout(8)).load()\n"
                for g in range(groups)
            )
            + f"{indent}for score_base in cutlass.range(start, cached_stop, {key_slots * score_unroll}):\n"
            + "".join(
                f"{indent}    score_key_{u} = score_base + {u * key_slots} + score_slot\n"
                f"{indent}    score_row = score_key_{u}\n"
                f"{indent}    if score_row >= cached_stop:\n"
                f"{indent}        score_row = cached_stop - cutlass.Int32(1)\n"
                f"{indent}    key_view = tensor_bindings[{k_index}][(b, score_row, kv_head, None)]\n"
                f"{indent}    key_part_{u} = cute.make_tensor(cute.make_ptr(cutlass.BFloat16, (key_view.iterator + score_vector * 8).toint(), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout(8)).load().to(cutlass.Float32)\n"
                for u in range(score_unroll)
            )
            + head_lines
        )

    def warp_statistics(name: str, combine: str) -> str:
        return (
            per_head(f"{name}_{{g}} = cute.arch.warp_reduction_{combine}({name}_{{g}})", "            ")
            + "\n            cute.arch.sync_threads()\n            if lane == 0:\n"
            + per_head(f"scratch[{layout.values} + warp * {groups} + {{g}}] = {name}_{{g}}", "                ")
            + "\n            cute.arch.sync_threads()\n"
        )

    def value_pass(tile: range) -> str:
        loads_ = "\n".join(
            f"                key_{u} = base + {u * warps}\n"
            + "\n".join(f"                probability_{u}_{g} = cutlass.Float32(0.0)" for g in tile)
            + f"\n                if key_{u} < cached_stop:\n"
            + "\n".join(f"                    probability_{u}_{g} = {score_load(g, f'key_{u}')}" for g in tile)
            + f"\n                else:\n                    key_{u} = cached_stop - cutlass.Int32(1)\n"
            + lines(
                "row_{u}_{j} = tensor_bindings[" + str(v_index) + "][b, key_{u}, kv_head, {d}].to(cutlass.Float32)",
                "                ",
                u=u,
            )
            for u in range(unroll)
        )
        sums = "\n".join(
            lines("value_{g}_{j} = value_{g}_{j} + probability_{u}_{g} * row_{u}_{j}", "                ", u=u, g=g)
            for u in range(unroll)
            for g in tile
        )
        new_value = "\n".join(
            f"                    probability = {score_load(g, 'current_length')}\n"
            + lines(
                "value_{g}_{j} = value_{g}_{j} + probability * tensor_bindings["
                + str(new_v_index)
                + "][b, 0, kv_head, {d}].to(cutlass.Float32)",
                "                    ",
                g=g,
            )
            for g in tile
        )
        reduce_heads = "".join(
            lines("scratch[" + str(layout.values) + " + warp * " + str(head_dim) + " + {d}] = value_{g}_{j}", "            ", store=True, g=g)
            + "\n            cute.arch.sync_threads()\n"
            f"            for d in cutlass.range(thread, {head_dim}, {threads}):\n"
            f"                value = scratch[{layout.values} + d]\n"
            f"                for other in cutlass.range_constexpr(1, {warps}):\n"
            f"                    value = value + scratch[{layout.values} + other * {head_dim} + d]\n"
            "                if in_use == cutlass.Int32(1):\n"
            f"                    tensor_bindings[{output_index}][b * {query_heads} + head_{g}, 0, d] = value.to(cutlass.BFloat16)\n"
            "                else:\n"
            "                    cute.arch.store(\n"
            f"                        floats.iterator + partials + (chunk * {groups} + {g}) * {head_dim} + d, value, cop=\"cg\"\n"
            "                    )\n"
            "            cute.arch.sync_threads()\n"
            for g in tile
        )
        return (
            "\n".join(lines("value_{g}_{j} = cutlass.Float32(0.0)", "            ", g=g) for g in tile)
            + f"\n            for base in cutlass.range(start + warp, cached_stop, {unroll * warps}):\n"
            + loads_
            + "\n"
            + sums
            + "\n            if stop == keys:\n                if warp == 0:\n"
            + new_value
            + "\n"
            + reduce_heads
        )

    exponentials = "\n".join(
        f"                total_{g} = total_{g} + cute.exp({score_load(g, 'key')} - scratch[{layout.row_stats + 2 * g}])"
        for g in range(groups)
    )
    probabilities = "\n".join(
        f"                denominator = scratch[{layout.row_stats + 2 * g + 1}]\n"
        "                if denominator <= cutlass.Float32(0.0):\n"
        "                    denominator = cutlass.Float32(1.0)\n"
        f"                probability = (cute.exp({score_load(g, 'key')} - scratch[{layout.row_stats + 2 * g}]) / denominator)"
        ".to(cutlass.BFloat16).to(cutlass.Float32)\n"
        f"                {score_store(g, 'key', 'probability')}"
        for g in range(groups)
    )

    def stamp(index: int, indent: str = "            ") -> str:
        # X2a: thread 0 records ns since the item started, after a CTA sync.
        if stamp_base is None:
            return ""
        return (
            f"{indent}if thread == 0:\n"
            f"{indent}    cute.arch.store(floats.iterator + float_base + {stamp_base} + block * {DECODE_STAMP_STRIDE} + {index}, "
            "(cute.arch.globaltimer_lo() - stamp_t0).to(cutlass.Float32), cop=\"cg\")\n"
        )

    def wait(counter: int) -> str:
        return (
            "                cute.arch.sync_threads()\n"
            "                if thread == 0:\n"
            "                    seen = cutlass.Int32(0)\n"
            "                    while seen < in_use:\n"
            "                        seen = cute.arch.atomic_cas(\n"
            f"                            counters.iterator + counter_base + row * 2 + {counter},\n"
            "                            cmp=in_use,\n"
            "                            val=in_use,\n"
            '                            sem="acquire",\n'
            '                            scope="gpu",\n'
            "                        )\n"
            "                cute.arch.sync_threads()\n"
            "                cute.arch.fence_acq_rel_gpu()\n"
        )

    cached_scores = (
        vector_scores("            ")
        if vector_keys
        else (
            f"{hoisted_query}            for key in cutlass.range(start + thread, cached_stop, {threads}):\n"
            f"                key_row = tensor_bindings[{k_index}][(b, key, kv_head, None)].load().to(cutlass.Float32)\n"
            f"{scores_for('key_row', 'key', '                ')}\n"
        )
    )
    # Merges read every chunk's statistics and partials with loads whose
    # count is the compile-time chunk bound, so they issue together; chunks
    # past ``in_use`` are read but not used. A loop to the run-time ``in_use``
    # issued one coherent L2 load at a time (X2a: ~2 µs per exchange).
    pad = "                    "
    stats_merge = (
        "".join(
            f"{pad}stat_max_{o} = cute.arch.load(floats.iterator + stats + ({o} * {groups} + thread) * 2, cutlass.Float32, cop=\"cg\")\n"
            f"{pad}stat_sum_{o} = cute.arch.load(floats.iterator + stats + ({o} * {groups} + thread) * 2 + 1, cutlass.Float32, cop=\"cg\")\n"
            for o in range(chunks)
        )
        + f"{pad}row_maximum = stat_max_0\n"
        + "".join(
            f"{pad}if in_use > cutlass.Int32({o}):\n"
            f"{pad}    row_maximum = cute.arch.fmax(row_maximum, stat_max_{o}, nan=True)\n"
            for o in range(1, chunks)
        )
        + f"{pad}row_total = cutlass.Float32(0.0)\n"
        + "".join(
            f"{pad}if in_use > cutlass.Int32({o}):\n"
            f"{pad}    row_total = row_total + stat_sum_{o} * cute.exp(stat_max_{o} - row_maximum)\n"
            for o in range(chunks)
        )
    )
    vpad = "                        "
    value_merge = (
        "".join(
            f"{vpad}part_{o} = cute.arch.load(floats.iterator + partials + {o * groups * head_dim} + index, cutlass.Float32, cop=\"cg\")\n"
            for o in range(chunks)
        )
        + f"{vpad}value = cutlass.Float32(0.0)\n"
        + "".join(
            f"{vpad}if in_use > cutlass.Int32({o}):\n{vpad}    value = value + part_{o}\n"
            for o in range(chunks)
        )
    )
    # The new key's score: in the coalesced path warp 0 computes it from the
    # shared key row like a cached key (thread 0 alone delayed the chunk that
    # holds it by ~1 µs, X2a); otherwise thread 0 uses the whole query row.
    if vector_keys and "attention:new_k" in prologues:
        shuffles = []
        offset = key_lanes // 2
        while offset:
            shuffles.append(offset)
            offset //= 2
        npad = "                    "
        new_key_score = (
            "            if stop == keys:\n"
            "                if warp == 0:\n"
            f"{npad}new_part = cute.make_tensor(scratch.iterator + {layout.key_row} + score_vector * 8, cute.make_layout(8)).load()\n"
            + "".join(
                f"{npad}partial = cutlass.Float32((query_part_{g} * new_part).reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0)){weight}\n"
                + "".join(f"{npad}partial = partial + cute.arch.shuffle_sync_bfly(partial, offset={o})\n" for o in shuffles)
                + f"{npad}if lane == 0:\n"
                f"{npad}    score = partial.to(cutlass.BFloat16).to(cutlass.Float32)\n"
                f"{npad}    score = score * cutlass.Float32({scale!r})\n"
                + capped(f"{npad}    ")
                + f"{npad}    {score_store(g, 'current_length', 'score')}\n"
                f"{npad}    maximum_{g} = cute.arch.fmax(maximum_{g}, score, nan=True)\n"
                for g in range(groups)
            )
        )
    else:
        new_key_score = (
            "            if stop == keys:\n"
            "                if thread == 0:\n"
            + (textwrap.indent(hoisted_query, "        ") if vector_keys else "")
            + f"                    key_row = {new_key_row}\n"
            + scores_for("key_row", "current_length", "                    ")
            + "\n"
        )
    # The new key's score (thread 0) uses the whole query row.
    new_key_query = hoisted_query if vector_keys else ""
    block = f'''{choose_in_use}    chunk_keys = (visible + in_use - cutlass.Int32(1)) // in_use
    # Only the chunks in use are items, row-major so a row's items are
    # consecutive (see the deadlock note above).
    for item in cutlass.range(block, cutlass.Int32({rows}) * in_use, workers):
        row = item // in_use
        chunk = item % in_use
        if True:
            stamp_t0 = cute.arch.globaltimer_lo()
{"" if stamp_base is None else f"""            if thread == 0:
                cute.arch.store(floats.iterator + float_base + {stamp_base} + block * {DECODE_STAMP_STRIDE} + 10, (stamp_t0 & cutlass.Int32(0xFFFF)).to(cutlass.Float32), cop="cg")
                cute.arch.store(floats.iterator + float_base + {stamp_base} + block * {DECODE_STAMP_STRIDE} + 11, ((stamp_t0 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)).to(cutlass.Float32), cop="cg")
"""}            b = row // {kv_heads * tiles}
            kv_head = row // {tiles} % {kv_heads}
            head_base = kv_head * {group_size} + row % {tiles} * {groups}
            start = first + chunk * chunk_keys
            stop = start + chunk_keys
            if stop > keys:
                stop = keys
            # Cached keys come from the cache; the new key at current_length
            # (in the last chunk) from new_k and new_v.
            cached_stop = stop
            if cached_stop > current_length:
                cached_stop = current_length
            scores = float_base + (row * {chunks} + chunk) * {groups * capacity}
            stats = float_base + {stats_base} + row * {chunks * groups * 2}
            partials = float_base + {partial_base} + row * {chunks * groups * head_dim}
{staging}            cute.arch.sync_threads()
{stamp(1)}            # Scores: one key per thread, whole rows per load, every head of
            # the group per row.
{per_head("maximum_{g} = cutlass.Float32(-1.0e30)", "            ")}
{cached_scores}{new_key_score}{warp_statistics("maximum", "max")}{stamp(2)}            if thread < {groups}:
                chunk_maximum = scratch[{layout.values} + thread]
                for other in cutlass.range_constexpr(1, {warps}):
                    chunk_maximum = cute.arch.fmax(chunk_maximum, scratch[{layout.values} + other * {groups} + thread], nan=True)
                scratch[{layout.row_stats} + thread * 2] = chunk_maximum
            cute.arch.sync_threads()
{stamp(3)}            # Exponentials against the chunk maximum and their sums.
{per_head("total_{g} = cutlass.Float32(0.0)", "            ")}
            for key in cutlass.range(start + thread, stop, {threads}):
{exponentials}
{warp_statistics("total", "sum")}{stamp(4)}            if thread < {groups}:
                chunk_total = scratch[{layout.values} + thread]
                for other in cutlass.range_constexpr(1, {warps}):
                    chunk_total = chunk_total + scratch[{layout.values} + other * {groups} + thread]
                scratch[{layout.row_stats} + thread * 2 + 1] = chunk_total
                if in_use > cutlass.Int32(1):
                    cute.arch.store(floats.iterator + stats + (chunk * {groups} + thread) * 2, scratch[{layout.row_stats} + thread * 2], cop="cg")
                    cute.arch.store(floats.iterator + stats + (chunk * {groups} + thread) * 2 + 1, chunk_total, cop="cg")
            if in_use > cutlass.Int32(1):
                cute.arch.fence_acq_rel_gpu()
                cute.arch.sync_threads()
                if thread == 0:
                    cute.arch.atomic_add(
                        counters.iterator + counter_base + row * 2,
                        cutlass.Int32(1),
                        sem="release",
                        scope="gpu",
                    )
{wait(0)}                if thread < {groups}:
{stats_merge}                    scratch[{layout.row_stats} + thread * 2] = row_maximum
                    scratch[{layout.row_stats} + thread * 2 + 1] = row_total
            cute.arch.sync_threads()
{stamp(5)}            # Normalized BF16 probabilities replace the scores.
            for key in cutlass.range(start + thread, stop, {threads}):
{probabilities}
            cute.arch.sync_threads()
{stamp(6)}            # Value sum: warps take keys, lanes the head dimensions; each
            # iteration issues the loads of several keys before accumulating
            # them in key order. Keys past the cached range read the last
            # cached key with probability 0. Each head's warp partials are
            # summed in warp order through shared scratch.
{"".join(value_pass(range(first, first + head_tile)) for first in range(0, groups, head_tile))}{stamp(7)}            if in_use > cutlass.Int32(1):
                cute.arch.fence_acq_rel_gpu()
                cute.arch.sync_threads()
                if thread == 0:
                    arrived = cute.arch.atomic_add(
                        counters.iterator + counter_base + row * 2 + 1,
                        cutlass.Int32(1),
                        sem="acq_rel",
                        scope="gpu",
                    )
                    scratch[0] = cutlass.Float32(0.0)
                    if arrived == in_use - cutlass.Int32(1):
                        scratch[0] = cutlass.Float32(1.0)
                cute.arch.sync_threads()
                if scratch[0] != cutlass.Float32(0.0):
                    cute.arch.fence_acq_rel_gpu()
                    for index in cutlass.range(thread, {groups * head_dim}, {threads}):
{value_merge}                        tensor_bindings[{output_index}][
                            b * {query_heads} + head_base + index // {head_dim}, 0, index % {head_dim}
                        ] = value.to(cutlass.BFloat16)
                    if thread == 0:
                        # Every chunk has passed the wait: reset for the next phase.
                        for counter in cutlass.range_constexpr(2):
                            cute.arch.atomic_exch(
                                counters.iterator + counter_base + row * 2 + counter,
                                cutlass.Int32(0),
                                sem="relaxed",
                                scope="gpu",
                            )
                cute.arch.sync_threads()
{stamp(8)}            # Keys before current_length come from the cache and the new key
            # from new_k, so appending here cannot race with readers.
            if chunk == 0 and row % {tiles} == 0:
                if warp == 0:
                    for d in cutlass.range(lane, {head_dim}, 32):
                        tensor_bindings[{k_index}][b, current_length, kv_head, d] = {new_key_value}
                        tensor_bindings[{v_index}][b, current_length, kv_head, d] = tensor_bindings[{new_v_index}][b, 0, kv_head, d]
            cute.arch.sync_threads()
{stamp(9)}'''
    return _DecodeVariant(block, rows * chunks, floats, rows * 2)
