"""Linear attention decode (``Attention(algorithm="linear")``): one CTA work
item per value head. ``AttentionBodyProvider`` proposes, widens and emits it
through this module; ``LinearRecurrence`` documents the computation.

The delta-rule path follows vLLM's fused GDN decode kernel (vllm v0.34.0,
62d4bc3c, ``csrc/gdn/fused_gdn_decode_kernel.cu``, Apache-2.0), changed for a
persistent megakernel and for every recurrence the op declares:

- a work item is one value head of the one decode token; CTAs take items from
  the phase's work iterator, with the launch's threads, whatever their number;
- the causal convolution and its SiLU run here as the prologue (vLLM runs
  ``causal_conv1d_update`` first); each item convolves only its head's q, k
  and v channels, so value heads sharing a key head convolve it redundantly
  and nothing writes the window (the model shifts it after the call);
- a thread owns a value column and walks key rows; the K reductions are
  partial sums over thread groups combined through shared scratch;
- decay per head or per key dimension, the delta or the additive (Mamba2,
  ``dt``-scaled) update, ``[k, v]`` or ``[v, k]`` states, the ``D`` skip and
  the per-head gated RMSNorm are options of one body.

The rounding points are the op reference's: BF16 convolution and SiLU
outputs, FP32 gates, state and sums, a BF16 head output before the norm.

A ``[k, v]`` state is walked with a thread per value column, a ``[v, k]``
state (Mamba2) with a warp per value column and lanes along the key rows, so
both read their contiguous axis.

Not ported yet: vLLM's multi-token (MTP) path, shared-memory state staging
with ``cp.async``, and its cluster split of one head over several CTAs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ....graph import Region
from ....ops import Attention
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
    SemanticContract,
    TargetFacts,
    TensorLayout,
    TileContract,
    access_effects,
    widened_participants,
)
from dataclasses import dataclass

BODY_NAME = "linear_attention_decode_heads"
_THREADS = 128
# Shared scratch slots for the norm, gate and step scalars.
_SCALARS = 8


class UnsupportedLinearAttentionBody(ValueError):
    pass


@dataclass(frozen=True)
class GeneratedLinearAttentionBody:
    source: str
    function_name: str = "run"
    threads_per_cta: int = _THREADS
    tile_count: int = 1
    work_step: int = 1
    work_iterator_dtype: str = "torch.int64"
    work_iterator_encoding: str = "one value head per CTA tile"
    phase_floats: int = 0
    phase_counters: int = 0


# Key-row thread groups per head, at most: each adds a value row of partial
# sums to the body's shared scratch, which shares the CTA with the GEMMs'
# staged tiles (small models leave a few KiB).
MAX_PARTITIONS = 2


def partitions(threads: int, head_v_dim: int) -> int:
    """Thread groups that split a head's key rows; each owns every value column."""
    if threads >= head_v_dim and threads % head_v_dim == 0:
        return min(MAX_PARTITIONS, threads // head_v_dim)
    return 1


def column_splits(attention: Attention, workers: int) -> int:
    """Work items per value head: a head's value columns split over CTAs, so
    small-head models fill the GPU. Columns are independent except under the
    per-head norm, which needs the whole head in one CTA."""
    heads, _, _, dv = _dims(attention)
    if attention.recurrent.norm:
        return 1
    splits = 1
    while heads * splits * 2 <= workers and dv % (splits * 2) == 0 and dv // (splits * 2) >= 16:
        splits *= 2
    return splits


def _dims(attention: Attention) -> tuple[int, int, int, int]:
    """(heads, key_heads, head_k_dim, head_v_dim)."""
    grouping = attention.grouping
    return grouping.query_heads, grouping.kv_heads, grouping.head_dim, attention.recurrent.head_v_dim


def scratch_floats(attention: Attention, threads: int, splits: int = 1) -> int:
    """q, k, v, the per-key decay (if any), the per-group partial sums, the
    update, and scalars."""
    _, _, dk, dv = _dims(attention)
    cols = dv // splits
    per_key = dk if attention.recurrent.decay_per_key else 0
    return 2 * dk + per_key + (partitions(threads, cols) + 2) * cols + _SCALARS


def widen(config: BodyConfig, threads: int) -> BodyConfig:
    return replace(
        config,
        participants=widened_participants(config, threads),
        scratch=replace(
            config.scratch,
            size_bytes=scratch_floats(config.semantic.semantics, threads, config.tiles.cta_shape[0]) * 4,
        ),
    )


def candidates(
    region: Region, program: Any, attention: Attention, target: TargetFacts
) -> CandidateSet:
    def reject(reason: str) -> CandidateSet:
        return CandidateSet((), (ProviderRejection("attention", reason),))

    if target.architecture != "sm90a":
        return reject("linear attention targets sm90a only")
    reason = _unsupported(program, attention)
    if reason:
        return reject(reason)
    return CandidateSet((_config(region, program, attention, column_splits(attention, target.multiprocessor_count)),), ())


def _unsupported(program: Any, attention: Attention) -> str:
    recurrence = attention.recurrent
    if recurrence is None:
        return "attention has no linear recurrence"
    loads = {load.id.removeprefix("attention:"): load for load in program.loads}
    _, _, dk, dv = _dims(attention)
    if loads["state"].dtype != "torch.float32":
        return "the recurrent state must be FP32"
    if loads["qkv"].dtype != "torch.bfloat16" or program.store.dtype != "torch.bfloat16":
        return "projected channels and the head output must be BF16"
    if recurrence.conv_kernel and not 2 <= recurrence.conv_kernel <= 8:
        return "convolution kernels of 2-8 taps only"
    if dk > 256 or dv > 256:
        return "head dims above 256 are not supported"
    if recurrence.output_gate not in ("none", "silu", "sigmoid"):
        return f"unknown output gate {recurrence.output_gate}"
    return ""


def _config(region: Region, program: Any, attention: Attention, splits: int = 1) -> BodyConfig:
    threads = _THREADS
    heads, _, dk, dv = _dims(attention)
    layouts = {
        load.value_id: TensorLayout(load.dtype, load.axes, load.strides, 4, load.source_shape)
        for load in program.loads
    }
    output = TensorLayout(program.store.dtype, ("b", "e"), program.store.strides, 4, program.store.shape)
    layouts[program.store.value_id] = output
    scratch_bytes = scratch_floats(attention, threads, splits) * 4
    return BodyConfig(
        provider="attention",
        name=BODY_NAME,
        semantic=SemanticContract(
            operation_family="attention",
            source_op_ids=region.source_op_ids,
            numerical_policy="source_semantics",
            expression_hooks=(),
            access_effects={
                **access_effects(program),
                "writes": (program.store.value_id, attention.recurrent.state_value_id),
            },
            semantics=attention,
        ),
        layouts=LayoutContract(layouts, {program.store.value_id: output}),
        tiles=TileContract(
            # (column splits per head, value columns per work item)
            cta_shape=(splits, dv // splits),
            cluster_shape=(1, 1),
            work_partition="a slice of one value head's columns per CTA tile",
            mma_family=None,
            reduction_tile=(dk,),
        ),
        participants=ParticipantContract(
            participating_threads=threads,
            waiting_threads=0,
            warp_roles=("value columns of one head; key rows split over thread groups",)
            * (threads // 32),
            registers_per_thread=64,
        ),
        pipeline=PipelineContract(
            stage_count=1,
            barriers=(),
            async_operations=(),
            drain_rule="every state and output store completes before run returns",
        ),
        scratch=ScratchContract(owner="cta", size_bytes=scratch_bytes, alignment_bytes=16),
        completion=CompletionContract(
            outputs_ready="the head's outputs and state are stored when run returns",
            scratch_reusable="scratch is reusable when run returns",
            register_fragments="per-column key-row partial sums",
        ),
        evidence=ResourceEvidence(
            measured=False,
            source="linear attention decode body; delta-rule path after vLLM's fused GDN decode kernel",
            revision="vllm@62d4bc3c7e33b1b6c906e233e6af2d3506c482ca (v0.34.0) csrc/gdn/fused_gdn_decode_kernel.cu",
            resources={"threads": threads, "shared_bytes": scratch_bytes, "registers_per_thread": 64, "items": heads},
            measurements={},
        ),
    )


def emit(program: Any, config: BodyConfig) -> GeneratedLinearAttentionBody:
    attention = config.semantic.semantics
    if program.family != "attention" or not isinstance(attention, Attention):
        raise UnsupportedLinearAttentionBody("body lacks its linear attention semantics")
    reason = _unsupported(program, attention)
    if reason:
        raise UnsupportedLinearAttentionBody(reason)
    recurrence = attention.recurrent
    threads = config.participants.participating_threads
    if threads < 64 or threads % 32:
        raise UnsupportedLinearAttentionBody("the body needs at least two warps")
    index = {load.id.removeprefix("attention:"): position for position, load in enumerate(program.loads)}
    out = len(program.loads)
    heads, key_heads, dk, dv = _dims(attention)
    ratio = heads // key_heads
    splits = config.tiles.cta_shape[0]
    cols = dv // splits
    if splits < 1 or dv % splits or (splits > 1 and recurrence.norm):
        raise UnsupportedLinearAttentionBody("column splits must divide head_v_dim and exclude the per-head norm")
    groups = partitions(threads, cols)
    kernel = recurrence.conv_kernel
    # Scratch offsets (FP32 elements).
    q, k, v = 0, dk, 2 * dk
    decay_k = 2 * dk + cols
    partial = decay_k + (dk if recurrence.decay_per_key else 0)
    update = partial + groups * cols
    scalars = update + cols
    warps = threads // 32
    f32 = "cutlass.Float32"

    def element_at(row: str, column: str) -> str:
        if recurrence.state_layout == "kv":
            return f"tensor_bindings[{index['state']}][0, h, {row}, {column}]"
        return f"tensor_bindings[{index['state']}][0, h, {column}, {row}]"

    def decay_at(row: str) -> str:
        return f"scratch[{decay_k} + {row}]" if recurrence.decay_per_key else "decay"

    # The state passes are bound by memory latency, so each thread issues its
    # loads of a chunk of key rows together, then updates and stores them.
    rows_per_thread = dk // groups
    chunk = next((size for size in (16, 8, 4, 2) if rows_per_thread % size == 0), 1)

    def kv_pass(update_state: bool) -> str:
        """A [k, v] thread's (key-row group, value column) item: k^T (decayed
        S) when not ``update_state``, else the update and q^T S."""
        i12, i16 = " " * 12, " " * 16
        rows = "".join(f"{i16}row_{u} = base + {u * groups}\n" for u in range(chunk))
        loads = "".join(f"{i16}held_{u} = {element_at(f'row_{u}', 'column')}\n" for u in range(chunk))
        if update_state:
            uses = "".join(
                f"{i16}updated_{u} = held_{u} * {decay_at(f'row_{u}')} + scratch[{k} + row_{u}] * step\n"
                f"{i16}{element_at(f'row_{u}', 'column')} = updated_{u}\n"
                f"{i16}total = total + updated_{u} * scratch[{q} + row_{u}]\n"
                for u in range(chunk)
            )
            head = f"{i12}step = scratch[{update} + c]\n"
        else:
            uses = "".join(
                f"{i16}total = total + (held_{u} * {decay_at(f'row_{u}')}) * scratch[{k} + row_{u}]\n"
                for u in range(chunk)
            )
            head = ""
        return (
            f"        # One (key-row group, value column) item per thread ({groups} x {cols} <= {threads}).\n"
            f"        if first < {groups * cols}:\n"
            f"{i12}item = first\n{i12}group = item // {cols}\n{i12}c = item % {cols}\n{i12}column = col0 + c\n"
            f"{head}{i12}total = {f32}(0.0)\n"
            f"{i12}for base in cutlass.range(group, {dk}, {groups * chunk}):\n{rows}{loads}{uses}"
            f"{i12}scratch[{partial} + item] = total\n"
        )

    kv_delta = kv_pass(False)
    kv_state = kv_pass(True)
    binding_type = ", ".join("cute.Tensor" for _ in range(out + 1))
    f32 = "cutlass.Float32"

    def binding(name: str) -> str:
        return f"tensor_bindings[{index[name]}]"

    def has(name: str) -> bool:
        return name in index

    def element(row: str, column: str) -> str:
        if recurrence.state_layout == "kv":
            return f"{binding('state')}[0, h, {row}, {column}]"
        return f"{binding('state')}[0, h, {column}, {row}]"

    def load_channels(first: str, count: int, destination: int) -> str:
        """``count`` channels from ``first``, convolved when the layer has a convolution."""
        if not kernel:
            return (
                f"        for i in cutlass.range(first, {count}, {threads}):\n"
                f"            scratch[{destination} + i] = {binding('qkv')}[0, {first} + i].to({f32})\n"
            )
        bias = f"{binding('conv_bias')}[channel].to({f32})" if has("conv_bias") else f"{f32}(0.0)"
        silu = (
            f"            x = (x / ({f32}(1.0) + cute.exp(-x))).to(cutlass.BFloat16).to({f32})\n"
            if recurrence.conv_silu
            else ""
        )
        taps = "\n".join(
            f"            total = total + {binding('conv_weight')}[channel, 0, {j}].to({f32}) * {binding('conv_state')}[0, channel, {j}].to({f32})"
            for j in range(kernel - 1)
        )
        return f'''        for i in cutlass.range(first, {count}, {threads}):
            channel = {first} + i
            total = {bias}
{taps}
            total = total + {binding('conv_weight')}[channel, 0, {kernel - 1}].to({f32}) * {binding('qkv')}[0, channel].to({f32})
            x = total.to(cutlass.BFloat16).to({f32})
{silu}            scratch[{destination} + i] = x
'''

    def load_all_channels(q_first: str, k_first: str, v_first: str) -> str:
        """q, k and this item's v channels in one loop (scratch slots q, k, v
        are contiguous), so each thread takes about one channel and the
        convolution costs one round of loads, not one per operand."""
        assert (q, k, v) == (0, dk, 2 * dk)
        total = 2 * dk + cols
        source = (
            f"            channel = {q_first} + j\n"
            f"            if j >= {2 * dk}:\n"
            f"                channel = {v_first} + (j - {2 * dk})\n"
            f"            elif j >= {dk}:\n"
            f"                channel = {k_first} + (j - {dk})\n"
        )
        if not kernel:
            return (
                f"        for j in cutlass.range(first, {total}, {threads}):\n{source}"
                f"            scratch[j] = {binding('qkv')}[0, channel].to({f32})\n"
            )
        bias = f"{binding('conv_bias')}[channel].to({f32})" if has("conv_bias") else f"{f32}(0.0)"
        silu = (
            f"            x = (x / ({f32}(1.0) + cute.exp(-x))).to(cutlass.BFloat16).to({f32})\n"
            if recurrence.conv_silu
            else ""
        )
        # A ring window (``conv_ring``) holds token t in slot t % kernel: the
        # taps read slots ring_0.. and the new token goes to ring_new, which no
        # tap reads, so items sharing a channel all write it without racing.
        tap_slot = (lambda j: f"ring_{j}") if recurrence.conv_ring else (lambda j: str(j))
        taps = "".join(
            f"            total = total + {binding('conv_weight')}[channel, 0, {j}].to({f32}) * {binding('conv_state')}[0, channel, {tap_slot(j)}].to({f32})\n"
            for j in range(kernel - 1)
        )
        ring_write = (
            f"            {binding('conv_state')}[0, channel, ring_new] = {binding('qkv')}[0, channel]\n"
            if recurrence.conv_ring
            else ""
        )
        return (
            f"        for j in cutlass.range(first, {total}, {threads}):\n{source}"
            f"            total = {bias}\n{taps}"
            f"            total = total + {binding('conv_weight')}[channel, 0, {kernel - 1}].to({f32}) * {binding('qkv')}[0, channel].to({f32})\n"
            f"            x = total.to(cutlass.BFloat16).to({f32})\n{silu}"
            "            scratch[j] = x\n"
            f"{ring_write}"
        )

    def softplus_gate(position: str, rate: str) -> tuple[str, str]:
        """(lines computing ``step`` and ``g`` at a decay position, indent 12)."""
        bias = f" + {binding('dt_bias')}[{position}].to({f32})" if has("dt_bias") else ""
        source = f"{binding('decay')}[0, {position}].to({f32})"
        if recurrence.decay == "log":
            return "", source
        return (
            f"            gate_input = {source}{bias}\n"
            "            step = gate_input\n"
            f"            if gate_input <= {f32}(20.0):\n"
            "                step = cute.math.log1p(cute.exp(gate_input))\n",
            f"{rate} * step",
        )

    rate = f"(-cute.exp({binding('a_log')}[h].to({f32})))" if has("a_log") else ""
    if recurrence.decay_per_key:
        lines, g = softplus_gate(f"h * {dk} + i", rate)
        per_key = (
            f"        for i in cutlass.range(first, {dk}, {threads}):\n"
            + lines
            + f"            scratch[{decay_k} + i] = cute.exp({g})\n"
        )
        scalar_decay = ""
        factor = f"scratch[{decay_k} + row]"
    else:
        lines, g = softplus_gate("h", rate)
        per_key = ""
        scalar_decay = (
            lines.replace("\n            ", "\n                ").replace("            gate_input", "                gate_input", 1)
            + f"                scratch[{scalars} + 3] = cute.exp({g})\n"
            + (f"                scratch[{scalars} + 5] = step\n" if recurrence.dt_update else "")
        )
        factor = "decay"
    beta = (
        f"                scratch[{scalars} + 2] = {f32}(1.0) / ({f32}(1.0) + cute.exp(-{binding('beta')}[0, h].to({f32})))\n"
        if recurrence.update == "delta"
        else ""
    )
    norms = ""
    if recurrence.qk_l2norm:
        norms = f'''        if warp < 2:
            square = {f32}(0.0)
            for i in cutlass.range(lane, {dk}, 32):
                x = scratch[warp * {dk} + i]
                square = square + x * x
            square = cute.arch.warp_reduction_sum(square)
            if lane == 0:
                scratch[{scalars} + warp] = cute.rsqrt(square + {f32}(1.0e-6))
'''
    query_scale = f"scratch[{scalars}] * {f32}({attention.scale!r})" if recurrence.qk_l2norm else f"{f32}({attention.scale!r})"
    key_scale = (
        f"        for i in cutlass.range(first, {dk}, {threads}):\n"
        f"            scratch[{k} + i] = scratch[{k} + i] * scratch[{scalars} + 1]\n"
        if recurrence.qk_l2norm
        else ""
    )
    if recurrence.update == "delta":
        updates = f'''        # k^T (decayed S), per key-row group; then the delta-rule update.
{kv_delta}        cute.arch.sync_threads()
        for c in cutlass.range(first, {cols}, {threads}):
            column = col0 + c
            remembered = {f32}(0.0)
            for group in cutlass.range_constexpr({groups}):
                remembered = remembered + scratch[{partial} + group * {cols} + c]
            scratch[{update} + c] = (scratch[{v} + c] - remembered) * scratch[{scalars} + 2]
        cute.arch.sync_threads()
'''
    else:
        step = f" * scratch[{scalars} + 5]" if recurrence.dt_update else ""
        updates = f'''        for c in cutlass.range(first, {cols}, {threads}):
            column = col0 + c
            scratch[{update} + c] = scratch[{v} + c]{step}
        cute.arch.sync_threads()
'''
    skip = " + scratch[{v} + c] * skip_value".format(v=v) if has("skip") else ""
    skip_load = f"        skip_value = {binding('skip')}[h].to({f32})\n" if has("skip") else ""
    ring_slots = ""
    if recurrence.conv_ring:
        position_load = next(load for load in program.loads if load.id == "attention:conv_position")
        position = f"{binding('conv_position')}[{', '.join('0' for _ in position_load.source_shape)}].to(cutlass.Int32)"
        ring_slots = f"        position = {position}\n" + "".join(
            f"        ring_{j} = (position + cutlass.Int32({j + 1})) % cutlass.Int32({kernel})\n" for j in range(kernel - 1)
        ) + f"        ring_new = position % cutlass.Int32({kernel})\n"
    gate_value = (
        f"            gate = {binding('gate')}[0, h * {dv} + column].to({f32})\n"
        + (
            f"            gate = gate / ({f32}(1.0) + cute.exp(-gate))\n"
            if recurrence.output_gate == "silu"
            else f"            gate = {f32}(1.0) / ({f32}(1.0) + cute.exp(-gate))\n"
        )
        if recurrence.output_gate != "none"
        else ""
    )
    if recurrence.norm:
        weighted = (
            f"            normalized = (scratch[{update} + c] * inverse).to(cutlass.BFloat16).to({f32})\n"
            f"            weighted = ({binding('norm_weight')}[column].to({f32}) * normalized).to(cutlass.BFloat16).to({f32})\n"
            if recurrence.norm_rounds
            else f"            weighted = {binding('norm_weight')}[column].to({f32}) * (scratch[{update} + c] * inverse)\n"
        )
        finish = f'''        # Per-head gated RMSNorm.
        if warp == 0:
            square = {f32}(0.0)
            for c in cutlass.range(lane, {cols}, 32):
                x = scratch[{update} + c]
                square = square + x * x
            square = cute.arch.warp_reduction_sum(square)
            if lane == 0:
                scratch[{scalars} + 4] = cute.rsqrt(square / {f32}({float(dv)!r}) + {f32}({recurrence.norm_eps!r}))
        cute.arch.sync_threads()
        inverse = scratch[{scalars} + 4]
        for c in cutlass.range(first, {cols}, {threads}):
            column = col0 + c
{weighted}{gate_value}            tensor_bindings[{out}][0, h * {dv} + column] = (weighted{" * gate" if gate_value else ""}).to(cutlass.BFloat16)
'''
    else:
        finish = f'''        for c in cutlass.range(first, {cols}, {threads}):
            column = col0 + c
            result = scratch[{update} + c]
{gate_value}            tensor_bindings[{out}][0, h * {dv} + column] = (result{" * gate" if gate_value else ""}).to(cutlass.BFloat16)
'''
    preload = ""
    if recurrence.state_layout == "kv":
        # A thread owns a value column of a key-row group: S's rows are
        # contiguous along V, so a warp's loads are coalesced.
        state_pass = f'''        # 4. S <- decay * S + k u^T, stored; q^T S per key-row group.
{kv_state}        cute.arch.sync_threads()
        for c in cutlass.range(first, {cols}, {threads}):
            column = col0 + c
            output = {f32}(0.0)
            for group in cutlass.range_constexpr({groups}):
                output = output + scratch[{partial} + group * {cols} + c]
            scratch[{update} + c] = (output{skip}).to(cutlass.BFloat16).to({f32})
        cute.arch.sync_threads()
'''
    else:
        # A [v, k] state is contiguous along K: a warp owns a value column and
        # its lanes walk the key rows (coalesced), then reduce across the warp.
        per_warp, per_lane = divmod(cols, warps), divmod(dk, 32)
        if per_warp[1] == 0 and per_lane[1] == 0 and per_warp[0] * per_lane[0] <= 32:
            # Unrolled: every state element a lane touches is loaded first, so
            # its loads overlap, then each column is updated and reduced.
            columns, rows = per_warp[0], per_lane[0]
            i12 = " " * 12
            # Issued at the item's start (``preload``), before the convolution:
            # nothing they read is written earlier in the item.
            preload = "".join(
                f"        held_{j}_{u} = {element_at(f'lane + {32 * u}', f'col0 + load_warp + {j * warps}')}\n"
                for j in range(columns) for u in range(rows)
            )
            uses = "".join(
                f"{i12}c = first_warp + {j * warps}\n"
                f"{i12}step = scratch[{update} + c]\n"
                f"{i12}total = {f32}(0.0)\n"
                + "".join(
                    f"{i12}updated = held_{j}_{u} * {decay_at(f'lane + {32 * u}')} + scratch[{k} + lane + {32 * u}] * step\n"
                    f"{i12}{element_at(f'lane + {32 * u}', 'col0 + c')} = updated\n"
                    f"{i12}total = total + updated * scratch[{q} + lane + {32 * u}]\n"
                    for u in range(rows)
                )
                + f"{i12}total = cute.arch.warp_reduction_sum(total)\n"
                f"{i12}if lane == 0:\n"
                f"{i12}    scratch[{update} + c] = (total{skip}).to(cutlass.BFloat16).to({f32})\n"
                for j in range(columns)
            )
            state_pass = (
                f"        # 4. S <- decay * S + k u^T, stored; q^T S: {columns} value column(s) per warp, {rows} key row(s) per lane.\n"
                f"        if first_warp < {warps}:\n{uses}"
                "        cute.arch.sync_threads()\n"
            )
        else:
            state_pass = f'''        # 4. S <- decay * S + k u^T, stored; q^T S, a warp per value column.
        for c in cutlass.range(first_warp, {cols}, {warps}):
            column = col0 + c
            step = scratch[{update} + c]
            output = {f32}(0.0)
            for row in cutlass.range(lane, {dk}, 32):
                updated = {element('row', 'column')} * {factor} + scratch[{k} + row] * step
                {element('row', 'column')} = updated
                output = output + updated * scratch[{q} + row]
            output = cute.arch.warp_reduction_sum(output)
            if lane == 0:
                scratch[{update} + c] = (output{skip}).to(cutlass.BFloat16).to({f32})
        cute.arch.sync_threads()
'''
        if recurrence.update == "delta":
            updates = f'''        # k^T (decayed S) and the delta-rule update, a warp per value column.
        for c in cutlass.range(first_warp, {cols}, {warps}):
            column = col0 + c
            remembered = {f32}(0.0)
            for row in cutlass.range(lane, {dk}, 32):
                remembered = remembered + ({element('row', 'column')} * {factor}) * scratch[{k} + row]
            remembered = cute.arch.warp_reduction_sum(remembered)
            if lane == 0:
                scratch[{update} + c] = (scratch[{v} + c] - remembered) * scratch[{scalars} + 2]
        cute.arch.sync_threads()
'''
    gates = beta + scalar_decay
    gates_block = f"        if thread == 0:\n{gates}" if gates else ""
    source = f'''import cutlass
import cutlass.cute as cute


@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{binding_type}],
    scratch: cute.Tensor,
    epilogue,
):
    # One value head per CTA tile ({ratio} value head(s) per key head). Thread t
    # owns value column t % {dv} and key rows t // {dv}, t // {dv} + {groups}, ...;
    # the {groups} groups' partial sums meet in scratch.
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    warp = thread // 32
    lane = thread % 32
    # Launches without GEMMs keep bodies at their own width, so a CTA may have
    # more threads than this body's {threads}: those take no work (the state
    # update is read-modify-write, so duplicated items would race).
    first = thread
    first_warp = warp
    load_warp = warp
    if thread >= cutlass.Int32({threads}):
        first = cutlass.Int32({1 << 20})
        first_warp = cutlass.Int32({1 << 20})
        load_warp = cutlass.Int32(0)
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        item = cutlass.Int32(work_iterator[tile])
        h = item // {splits}
        col0 = (item % {splits}) * {cols}
        kh = h // {ratio}
{preload}{skip_load}{ring_slots}        # 1. This head's q, k and v channels{(" (causal convolution and SiLU)" if recurrence.conv_silu else " (causal convolution)") if kernel else ""}.
{gates_block}{load_all_channels(f"{recurrence.q_offset} + kh * {dk}", f"{recurrence.k_offset} + kh * {dk}", f"{recurrence.v_offset} + h * {dv} + col0")}{per_key}        cute.arch.sync_threads()
        # 2. q/k norms.
{norms}        cute.arch.sync_threads()
        query_scale = {query_scale}
        for i in cutlass.range(first, {dk}, {threads}):
            scratch[{q} + i] = scratch[{q} + i] * query_scale
{key_scale}        decay = scratch[{scalars} + 3]
        cute.arch.sync_threads()
        # 3. The update u of each value column.
{updates}{state_pass}        # 5. Output.
{finish}        cute.arch.sync_threads()
'''
    return GeneratedLinearAttentionBody(source=source, threads_per_cta=threads, tile_count=heads * splits)
