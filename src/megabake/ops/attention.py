"""Softmax attention: the ops a model declares (``megabake::attention`` and
``megabake::kv_attention``), their programs, and the ``Attention`` region
descriptor that device bodies implement.

A region is attention in one of three ways: the model declares it with one
of the two ops (Hugging Face models do, through ``fx_handler``), or the
lowered graph spells it out as ``bmm -> softmax -> bmm`` (eager attention).
Stateful decode attention can also absorb the elementwise producers of its
query and new key (RoPE) as prologues.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import operator
from typing import Any, Literal

import torch

from ..graph import ComputeGraph, Op, RegionCandidate, Value
from ..ir import (
    AxisAffine,
    AxisExpression,
    AxisPredicate,
    AxisTerm,
    Constant,
    IndexedLoad,
    IterationDomain,
    KVCacheAccess,
    KVCacheLayout,
    Load,
    ScalarExpression,
    TensorProgram,
    TensorStore,
    UnsupportedLowering,
    Where,
)
from .matching import (
    has_illegal_effects,
    is_contraction,
    live_outputs_of,
    matches_target,
    operation_target,
    ordered_operations,
    producer_of,
    scalar_operand,
    value_bytes,
)


@torch.library.custom_op(
    "megabake::kv_attention",
    mutates_args=("key_cache", "value_cache", "valid_length"),
)
def kv_attention(
    source: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    valid_length: torch.Tensor,
    scale: float = 0.0,
    window: int = 0,
    softcap: float = 0.0,
) -> torch.Tensor:
    """One-token decode attention that appends to bounded KV caches.

    Scores are the BF16 q·k products scaled in FP32 by ``scale`` (0 means
    head_dim**-0.5), then mapped to softcap * tanh(s / softcap) when
    ``softcap`` is set. With ``window`` set the query sees only the last
    ``window`` keys (its own included): sliding-window attention over a
    linear cache."""
    if source.shape[1] != 1:
        raise RuntimeError("megabake kv_attention supports one-token decode only")
    start = int(valid_length.item())
    end = start + source.shape[1]
    if start < 0 or end > key_cache.shape[1]:
        raise RuntimeError(
            f"KV cache length {start} plus {source.shape[1]} exceeds capacity "
            f"{key_cache.shape[1]}"
        )
    key_cache[:, start:end].copy_(key)
    value_cache[:, start:end].copy_(value)
    valid_length.fill_(end)
    query = source.permute(0, 2, 1, 3)
    keys = key_cache[:, :end].permute(0, 2, 1, 3)
    values = value_cache[:, :end].permute(0, 2, 1, 3)
    if query.shape[1] != keys.shape[1]:
        group_size = query.shape[1] // keys.shape[1]
        keys = keys.repeat_interleave(group_size, dim=1)
        values = values.repeat_interleave(group_size, dim=1)
    scores = (query @ keys.transpose(-2, -1)).float() * (scale or source.shape[-1] ** -0.5)
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    if window and end > window:
        scores[..., : end - window] = float("-inf")
    probabilities = torch.softmax(scores, dim=-1).to(source.dtype)
    output = probabilities @ values
    return output.permute(0, 2, 1, 3).reshape(
        source.shape[0] * source.shape[2], source.shape[1], source.shape[3]
    )


@kv_attention.register_fake
def _kv_attention_fake(
    source: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    valid_length: torch.Tensor,
    scale: float = 0.0,
    window: int = 0,
    softcap: float = 0.0,
) -> torch.Tensor:
    return torch.empty(
        (
            source.shape[0] * source.shape[2],
            source.shape[1],
            source.shape[3],
        ),
        dtype=source.dtype,
        device=source.device,
    )


KV_ATTENTION_OP = torch.ops.megabake.kv_attention.default


@torch.library.custom_op("megabake::attention", mutates_args=())
def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Softmax attention with Hugging Face eager numerics.

    ``query`` is ``[batch, query, heads, dim]``; ``key`` and ``value`` are
    ``[batch, key, kv_heads, dim]`` with ``heads`` a multiple of ``kv_heads``;
    ``mask`` is additive and broadcasts to ``[batch, heads, query, key]``.
    Returns ``[batch * heads, query, dim]``, the layout of ``kv_attention``.
    """
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if query.shape[1] != key.shape[1]:
        group_size = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(group_size, dim=1)
        value = value.repeat_interleave(group_size, dim=1)
    scores = torch.matmul(query, key.transpose(2, 3)) * scale
    if mask is not None:
        scores = scores + mask
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probabilities, value)
    return output.reshape(-1, output.shape[2], output.shape[3])


@attention.register_fake
def _attention_fake(query, key, value, mask, scale):
    return torch.empty(
        (query.shape[0] * query.shape[2], query.shape[1], query.shape[3]),
        dtype=query.dtype,
        device=query.device,
    )


ATTENTION_OP = torch.ops.megabake.attention.default


@dataclass(frozen=True)
class HeadGrouping:
    batch: int
    query_heads: int
    kv_heads: int
    head_dim: int

    @property
    def group_size(self) -> int:
        """Query heads per KV head."""
        return self.query_heads // self.kv_heads


@dataclass(frozen=True)
class NumericalPolicy:
    """The dtypes the scores, softmax statistics, probabilities and value
    accumulators are rounded to."""

    score_dtype: str | None
    statistics_dtype: str | None
    probability_dtype: str | None
    accumulator_dtype: str | None


# Hugging Face eager attention, which the declared ops' references implement.
HF_EAGER_NUMERICS = NumericalPolicy(
    "torch.bfloat16", "torch.float32", "torch.bfloat16", "torch.float32"
)


@dataclass(frozen=True)
class Mask:
    """What is added to the scores before the softmax.

    ``additive_tensor`` adds a bound mask tensor; ``stateful_causal`` masks
    keys past the cache's valid length. ``typed_where`` adds ``false_value``
    where a condition fails: a bound boolean tensor, or keys past
    ``causal_offset`` (or outside ``window_offsets``) of the query.
    """

    kind: Literal["none", "additive_tensor", "typed_where", "stateful_causal"]
    causal_offset: int | None = None
    window_offsets: tuple[int, int] | None = None
    true_value: float | None = None
    false_value: float | None = None


@dataclass(frozen=True)
class CacheEffects:
    """How the attention reads and changes a KV cache.

    ``policy`` is ``read_only``, ``functional_append`` (the appended cache is
    a graph output), ``in_place`` (another op updates the cache) or
    ``stateful`` (the attention appends to the cache itself).
    """

    policy: str
    reads: tuple[str, ...] = ()
    appends: tuple[str, ...] = ()
    updates: tuple[str, ...] = ()


@dataclass(frozen=True)
class Prologue:
    """An elementwise producer of an attention operand that the body
    evaluates in place of reading its result."""

    combine: ScalarExpression
    loads: tuple[IndexedLoad, ...]
    dtype: str


@dataclass(frozen=True)
class Attention:
    """Softmax attention over the query, key and value accesses."""

    algorithm: Literal["stable_softmax", "online_softmax"]
    numerics: NumericalPolicy
    q: IndexedLoad
    k: IndexedLoad
    v: IndexedLoad
    grouping: HeadGrouping
    scale: float
    mask: Mask
    # The values the region must produce; ``boundary_output`` is the graph
    # output a trailing view of the attention result reaches, if any.
    outputs: tuple[str, ...]
    boundary_output: str | None
    cache: CacheEffects
    kv_state: KVCacheAccess | None = None
    # Keyed by the operand load they replace: attention:q or attention:new_k.
    prologues: Mapping[str, Prologue] = field(default_factory=dict)
    # Scaled scores s become softcap * tanh(s / softcap) when set.
    softcap: float = 0.0


def _declared_grouping(query: IndexedLoad, key: IndexedLoad) -> HeadGrouping:
    """Grouping of the declared ops' [batch, sequence, heads, dim] operands."""
    batch, _, query_heads, head_dim = query.source_shape
    return HeadGrouping(batch, query_heads, key.source_shape[2], head_dim)


def kv_attention_program(
    source: Value,
    new_key: Value,
    new_value: Value,
    key_cache: Value,
    value_cache: Value,
    valid_length: Value,
    output: Value,
) -> TensorProgram:
    """The program of ``megabake::kv_attention`` over its operand values."""
    if any(
        value.shape is None or value.strides is None or value.dtype is None
        for value in (source, new_key, new_value, key_cache, value_cache, valid_length)
    ):
        raise UnsupportedLowering("KV attention operands lack static tensor metadata")

    batch, query_length, query_heads, head_dim = source.shape
    _, new_length, kv_heads, _ = new_key.shape
    _, capacity, cache_kv_heads, _ = key_cache.shape
    group_size = query_heads // kv_heads
    contiguous_cache = (
        capacity * kv_heads * head_dim,
        kv_heads * head_dim,
        head_dim,
        1,
    )
    if (
        query_length != 1
        or new_length != query_length
        or new_key.shape != new_value.shape
        or key_cache.shape != value_cache.shape
        or cache_kv_heads != kv_heads
        or query_heads % kv_heads
        or group_size <= 0
        or key_cache.strides != contiguous_cache
        or value_cache.strides != contiguous_cache
        or valid_length.shape != (1,)
        or valid_length.dtype != "torch.int32"
        or output.shape != (batch * query_heads, query_length, head_dim)
        or source.dtype != "torch.bfloat16"
        or new_key.dtype != "torch.bfloat16"
        or key_cache.dtype != "torch.bfloat16"
    ):
        raise UnsupportedLowering(
            "KV attention supports one-token decode over two contiguous BF16 caches"
        )

    batch_head = AxisAffine((AxisTerm("bh", divisor=query_heads),))
    query_head = AxisAffine((AxisTerm("bh", modulus=query_heads),))
    kv_head = AxisAffine(
        (AxisTerm("bh", divisor=query_heads, modulus=group_size),)
    )
    loads = (
        IndexedLoad(
            "attention:q", source.id, (batch_head, "s", query_head, "d"),
            source.strides, source.dtype, source.shape,
        ),
        IndexedLoad(
            "attention:new_k", new_key.id, (batch_head, 0, kv_head, "d"),
            new_key.strides, new_key.dtype, new_key.shape,
        ),
        IndexedLoad(
            "attention:new_v", new_value.id, (batch_head, 0, kv_head, "d"),
            new_value.strides, new_value.dtype, new_value.shape,
        ),
        IndexedLoad(
            "attention:k", key_cache.id, (batch_head, "k", kv_head, "d"),
            key_cache.strides, key_cache.dtype, key_cache.shape,
        ),
        IndexedLoad(
            "attention:v", value_cache.id, (batch_head, "k", kv_head, "d"),
            value_cache.strides, value_cache.dtype, value_cache.shape,
        ),
        IndexedLoad(
            "attention:valid_length", valid_length.id, (0,),
            valid_length.strides, valid_length.dtype, valid_length.shape,
        ),
    )
    zero = Constant(0.0, "torch.float32")
    state_access = KVCacheAccess(
        key_cache_value_id=key_cache.id,
        value_cache_value_id=value_cache.id,
        valid_length_value_id=valid_length.id,
        key_cache_load_id="attention:k",
        value_cache_load_id="attention:v",
        new_key_load_id="attention:new_k",
        new_value_load_id="attention:new_v",
        valid_length_load_id="attention:valid_length",
        capacity=capacity,
        query_length=query_length,
        key_layout=KVCacheLayout(key_cache.shape, key_cache.strides, key_cache.dtype),
        value_layout=KVCacheLayout(
            value_cache.shape, value_cache.strides, value_cache.dtype
        ),
        storage_identity=(key_cache.id, value_cache.id),
    )
    return TensorProgram(
        family="attention",
        source_target="megabake.kv_attention",
        decomposition="stateful decode attention with bounded contiguous KV cache",
        iteration_domains=(
            IterationDomain("bh", batch * query_heads),
            IterationDomain("s", query_length),
            IterationDomain("d", head_dim),
            IterationDomain("k", capacity),
        ),
        reduction_axes=("k",),
        loads=loads,
        init=zero,
        combine=zero,
        post=zero,
        accumulator_dtype="torch.float32",
        store=TensorStore(
            output.id, output.shape, output.dtype, output.strides, True
        ),
        predicates=(
            AxisPredicate("bh", batch * query_heads),
            AxisPredicate("s", query_length),
            AxisPredicate("d", head_dim),
        ),
        state_accesses=(state_access,),
    )


def attention_access_axes(
    head_axis: int, batch_axis: AxisAffine, head: AxisAffine, sequence_axis: str
) -> tuple[AxisExpression, ...]:
    if head_axis == 2:
        return (batch_axis, sequence_axis, head, "d")
    return (batch_axis, head, sequence_axis, "d")


def attention_tensor_program(
    output_id: str,
    q_root: str,
    k_root: str,
    v_root: str,
    q: Any,
    k: Any,
    v: Any,
    axes: tuple[int, int, int],
    mask_value: str | None,
    graph: ComputeGraph,
) -> TensorProgram | None:
    output = graph.values[output_id]
    if output.shape is None or output.dtype is None or output.strides is None:
        return None
    if q.shape is None or k.shape is None or v.shape is None:
        return None
    q_head_axis, k_head_axis, v_head_axis = axes
    batch = q.shape[0]
    query_heads = q.shape[q_head_axis]
    query_length = q.shape[3 - q_head_axis]
    kv_heads = k.shape[k_head_axis]
    kv_length = k.shape[3 - k_head_axis]
    head_dim = q.shape[-1]
    group_size = query_heads // kv_heads
    batch_head = AxisAffine((AxisTerm("bh", divisor=query_heads),))
    query_head = AxisAffine((AxisTerm("bh", modulus=query_heads),))
    kv_head = AxisAffine(
        (AxisTerm("bh", divisor=query_heads, modulus=group_size),)
    )

    loads = (
        IndexedLoad(
            "attention:q",
            q_root,
            attention_access_axes(q_head_axis, batch_head, query_head, "s"),
            q.strides or (),
            q.dtype or "",
            q.shape,
        ),
        IndexedLoad(
            "attention:k",
            k_root,
            attention_access_axes(k_head_axis, batch_head, kv_head, "k"),
            k.strides or (),
            k.dtype or "",
            k.shape,
        ),
        IndexedLoad(
            "attention:v",
            v_root,
            attention_access_axes(v_head_axis, batch_head, kv_head, "k"),
            v.strides or (),
            v.dtype or "",
            v.shape,
        ),
    )
    if mask_value is not None:
        mask = graph.values[mask_value]
        if mask.shape is None or mask.strides is None or mask.dtype is None:
            return None
        mask_head = 0 if mask.shape[1] == 1 else query_head
        loads = (
            *loads,
            IndexedLoad(
                "attention:mask",
                mask_value,
                (batch_head, mask_head, "s", "k"),
                mask.strides,
                mask.dtype,
                mask.shape,
            ),
        )
    zero = Constant(0.0, "torch.float32")
    return TensorProgram(
        family="attention",
        source_target="megabake.attention",
        decomposition="stable softmax attention with grouped heads and typed masks",
        iteration_domains=(
            IterationDomain("bh", batch * query_heads),
            IterationDomain("s", query_length),
            IterationDomain("d", head_dim),
            IterationDomain("k", kv_length),
        ),
        reduction_axes=("k",),
        loads=loads,
        init=zero,
        combine=zero,
        post=zero,
        accumulator_dtype="torch.float32",
        store=TensorStore(
            output.id,
            output.shape,
            output.dtype,
            output.strides,
            True,
        ),
        predicates=(
            AxisPredicate("bh", batch * query_heads),
            AxisPredicate("s", query_length),
            AxisPredicate("d", head_dim),
        ),
    )


def attention_program(
    query: Value,
    key: Value,
    value: Value,
    mask: Value | None,
    output: Value,
    graph: ComputeGraph,
) -> TensorProgram:
    """The program of ``megabake::attention`` over its operand values."""
    if any(
        operand.shape is None or len(operand.shape) != 4 or operand.dtype != "torch.bfloat16"
        for operand in (query, key, value)
    ):
        raise UnsupportedLowering("attention supports four-dimensional BF16 operands")
    if query.shape[2] % key.shape[2] or key.shape != value.shape:
        raise UnsupportedLowering("attention needs whole KV-head groups and matching K and V")
    # Operands are [batch, sequence, heads, dim] by the op contract, so the
    # head axis is known and no stride heuristic is needed.
    program = attention_tensor_program(
        output.id,
        query.id,
        key.id,
        value.id,
        query,
        key,
        value,
        (2, 2, 2),
        None if mask is None else mask.id,
        graph,
    )
    if program is None:
        raise UnsupportedLowering("attention operands lack static layouts")
    return program


def declared_candidate(operation: Op, graph: ComputeGraph) -> RegionCandidate | None:
    """``megabake::attention``: the model declares its attention, so the
    region needs no pattern matching and no layout inference."""
    if not matches_target(operation, "megabake.attention.default"):
        return None
    program = operation.tensor_program
    if not isinstance(program, TensorProgram) or program.family != "attention":
        return None
    loads = {load.id: load for load in program.loads}
    query, key = loads["attention:q"], loads["attention:k"]
    output_id = operation.results[0]
    return RegionCandidate(
        id=f"region:{operation.id.removeprefix('op:')}:attention",
        source_op_ids=(operation.id,),
        external_inputs=operation.operands,
        live_outputs=(output_id,),
        tensor_program=program,
        semantics=Attention(
            algorithm="stable_softmax",
            numerics=HF_EAGER_NUMERICS,
            q=query,
            k=key,
            v=loads["attention:v"],
            grouping=_declared_grouping(query, key),
            scale=scalar_operand(operation, 4),
            mask=Mask("additive_tensor" if "attention:mask" in loads else "none"),
            outputs=(output_id,),
            boundary_output=output_id if output_id in graph.outputs else None,
            cache=CacheEffects("read_only"),
        ),
        cost_bytes=0,
    )


def stateful_candidate(operation: Op, graph: ComputeGraph) -> RegionCandidate | None:
    """``megabake::kv_attention``: one-token decode over a bounded KV cache it
    appends to."""
    if not matches_target(operation, "megabake.kv_attention.default"):
        return None
    program = operation.tensor_program
    if (
        not isinstance(program, TensorProgram)
        or program.family != "attention"
        or not program.state_accesses
    ):
        return None
    state = program.state_accesses[0]
    loads = {load.id: load for load in program.loads}
    output_id = operation.results[0]
    scale, window, softcap = _kv_attention_options(operation)
    head_dim = loads["attention:q"].source_shape[3]
    return RegionCandidate(
        id=f"region:{operation.id.removeprefix('op:')}:kv_attention",
        source_op_ids=(operation.id,),
        external_inputs=operation.operands,
        live_outputs=(output_id,),
        tensor_program=program,
        semantics=Attention(
            algorithm="stable_softmax",
            numerics=HF_EAGER_NUMERICS,
            q=loads["attention:q"],
            k=loads["attention:k"],
            v=loads["attention:v"],
            grouping=_declared_grouping(loads["attention:q"], loads["attention:k"]),
            scale=scale or head_dim**-0.5,
            mask=Mask(
                "stateful_causal",
                causal_offset=0,
                window_offsets=(1 - window, 0) if window else None,
            ),
            outputs=(output_id,),
            boundary_output=output_id if output_id in graph.outputs else None,
            cache=CacheEffects(
                "stateful",
                reads=(state.key_cache_value_id, state.value_cache_value_id),
                appends=(
                    loads["attention:new_k"].value_id,
                    loads["attention:new_v"].value_id,
                ),
                updates=(
                    state.key_cache_value_id,
                    state.value_cache_value_id,
                    state.valid_length_value_id,
                ),
            ),
            kv_state=state,
            softcap=softcap,
        ),
        cost_bytes=(
            2
            * loads["attention:q"].source_shape[0]
            * state.query_length
            * loads["attention:k"].source_shape[2]
            * loads["attention:q"].source_shape[3]
            * 2
        ),
    )


def _kv_attention_options(operation: Op) -> tuple[float, int, float]:
    """``scale``, ``window`` and ``softcap`` of a ``megabake::kv_attention``
    call, positional or keyword, with the op's defaults."""
    arguments = tuple(operation.attributes.get("fx_args", ()))[6:]
    keywords = dict(operation.attributes.get("fx_kwargs", {}) or {})
    names = ("scale", "window", "softcap")
    values = dict(zip(names, arguments))
    values.update({name: keywords[name] for name in names if name in keywords})
    scale, window, softcap = (values.get(name, 0) for name in names)
    if not all(isinstance(value, (int, float)) for value in (scale, window, softcap)):
        raise UnsupportedLowering("KV attention scale, window and soft cap must be static")
    return float(scale), int(window), float(softcap)


def prologue_candidates(
    graph: ComputeGraph, producers: tuple[RegionCandidate, ...]
) -> tuple[RegionCandidate, ...]:
    """Stateful decode attention that computes its query and new key from
    their elementwise producers (RoPE) instead of reading them.

    The attention then binds the producer's input in place of its result,
    which never materializes, and the producer's other operands (cos, sin)
    as extra loads. Its body evaluates each producer's program for the one
    row it needs (``Attention.prologues``).
    """
    by_output = {
        candidate.live_outputs[0]: candidate
        for candidate in producers
        if candidate.executable
        and len(candidate.live_outputs) == 1
        and candidate.tensor_program is not None
        and candidate.tensor_program.family == "elementwise"
    }
    candidates = []
    for operation in graph.ops.values():
        attention = stateful_candidate(operation, graph)
        if attention is None:
            continue
        program = attention.tensor_program
        assert program is not None
        loads = list(program.loads)
        prologues: dict[str, Prologue] = {}
        operations: list[Op] = []
        extra: dict[str, IndexedLoad] = {}
        for load_id in ("attention:q", "attention:new_k"):
            position = next(index for index, load in enumerate(loads) if load.id == load_id)
            load = loads[position]
            producer = by_output.get(load.value_id)
            if (
                producer is None
                or load.value_id in graph.outputs
                or graph.values[load.value_id].users != (operation.id,)
            ):
                continue
            producer_program = producer.tensor_program
            assert producer_program is not None
            if (
                producer_program.store.shape != load.source_shape
                or producer_program.store.strides != load.strides
            ):
                continue
            identity = tuple(domain.axis for domain in producer_program.iteration_domains)
            # The producer input read at the output position takes the
            # result's place in the attention's bindings.
            source = next(
                (
                    candidate
                    for candidate in producer_program.loads
                    if candidate.axes == identity
                    and candidate.strides == load.strides
                    and candidate.dtype == load.dtype
                ),
                None,
            )
            if source is None:
                continue
            loads[position] = replace(load, value_id=source.value_id)
            prologues[load_id] = Prologue(
                producer_program.combine, producer_program.loads, producer_program.store.dtype
            )
            operations.extend(graph.ops[op_id] for op_id in producer.source_op_ids)
            for candidate in producer_program.loads:
                extra.setdefault(candidate.value_id, candidate)
        if not prologues:
            continue
        bound = {load.value_id for load in loads}
        for value_id, candidate in extra.items():
            if value_id not in bound:
                # Point accesses keep the attention's access maps valid; the
                # body reads these through the prologue's own loads.
                loads.append(
                    replace(
                        candidate,
                        id=f"prologue:{len(loads)}",
                        axes=(0,) * len(candidate.strides),
                    )
                )
                bound.add(value_id)
        fused = replace(program, loads=tuple(loads))
        fused_loads = {load.id: load for load in fused.loads}
        operation_ids = {item.id for item in operations}
        if len(operation_ids) != len(operations):
            continue
        candidates.append(
            RegionCandidate(
                id=f"region:{operation.id.removeprefix('op:')}:attention_prologue",
                source_op_ids=tuple(
                    item.id
                    for item in graph.ops.values()
                    if item.id in operation_ids or item.id == operation.id
                ),
                external_inputs=tuple(dict.fromkeys(load.value_id for load in fused.loads)),
                live_outputs=attention.live_outputs,
                tensor_program=fused,
                semantics=replace(
                    attention.semantics, q=fused_loads["attention:q"], prologues=prologues
                ),
                cost_bytes=attention.cost_bytes
                + sum(
                    2 * value_bytes(load.value_id, graph)
                    for load in program.loads
                    if load.id in prologues
                ),
            )
        )
    return tuple(candidates)


def pattern_candidates(graph: ComputeGraph) -> tuple[RegionCandidate, ...]:
    """Attention written out as ``bmm -> softmax -> bmm`` (eager attention),
    recognized at its value ``bmm``."""
    candidates = []
    for operation in graph.ops.values():
        if matches_target(operation, "aten.bmm.default"):
            candidate = _pattern_candidate(operation, graph)
            if candidate is not None:
                candidates.append(candidate)
    return tuple(candidates)


def _pattern_candidate(value_bmm: Op, graph: ComputeGraph) -> RegionCandidate | None:
    value_program = value_bmm.tensor_program
    if not is_contraction(value_bmm) or not isinstance(value_program, TensorProgram):
        return None
    if len(value_program.loads) != 2:
        return None

    probability_root = _producer_until(
        value_program.loads[0].value_id,
        lambda operation: matches_target(operation, "aten.div.Tensor"),
        graph,
    )
    if probability_root is None:
        return None
    division = probability_root
    softmax = _softmax_operations(division, graph)
    if softmax is None:
        return None
    algorithm, score_value, softmax_operations = softmax
    score_chain = _score_chain(score_value, graph)
    if score_chain is None:
        return None
    score_add, scale_operation, mask_value, score_bmm, score_path = score_chain
    score_program = score_bmm.tensor_program
    if (
        not is_contraction(score_bmm)
        or not isinstance(score_program, TensorProgram)
        or len(score_program.loads) != 2
    ):
        return None
    softmax_input = (
        score_add.results[0] if score_add is not None else scale_operation.results[0]
    )
    softmax_input_cast = producer_of(score_value, graph)
    if (
        softmax_input_cast is not None
        and matches_target(softmax_input_cast, "prims.convert_element_type.default")
        and softmax_input_cast.operands == (softmax_input,)
    ):
        softmax_input_operations = (softmax_input_cast,)
    else:
        softmax_input_operations = ()

    q_root, q_path = _value_root_path(score_program.loads[0].value_id, graph)
    k_root, k_path = _value_root_path(score_program.loads[1].value_id, graph)
    v_root, v_path = _value_root_path(value_program.loads[1].value_id, graph)
    q = graph.values[q_root]
    k = graph.values[k_root]
    v = graph.values[v_root]
    if any(value.shape is None or len(value.shape) != 4 for value in (q, k, v)):
        return None
    scores = graph.values[score_bmm.results[0]].shape
    axes = _attention_axes(q, k, v, scores)
    if axes is None:
        return None
    q_head_axis, k_head_axis, v_head_axis = axes
    query_heads = q.shape[q_head_axis]
    kv_heads = k.shape[k_head_axis]
    head_dim = q.shape[-1]
    if (
        query_heads % kv_heads
        or head_dim != k.shape[-1]
        or head_dim != v.shape[-1]
        or v.shape[v_head_axis] != kv_heads
    ):
        return None

    group_size = query_heads // kv_heads
    batch_head = AxisAffine((AxisTerm("bh", divisor=query_heads),))
    query_head = AxisAffine((AxisTerm("bh", modulus=query_heads),))
    kv_head = AxisAffine(
        (AxisTerm("bh", divisor=query_heads, modulus=group_size),)
    )
    q_load = IndexedLoad(
        "attention:q",
        q_root,
        attention_access_axes(q_head_axis, batch_head, query_head, "s"),
        q.strides or (),
        q.dtype or "",
        q.shape,
    )
    k_load = IndexedLoad(
        "attention:k",
        k_root,
        attention_access_axes(k_head_axis, batch_head, kv_head, "k"),
        k.strides or (),
        k.dtype or "",
        k.shape,
    )
    v_load = IndexedLoad(
        "attention:v",
        v_root,
        attention_access_axes(v_head_axis, batch_head, kv_head, "k"),
        v.strides or (),
        v.dtype or "",
        v.shape,
    )
    score_dtype = graph.values[scale_operation.results[0]].dtype or ""
    mask, attention_mask = _mask(mask_value, graph)

    probability_value = graph.values[value_program.loads[0].value_id]
    probability_path = _producer_path(value_program.loads[0].value_id, division, graph)
    if probability_path is None:
        return None
    boundary_output, output_path = _boundary_output_path(
        value_bmm.results[0], graph
    )
    operations = ordered_operations(
        (
            *q_path,
            *k_path,
            *v_path,
            score_bmm,
            *score_path,
            scale_operation,
            *(() if score_add is None else (score_add,)),
            *softmax_input_operations,
            *softmax_operations,
            *probability_path,
            value_bmm,
            *output_path,
        ),
        graph,
    )
    attention_live_outputs = live_outputs_of(operations, graph)
    if value_bmm.results[0] not in attention_live_outputs:
        attention_live_outputs = (
            value_bmm.results[0],
            *attention_live_outputs,
        )
    rejections: list[str] = []
    if any(has_illegal_effects(operation) for operation in operations):
        rejections.append("source effects do not fit a composable attention body")
    external_inputs = tuple(
        dict.fromkeys(
            (
                q_root,
                k_root,
                v_root,
                *((attention_mask,) if attention_mask is not None else ()),
            )
        )
    )
    return RegionCandidate(
        id=f"region:{value_bmm.id.removeprefix('op:')}:attention",
        source_op_ids=tuple(operation.id for operation in operations),
        external_inputs=external_inputs,
        live_outputs=attention_live_outputs,
        tensor_program=attention_tensor_program(
            value_bmm.results[0],
            q_root,
            k_root,
            v_root,
            q,
            k,
            v,
            axes,
            attention_mask,
            graph,
        ),
        semantics=Attention(
            algorithm=algorithm,
            numerics=NumericalPolicy(
                score_dtype,
                graph.values[score_value].dtype,
                probability_value.dtype,
                score_program.accumulator_dtype,
            ),
            q=q_load,
            k=k_load,
            v=v_load,
            grouping=HeadGrouping(q.shape[0], query_heads, kv_heads, head_dim),
            scale=scalar_operand(scale_operation, 1),
            mask=mask,
            outputs=attention_live_outputs,
            boundary_output=boundary_output,
            cache=_cache_effects(k_root, v_root, graph),
        ),
        cost_bytes=sum(
            value_bytes(operation.results[0], graph)
            for operation in operations[:-1]
            if isinstance(operation.tensor_program, TensorProgram)
            and operation.tensor_program.store.owns_storage
        ),
        rejections=tuple(rejections),
    )


def _attention_head_axes(value: Any) -> tuple[int, ...]:
    """Axes that can hold the heads: their stride steps over one head dim.

    A size-1 axis has no meaningful stride, so with one KV head (MQA) or one
    token both axes can qualify; the caller resolves that with the score shape.
    """
    if value.strides is None or len(value.strides) != 4:
        return ()
    head_stride = value.shape[3] * value.strides[3]
    return tuple(
        axis
        for axis in (2, 1)
        if value.strides[axis] == head_stride or value.shape[axis] == 1
    )


def _attention_axes(
    q: Any, k: Any, v: Any, scores: tuple[int, ...] | None
) -> tuple[int, int, int] | None:
    """Head axes of q, k and v that agree with each other and with the
    ``[batch * query_heads, query_length, kv_length]`` score matrix."""
    for q_axis in _attention_head_axes(q):
        query_heads, query_length = q.shape[q_axis], q.shape[3 - q_axis]
        for k_axis in _attention_head_axes(k):
            kv_heads, kv_length = k.shape[k_axis], k.shape[3 - k_axis]
            for v_axis in _attention_head_axes(v):
                if (
                    query_heads % kv_heads
                    or v.shape[v_axis] != kv_heads
                    or v.shape[3 - v_axis] != kv_length
                ):
                    continue
                if scores is not None and tuple(scores) != (
                    q.shape[0] * query_heads,
                    query_length,
                    kv_length,
                ):
                    continue
                return q_axis, k_axis, v_axis
    return None


def _producer_until(
    value_id: str, predicate: Callable[[Op], bool], graph: ComputeGraph
) -> Op | None:
    visited: set[str] = set()
    while value_id not in visited:
        visited.add(value_id)
        operation = producer_of(value_id, graph)
        if operation is None:
            return None
        if predicate(operation):
            return operation
        program = operation.tensor_program
        if (
            not isinstance(program, TensorProgram)
            or program.family not in {"view", "copy", "elementwise"}
            or len(operation.operands) != 1
        ):
            return None
        value_id = operation.operands[0]
    return None


def _producer_path_until(
    value_id: str, predicate: Callable[[Op], bool], graph: ComputeGraph
) -> tuple[Op, ...] | None:
    operations: list[Op] = []
    visited: set[str] = set()
    while value_id not in visited:
        visited.add(value_id)
        operation = producer_of(value_id, graph)
        if operation is None:
            return None
        if predicate(operation):
            return (*operations, operation)
        program = operation.tensor_program
        if (
            not isinstance(program, TensorProgram)
            or program.family not in {"view", "copy"}
            or len(operation.operands) != 1
        ):
            return None
        operations.append(operation)
        value_id = operation.operands[0]
    return None


def _producer_path(value_id: str, stop: Op, graph: ComputeGraph) -> tuple[Op, ...] | None:
    operations: list[Op] = []
    visited: set[str] = set()
    while value_id not in visited:
        visited.add(value_id)
        operation = producer_of(value_id, graph)
        if operation is None:
            return None
        if operation.id == stop.id:
            return tuple(operations)
        program = operation.tensor_program
        if (
            not isinstance(program, TensorProgram)
            or program.family not in {"view", "copy", "elementwise"}
            or len(operation.operands) != 1
        ):
            return None
        operations.append(operation)
        value_id = operation.operands[0]
    return None


def _boundary_output_path(
    value_id: str, graph: ComputeGraph
) -> tuple[str, tuple[Op, ...]]:
    operations: list[Op] = []
    current = value_id
    while current not in graph.outputs:
        users = graph.values[current].users
        if len(users) != 1:
            break
        operation = graph.ops.get(users[0])
        program = operation.tensor_program if operation is not None else None
        if (
            operation is None
            or not isinstance(program, TensorProgram)
            or program.family != "view"
            or len(operation.operands) != 1
        ):
            break
        operations.append(operation)
        current = operation.results[0]
    return current, tuple(operations)


def _value_root_path(
    value_id: str, graph: ComputeGraph
) -> tuple[str, tuple[Op, ...]]:
    operations: list[Op] = []
    current = value_id
    while (
        graph.values[current].shape is None
        or len(graph.values[current].shape) != 4
    ):
        operation = producer_of(current, graph)
        program = operation.tensor_program if operation is not None else None
        if (
            operation is None
            or not isinstance(program, TensorProgram)
            or program.family not in {"view", "copy"}
            or len(operation.operands) != 1
        ):
            return current, tuple(operations)
        operations.append(operation)
        current = operation.operands[0]
    while True:
        operation = producer_of(current, graph)
        program = operation.tensor_program if operation is not None else None
        operand_shape = (
            graph.values[operation.operands[0]].shape
            if operation is not None and operation.operands
            else None
        )
        if (
            operation is None
            or not isinstance(program, TensorProgram)
            or program.family not in {"view", "copy"}
            or len(operation.operands) != 1
            or operand_shape is None
            or len(operand_shape) < 4
        ):
            return current, tuple(operations)
        operations.append(operation)
        current = operation.operands[0]


def _softmax_operations(
    division: Op, graph: ComputeGraph
) -> tuple[str, str, tuple[Op, ...]] | None:
    if not matches_target(division, "aten.div.Tensor") or len(division.operands) != 2:
        return None
    exponential = producer_of(division.operands[0], graph)
    subtraction = (
        producer_of(exponential.operands[0], graph)
        if exponential is not None
        else None
    )
    if (
        exponential is None
        or subtraction is None
        or not matches_target(exponential, "aten.exp.default")
        or not matches_target(subtraction, "aten.sub.Tensor")
        or len(subtraction.operands) != 2
    ):
        return None

    maximum = producer_of(subtraction.operands[1], graph)
    if (
        maximum is not None
        and matches_target(maximum, "aten.amax.default")
        and exponential.operands == (subtraction.results[0],)
        and producer_of(division.operands[1], graph) is not None
        and matches_target(producer_of(division.operands[1], graph), "aten.sum.dim_IntList")
        and producer_of(producer_of(division.operands[1], graph).operands[0], graph)
        is exponential
    ):
        return "stable_softmax", subtraction.operands[0], (
            maximum,
            subtraction,
            exponential,
            producer_of(division.operands[1], graph),
            division,
        )

    maximum_projection = producer_of(subtraction.operands[1], graph)
    sum_projection = producer_of(division.operands[1], graph)
    if (
        maximum_projection is None
        or sum_projection is None
        or operation_target(maximum_projection) != str(operator.getitem)
        or operation_target(sum_projection) != str(operator.getitem)
    ):
        return None
    statistics = producer_of(maximum_projection.operands[0], graph)
    if (
        statistics is None
        or not matches_target(statistics, "prims.prepare_softmax_online.default")
        or maximum_projection.operands[0] != statistics.results[0]
        or maximum_projection.attributes.get("fx_args", (None, None))[1] != 0
        or sum_projection.attributes.get("fx_args", (None, None))[1] != 1
        or statistics.results[1] not in sum_projection.operands
        or statistics.operands != (subtraction.operands[0],)
        or exponential.operands != (subtraction.results[0],)
    ):
        return None
    return "online_softmax", subtraction.operands[0], (
        statistics,
        maximum_projection,
        sum_projection,
        subtraction,
        exponential,
        division,
    )


def _score_chain(
    score_value: str, graph: ComputeGraph
) -> tuple[Op | None, Op, str | None, Op, tuple[Op, ...]] | None:
    operation = producer_of(score_value, graph)
    if (
        operation is not None
        and matches_target(operation, "prims.convert_element_type.default")
        and len(operation.operands) == 1
    ):
        score_value = operation.operands[0]
        operation = producer_of(score_value, graph)
    mask_value: str | None = None
    if operation is not None and matches_target(operation, "aten.add.Tensor"):
        if len(operation.operands) != 2:
            return None
        mask_value = operation.operands[1]
        scale_operation = producer_of(operation.operands[0], graph)
    else:
        scale_operation = operation
    if scale_operation is None or not matches_target(scale_operation, "aten.mul.Tensor"):
        return None
    score_path = _producer_path_until(
        scale_operation.operands[0],
        lambda candidate: matches_target(candidate, "aten.bmm.default"),
        graph,
    )
    score_bmm = score_path[-1] if score_path is not None else None
    if score_bmm is None or not isinstance(scalar_operand(scale_operation, 1), (int, float)):
        return None
    return (
        operation if mask_value is not None else None,
        scale_operation,
        mask_value,
        score_bmm,
        score_path[:-1],
    )


def _mask(value_id: str | None, graph: ComputeGraph) -> tuple[Mask, str | None]:
    """The mask added to the scores, and the tensor the body reads for it."""
    if value_id is None:
        return Mask("none"), None
    operation = producer_of(value_id, graph)
    if operation is not None and matches_target(operation, "aten.where.self"):
        program = operation.tensor_program
        if isinstance(program, TensorProgram):
            condition = producer_of(_condition_value(program, graph), graph)
            condition_program = condition.tensor_program if condition is not None else None
            condition_loads = (
                condition_program.loads if isinstance(condition_program, TensorProgram) else ()
            )
            mask = Mask(
                "typed_where",
                causal_offset=_causal_offset(condition, graph),
                window_offsets=_window_offsets(condition, graph),
                true_value=_scalar_mask_value(program.loads[1], graph),
                false_value=_scalar_mask_value(program.loads[2], graph),
            )
            return mask, condition_loads[0].value_id if condition_loads else None
    return Mask("additive_tensor"), value_id


def _scalar_mask_value(load: IndexedLoad, graph: ComputeGraph) -> float | None:
    operation = producer_of(load.value_id, graph)
    if (
        operation is not None
        and isinstance(operation.tensor_program, TensorProgram)
        and operation.tensor_program.family == "copy"
    ):
        source = producer_of(operation.tensor_program.loads[0].value_id, graph)
        if (
            source is not None
            and isinstance(source.tensor_program, TensorProgram)
            and isinstance(source.tensor_program.combine, Constant)
        ):
            value = source.tensor_program.combine.value
            return value if isinstance(value, (int, float)) else None
    if (
        operation is not None
        and matches_target(operation, "aten.full.default")
        and len(operation.attributes.get("fx_args", ())) >= 2
    ):
        value = operation.attributes["fx_args"][1]
        return value if isinstance(value, (int, float)) else None
    if (
        operation is None
        or not matches_target(operation, "aten.scalar_tensor.default")
        or not operation.attributes.get("fx_args")
    ):
        return None
    value = operation.attributes["fx_args"][0]
    return value if isinstance(value, (int, float)) else None


def _condition_value(program: TensorProgram, graph: ComputeGraph) -> str:
    condition = program.combine
    if isinstance(condition, Where) and isinstance(condition.condition, Load):
        return next(
            load.value_id for load in program.loads if load.id == condition.condition.load_id
        )
    return program.loads[0].value_id


def _causal_offset(condition: Op | None, graph: ComputeGraph) -> int | None:
    if condition is None or not matches_target(condition, "aten.le.Tensor"):
        return None
    return _comparison_offset(condition, graph)


def _window_offsets(condition: Op | None, graph: ComputeGraph) -> tuple[int, int] | None:
    if condition is None or not matches_target(condition, "aten.bitwise_and.Tensor"):
        return None
    if len(condition.operands) != 2:
        return None
    upper = producer_of(condition.operands[0], graph)
    lower = producer_of(condition.operands[1], graph)
    if (
        upper is None
        or lower is None
        or not matches_target(upper, "aten.le.Tensor")
        or not matches_target(lower, "aten.ge.Tensor")
    ):
        return None
    lower_offset = _comparison_offset(lower, graph)
    upper_offset = _comparison_offset(upper, graph)
    if lower_offset is None or upper_offset is None:
        return None
    return lower_offset, upper_offset


def _comparison_offset(condition: Op, graph: ComputeGraph) -> int | None:
    program = condition.tensor_program
    if not isinstance(program, TensorProgram) or len(program.loads) != 2:
        return None
    offset = producer_of(program.loads[1].value_id, graph)
    if offset is None or not (
        matches_target(offset, "aten.add.Tensor")
        or matches_target(offset, "aten.sub.Tensor")
    ):
        return None
    value = scalar_operand(offset, 1)
    if not isinstance(value, int):
        return None
    return -value if matches_target(offset, "aten.sub.Tensor") else value


def _cache_effects(key_root: str, value_root: str, graph: ComputeGraph) -> CacheEffects:
    key_state, key_append = _cache_access(key_root, graph)
    value_state, value_append = _cache_access(value_root, graph)
    updates = tuple(
        value_id
        for value_id in (key_root, value_root)
        if value_id in graph.outputs
    )
    if updates:
        policy = "functional_append"
    elif _in_place_cache_update((key_state, value_state, key_root, value_root), graph):
        policy = "in_place"
    else:
        policy = "read_only"
    return CacheEffects(
        policy,
        reads=(key_state, value_state),
        appends=tuple(
            value_id
            for value_id in (key_append, value_append)
            if value_id is not None
        ),
        updates=updates,
    )


def _cache_access(value_id: str, graph: ComputeGraph) -> tuple[str, str | None]:
    operation = producer_of(value_id, graph)
    if (
        operation is not None
        and matches_target(operation, "aten.cat.default")
        and len(operation.operands) == 2
        and _is_sequence_append(operation, graph)
    ):
        return operation.operands[0], operation.operands[1]
    return value_id, None


def _is_sequence_append(operation: Op, graph: ComputeGraph) -> bool:
    # A cache append extends a boundary cache along a non-feature axis; a
    # feature-axis cat (for example rotate-half RoPE) is ordinary computation.
    cache = graph.values.get(operation.operands[0])
    if cache is None or cache.shape is None:
        return False
    arguments = operation.attributes.get("fx_args", ())
    dimension = arguments[1] if len(arguments) > 1 else 0
    if not isinstance(dimension, int):
        return False
    if dimension % len(cache.shape) == len(cache.shape) - 1:
        return False
    seen: set[str] = set()
    pending = [cache.id]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in graph.inputs:
            return True
        value = graph.values.get(current)
        if value is None:
            continue
        pending.extend(value.storage_relation.aliases)
        producer = producer_of(current, graph)
        if producer is not None and len(producer.operands) == 1:
            program = producer.tensor_program
            if isinstance(program, TensorProgram) and program.family in {"view", "copy"}:
                pending.append(producer.operands[0])
    return False


def _in_place_cache_update(cache_values: tuple[str, ...], graph: ComputeGraph) -> bool:
    return any(
        operation.effects.get("mutates_inputs")
        and operation.operands
        and operation.operands[0] in cache_values
        for operation in graph.ops.values()
    )

