"""Linear attention, one-token decode, as an attention algorithm: the op a
model declares (``megabake::linear_attention``), its eager reference, its
program (family ``attention``), and its region, an ``Attention`` with
``algorithm="linear"`` and a ``LinearRecurrence`` (which documents the
computation).

One call is a linear-attention layer's token mixer between its input
projections and its output projection: Gated DeltaNet (Qwen3.5, Qwen3-Next),
Kimi Delta Attention, Mamba2 and gated linear attention differ only in the
recurrence's options. Decomposed, a layer is tens of regions; declared, it is
one, which ``AttentionBodyProvider`` serves.

The causal-convolution window is a ring indexed by the token position
(``conv_position``), so the op writes the new token into the one slot no tap
reads: value heads that share key channels (Mamba2's B and C, grouped GDN)
read the same window without racing on a shift, and no separate shift ops (a
concat and an in-place copy, which cost a barrier interval per layer) remain.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..graph import ComputeGraph, Op, RegionCandidate, Value
from ..ir import (
    AxisPredicate,
    Constant,
    IndexedLoad,
    IterationDomain,
    TensorProgram,
    TensorStore,
    UnsupportedLowering,
)
from .attention import Attention, CacheEffects, HeadGrouping, LinearRecurrence, Mask, NumericalPolicy
from .matching import matches_target

# The tensor operands, in schema order; all but the first two are optional.
TENSOR_ARGUMENTS = (
    "qkv",
    "state",
    "decay",
    "conv_state",
    "conv_weight",
    "conv_bias",
    "a_log",
    "dt_bias",
    "beta",
    "skip",
    "gate",
    "norm_weight",
    "conv_position",
)


@torch.library.custom_op("megabake::linear_attention", mutates_args=("state", "conv_state"))
def linear_attention(
    qkv: torch.Tensor,
    state: torch.Tensor,
    decay: torch.Tensor,
    conv_state: torch.Tensor | None,
    conv_weight: torch.Tensor | None,
    conv_bias: torch.Tensor | None,
    a_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    beta: torch.Tensor | None,
    skip: torch.Tensor | None,
    gate: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    key_heads: int,
    head_k_dim: int,
    q_offset: int,
    k_offset: int,
    v_offset: int,
    scale: float,
    qk_l2norm: bool,
    state_layout: str,
    dt_update: bool,
    gate_activation: str,
    norm_eps: float,
    norm_rounds: bool,
    conv_silu: bool = True,
    conv_position: torch.Tensor | None = None,
) -> torch.Tensor:
    """One decode token of a linear-attention layer (see ``LinearRecurrence``).

    ``qkv`` is ``[1, channels]``, the projected channels before the causal
    convolution; q, k (``key_heads`` heads of ``head_k_dim``) and v
    (``state.shape[1]`` heads) start at their offsets. ``state`` is the FP32
    ``[1, heads, head_k_dim, head_v_dim]`` (``state_layout="kv"``) or ``[1,
    heads, head_v_dim, head_k_dim]`` (``"vk"``) state, updated in place.
    ``decay`` is ``[1, heads]`` or ``[1, heads * head_k_dim]``: the gate
    input ``a`` when ``a_log`` is given, else the log decay. ``beta`` selects
    the delta rule; ``skip`` is Mamba2's ``D``; ``norm_weight`` the per-head
    gated RMSNorm of the output with ``gate``. Returns ``[1, heads *
    head_v_dim]`` in ``qkv``'s dtype.

    With ``conv_position`` (the token's position p, an integer tensor) the
    convolution window is a ring ``[1, channels, kernel]``: token t sits in
    slot t % kernel, the taps read the previous ``kernel - 1`` tokens' slots
    and the new token is written to slot p % kernel, which no tap reads, so
    the op updates the window itself (heads sharing channels write the same
    value). Without it the window is ``[1, channels, kernel - 1]``, read only;
    the caller shifts it (``shift_conv_window``).
    """
    heads, head_v_dim = _heads(state, head_k_dim, state_layout)
    x = qkv
    if conv_weight is not None and conv_position is not None:
        kernel = conv_weight.shape[2]
        # Indexed on the device: no host read of the position, so the
        # reference runs inside CUDA graphs (the Inductor baselines).
        position = conv_position.reshape(-1)[:1].long()
        taps = (position + 1 + torch.arange(kernel - 1, device=position.device)) % kernel
        window = torch.cat([conv_state.index_select(2, taps), qkv[:, :, None]], dim=-1)
        conv_state.index_copy_(2, position % kernel, qkv[:, :, None])
        x = F.conv1d(window, conv_weight, conv_bias, groups=qkv.shape[1])[..., 0]
        if conv_silu:
            x = F.silu(x)
    elif conv_weight is not None:
        window = torch.cat([conv_state, qkv[:, :, None]], dim=-1)
        x = F.conv1d(window, conv_weight, conv_bias, groups=qkv.shape[1])[..., 0]
        if conv_silu:
            x = F.silu(x)
    query = x[:, q_offset : q_offset + key_heads * head_k_dim].reshape(1, key_heads, head_k_dim).float()
    key = x[:, k_offset : k_offset + key_heads * head_k_dim].reshape(1, key_heads, head_k_dim).float()
    value = x[:, v_offset : v_offset + heads * head_v_dim].reshape(1, heads, head_v_dim).float()
    if heads != key_heads:
        query = query.repeat_interleave(heads // key_heads, dim=1)
        key = key.repeat_interleave(heads // key_heads, dim=1)
    if qk_l2norm:
        query = _l2norm(query)
        key = _l2norm(key)
    query = query * scale
    per_key = decay.shape[1] == heads * head_k_dim
    gate_input = decay.float()
    step = None
    if a_log is not None:
        if dt_bias is not None:
            gate_input = gate_input + dt_bias
        step = F.softplus(gate_input)
        rate = -a_log.float().exp()
        log_decay = (rate[:, None] if per_key else rate) * step.reshape(1, heads, -1).squeeze(-1)
    else:
        log_decay = gate_input.reshape(1, heads, -1).squeeze(-1)
    factor = log_decay.exp()
    current = state if state_layout == "kv" else state.transpose(-1, -2)
    current = current * (factor[..., None] if per_key else factor[..., None, None])
    if beta is not None:
        remembered = (current * key[..., None]).sum(dim=-2)
        update = (value - remembered) * beta.float().sigmoid()[..., None]
    elif dt_update:
        update = value * step[..., None]
    else:
        update = value
    current = current + key[..., None] * update[..., None, :]
    state.copy_(current if state_layout == "kv" else current.transpose(-1, -2))
    output = (current * query[..., None]).sum(dim=-2)
    if skip is not None:
        output = output + value * skip.float()[:, None]
    output = output.to(qkv.dtype)
    if norm_weight is not None:
        output = _gated_rms_norm(
            output.reshape(-1, head_v_dim), gate.reshape(-1, head_v_dim), norm_weight,
            norm_eps, gate_activation, norm_rounds,
        )
    elif gate is not None:
        output = (output.reshape(-1, head_v_dim).float() * _activation(gate.reshape(-1, head_v_dim).float(), gate_activation)).to(qkv.dtype)
    return output.reshape(1, heads * head_v_dim)


@linear_attention.register_fake
def _linear_attention_fake(qkv, state, *arguments):
    head_k_dim, state_layout = arguments[11], arguments[17]
    heads, head_v_dim = _heads(state, head_k_dim, state_layout)
    return torch.empty((1, heads * head_v_dim), dtype=qkv.dtype, device=qkv.device)


LINEAR_ATTENTION_OP = torch.ops.megabake.linear_attention.default
LINEAR_ATTENTION_TARGET = "megabake.linear_attention.default"

# The rounding points: BF16 convolution and head outputs, FP32 normalization,
# gates, state and accumulation.
LINEAR_NUMERICS = NumericalPolicy("torch.bfloat16", "torch.float32", None, "torch.float32")


def shift_conv_window(conv_state: torch.Tensor, qkv: torch.Tensor) -> None:
    """Shift the new token's channels into a causal-convolution window, after
    ``linear_attention`` has read it."""
    conv_state.copy_(torch.cat([conv_state[:, :, 1:], qkv[:, :, None]], dim=-1))


def _heads(state: torch.Tensor, head_k_dim: int, state_layout: str) -> tuple[int, int]:
    if state.dim() != 4 or state.shape[0] != 1 or state_layout not in ("kv", "vk"):
        raise RuntimeError("linear attention state must be [1, heads, ., .] in a kv or vk layout")
    key_axis = 2 if state_layout == "kv" else 3
    if state.shape[key_axis] != head_k_dim:
        raise RuntimeError("linear attention state does not match head_k_dim")
    return state.shape[1], state.shape[5 - key_axis]


def _l2norm(source: torch.Tensor) -> torch.Tensor:
    return source * torch.rsqrt((source * source).sum(dim=-1, keepdim=True) + 1e-6)


def _activation(gate: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "silu":
        return F.silu(gate)
    if activation == "sigmoid":
        return torch.sigmoid(gate)
    raise RuntimeError(f"unknown output gate activation {activation!r}")


def _gated_rms_norm(
    hidden: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float,
    activation: str, rounds: bool,
) -> torch.Tensor:
    """Per-head RMSNorm, then the gate: ``Qwen3_5RMSNormGated`` when
    ``rounds`` (BF16 normalized head times the weight), else
    ``KimiLinearRMSNormGated`` (FP32 throughout)."""
    dtype = hidden.dtype
    hidden = hidden.to(torch.float32)
    hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps)
    hidden = weight * hidden.to(dtype) if rounds else weight.to(torch.float32) * hidden
    return (hidden * _activation(gate.to(torch.float32), activation)).to(dtype)


# One iteration axis per dimension a load indexes; "b" and "e" are the output's.
_AXES = {
    "qkv": (0, "c"),
    "state": (0, "h", "s0", "s1"),
    "decay": (0, "g"),
    "conv_state": (0, "c", "t"),
    "conv_weight": ("c", 0, "w"),
    "conv_bias": ("c",),
    "a_log": ("h",),
    "dt_bias": ("g",),
    "beta": (0, "h"),
    "skip": ("h",),
    "gate": (0, "e"),
    "norm_weight": ("v",),
    "conv_position": None,  # all-constant axes, by rank
}


def arguments_of(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """The op's arguments by schema name, from a call's ``args`` and ``kwargs``."""
    return {
        argument.name: args[index] if index < len(args) else kwargs.get(argument.name, argument.default_value)
        for index, argument in enumerate(LINEAR_ATTENTION_OP._schema.arguments)
    }


def static_attributes(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """The op's non-tensor arguments, which must be static."""
    attributes = {
        name: value for name, value in arguments_of(args, kwargs).items() if name not in TENSOR_ARGUMENTS
    }
    if not all(isinstance(value, (bool, int, float, str)) for value in attributes.values()):
        raise UnsupportedLowering("linear attention options must be static")
    return attributes


def linear_attention_program(
    operands: dict[str, Value], attributes: dict[str, Any], output: Value
) -> TensorProgram:
    """The program of ``megabake::linear_attention`` over its present tensor
    operands (by argument name)."""
    if any(
        value.shape is None or value.strides is None or value.dtype is None
        for value in (*operands.values(), output)
    ):
        raise UnsupportedLowering("linear attention operands lack static tensor metadata")
    qkv, state = operands["qkv"], operands["state"]
    layout = attributes["state_layout"]
    head_k_dim, key_heads = attributes["head_k_dim"], attributes["key_heads"]
    if len(state.shape) != 4 or layout not in ("kv", "vk") or len(qkv.shape) != 2 or qkv.shape[0] != 1:
        raise UnsupportedLowering("linear attention state or qkv does not match the op contract")
    heads = state.shape[1]
    head_v_dim = state.shape[3] if layout == "kv" else state.shape[2]
    channels = qkv.shape[1]
    kernel = operands["conv_weight"].shape[2] if "conv_weight" in operands else 0
    shapes = {
        "decay": {(1, heads), (1, heads * head_k_dim)},
        "conv_state": {(1, channels, kernel if "conv_position" in operands else kernel - 1)},
        "conv_position": {(1,), (1, 1)},
        "conv_weight": {(channels, 1, kernel)},
        "conv_bias": {(channels,)},
        "a_log": {(heads,)},
        "dt_bias": {(operands["decay"].shape[1],)},
        "beta": {(1, heads)},
        "skip": {(heads,)},
        "gate": {(1, heads * head_v_dim)},
        "norm_weight": {(head_v_dim,)},
    }
    ranges = (
        attributes["q_offset"] + key_heads * head_k_dim,
        attributes["k_offset"] + key_heads * head_k_dim,
        attributes["v_offset"] + heads * head_v_dim,
    )
    if (
        key_heads <= 0
        or heads % key_heads
        or max(ranges) > channels
        or ("conv_weight" in operands) != ("conv_state" in operands)
        or ("norm_weight" in operands and "gate" not in operands)
        or attributes["dt_update"] and ("a_log" not in operands or operands["decay"].shape != (1, heads))
        or any(operands[name].shape not in allowed for name, allowed in shapes.items() if name in operands)
        or output.shape != (1, heads * head_v_dim)
        or state.dtype != "torch.float32"
        or ("conv_position" in operands and operands["conv_position"].dtype not in ("torch.int64", "torch.int32"))
        or ("conv_position" in operands and "conv_weight" not in operands)
    ):
        raise UnsupportedLowering("linear attention operand shapes do not match the op contract")
    loads = tuple(
        IndexedLoad(
            f"attention:{name}", value.id,
            _AXES[name] if _AXES[name] is not None else (0,) * len(value.shape),
            value.strides, value.dtype, value.shape,
        )
        for name, value in operands.items()
    )
    sizes = {
        "c": channels, "h": heads, "s0": state.shape[2], "s1": state.shape[3],
        "g": operands["decay"].shape[1],
        "t": max(kernel if "conv_position" in operands else kernel - 1, 1), "w": max(kernel, 1),
        "v": head_v_dim,
    }
    used = {axis for load in loads for axis in load.axes if isinstance(axis, str)} - {"e"}
    reductions = tuple(axis for axis in sizes if axis in used)
    zero = Constant(0.0, "torch.float32")
    return TensorProgram(
        family="attention",
        source_target="megabake.linear_attention",
        decomposition="one-token linear-attention recurrence over an in-place per-head state",
        # The output is [1, heads * head_v_dim]; every other axis indexes an operand.
        iteration_domains=(
            IterationDomain("b", 1),
            IterationDomain("e", heads * head_v_dim),
            *(IterationDomain(axis, sizes[axis]) for axis in reductions),
        ),
        reduction_axes=reductions,
        loads=loads,
        init=zero,
        combine=zero,
        post=None,
        accumulator_dtype="torch.float32",
        store=TensorStore(output.id, output.shape, output.dtype, output.strides, True),
        predicates=(AxisPredicate("e", heads * head_v_dim),),
    )


def candidate(operation: Op, graph: ComputeGraph) -> RegionCandidate | None:
    """``megabake::linear_attention``: the model declares the layer's token
    mixer, so the region needs no pattern matching."""
    if not matches_target(operation, LINEAR_ATTENTION_TARGET):
        return None
    program = operation.tensor_program
    if not isinstance(program, TensorProgram) or program.source_target != "megabake.linear_attention":
        return None
    attributes = static_attributes(
        tuple(operation.attributes.get("fx_args", ())), dict(operation.attributes.get("fx_kwargs", {}) or {})
    )
    loads = {load.id.removeprefix("attention:"): load for load in program.loads}
    state = loads["state"]
    layout = attributes["state_layout"]
    heads = state.source_shape[1]
    head_k_dim = attributes["head_k_dim"]
    head_v_dim = state.source_shape[3] if layout == "kv" else state.source_shape[2]
    qkv = loads["qkv"]
    output_id = operation.results[0]
    recurrence = LinearRecurrence(
        state_value_id=state.value_id,
        state_layout=layout,
        head_v_dim=head_v_dim,
        conv_kernel=loads["conv_weight"].source_shape[2] if "conv_weight" in loads else 0,
        q_offset=attributes["q_offset"],
        k_offset=attributes["k_offset"],
        v_offset=attributes["v_offset"],
        decay="softplus_gate" if "a_log" in loads else "log",
        decay_per_key=loads["decay"].source_shape[1] == heads * head_k_dim,
        update="delta" if "beta" in loads else "additive",
        dt_update=bool(attributes["dt_update"]),
        qk_l2norm=bool(attributes["qk_l2norm"]),
        skip="skip" in loads,
        output_gate=attributes["gate_activation"] if "gate" in loads else "none",
        norm="norm_weight" in loads,
        norm_eps=float(attributes["norm_eps"]),
        norm_rounds=bool(attributes["norm_rounds"]),
        conv_silu=bool(attributes["conv_silu"]),
        conv_ring="conv_position" in loads,
    )
    reads = tuple(load.value_id for name, load in loads.items() if name in ("state", "conv_state"))
    updates = (state.value_id, *((loads["conv_state"].value_id,) if "conv_position" in loads else ()))
    return RegionCandidate(
        id=f"region:{operation.id.removeprefix('op:')}:linear_attention",
        source_op_ids=(operation.id,),
        external_inputs=operation.operands,
        live_outputs=(output_id,),
        tensor_program=program,
        semantics=Attention(
            algorithm="linear",
            numerics=LINEAR_NUMERICS,
            q=qkv,
            k=qkv,
            v=qkv,
            grouping=HeadGrouping(1, heads, attributes["key_heads"], head_k_dim),
            scale=float(attributes["scale"]),
            mask=Mask("none"),
            outputs=(output_id,),
            boundary_output=output_id if output_id in graph.outputs else None,
            cache=CacheEffects("stateful", reads=reads, updates=updates),
            recurrent=recurrence,
        ),
        # The state is read and written once.
        cost_bytes=2 * heads * head_k_dim * head_v_dim * 4,
    )
