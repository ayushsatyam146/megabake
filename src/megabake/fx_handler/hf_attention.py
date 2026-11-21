"""Route Hugging Face attention through ``megabake::attention``.

Every transformers attention module dispatches through ``AttentionInterface``
by ``config._attn_implementation``. Registering a ``megabake`` implementation
therefore declares attention for all such models at once, in place of
recovering it from decomposed ATen by pattern matching:

    register_transformers_attention()
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="megabake")

Prefill calls ``megabake::attention``. ``DecodeStep`` wraps the same model for
one-token decode over explicit KV state through ``megabake::kv_attention``,
and declares linear-attention layers (Gated DeltaNet, Kimi Delta Attention,
Mamba2) through ``megabake::linear_attention`` over their convolution window
and recurrent state.
"""

from __future__ import annotations

import inspect
import sys
import types
from typing import Any

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from ..ops.attention import attention
from ..ops.linear_attention import linear_attention

ATTENTION_IMPLEMENTATION = "megabake"


def megabake_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Hugging Face attention-interface signature: ``query`` is
    ``[batch, heads, query, dim]``, ``key`` and ``value`` are
    ``[batch, kv_heads, key, dim]``; returns ``[batch, query, heads, dim]``."""
    if (
        (dropout and module.training)
        or softcap is not None
        or kwargs.get("s_aux") is not None
        or kwargs.get("output_attentions")
    ):
        # Not covered by the declared op yet: keep the model's own eager path.
        eager = sys.modules[type(module).__module__].eager_attention_forward
        return eager(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, softcap=softcap, **kwargs,
        )
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    mask = None if attention_mask is None else attention_mask[:, :, :, : key.shape[-2]]
    batch, heads, length, dim = query.shape
    output = attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), mask, float(scaling)
    )
    return output.view(batch, heads, length, dim).transpose(1, 2).contiguous(), None


def register_transformers_attention() -> str:
    """Register the ``megabake`` attention and mask implementations; returns the name."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, eager_mask

    AttentionInterface.register(ATTENTION_IMPLEMENTATION, megabake_decode_attention)
    # The op takes the eager additive mask, which already encodes causal,
    # sliding-window and padding rules.
    AttentionMaskInterface.register(ATTENTION_IMPLEMENTATION, eager_mask)
    return ATTENTION_IMPLEMENTATION


class _PassThroughLayer(CacheLayerMixin):
    """A cache layer that leaves storage to ``megabake::kv_attention``: it hands
    the new token's key and value straight to the attention call."""

    is_compileable = True
    is_sliding = False

    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        self.is_initialized = True

    def lazy_initialization(self, key_states, value_states) -> None:
        pass

    def update(self, key_states, value_states, *args, **kwargs):
        return key_states, value_states

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.capacity, 0

    def get_seq_length(self) -> int:
        return 0

    def get_max_length(self) -> int:
        return self.capacity


def _decode_cache(layer_count: int, capacity: int) -> Cache:
    return Cache(layers=[_PassThroughLayer(capacity) for _ in range(layer_count)])


def megabake_decode_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    megabake_state: tuple[tuple[torch.Tensor, ...], ...] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """One-token decode through ``megabake::kv_attention`` on the layer's own
    cache. The cache holds every earlier token, so the causal mask is implied."""
    if megabake_state is None:
        return megabake_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, softcap=softcap, **kwargs,
        )
    if kwargs.get("s_aux") is not None:
        raise NotImplementedError("decode attention with sinks is not declared yet")
    key_cache, value_cache, valid_length = megabake_state[module.layer_idx]
    # A sliding layer sees its last ``window`` keys of the linear cache. The
    # attention interface passes the window (Mistral keeps it only in its
    # config); older modules carry it as an attribute.
    window = kwargs.get("sliding_window", getattr(module, "sliding_window", None)) or 0
    dim = query.shape[-1]
    batch, heads, length, _ = query.shape
    output = torch.ops.megabake.kv_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        key_cache, value_cache, valid_length,
        float(scaling or 0.0), int(window), float(softcap or 0.0),
    )
    return output.view(batch, heads, length, dim).transpose(1, 2).contiguous(), None


_ATTENTION_LAYERS = {"full_attention", "sliding_attention"}
_LINEAR_LAYERS = {"linear_attention", "mamba"}
# Attention modules that bypass transformers' attention interface, declared
# by an adapter (``MIXERS``) instead.
_DECLARED_ATTENTION = "declared_attention"


def _has(module: torch.nn.Module, *names: str) -> bool:
    return all(hasattr(module, name) for name in names)


class _Mixer:
    """How one family of token-mixing modules maps onto MegaBake's declared
    decode ops: the layer kind, its state shapes, and its decode step, which
    returns what the module's own forward returns."""

    kind = "linear_attention"

    def matches(self, module: torch.nn.Module) -> bool:
        raise NotImplementedError

    def state_shapes(self, module: torch.nn.Module, dtype: torch.dtype, capacity: int):
        raise NotImplementedError

    def decode(self, module, hidden_states, *state):
        raise NotImplementedError


def _silu_conv(module: torch.nn.Module) -> None:
    if getattr(module, "activation", "silu") != "silu":
        raise NotImplementedError("linear attention decode is declared for a SiLU convolution")


class _GatedDeltaNet(_Mixer):
    """Qwen3.5's Gated DeltaNet: scalar decay, the delta rule, a SiLU-gated norm."""

    def matches(self, module):
        return _has(module, "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "conv1d", "A_log", "dt_bias", "norm")

    def state_shapes(self, module, dtype, capacity):
        return [
            ((1, module.conv_dim, module.conv_kernel_size), dtype),
            ((1, module.num_v_heads, module.head_k_dim, module.head_v_dim), torch.float32),
        ]

    def decode(self, module, hidden_states, conv_state, state, position):
        _silu_conv(module)
        qkv = module.in_proj_qkv(hidden_states).reshape(1, -1)
        key_dim = module.num_k_heads * module.head_k_dim
        output = linear_attention(
            qkv, state, module.in_proj_a(hidden_states).reshape(1, -1),
            conv_state, module.conv1d.weight, module.conv1d.bias, module.A_log, module.dt_bias,
            module.in_proj_b(hidden_states).reshape(1, -1), None,
            module.in_proj_z(hidden_states).reshape(1, -1), module.norm.weight,
            module.num_k_heads, module.head_k_dim, 0, key_dim, 2 * key_dim,
            module.head_k_dim**-0.5, True, "kv", False, "silu",
            float(module.norm.variance_epsilon), True, True, position,
        )
        return module.out_proj(output.reshape(1, 1, -1))


class _KimiDeltaAttention(_Mixer):
    """Kimi Delta Attention: GDN with a decay per key dimension and a
    sigmoid-gated FP32 norm."""

    def matches(self, module):
        return _has(module, "q_proj", "k_proj", "v_proj", "conv1d", "forget_gate", "b_proj", "g_a_proj", "g_b_proj", "o_norm", "o_proj")

    def state_shapes(self, module, dtype, capacity):
        return [
            ((1, module.conv_dim, module.conv_kernel_size), dtype),
            ((1, module.num_heads, module.head_dim, module.head_dim), torch.float32),
        ]

    def decode(self, module, hidden_states, conv_state, state, position):
        _silu_conv(module)
        qkv = torch.cat(
            [module.q_proj(hidden_states), module.k_proj(hidden_states), module.v_proj(hidden_states)], dim=-1
        ).reshape(1, -1)
        forget = module.forget_gate
        width = module.qkv_dim
        output = linear_attention(
            qkv, state, forget.f_b_proj(forget.f_a_proj(hidden_states)).reshape(1, -1),
            conv_state, module.conv1d.weight, module.conv1d.bias, forget.A_log.reshape(-1), forget.dt_bias,
            module.b_proj(hidden_states).reshape(1, -1), None,
            module.g_b_proj(module.g_a_proj(hidden_states)).reshape(1, -1), module.o_norm.weight,
            module.num_heads, module.head_dim, 0, width, 2 * width,
            module.head_dim**-0.5, True, "kv", False, "sigmoid",
            float(module.o_norm.variance_epsilon), False, True, position,
        )
        return module.o_proj(output.reshape(1, 1, -1))


class _ComplexKDA(_Mixer):
    """ComplexKDA checkpoints' (OpenEuroLLM) Kimi Delta Attention with an
    unsigned gate: separate q/k/v short convolutions (declared as one), a
    per-key log decay ``lower_bound * sigmoid(exp(A_log) * (z + dt_bias))``
    ("sigmoid") or ``-exp(A_log) * softplus(z + dt_bias)`` ("softplus"), the
    delta rule and a sigmoid-gated FP32 norm. Returns the module's
    ``(output, None, cache)`` triple."""

    def matches(self, module):
        return _has(module, "q_conv1d", "k_conv1d", "v_conv1d", "f_proj", "b_proj", "g_proj", "o_norm", "o_proj", "lower_bound", "gate")

    def state_shapes(self, module, dtype, capacity):
        return [
            ((1, 2 * module.key_dim + module.value_dim, module.conv_size), dtype),
            ((1, module.num_v_heads, module.head_k_dim, module.head_v_dim), torch.float32),
        ]

    def decode(self, module, hidden_states, conv_state, state, position):
        if module.gate not in ("sigmoid", "softplus"):
            raise NotImplementedError(f"ComplexKDA's signed gate {module.gate!r} is not declared yet")
        if module.allow_neg_eigval or module.conv_silu not in ("", "qkv"):
            raise NotImplementedError("ComplexKDA decode is declared for beta in (0, 1) and all-or-none conv SiLU")
        convs = (module.q_conv1d, module.k_conv1d, module.v_conv1d)
        qkv = torch.cat(
            [module.q_proj(hidden_states), module.k_proj(hidden_states), module.v_proj(hidden_states)], dim=-1
        ).reshape(1, -1)
        weight = torch.cat([conv.weight for conv in convs], dim=0)
        bias = torch.cat([conv.bias for conv in convs], dim=0) if module.conv_bias else None
        heads, head_k_dim = module.num_v_heads, module.head_k_dim
        gate = module.f_proj(hidden_states).reshape(1, heads, head_k_dim).float() + module.dt_bias.view(1, heads, head_k_dim)
        rate = module.A_log.float().exp().view(1, heads, 1)
        log_decay = (
            module.lower_bound * torch.sigmoid(rate * gate)
            if module.gate == "sigmoid"
            else -rate * torch.nn.functional.softplus(gate)
        ).reshape(1, heads * head_k_dim)
        output = linear_attention(
            qkv, state, log_decay, conv_state, weight, bias, None, None,
            module.b_proj(hidden_states).reshape(1, -1), None,
            module.g_proj(hidden_states).reshape(1, -1), module.o_norm.weight,
            module.num_heads, head_k_dim, 0, module.key_dim, 2 * module.key_dim,
            head_k_dim**-0.5, True, "kv", False, "sigmoid",
            float(module.o_norm.eps), False, module.conv_silu == "qkv", position,
        )
        return module.o_proj(output.reshape(1, 1, -1)), None, None


class _ComplexKDAAttention(_Mixer):
    """The softmax attention of ComplexKDA's hybrids: NoPE (no rotary), an
    optional output gate ``sigmoid(g_proj(x))`` before ``o_proj``, through
    SDPA rather than transformers' attention interface, so it is declared
    here as ``megabake::kv_attention`` over its own KV cache."""

    kind = _DECLARED_ATTENTION

    def matches(self, module):
        return _has(module, "q_proj", "k_proj", "v_proj", "o_proj", "num_kv_heads", "rotary", "window_size", "output_gate") and not hasattr(module, "q_conv1d")

    def state_shapes(self, module, dtype, capacity):
        cache = ((1, capacity, module.num_kv_heads, module.head_dim), dtype)
        return [cache, cache, ((1,), torch.int32)]

    def decode(self, module, hidden_states, key_cache, value_cache, valid_length):
        if module.rotary is not None:
            raise NotImplementedError("ComplexKDA attention with rotary embeddings is not declared yet")
        heads, kv_heads, dim = module.num_heads, module.num_kv_heads, module.head_dim
        query = module.q_proj(hidden_states).view(1, 1, heads, dim)
        key = module.k_proj(hidden_states).view(1, 1, kv_heads, dim)
        value = module.v_proj(hidden_states).view(1, 1, kv_heads, dim)
        if module.qk_norm:
            query, key = module.q_norm(query), module.k_norm(key)
        output = torch.ops.megabake.kv_attention(
            query, key, value, key_cache, value_cache, valid_length, 0.0, int(module.window_size or 0), 0.0
        ).reshape(1, 1, heads * dim)
        if module.output_gate:
            output = output * torch.sigmoid(module.g_proj(hidden_states))
        return module.o_proj(output), None, None


class _Mamba2(_Mixer):
    """Mamba2's selective state update: scalar decay exp(dt * A), the
    additive dt-scaled update of a [v, k] state, the D skip and the SiLU
    output gate; its RMSNorm over all heads stays in the graph."""

    def matches(self, module):
        return _has(module, "in_proj", "conv1d", "dt_bias", "A_log", "D", "norm", "out_proj", "ssm_state_size", "n_groups")

    def state_shapes(self, module, dtype, capacity):
        return [
            ((1, module.conv_dim, module.conv_kernel_size), dtype),
            ((1, module.num_heads, module.head_dim, module.ssm_state_size), torch.float32),
        ]

    def decode(self, module, hidden_states, conv_state, state, position):
        _silu_conv(module)
        projected = module.in_proj(hidden_states).reshape(1, -1)
        gate, qkv, dt = projected.split([module.intermediate_size, module.conv_dim, module.num_heads], dim=-1)
        groups = module.n_groups * module.ssm_state_size
        # MambaRMSNormGated gates before it normalizes, and the gate is per
        # head, so the body applies x * silu(z); only the norm over all heads
        # stays in the graph.
        output = linear_attention(
            qkv, state, dt, conv_state, module.conv1d.weight, module.conv1d.bias, module.A_log,
            module.dt_bias, None, module.D, gate, None,
            module.n_groups, module.ssm_state_size,
            module.intermediate_size + groups, module.intermediate_size, 0,
            1.0, False, "vk", True, "silu", 1e-6, True, True, position,
        )
        output = module.norm(output.reshape(1, 1, -1))
        return module.out_proj(output.to(hidden_states.dtype))


MIXERS: tuple[_Mixer, ...] = (
    _GatedDeltaNet(), _KimiDeltaAttention(), _ComplexKDA(), _Mamba2(), _ComplexKDAAttention(),
)


def mixer_of(module: torch.nn.Module) -> _Mixer | None:
    """The declaration of a token-mixing module, if any."""
    if not hasattr(module, "layer_idx"):
        return None
    return next((mixer for mixer in MIXERS if mixer.matches(module)), None)


def linear_mixer(module: torch.nn.Module) -> _Mixer | None:
    """The declaration of a linear-attention module, if any."""
    mixer = mixer_of(module)
    return mixer if mixer is not None and mixer.kind != _DECLARED_ATTENTION else None


class _LinearState:
    """``cache_params`` of an all-linear decode step: models that do not pass
    their keyword arguments down (Mamba2) still hand it to every mixer."""

    def __init__(self, layers: tuple[tuple[torch.Tensor, ...], ...]):
        self.layers = layers


def _declared_forward(module: torch.nn.Module, *args: Any, megabake_state=None, **kwargs: Any):
    """A token mixer's forward: with ``megabake_state`` one decode token
    through its declared op on the layer's state, else the module's own
    forward. Bound to the module (not a closure), so ``copy.deepcopy`` of a
    DecodeStep rebinds it to the copy."""
    if megabake_state is None and isinstance(kwargs.get("cache_params"), _LinearState):
        megabake_state = kwargs["cache_params"].layers
    if megabake_state is None:
        return module._megabake_original_forward(*args, **kwargs)
    hidden_states = args[0] if args else kwargs["hidden_states"]
    if hidden_states.shape[:2] != (1, 1):
        raise NotImplementedError("declared decode takes one batch-1 token")
    return mixer_of(module).decode(module, hidden_states, *megabake_state[module.layer_idx])


class DecodeStep(torch.nn.Module):
    """One-token decode for any Hugging Face attention model, with explicit state.

    ``forward(input_ids, position_ids, *state)`` takes, per attention layer, a
    key cache and a value cache of shape ``[batch, capacity, kv_heads,
    head_dim]`` and an int32 ``[1]`` valid length; per linear-attention layer
    (``MIXERS``), a convolution window ``[1, channels, kernel]`` (a ring
    indexed by the token position) and an FP32 recurrent state;
    ``state_shapes`` lists them. It replaces the
    hand-written per-architecture decode wrappers. It switches ``model`` to
    the ``megabake`` attention implementation and declares its
    linear-attention layers.
    """

    def __init__(self, model: torch.nn.Module, capacity: int):
        super().__init__()
        register_transformers_attention()
        model.set_attn_implementation(ATTENTION_IMPLEMENTATION)
        self.model = model
        self.capacity = capacity
        config = model.config.get_text_config(decoder=True)
        self.layer_count = config.num_hidden_layers
        self.layer_types = list(getattr(config, "layer_types", None) or ["full_attention"] * self.layer_count)
        unsupported = set(self.layer_types) - _ATTENTION_LAYERS - _LINEAR_LAYERS - {_DECLARED_ATTENTION}
        if unsupported:
            raise NotImplementedError(f"decode of {sorted(unsupported)} layers is not declared yet")
        self.kv_heads = getattr(config, "num_key_value_heads", 0)
        heads = getattr(config, "num_attention_heads", 0)
        self.head_dim = getattr(config, "head_dim", None) or (config.hidden_size // heads if heads else 0)
        self.mixers: dict[int, torch.nn.Module] = {}
        for module in model.modules():
            if mixer_of(module) is not None:
                if module.layer_idx in self.mixers:
                    raise NotImplementedError(f"layer {module.layer_idx} has two declared token mixers")
                module._megabake_original_forward = module.forward
                module.forward = types.MethodType(_declared_forward, module)
                self.mixers[module.layer_idx] = module
        self.takes_cache_params = "cache_params" in inspect.signature(model.forward).parameters
        if not getattr(config, "layer_types", None) and self.mixers:
            self.layer_types = ["mamba" if index in self.mixers else "full_attention" for index in range(self.layer_count)]
        for index, module in self.mixers.items():
            if mixer_of(module).kind == _DECLARED_ATTENTION:
                self.layer_types[index] = _DECLARED_ATTENTION
        missing = [
            index for index, kind in enumerate(self.layer_types)
            if kind in _LINEAR_LAYERS and index not in self.mixers
        ]
        if missing:
            raise NotImplementedError(f"linear-attention layers {missing} have no declared mixer")

    def state_shapes(self, batch: int = 1, dtype: torch.dtype = torch.bfloat16):
        cache = ((batch, self.capacity, self.kv_heads, self.head_dim), dtype)
        shapes = []
        for index, kind in enumerate(self.layer_types):
            if index in self.mixers:
                if batch != 1:
                    raise NotImplementedError("declared token mixers decode batch 1")
                module = self.mixers[index]
                shapes.append(mixer_of(module).state_shapes(module, dtype, self.capacity))
            else:
                shapes.append([cache, cache, ((1,), torch.int32)])
        return shapes

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, *state: torch.Tensor):
        layers, offset = [], 0
        for kind in self.layer_types:
            count = 2 if kind in _LINEAR_LAYERS else 3
            # Linear layers also take the token position: their convolution
            # windows are rings indexed by it.
            extra = (position_ids,) if kind in _LINEAR_LAYERS else ()
            layers.append((*state[offset : offset + count], *extra))
            offset += count
        if offset != len(state):
            raise ValueError(f"expected {offset} state tensors, got {len(state)}")
        if len(self.mixers) == self.layer_count:
            # Every layer is declared here: no transformers KV cache. A model
            # taking ``cache_params`` may not pass keywords down (Mamba2).
            carrier = {"cache_params": _LinearState(tuple(layers))} if self.takes_cache_params else {}
            return self.model(
                input_ids=input_ids, use_cache=False, megabake_state=tuple(layers), **carrier
            ).logits
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=position_ids[0],
            past_key_values=_decode_cache(self.layer_count, self.capacity),
            use_cache=True,
            megabake_state=tuple(layers),
        ).logits
