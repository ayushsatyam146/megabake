"""Route Hugging Face attention through ``megabake::attention``.

Every transformers attention module dispatches through ``AttentionInterface``
by ``config._attn_implementation``. Registering a ``megabake`` implementation
therefore declares attention for all such models at once, in place of
recovering it from decomposed ATen by pattern matching:

    register_transformers_attention()
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="megabake")

Prefill calls ``megabake::attention``. ``DecodeStep`` wraps the same model for
one-token decode over explicit KV state through ``megabake::kv_attention``.
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from ..ops.attention import attention

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


class DecodeStep(torch.nn.Module):
    """One-token decode for any Hugging Face attention model, with explicit state.

    ``forward(input_ids, position_ids, *state)`` takes, per layer, a key cache
    and a value cache of shape ``[batch, capacity, kv_heads, head_dim]`` and an
    int32 ``[1]`` valid length; ``state_shapes`` lists them. It replaces the
    hand-written per-architecture decode wrappers. It switches ``model`` to
    the ``megabake`` attention implementation.
    """

    def __init__(self, model: torch.nn.Module, capacity: int):
        super().__init__()
        register_transformers_attention()
        model.set_attn_implementation(ATTENTION_IMPLEMENTATION)
        self.model = model
        self.capacity = capacity
        config = model.config.get_text_config(decoder=True)
        unsupported = set(getattr(config, "layer_types", None) or ()) - _ATTENTION_LAYERS
        if unsupported:
            raise NotImplementedError(f"decode of {sorted(unsupported)} layers is not declared yet")
        self.layer_count = config.num_hidden_layers
        self.kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

    def state_shapes(self, batch: int = 1, dtype: torch.dtype = torch.bfloat16):
        cache = ((batch, self.capacity, self.kv_heads, self.head_dim), dtype)
        return [[cache, cache, ((1,), torch.int32)] for _ in range(self.layer_count)]

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, *state: torch.Tensor):
        layers = tuple(tuple(state[3 * index : 3 * index + 3]) for index in range(self.layer_count))
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=position_ids[0],
            past_key_values=_decode_cache(self.layer_count, self.capacity),
            use_cache=True,
            megabake_state=layers,
        ).logits
