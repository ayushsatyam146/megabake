"""One-token decode wrappers for Hugging Face causal LMs with explicit state tensors.

`StatefulCausalLM` generalizes `tests.fixtures.StatefulSmolLM` beyond Llama:
attention biases (Qwen2), q/k norms and sandwich norms with per-layer-type
RoPE (Gemma 3), and Qwen3.5's output-gated attention plus Gated DeltaNet
linear-attention layers. Attention layers own a KV cache and valid length;
linear-attention layers own a convolution window and a recurrent state.
"""

from __future__ import annotations

import inspect
import os
import sys

import torch
from transformers.activations import ACT2FN

from megabake.ops.attention import kv_attention
from megabake.fx_handler.hf_attention import linear_mixer

# MEGABAKE_LINEAR_ATTENTION=0 runs linear-attention decode as its decomposed
# ops (an A/B switch); by default each layer's token mixer is one declared
# ``megabake::linear_attention``.
DECLARED_LINEAR_ATTENTION = os.environ.get("MEGABAKE_LINEAR_ATTENTION", "1") != "0"


def text_model(model: torch.nn.Module) -> torch.nn.Module:
    inner = model.model
    return getattr(inner, "language_model", inner)


def text_config(model: torch.nn.Module):
    config = model.config
    return getattr(config, "text_config", None) or config


def layer_types(model: torch.nn.Module) -> list[str]:
    config = text_config(model)
    return list(
        getattr(config, "layer_types", None) or ["full_attention"] * config.num_hidden_layers
    )


def _l2norm(source: torch.Tensor) -> torch.Tensor:
    return source * torch.rsqrt((source * source).sum(dim=-1, keepdim=True) + 1e-6)


def _rotary_positions(config, position_ids: torch.Tensor) -> torch.Tensor:
    # Qwen3.5 text rotary embeddings expect the three spatial/temporal axes.
    # For text-only decoding HF expands the same text position across them.
    if getattr(config, "model_type", None) == "qwen3_5_text":
        return position_ids.unsqueeze(0).expand(3, -1, -1)
    return position_ids


class StatefulCausalLM(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = text_model(model)
        self.lm_head = model.lm_head
        self.config = text_config(model)
        self.layer_types = layer_types(model)
        self.rotary_takes_type = (
            "layer_type" in inspect.signature(self.model.rotary_emb.forward).parameters
        )
        self.softcap = getattr(self.config, "final_logit_softcapping", None)

    def state_shapes(self, capacity: int) -> list[list[tuple[tuple[int, ...], torch.dtype]]]:
        """Per layer, the shape and dtype of each state tensor `forward` expects, in order."""
        shapes = []
        for layer, kind in zip(self.model.layers, self.layer_types, strict=True):
            if kind == "linear_attention":
                mixer = layer.linear_attn
                shapes.append(
                    [
                        # The declared op's window is a ring of kernel slots; the
                        # decomposed step shifts kernel - 1 of them.
                        ((1, mixer.conv_dim, mixer.conv_kernel_size - (0 if DECLARED_LINEAR_ATTENTION else 1)), torch.bfloat16),
                        ((1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim), torch.float32),
                    ]
                )
            else:
                attention = layer.self_attn
                window = getattr(attention, "sliding_window", None)
                if window is not None and capacity > window:
                    raise ValueError(
                        f"capacity {capacity} exceeds the {window}-token sliding window"
                    )
                cache = (1, capacity, self.config.num_key_value_heads, attention.head_dim)
                shapes.append(
                    [(cache, torch.bfloat16), (cache, torch.bfloat16), ((1,), torch.int32)]
                )
        return shapes

    def _attention(
        self, attention, normalized, position_embeddings, key_cache, value_cache, valid_length
    ):
        batch, sequence, _ = normalized.shape
        head_dim = attention.head_dim
        projected = attention.q_proj(normalized)
        gate = None
        if projected.shape[-1] == 2 * self.config.num_attention_heads * head_dim:
            # Qwen3.5 interleaves an output gate with each query head.
            query, gate = torch.chunk(projected.view(batch, sequence, -1, 2 * head_dim), 2, dim=-1)
            gate = gate.reshape(batch, sequence, -1)
        else:
            query = projected.view(batch, sequence, -1, head_dim)
        key = attention.k_proj(normalized).view(batch, sequence, -1, head_dim)
        value = attention.v_proj(normalized).view(batch, sequence, -1, head_dim)
        if hasattr(attention, "q_norm"):
            query = attention.q_norm(query)
            key = attention.k_norm(key)
        cosine, sine = position_embeddings
        rotate = sys.modules[type(attention).__module__].apply_rotary_pos_emb
        query, key = rotate(query, key, cosine, sine, unsqueeze_dim=2)
        # kv_attention scales scores by head_dim**-0.5; fold any other scale into the query.
        correction = attention.scaling * head_dim**0.5
        if abs(correction - 1.0) > 1e-6:
            query = query * correction
        attended = kv_attention(query, key, value, key_cache, value_cache, valid_length)
        attended = attended.reshape(batch, sequence, -1)
        if gate is not None:
            attended = attended * torch.sigmoid(gate)
        return attention.o_proj(attended)

    def _linear_attention(self, mixer, normalized, conv_state, recurrent_state, position_ids):
        if DECLARED_LINEAR_ATTENTION:
            return linear_mixer(mixer).decode(mixer, normalized, conv_state, recurrent_state, position_ids)
        return self._decomposed_linear_attention(mixer, normalized, conv_state, recurrent_state)

    def _decomposed_linear_attention(self, mixer, normalized, conv_state, recurrent_state):
        batch, sequence, _ = normalized.shape
        mixed = mixer.in_proj_qkv(normalized).transpose(1, 2)
        window = torch.cat([conv_state, mixed], dim=-1)
        conv_state.copy_(window[:, :, 1:])
        mixed = torch.nn.functional.conv1d(window, mixer.conv1d.weight, groups=mixer.conv_dim)
        mixed = ACT2FN[mixer.activation](mixed).transpose(1, 2)
        query, key, value = torch.split(
            mixed, [mixer.key_dim, mixer.key_dim, mixer.value_dim], dim=-1
        )
        query = query.reshape(batch, mixer.num_k_heads, mixer.head_k_dim).float()
        key = key.reshape(batch, mixer.num_k_heads, mixer.head_k_dim).float()
        value = value.reshape(batch, mixer.num_v_heads, mixer.head_v_dim).float()
        if mixer.num_v_heads != mixer.num_k_heads:
            repeat = mixer.num_v_heads // mixer.num_k_heads
            query = query.repeat_interleave(repeat, dim=1)
            key = key.repeat_interleave(repeat, dim=1)
        query = _l2norm(query) * mixer.head_k_dim**-0.5
        key = _l2norm(key)
        beta = mixer.in_proj_b(normalized).reshape(batch, -1).float().sigmoid()
        decay = -mixer.A_log.float().exp() * torch.nn.functional.softplus(
            mixer.in_proj_a(normalized).reshape(batch, -1).float() + mixer.dt_bias
        )
        # One step of the gated delta rule.
        state = recurrent_state * decay.exp()[..., None, None]
        remembered = (state * key[..., None]).sum(dim=-2)
        delta = (value - remembered) * beta[..., None]
        state = state + key[..., None] * delta[..., None, :]
        recurrent_state.copy_(state)
        output = (state * query[..., None]).sum(dim=-2).to(normalized.dtype)
        gate = mixer.in_proj_z(normalized).reshape(-1, mixer.head_v_dim)
        output = mixer.norm(output.reshape(-1, mixer.head_v_dim), gate)
        return mixer.out_proj(output.reshape(batch, sequence, -1))

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, *state: torch.Tensor):
        hidden = self.model.embed_tokens(input_ids)
        position_ids = position_ids.to(torch.int64)
        rotary_positions = _rotary_positions(self.config, position_ids)
        rotary = {}
        rotary_positions = position_ids
        if hasattr(self.model.rotary_emb, "mrope_section") and position_ids.ndim == 2:
            # Multimodal RoPE (Qwen3.5) takes (3, batch, positions); text-only
            # input repeats the same positions, as the Hugging Face model does.
            rotary_positions = position_ids[None].expand(3, *position_ids.shape)
        for kind in set(self.layer_types) - {"linear_attention"}:
            arguments = (kind,) if self.rotary_takes_type else ()
            rotary[kind] = self.model.rotary_emb(hidden, rotary_positions, *arguments)
        offset = 0
        for layer, kind in zip(self.model.layers, self.layer_types, strict=True):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            if kind == "linear_attention":
                mixed = self._linear_attention(layer.linear_attn, normalized, *state[offset : offset + 2], position_ids)
                offset += 2
            else:
                mixed = self._attention(
                    layer.self_attn, normalized, rotary[kind], *state[offset : offset + 3]
                )
                offset += 3
            if hasattr(layer, "pre_feedforward_layernorm"):
                # Gemma sandwich norms around both sublayers.
                hidden = residual + layer.post_attention_layernorm(mixed)
                mlp = layer.mlp(layer.pre_feedforward_layernorm(hidden))
                hidden = hidden + layer.post_feedforward_layernorm(mlp)
            else:
                hidden = residual + mixed
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        if offset != len(state):
            raise ValueError(f"expected {offset} state tensors, got {len(state)}")
        logits = self.lm_head(self.model.norm(hidden))
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits
