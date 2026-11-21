from __future__ import annotations

import torch

from megabake.ops.attention import kv_attention


class CompleteMLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 576,
        intermediate_size: int = 1536,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.gate_weight = torch.nn.Parameter(
            torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
        )
        self.up_weight = torch.nn.Parameter(
            torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
        )
        self.down_weight = torch.nn.Parameter(
            torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
        )

    def forward(self, source: torch.Tensor):
        gate = torch.nn.functional.linear(source, self.gate_weight)
        up = torch.nn.functional.linear(source, self.up_weight)
        activated = torch.nn.functional.silu(gate)
        fused = activated * up
        output = torch.nn.functional.linear(fused, self.down_weight)
        return output, gate, activated, fused


def rms_norm(source, weight, residual):
    widened = source.float()
    variance = torch.pow(widened, 2).mean(dim=-1, keepdim=True)
    normalized = torch.rsqrt(variance + 1e-5) * widened
    return (normalized * weight).to(torch.bfloat16) + residual


def rotate_half(source, cosine, sine):
    half = source.shape[-1] // 2
    first = source[..., :half] * cosine[..., :half] - source[..., half:] * sine[..., :half]
    second = source[..., half:] * cosine[..., :half] + source[..., :half] * sine[..., :half]
    return torch.cat((first, second), dim=-1)


def grouped_attention(query, key, value, mask):
    query = query.permute(0, 2, 1, 3)
    grouped_key = key.permute(0, 2, 1, 3).repeat_interleave(query.shape[1] // key.shape[2], dim=1)
    grouped_value = value.permute(0, 2, 1, 3).repeat_interleave(
        query.shape[1] // value.shape[2], dim=1
    )
    scores = (query @ grouped_key.transpose(-2, -1)) * query.shape[-1] ** -0.5 + mask
    probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    return probabilities @ grouped_value


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 16,
        heads: int = 2,
        key_value_heads: int = 1,
        intermediate_size: int = 32,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if hidden_size % heads or heads % key_value_heads:
            raise ValueError("TransformerBlock requires divisible head counts")
        self.heads = heads
        self.key_value_heads = key_value_heads
        self.head_dim = hidden_size // heads
        self.input_norm_weight = torch.nn.Parameter(
            torch.randn(hidden_size, device=device, dtype=dtype)
        )
        self.query_weight = torch.nn.Parameter(
            torch.randn(hidden_size, hidden_size, device=device, dtype=dtype) / hidden_size**0.5
        )
        self.key_weight = torch.nn.Parameter(
            torch.randn(
                key_value_heads * self.head_dim,
                hidden_size,
                device=device,
                dtype=dtype,
            )
            / hidden_size**0.5
        )
        self.value_weight = torch.nn.Parameter(
            torch.randn(
                key_value_heads * self.head_dim,
                hidden_size,
                device=device,
                dtype=dtype,
            )
            / hidden_size**0.5
        )
        self.output_weight = torch.nn.Parameter(
            torch.randn(hidden_size, hidden_size, device=device, dtype=dtype) / hidden_size**0.5
        )
        self.post_norm_weight = torch.nn.Parameter(
            torch.randn(hidden_size, device=device, dtype=dtype)
        )
        self.gate_weight = torch.nn.Parameter(
            torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
            / hidden_size**0.5
        )
        self.up_weight = torch.nn.Parameter(
            torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
            / hidden_size**0.5
        )
        self.down_weight = torch.nn.Parameter(
            torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
            / intermediate_size**0.5
        )

    def _attention(
        self,
        source,
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        mask,
    ):
        batch, sequence, _ = source.shape
        query = torch.nn.functional.linear(source, self.query_weight).view(
            batch, sequence, self.heads, self.head_dim
        )
        key = torch.nn.functional.linear(source, self.key_weight).view(
            batch, sequence, self.key_value_heads, self.head_dim
        )
        query = rotate_half(query, query_cosine, query_sine)
        key = rotate_half(key, key_cosine, key_sine)
        value = torch.nn.functional.linear(source, self.value_weight).view(
            batch, sequence, self.key_value_heads, self.head_dim
        )
        attention = grouped_attention(query, key, value, mask)
        return attention.permute(0, 2, 1, 3).reshape(batch, sequence, -1)

    def _mlp(self, source):
        gate = torch.nn.functional.linear(source, self.gate_weight)
        up = torch.nn.functional.linear(source, self.up_weight)
        activated = torch.nn.functional.silu(gate)
        fused = activated * up
        return torch.nn.functional.linear(fused, self.down_weight), gate, activated, fused

    def forward(
        self,
        source,
        residual,
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        mask,
    ):
        normalized = rms_norm(source, self.input_norm_weight, residual)
        projected = torch.nn.functional.linear(
            self._attention(
                normalized,
                query_cosine,
                query_sine,
                key_cosine,
                key_sine,
                mask,
            ),
            self.output_weight,
        )
        attention_hidden = projected + source
        normalized_hidden = rms_norm(attention_hidden, self.post_norm_weight, attention_hidden)
        mlp_output, gate, activated, fused = self._mlp(normalized_hidden)
        return mlp_output + attention_hidden, gate, activated, fused


class StatefulTransformerBlock(TransformerBlock):
    def forward(
        self,
        source,
        residual,
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        key_cache,
        value_cache,
        valid_length,
    ):
        batch = source.shape[0]
        normalized = rms_norm(source, self.input_norm_weight, residual)
        query = torch.nn.functional.linear(normalized, self.query_weight).view(
            batch, 1, self.heads, self.head_dim
        )
        key = torch.nn.functional.linear(normalized, self.key_weight).view(
            batch, 1, self.key_value_heads, self.head_dim
        )
        query = rotate_half(query, query_cosine, query_sine)
        key = rotate_half(key, key_cosine, key_sine)
        value = torch.nn.functional.linear(normalized, self.value_weight).view(
            batch, 1, self.key_value_heads, self.head_dim
        )
        attention = kv_attention(query, key, value, key_cache, value_cache, valid_length)
        attention_flat = attention.reshape(batch, 1, -1)
        projected = torch.nn.functional.linear(attention_flat, self.output_weight)
        attention_hidden = projected + source
        normalized_hidden = rms_norm(attention_hidden, self.post_norm_weight, attention_hidden)
        mlp_output, gate, activated, fused = self._mlp(normalized_hidden)
        return mlp_output + attention_hidden, gate, activated, fused


SMOLLM_MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
SMOLLM_REVISION = "1d461723eec654e65efdc40cf49301c89c0c92f4"


class StatefulSmolLM(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head
        self.query_heads = model.config.num_attention_heads
        self.key_value_heads = model.config.num_key_value_heads
        self.head_dim = model.config.hidden_size // self.query_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        *state: torch.Tensor,
    ):
        layer_count = len(self.model.layers)
        key_caches = state[:layer_count]
        value_caches = state[layer_count : 2 * layer_count]
        valid_lengths = state[2 * layer_count :]
        hidden = self.model.embed_tokens(input_ids)
        position_ids = position_ids.to(torch.int64)
        for layer, key_cache, value_cache, valid_length in zip(
            self.model.layers,
            key_caches,
            value_caches,
            valid_lengths,
            strict=True,
        ):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attention = layer.self_attn
            batch, sequence, _ = normalized.shape
            query = torch.nn.functional.linear(normalized, attention.q_proj.weight).view(
                batch, sequence, self.query_heads, self.head_dim
            )
            key = torch.nn.functional.linear(normalized, attention.k_proj.weight).view(
                batch, sequence, self.key_value_heads, self.head_dim
            )
            value = torch.nn.functional.linear(normalized, attention.v_proj.weight).view(
                batch, sequence, self.key_value_heads, self.head_dim
            )
            cosine, sine = self.model.rotary_emb(normalized, position_ids)
            query_cosine = cosine[:, :, None, :].expand(
                batch, sequence, self.query_heads, self.head_dim
            )
            query_sine = sine[:, :, None, :].expand(
                batch, sequence, self.query_heads, self.head_dim
            )
            key_cosine = cosine[:, :, None, :].expand(
                batch, sequence, self.key_value_heads, self.head_dim
            )
            key_sine = sine[:, :, None, :].expand(
                batch, sequence, self.key_value_heads, self.head_dim
            )
            query = rotate_half(query, query_cosine, query_sine)
            key = rotate_half(key, key_cosine, key_sine)
            attended = kv_attention(query, key, value, key_cache, value_cache, valid_length)
            projected = torch.nn.functional.linear(
                attended.reshape(batch, sequence, -1), attention.o_proj.weight
            )
            hidden = residual + projected
            residual = hidden
            normalized = layer.post_attention_layernorm(hidden)
            gate = torch.nn.functional.linear(normalized, layer.mlp.gate_proj.weight)
            up = torch.nn.functional.linear(normalized, layer.mlp.up_proj.weight)
            fused = torch.nn.functional.silu(gate) * up
            hidden = residual + torch.nn.functional.linear(fused, layer.mlp.down_proj.weight)
        logits = self.lm_head(self.model.norm(hidden))
        return logits


def SmolLMArchitecture(
    layers: int = 30,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=576,
        intermediate_size=1536,
        num_attention_heads=9,
        num_key_value_heads=3,
        num_hidden_layers=layers,
        vocab_size=49152,
        max_position_embeddings=4,
    )
    with torch.device(device):
        model = LlamaForCausalLM(config)
    model.set_attn_implementation("eager")
    model.config.use_cache = False
    return model.to(dtype=dtype).eval()


def StatefulSmolLMArchitecture(
    layers: int = 30,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulSmolLM:
    return StatefulSmolLM(SmolLMArchitecture(layers=layers, device=device, dtype=dtype)).eval()
