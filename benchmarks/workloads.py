from __future__ import annotations

import torch


def mlp(source, gate_weight, up_weight, down_weight):
    return (
        torch.nn.functional.silu(source @ gate_weight.T) * (source @ up_weight.T)
    ) @ down_weight.T


def transformer_block(mode, batch, sequence, device="cuda"):
    from tests.fixtures import StatefulTransformerBlock, TransformerBlock

    dimensions = {
        "hidden_size": 576,
        "heads": 9,
        "key_value_heads": 3,
        "intermediate_size": 1536,
        "device": device,
    }
    dtype = torch.bfloat16
    source = torch.randn(batch, sequence, 576, device=device, dtype=dtype) / 24
    residual = torch.randn_like(source)
    query_cosine = torch.randn(batch, sequence, 9, 64, device=device, dtype=dtype)
    query_sine = torch.randn_like(query_cosine)
    key_cosine = torch.randn(batch, sequence, 3, 64, device=device, dtype=dtype)
    key_sine = torch.randn_like(key_cosine)
    if mode == "prefill":
        mask = torch.randn(batch, 1, sequence, sequence, device=device, dtype=dtype)
        return TransformerBlock(**dimensions).eval(), (
            source,
            residual,
            query_cosine,
            query_sine,
            key_cosine,
            key_sine,
            mask,
        )

    capacity = 1024
    key_cache = torch.randn(batch, capacity, 3, 64, device=device, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    valid_length = torch.zeros(1, device=device, dtype=torch.int32)
    return StatefulTransformerBlock(**dimensions).eval(), (
        source,
        residual,
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        key_cache,
        value_cache,
        valid_length,
    )
