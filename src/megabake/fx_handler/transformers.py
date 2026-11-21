from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from ..ops.attention import kv_attention
from .export import ExportedCapture, capture_exported_program


@dataclass
class TransformerCapture(ExportedCapture):
    model_name: str | None = None
    revision: str | None = None


@dataclass
class KVCache:
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    valid_length: torch.Tensor

    @property
    def capacity(self) -> int:
        return self.key_cache.shape[1]

    def validate(self) -> None:
        if (
            self.key_cache.ndim != 4
            or self.value_cache.shape != self.key_cache.shape
            or not self.key_cache.is_contiguous()
            or not self.value_cache.is_contiguous()
            or self.key_cache.dtype != self.value_cache.dtype
            or self.valid_length.shape != (1,)
            or self.valid_length.dtype != torch.int32
            or self.key_cache.device != self.value_cache.device
            or self.key_cache.device != self.valid_length.device
        ):
            raise ValueError("KV cache must be two contiguous same-device tensors")
        length = int(self.valid_length.item())
        if length < 0 or length > self.capacity:
            raise ValueError(f"KV length {length} is outside capacity {self.capacity}")


class _BoundKVCache(torch.nn.Module):
    def __init__(self, model: Callable[..., Any], cache: KVCache):
        super().__init__()
        self.model = model
        self.register_buffer("key_cache", cache.key_cache)
        self.register_buffer("value_cache", cache.value_cache)
        self.register_buffer("valid_length", cache.valid_length)

    def forward(self, source: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Any:
        return self.model(source, key, value, self.key_cache, self.value_cache, self.valid_length)


def bind_kv_cache(model: Callable[..., Any], cache: KVCache) -> torch.nn.Module:
    cache.validate()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("bind_kv_cache expects a torch.nn.Module")
    return _BoundKVCache(model, cache).eval()


class StatefulAttention(torch.nn.Module):
    def forward(
        self,
        source: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        valid_length: torch.Tensor,
    ) -> torch.Tensor:
        return kv_attention(source, key, value, key_cache, value_cache, valid_length)


def capture_transformer(
    model: str | torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    revision: str | None = None,
    device: torch.device | str | None = None,
    backend: Callable[..., Any] | None = None,
    **backend_options: Any,
) -> TransformerCapture:
    model_name = model if isinstance(model, str) else None
    if isinstance(model, str):
        loaded = AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            attn_implementation="eager",
        )
    else:
        loaded = model
    target_device = (
        torch.device(device)
        if device is not None
        else next(loaded.parameters(), torch.empty(0)).device
    )
    loaded = loaded.to(target_device).eval()
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    input_ids = input_ids.to(target_device)
    attention_mask = attention_mask.to(target_device)
    with torch.no_grad():
        exported = torch.export.export(
            loaded,
            (input_ids,),
            kwargs={
                "attention_mask": attention_mask,
                "use_cache": False,
                "return_dict": False,
            },
        )
        capture = capture_exported_program(
            exported,
            args=(input_ids,),
            kwargs={"attention_mask": attention_mask},
            model=loaded,
            backend=backend,
            model_name=model_name or type(loaded).__name__,
            input_defaults={
                "attention_mask": torch.ones_like(input_ids),
                "use_cache": False,
                "return_dict": False,
            },
            **backend_options,
        )
    return TransformerCapture(
        exported_program=capture.exported_program,
        model=capture.model,
        graph=capture.graph,
        context=capture.context,
        forward=capture.forward,
        input_defaults=capture.input_defaults,
        model_name=model_name,
        revision=revision,
    )
