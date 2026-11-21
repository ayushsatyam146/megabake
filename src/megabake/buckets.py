"""Variable-length calls through a few fixed-size compiled kernels.

MegaBake kernels have static shapes. An inference engine sends prompts and
batches of any length, so ``bucketed`` serves lengths up to the largest bucket
with one compile per bucket, as vLLM does with CUDA-graph capture sizes: each
call pads its sequence arguments at the end to the smallest bucket that holds
them, runs that bucket's kernel, and cuts the outputs back to the real length.

Padding is exact only when outputs at real positions cannot depend on padded
positions, which holds for causal decoders (per-token work plus causal
attention, with padding after the real tokens). That is checked, not
assumed, on the first call of each bucket: eager calls with two different
paddings must agree bit for bit at the real positions (no dependence on the
padding values), and eager padded and unpadded calls must agree there up to
rounding (no dependence on the padded length). The kernel itself was gated
against eager on the padded inputs it was compiled for.

    from megabake import bucketed, make_backend

    prefill = bucketed(
        model,
        backend=make_backend(mode="prefill"),
        sequence_args={"input_ids": 1, "attention_mask": 1},
        pad_values={"input_ids": 0, "attention_mask": 0},
    )
    logits = prefill(input_ids=ids, attention_mask=mask).logits  # any length <= 512
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch.utils import _pytree as pytree

DEFAULT_BUCKETS = (16, 32, 64, 128, 256, 512)
# Eager padded vs unpadded at the real positions: kernels picked for the two
# shapes round differently, while reading the padded length changes outputs
# outright.
_RELATIVE_L2_LIMIT = 0.05


class PaddingDependence(ValueError):
    """Outputs at real positions change with the padding, so buckets are inexact."""


class bucketed:
    """Call ``module`` at any sequence length up to ``max(buckets)``.

    ``sequence_args`` maps argument names to their sequence dimension; every
    tensor output whose size along ``output_dim`` equals the bucket is cut back
    to the real length there. ``pad_values`` gives each argument's padding
    value (default 0). The compiled kernels are cached per bucket by Dynamo.
    """

    def __init__(
        self,
        module: torch.nn.Module | Callable[..., Any],
        *,
        backend: Any,
        sequence_args: Mapping[str, int],
        pad_values: Mapping[str, int | float] | None = None,
        buckets: Sequence[int] = DEFAULT_BUCKETS,
        output_dim: int = 1,
        verify: bool = True,
    ) -> None:
        if not sequence_args:
            raise ValueError("bucketed needs at least one sequence argument")
        if not buckets or any(bucket < 1 for bucket in buckets):
            raise ValueError("buckets must be positive lengths")
        self.module = module
        self.sequence_args = dict(sequence_args)
        self.pad_values = dict(pad_values or {})
        self.buckets = tuple(sorted(set(buckets)))
        self.output_dim = output_dim
        self.verify = verify
        self._signature = inspect.signature(
            module.forward if isinstance(module, torch.nn.Module) else module
        )
        # One recompile per bucket shape, plus the first compile.
        limit = max(torch._dynamo.config.recompile_limit, len(self.buckets) + 1)
        torch._dynamo.config.recompile_limit = limit
        self.compiled = torch.compile(module, backend=backend, fullgraph=True, dynamic=False)
        self.verified: set[int] = set()

    def bucket_for(self, length: int) -> int:
        for bucket in self.buckets:
            if length <= bucket:
                return bucket
        raise ValueError(f"length {length} exceeds the largest bucket {self.buckets[-1]}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        bound = self._signature.bind(*args, **kwargs)
        arguments = dict(bound.arguments)
        lengths = {
            name: arguments[name].shape[dim]
            for name, dim in self.sequence_args.items()
            if isinstance(arguments.get(name), torch.Tensor)
        }
        if not lengths:
            raise ValueError("no sequence argument was passed")
        if len(set(lengths.values())) != 1:
            raise ValueError(f"sequence arguments disagree on the length: {lengths}")
        length = next(iter(lengths.values()))
        bucket = self.bucket_for(length)
        padded = self._pad(arguments, length, bucket, fill=None)
        with torch.no_grad():
            output = self.compiled(**padded)
        result = self._cut(output, length, bucket)
        if self.verify and bucket not in self.verified:
            self._verify(arguments, length, bucket)
            self.verified.add(bucket)
        return result

    def _pad(
        self, arguments: Mapping[str, Any], length: int, bucket: int, fill: int | None
    ) -> dict[str, Any]:
        padded = dict(arguments)
        for name, dim in self.sequence_args.items():
            value = arguments.get(name)
            if not isinstance(value, torch.Tensor) or bucket == length:
                continue
            shape = list(value.shape)
            shape[dim] = bucket - length
            if fill is None:
                pad = torch.full(
                    shape, self.pad_values.get(name, 0), dtype=value.dtype, device=value.device
                )
            else:
                # A second, different padding for the dependence check.
                generator = torch.Generator(device="cpu").manual_seed(fill)
                if value.dtype.is_floating_point:
                    pad = torch.randn(shape, generator=generator).to(value)
                else:
                    pad = torch.randint(0, 2, shape, generator=generator).to(value)
            padded[name] = torch.cat((value, pad), dim=dim)
        return padded

    def _cut(self, output: Any, length: int, bucket: int) -> Any:
        dim = self.output_dim

        def cut(value: Any) -> Any:
            if (
                isinstance(value, torch.Tensor)
                and value.dim() > dim
                and value.shape[dim] == bucket
                and bucket != length
            ):
                return value.narrow(dim, 0, length)
            return value

        return pytree.tree_map(cut, output)

    def _verify(self, arguments: Mapping[str, Any], length: int, bucket: int) -> None:
        with torch.no_grad():
            first = self._cut(
                self.module(**self._pad(arguments, length, bucket, None)), length, bucket
            )
            second = self._cut(
                self.module(**self._pad(arguments, length, bucket, 1)), length, bucket
            )
            reference = self.module(**arguments)
        if not all(
            torch.equal(left, right)
            for left, right in zip(_tensors(first), _tensors(second), strict=True)
        ):
            raise PaddingDependence(
                "outputs at real positions change with the padding; "
                "this callable cannot run through padded buckets"
            )
        for expected, padded in zip(_tensors(reference), _tensors(first), strict=True):
            if expected.shape != padded.shape:
                raise PaddingDependence(
                    f"padded output has shape {tuple(padded.shape)}, unpadded {tuple(expected.shape)}"
                )
            if expected.is_floating_point():
                scale = expected.float().norm()
                error = (padded.float() - expected.float()).norm() / scale if scale > 0 else 0.0
                if error > _RELATIVE_L2_LIMIT:
                    raise PaddingDependence(
                        f"bucket {bucket}: padded eager is {float(error):.3g} from unpadded eager"
                    )
            elif not torch.equal(expected, padded):
                raise PaddingDependence(
                    f"bucket {bucket}: padded eager differs from unpadded eager"
                )


def _tensors(value: Any) -> list[torch.Tensor]:
    return [leaf for leaf in pytree.tree_leaves(value) if isinstance(leaf, torch.Tensor)]
