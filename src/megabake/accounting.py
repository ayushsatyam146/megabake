"""Bytes and FLOPs each scheduled phase must move (X0.1).

This is the numerator of a speed-of-light time: what the phase cannot avoid
moving through HBM, with every byte counted once. It is derived from the
plan only, so the same numbers can rank plans (X2.3) before anything runs.

Rules:

- Every distinct input value is read once, every output written once.
- A KV cache (``TensorProgram.state_accesses``) is read only up to the valid
  length and written for ``query_length`` rows, not at its capacity.
- A gather (embedding lookup) reads as many table bytes as it writes.
- A contraction over two 2-D operands costs ``2·M·N·K`` FLOPs; everything
  else is counted as bandwidth only (decode-sized elementwise FLOPs are
  negligible next to their bytes).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .plan import KernelPlan
from .schedule import ScheduledPhase


@dataclass(frozen=True)
class PhaseTraffic:
    bytes_read: int
    bytes_written: int
    flops: int
    weight_bytes: int  # the part of bytes_read that is parameters
    kv_bytes: int  # the part of bytes_read/written that is KV cache

    @property
    def bytes_moved(self) -> int:
        return self.bytes_read + self.bytes_written


def _itemsize(dtype: str | None) -> int:
    if dtype is None:
        return 0
    name = dtype.removeprefix("torch.")
    return torch.empty((), dtype=getattr(torch, name)).element_size()


def _value_bytes(plan: KernelPlan, value_id: str) -> int:
    value = plan.graph.values[value_id]
    if value.shape is None:
        return 0
    return math.prod(value.shape) * _itemsize(value.dtype)


def _is_parameter(plan: KernelPlan, value_id: str, depth: int = 0) -> bool:
    """A parameter, or a view (permute/view/expand) of one."""
    graph = plan.graph
    value = graph.values[value_id]
    if value.kind == "parameter":
        return True
    if depth > 8 or value.producer is None or value.producer not in graph.ops:
        return False
    op = graph.ops[value.producer]
    view_like = op.attributes.get("target", "") if isinstance(op.attributes, dict) else ""
    operands = [operand for operand in op.operands if operand in graph.values]
    if len(operands) != 1:
        return False
    if view_like and not any(name in str(view_like) for name in ("permute", "view", "expand", "t.", "transpose", "reshape", "alias")):
        return False
    return _is_parameter(plan, operands[0], depth + 1)


def phase_traffic(
    plan: KernelPlan, phase: ScheduledPhase, kv_valid_length: int | None = None
) -> PhaseTraffic:
    """Bytes and FLOPs of one phase. ``kv_valid_length`` is the number of
    cached keys the call attends over (None: the full capacity)."""
    selection = next(s for s in plan.selection.selections if s.region_id == phase.region_id)
    program = selection.program
    inputs = list(dict.fromkeys(selection.inputs))
    outputs = list(dict.fromkeys(selection.outputs))
    read = written = weights = kv = flops = 0

    kv_reads: dict[str, int] = {}
    for access in getattr(program, "state_accesses", ()) or ():
        valid = access.capacity if kv_valid_length is None else min(kv_valid_length, access.capacity)
        for cache_id, layout in (
            (access.key_cache_value_id, access.key_layout),
            (access.value_cache_value_id, access.value_layout),
        ):
            total = _value_bytes(plan, cache_id)
            per_row = total // access.capacity
            kv_reads[cache_id] = per_row * valid
            kv_written = per_row * access.query_length
            written += kv_written
            kv += kv_written

    gather = phase.body.name.startswith("generic_gather") or getattr(program, "family", "") == "gather"
    output_bytes = sum(_value_bytes(plan, value_id) for value_id in outputs)
    for value_id in inputs:
        if value_id in kv_reads:
            read += kv_reads[value_id]
            kv += kv_reads[value_id]
            continue
        size = _value_bytes(plan, value_id)
        is_weight = _is_parameter(plan, value_id)
        if gather and is_weight:
            size = min(size, output_bytes)
        read += size
        if is_weight:
            weights += size
    written += output_bytes

    shapes = [plan.graph.values[value_id].shape for value_id in inputs]
    two_d = [shape for shape in shapes if shape is not None and len(shape) == 2]
    if len(two_d) >= 2 and two_d[0][1] == two_d[1][0]:
        (m, k), (_, n) = two_d[0], two_d[1]
        # Paired (gate·up) bodies contract the same activation with two weights.
        flops = 2 * m * k * n * sum(1 for shape in two_d[1:] if shape[0] == k)
    return PhaseTraffic(read, written, flops, weights, kv)


def plan_traffic(plan: KernelPlan, kv_valid_length: int | None = None) -> list[PhaseTraffic]:
    return [phase_traffic(plan, phase, kv_valid_length) for phase in plan.phases]
