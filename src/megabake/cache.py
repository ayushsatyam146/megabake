from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import os
import json
import platform
import threading
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import cutlass
import torch

from .graph import ComputeGraph
from .plan import BodySelectionPlan, KernelPlan
from .runtime import (
    PHASE_TIMER_SLOTS,
    WEIGHT_STREAM_STAGES,
    producer_ring_pages,
    producer_warp_enabled,
    stall_inject_phase,
    stall_timeout_ns,
)

if TYPE_CHECKING:
    from .frontend import CaptureContext


@dataclass(frozen=True)
class PlanCacheResult:
    compiled: Any
    hit: bool
    diagnostics: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class PlanCacheEntry:
    key: str
    compiled: Any
    diagnostics: Mapping[str, Any] = MappingProxyType({})


@dataclass
class PlanCache:
    entries: dict[str, PlanCacheEntry] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def get_or_compile(
        self, key: str, compiler: Callable[[], Any]
    ) -> PlanCacheResult:
        with self._lock:
            entry = self.entries.get(key)
            if entry is not None:
                self.hits += 1
                return PlanCacheResult(entry.compiled, True, entry.diagnostics)
            self.misses += 1
            compiled = compiler()
            diagnostics = _diagnostics(compiled)
            entry = PlanCacheEntry(key, compiled, diagnostics)
            self.entries[key] = entry
            return PlanCacheResult(compiled, False, diagnostics)


def make_plan_key(
    plan: KernelPlan | BodySelectionPlan,
    context: CaptureContext,
    *,
    scheduler: str = "phase",
    toolchain: Mapping[str, str] | None = None,
    body_revisions: Sequence[tuple[str, str, str]] | None = None,
) -> str:
    payload = {
        "plan": _plan_data(plan, body_revisions),
        "context": _context_data(context),
        "scheduler": scheduler,
        "toolchain": dict(toolchain or {}) or _toolchain_versions(),
        # Timed kernels write stamps past the barrier state of an untimed buffer.
        "phase_timer_slots": PHASE_TIMER_SLOTS,
        "weight_stream_stages": WEIGHT_STREAM_STAGES,
        # Stall checks (X0.3) are compiled in or out when the kernel is generated.
        "stall_timeout_ns": stall_timeout_ns(),
        "stall_inject": stall_inject_phase(),
        "producer_warp": producer_warp_enabled(),
        "ring_pages": producer_ring_pages(),
        "stall_inject_ring": os.environ.get("MEGABAKE_STALL_INJECT_RING"),
    }
    encoded = json.dumps(
        _canonical(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _context_data(context: CaptureContext) -> dict[str, Any]:
    requirements = {
        key: value
        for key, value in context.requirements.items()
        if key != "execution"
    }
    return {
        "mode": context.mode,
        "fallback": context.fallback,
        "input_order": context.input_order,
        "input_specs": context.input_specs,
        "input_kinds": context.input_kinds,
        "parameter_names": context.parameter_names,
        "output_structure": context.output_structure,
        "output_count": context.output_count,
        "num_model_outputs": context.num_model_outputs,
        "prepared_graph_hash": context.prepared_graph_hash,
        "requirements": requirements,
    }


def _plan_data(
    plan: KernelPlan | BodySelectionPlan,
    body_revisions: Sequence[tuple[str, str, str]] | None,
) -> dict[str, Any]:
    selection = plan.selection if isinstance(plan, KernelPlan) else plan
    graph = selection.graph
    if graph is None:
        raise ValueError("cannot key a plan without its source graph")
    return {
        "graph": graph,
        "target": selection.target,
        "body_revisions": tuple(body_revisions or ())
        or tuple(
            (
                body.provider,
                body.evidence.revision,
                body.semantic.numerical_policy,
            )
            for body in (
                selected.body for selected in selection.selections
            )
        ),
    }


def _toolchain_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cutlass": str(getattr(cutlass, "__version__", "unknown")),
    }


def _diagnostics(compiled: Any) -> Mapping[str, Any]:
    search = getattr(compiled, "search", None)
    report = getattr(search, "report", None)
    if callable(report):
        try:
            return report()
        except Exception:
            return {}
    return {}


def _canonical(value: Any) -> Any:
    if isinstance(value, ComputeGraph):
        excluded = {
            "output_reconstruction",
            "output_tree_spec",
            "runtime_context",
            "source_nodes",
        }
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
            if field.name not in excluded
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": type(value).__qualname__,
            **{
                field.name: _canonical(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        items = [(_canonical(key), _canonical(item)) for key, item in value.items()]
        return {"__map__": sorted(items, key=lambda item: repr(item[0]))}
    if isinstance(value, (set, frozenset)):
        return {"__set__": sorted(_canonical(item) for item in value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, torch.Tensor):
        return {
            "__tensor__": str(value.dtype),
            "shape": tuple(value.shape),
            "strides": tuple(value.stride()),
            "device": str(value.device),
        }
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else str(value)
    if isinstance(value, (type, Callable)):
        return {
            "__callable__": type(value).__qualname__,
            "module": getattr(value, "__module__", None),
            "name": getattr(value, "__qualname__", None),
        }
    return {"__object__": type(value).__qualname__, "repr": repr(value)}
