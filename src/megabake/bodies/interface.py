"""The contract between the planner, codegen and device bodies.

A body provider proposes ``BodyConfig``s for a region and emits the device
code of the one the planner selects. Providers are grouped by GPU
generation (``Generation``); ``catalog`` offers a target the generic
providers plus its generation's.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

from ..ir import AxisAffine, DataIndex, TensorProgram

if TYPE_CHECKING:
    from ..graph import ComputeGraph, Region
    from ..ops import Semantics

AxisExpression = str | int | AxisAffine | DataIndex


@dataclass(frozen=True)
class TargetFacts:
    architecture: str
    threads_per_warp: int
    max_threads_per_cta: int
    max_shared_memory_per_cta: int
    max_registers_per_thread: int
    multiprocessor_count: int
    supports_tma: bool
    supports_wgmma: bool


@dataclass(frozen=True)
class SelectionPolicy:
    mode: str = "prefill"
    allow_generic: bool = True
    allow_tuned: bool = True
    # Large graphs use a table-driven launch: tensors come from a device
    # address table and TMA GEMM descriptors are written per phase in the
    # kernel prologue, so bodies must share one GEMM shared-memory layout.
    table_launch: bool = False
    # Launch-wide choices of the target generation's providers, made once
    # per graph (``Generation.launch_options``); each generation defines its
    # own type, and None means its defaults.
    target_options: Any = None
    numerical_policy: str = "source_semantics"
    require_measured: bool = False
    # The launch's ``device.DeviceTree``: the rates and capacities providers
    # and the planner decide with. None plans as before the tree existed.
    device: Any = None


@dataclass(frozen=True)
class WorkItem:
    output_tile: tuple[int, ...]
    reduction_range: tuple[int, int] | None = None


@dataclass(frozen=True)
class DeviceEntry:
    name: str = "run"
    arguments: tuple[str, ...] = (
        "work_iterator",
        "tensor_bindings",
        "scratch",
        "epilogue",
    )
    whole_kernel_launch: bool = False


@dataclass(frozen=True)
class SemanticContract:
    operation_family: str
    source_op_ids: tuple[str, ...]
    numerical_policy: str
    expression_hooks: tuple[str, ...]
    access_effects: Mapping[str, Any]
    # The region's descriptor (an ``ops`` family) when the body implements one.
    semantics: Semantics | None = None


@dataclass(frozen=True)
class TensorLayout:
    dtype: str
    axes: tuple[AxisExpression, ...]
    strides: tuple[int, ...]
    alignment_bytes: int
    shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class LayoutContract:
    inputs: Mapping[str, TensorLayout]
    outputs: Mapping[str, TensorLayout]


@dataclass(frozen=True)
class TileContract:
    cta_shape: tuple[int, ...]
    cluster_shape: tuple[int, ...]
    work_partition: str
    mma_family: str | None
    reduction_tile: tuple[int, ...]


@dataclass(frozen=True)
class ParticipantContract:
    participating_threads: int
    waiting_threads: int
    warp_roles: tuple[str, ...]
    registers_per_thread: int


@dataclass(frozen=True)
class PipelineContract:
    stage_count: int
    barriers: tuple[str, ...]
    async_operations: tuple[str, ...]
    drain_rule: str
    descriptors: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ScratchContract:
    owner: str
    size_bytes: int
    alignment_bytes: int


@dataclass(frozen=True)
class CompletionContract:
    outputs_ready: str
    scratch_reusable: str
    register_fragments: str


@dataclass(frozen=True)
class ResourceEvidence:
    measured: bool
    source: str
    revision: str
    resources: Mapping[str, Any]
    measurements: Mapping[str, Any]


@dataclass(frozen=True)
class BodyConfig:
    provider: str
    name: str
    semantic: SemanticContract
    layouts: LayoutContract
    tiles: TileContract
    participants: ParticipantContract
    pipeline: PipelineContract
    scratch: ScratchContract
    completion: CompletionContract
    evidence: ResourceEvidence
    entrypoint: DeviceEntry = DeviceEntry()


@dataclass(frozen=True)
class ProviderRejection:
    provider: str
    reason: str


@dataclass(frozen=True)
class CandidateSet:
    configurations: tuple[BodyConfig, ...]
    rejections: tuple[ProviderRejection, ...]


class GeneratedBody(Protocol):
    """Device source of one body variant and how its work is iterated."""

    source: str
    threads_per_cta: int
    tile_count: int
    # The difference between consecutive work-iterator entries: tile base
    # indices step by the elements or rows one tile covers, tile indices by 1.
    work_step: int
    work_iterator_encoding: str


class BodyProvider(Protocol):
    provider_name: str

    def get_candidates(
        self, region: Region, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        """Legal configs for the region, or rejections saying why none fit."""

    def emit(
        self,
        program: TensorProgram,
        config: BodyConfig,
        *,
        defer_length_publish: bool = False,
        workers: int | None = None,
    ) -> GeneratedBody:
        """The device body of a selected config. ``workers`` is given for
        table launches, whose bodies may split work across CTAs; with
        ``defer_length_publish`` a stateful body leaves its KV-length
        increment to the kernel (see ``emit_attention_body``)."""

    def widen(self, program: TensorProgram, config: BodyConfig, threads: int) -> BodyConfig:
        """The config run by CTAs of ``threads`` threads, so every phase of a
        launch shares one block size."""


@dataclass(frozen=True)
class Generation:
    """The providers a GPU generation adds to the generic ones."""

    architectures: frozenset[str]
    providers: tuple[BodyProvider, ...]
    # The ``SelectionPolicy.target_options`` of a graph:
    # (graph, table_launch, device tree or None).
    launch_options: Callable[[ComputeGraph, bool, Any], Any] | None = None


def widened_participants(config: BodyConfig, threads: int) -> ParticipantContract:
    """``config``'s participants repeated to fill ``threads`` threads."""
    current = config.participants.participating_threads
    if threads % current:
        raise InvalidBodyConfig(f"body uses {current} threads; cannot share {threads}")
    return replace(
        config.participants,
        participating_threads=threads,
        warp_roles=config.participants.warp_roles * (threads // current),
    )


def access_effects(program: TensorProgram) -> dict[str, Any]:
    """The values a body reads and writes for ``program``."""
    return {
        "reads": tuple(load.value_id for load in program.loads),
        "writes": (program.store.value_id,),
        "owns_storage": program.store.owns_storage,
    }


class InvalidBodyConfig(ValueError):
    pass


def validate_body_config(
    config: BodyConfig,
    target: TargetFacts | None = None,
    policy: SelectionPolicy | None = None,
) -> None:
    if policy is not None and policy.mode not in {"prefill", "decode"}:
        raise InvalidBodyConfig(
            f"selection mode {policy.mode!r} must be 'prefill' or 'decode'"
        )
    if config.entrypoint.name != "run" or config.entrypoint.arguments != (
        "work_iterator",
        "tensor_bindings",
        "scratch",
        "epilogue",
    ):
        raise InvalidBodyConfig(
            "device entry must be run(work_iterator, tensor_bindings, scratch, epilogue)"
        )
    if config.entrypoint.whole_kernel_launch:
        raise InvalidBodyConfig("whole-kernel launch wrappers are not device bodies")

    for label, layouts in (
        ("input", config.layouts.inputs),
        ("output", config.layouts.outputs),
    ):
        if not layouts and label == "output":
            raise InvalidBodyConfig(f"body has no {label} layouts")
        for value_id, layout in layouts.items():
            if not value_id or not layout.dtype:
                raise InvalidBodyConfig(f"body has an incomplete {label} layout")
            if len(layout.axes) != len(layout.strides):
                raise InvalidBodyConfig(
                    f"{label} layout {value_id} has mismatched axes and strides"
                )
            if layout.shape is not None and len(layout.shape) != len(layout.axes):
                raise InvalidBodyConfig(
                    f"{label} layout {value_id} has mismatched shape and axes"
                )
            if layout.alignment_bytes <= 0 or layout.alignment_bytes & (
                layout.alignment_bytes - 1
            ):
                raise InvalidBodyConfig(f"{label} layout {value_id} has invalid alignment")

    for dimensions in (config.tiles.cta_shape, config.tiles.cluster_shape):
        if not dimensions or any(dimension <= 0 for dimension in dimensions):
            raise InvalidBodyConfig("body tile dimensions must be positive")

    participants = config.participants
    if participants.participating_threads <= 0:
        raise InvalidBodyConfig("body must have participating threads")
    if (
        participants.waiting_threads < 0
        or participants.waiting_threads > participants.participating_threads
    ):
        raise InvalidBodyConfig("body waiting threads are outside the participant set")
    if participants.registers_per_thread <= 0:
        raise InvalidBodyConfig("body register requirement must be positive")
    if not participants.warp_roles:
        raise InvalidBodyConfig("body must declare warp roles")
    if config.pipeline.stage_count <= 0:
        raise InvalidBodyConfig("body must declare at least one pipeline stage")
    if config.scratch.size_bytes < 0 or config.scratch.alignment_bytes <= 0:
        raise InvalidBodyConfig("body scratch requirements are invalid")
    if not config.completion.outputs_ready or not config.completion.scratch_reusable:
        raise InvalidBodyConfig("body must declare output and scratch completion rules")

    if target is not None:
        if participants.participating_threads > target.max_threads_per_cta:
            raise InvalidBodyConfig("body requires more threads than the target allows")
        if participants.registers_per_thread > target.max_registers_per_thread:
            raise InvalidBodyConfig("body requires more registers than the target allows")
        if config.scratch.size_bytes > target.max_shared_memory_per_cta:
            raise InvalidBodyConfig("body requires more shared memory than the target allows")
        if len(participants.warp_roles) != participants.participating_threads // target.threads_per_warp:
            raise InvalidBodyConfig("body warp roles do not match participating threads")

    if policy is not None and config.semantic.numerical_policy != policy.numerical_policy:
        raise InvalidBodyConfig("body numerical policy does not match the selection policy")
    if policy is not None and policy.require_measured and not config.evidence.measured:
        raise InvalidBodyConfig("body lacks required measured evidence")
