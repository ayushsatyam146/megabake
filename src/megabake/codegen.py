from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from typing import Any
from collections.abc import Mapping, Sequence

from .bodies.cuda.sm90.gemm import (
    TABLE_DESCRIPTORS_PER_GEMM,
    HopperWgmmaGemmPersistentKernel,
    epilogue_loads,
    epilogue_source,
    has_bias,
    has_fused_epilogue,
    is_paired,
    is_swap_ab,
    kernel_tile,
    split_k,
    table_gemm_shared_bytes,
)
from .bodies.catalog import provider
from .bodies.cuda.sm90.cutlass_gemm import STREAM_TAG_SLOT, gemm_stamp_address
from .bodies.cuda.generic import emit_map_body
from .bodies.interface import (
    BodyConfig,
    LayoutContract,
    TensorLayout,
)
from .ir import (
    AxisPredicate,
    IndexedLoad,
    IterationDomain,
    Load,
    TensorProgram,
    TensorStore,
)
from .graph import DTYPE_BYTES
from .memory import _round_up, verify_plan
from .plan import ConversionBody, CuTeLayout, KernelPlan, SelectedBody
from .runtime import (
    CTA_TIMER,
    CTA_TIMER_PHASES,
    CTA_TIMER_STRIDE,
    CTA_TIMER_WORKERS,
    DYNAMIC_ADDRESS_RING,
    GRID_TIMER_SLOTS,
    PHASE_TIMER_SLOTS,
    PRODUCER_START_BARRIER,
    STALL_COUNTER_SLOTS,
    WEIGHT_STREAM_STAGES,
    call_counter_offset,
    producer_ring_pages,
    producer_warp_enabled,
    stall_inject_phase,
    stall_timeout_ns,
)
from .schedule import EventSchedule


class UnsupportedCodegen(ValueError):
    pass


@dataclass(frozen=True)
class WorkIteratorBinding:
    name: str
    phase_index: int
    region_id: str
    dtype: str
    length: int
    start: int
    step: int
    encoding: str


@dataclass(frozen=True)
class TensorArgument:
    name: str
    value_id: str
    storage_id: str
    kind: str
    layout: Any
    binding_name: str | None
    region_ids: tuple[str, ...]


@dataclass(frozen=True)
class PhaseEmission:
    index: int
    region_id: str
    body_symbol: str
    body_variant_id: int
    provider: str
    body_name: str
    work_item_count: int
    device_tile_count: int
    conversion_symbols: tuple[str, ...]
    barrier: str | None


@dataclass(frozen=True)
class LaunchMetadata:
    kernel_name: str
    launch_function_name: str
    grid_size: int
    threads_per_block: int
    cooperative: bool
    shared_memory_bytes: int
    architecture: str
    arguments: tuple[str, ...]
    work_iterators: tuple[WorkIteratorBinding, ...]
    tensors: tuple[TensorArgument, ...]
    phases: tuple[PhaseEmission, ...]
    launch_policy: str = "megabake_cooperative"
    body_variant_count: int = 0
    phase_runner_count: int = 0
    source_bytes: int = 0
    # Table-driven launches read tensor addresses and phase records from
    # device memory instead of passing one kernel parameter per tensor.
    phase_table: tuple[tuple[int, ...], ...] = ()
    dynamic_slots: tuple[int, ...] = ()
    # Slots whose addresses the host appends to the address table, one per
    # (phase, tensor column) of table launches (see _slot).
    resolved_slots: tuple[int, ...] = ()
    barrier_epochs: int = 0
    # Table-launched TMA GEMMs: one row per GEMM phase (variant, A, B, and C
    # address slots), and the phase whose tensors build each variant's
    # descriptor templates on the host.
    gemm_table: tuple[tuple[int, ...], ...] = ()
    gemm_template_phases: tuple[int, ...] = ()
    # Weight streaming: per barrier that streams, three columns per CTA (the
    # GEMM row + 1 whose stages it loads, 0 for none; that row's variant;
    # the CTA's rotated block index in its phase). Row 0 is all zero; empty
    # without streaming.
    stream_table: tuple[tuple[int, ...], ...] = ()
    # K splits per GEMM variant (split-K).
    gemm_splits: tuple[int, ...] = ()
    # Phase scratch: FP32 and Int32 global buffers for bodies whose CTAs
    # cooperate within a phase (split-K GEMMs, split-context attention).
    # Counters start at zero and each phase leaves them at zero. Phases of
    # one barrier interval get disjoint regions; columns 4 and 5 of a phase
    # row hold its float and counter offsets.
    phase_float_elements: int = 0
    phase_counter_count: int = 0
    # Producer warp (X1.1): one extra warpgroup after threads_per_block,
    # outside every CTA-wide barrier, whose first warp produces.
    producer_warp: bool = False

    @property
    def launch_threads(self) -> int:
        return self.threads_per_block + (128 if self.producer_warp else 0)


@dataclass(frozen=True)
class SourceMapEntry:
    symbol: str
    kind: str
    line_start: int
    line_end: int
    phase_index: int | None
    region_id: str | None
    body_provider: str | None
    body_name: str | None
    source_op_ids: tuple[str, ...] = ()
    source_path: str = "module.py"


@dataclass(frozen=True)
class GeneratedCuTeModule:
    source: str
    module_name: str
    source_path: Path
    source_map_path: Path
    source_map: tuple[SourceMapEntry, ...]
    launch: LaunchMetadata


_CUTLASS_TYPES = {
    "torch.bool": "cutlass.Boolean",
    "torch.uint8": "cutlass.UInt8",
    "torch.int8": "cutlass.Int8",
    "torch.int16": "cutlass.Int16",
    "torch.int32": "cutlass.Int32",
    "torch.int64": "cutlass.Int64",
    "torch.float16": "cutlass.Float16",
    "torch.bfloat16": "cutlass.BFloat16",
    "torch.float32": "cutlass.Float32",
    "torch.float64": "cutlass.Float64",
}


def emit_cute(
    plan: KernelPlan,
    output_dir: Path | str | None = Path("artifacts/generated"),
    *,
    event_schedule: EventSchedule | None = None,
    table: bool = False,
) -> GeneratedCuTeModule:
    """Emit one CuTe module for ``plan``.

    ``table`` selects the table-driven launch even when the plan contains TMA
    GEMM bodies; it is required once per-phase kernel parameters would exceed
    CUDA's parameter limit. Without it, plans with TMA GEMMs pass their
    descriptors as kernel parameters.
    """
    verify_plan(plan)
    if plan.launch.architecture not in {"sm90a", "sm100", "sm103", "sm120", "sm121"}:
        raise UnsupportedCodegen("codegen supports SM90 and Blackwell plans only")

    if event_schedule is None and (
        table or not any(phase.body.provider == "g0_range" for phase in plan.phases)
    ):
        source, launch, source_map = _emit_table_module(plan)
    else:
        source, launch, source_map = _emit_generic_module(plan, event_schedule)

    module_name = (
        f"megabake_{hashlib.sha256((source + repr(launch)).encode()).hexdigest()[:16]}"
    )
    source_path = Path(output_dir or Path(".")) / module_name / "module.py"
    source_map_path = source_path.with_name("source_map.json")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    source_map_path.write_text(
        json.dumps(
            {
                "module_name": module_name,
                "source_file": source_path.name,
                "launch": launch,
                "entries": source_map,
            },
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return GeneratedCuTeModule(
        source=source,
        module_name=module_name,
        source_path=source_path,
        source_map_path=source_map_path,
        source_map=source_map,
        launch=launch,
    )


def _emit_generic_module(
    plan: KernelPlan,
    event_schedule: EventSchedule | None = None,
) -> tuple[str, LaunchMetadata, tuple[SourceMapEntry, ...]]:
    selected = {
        selection.region_id: selection for selection in plan.selection.selections
    }
    has_g0 = any(phase.body.provider == "g0_range" for phase in plan.phases)
    normalized = (
        {
            region_id: _widened_selection(
                selection, plan.launch.threads_per_block
            )
            for region_id, selection in selected.items()
        }
        if has_g0
        else selected
    )
    tensor_arguments = tuple(
        TensorArgument(
            f"tensor_{index}",
            binding.value_id,
            binding.storage_id,
            binding.kind,
            binding.layout,
            binding.binding_name,
            binding.region_ids,
        )
        for index, binding in enumerate(plan.storage.bindings)
    )
    work_iterators: list[WorkIteratorBinding] = []
    phase_records: list[PhaseEmission] = []
    variants: dict[tuple[str, str, str], tuple[str, int, Any]] = {}
    body_definitions: list[str] = []
    g0_definitions: list[str] = []
    conversion_definitions: list[str] = []
    phase_runners: list[str] = []
    phase_runner_calls: dict[int, list[tuple[int, list[str]]]] = {}
    phase_runner_symbols: dict[int, str] = {}
    phase_publishers: list[str] = []
    g0_tensor_arguments: list[TensorArgument] = []
    g0_prepared: list[tuple[str, str, str, str, str, str, str]] = []
    g0_scratch: list[str] = []
    g0_scratch_by_body: dict[str, str] = {}
    scratch_elements: dict[int, int] = {}
    g0_shared_bytes = 0

    for phase in plan.phases:
        selection = normalized[phase.region_id]
        if phase.body.provider == "g0_range":
            variant_key = (
                phase.body.provider,
                phase.body.tiles.cta_shape[:2],
                phase.body.tiles.cluster_shape[:2],
                phase.body.scratch.size_bytes,
            )
            if variant_key not in variants:
                variant_id = len(variants)
                body_symbol = f"_g0_body_{variant_id}"
                variants[variant_key] = (body_symbol, variant_id, phase.body)
                tile = phase.body.tiles.cta_shape[:2]
                cluster = phase.body.tiles.cluster_shape[:2]
                g0_definitions.append(
                    f"{body_symbol} = HopperWgmmaGemmPersistentKernel(\n"
                    "    cutlass.Float32,\n"
                    f"    {tile!r},\n"
                    f"    {cluster!r},\n"
                    "    1,\n"
                    "    True,\n"
                    ")"
                )
                g0_shared_bytes += phase.body.scratch.size_bytes
            body_symbol, variant_id, _ = variants[variant_key]

            g0_group_index = len(g0_prepared)
            g0_indices = []
            for value_id in (*selection.inputs, selection.outputs[0]):
                g0_indices.append(len(g0_tensor_arguments))
                argument_name = f"g0_tensor_{len(g0_tensor_arguments)}"
                layout = (
                    phase.body.layouts.inputs.get(value_id)
                    if value_id in selection.inputs
                    else phase.body.layouts.outputs[value_id]
                )
                g0_tensor_arguments.append(
                    TensorArgument(
                        argument_name,
                        value_id,
                        value_id,
                        "g0_input" if value_id in selection.inputs else "g0_output",
                        layout,
                        None,
                        (phase.region_id,),
                    )
                )

            scratch = g0_scratch_by_body.get(body_symbol)
            if scratch is None:
                scratch = f"_g0_scratch_{len(g0_scratch)}"
                g0_scratch_by_body[body_symbol] = scratch
                g0_scratch.append(
                    f"    {scratch} = cutlass.utils.SmemAllocator().allocate"
                    f"({body_symbol}.shared_storage)"
                )
            g0_prepared.append(
                (
                    body_symbol,
                    *g0_indices,
                )
            )
            runner_lines = [
                f"        {body_symbol}.run(\n"
                f"            g0_values[{g0_group_index}][2],\n"
                f"            g0_values[{g0_group_index}][0],\n"
                f"            {scratch},\n"
                f"            g0_values[{g0_group_index}][1],\n"
                "        )"
            ]
            phase_runner_symbols[phase.index] = f"_run_variant_{variant_id}"
            phase_runner_calls.setdefault(variant_id, []).append(
                (phase.index, runner_lines)
            )
            phase_records.append(
                PhaseEmission(
                    phase.index,
                    phase.region_id,
                    f"{body_symbol}.run",
                    variant_id,
                    phase.body.provider,
                    phase.body.name,
                    len(phase.work_items),
                    len(phase.work_items),
                    (),
                    phase.barrier.id if phase.barrier else None,
                )
            )
            if phase.barrier is not None:
                phase_publishers.append(
                    f"    @cute.jit\n"
                    f"    def _publish_phase_{phase.index}(epoch: cutlass.Int32):\n"
                    f"        grid_barrier(ready, workers, epoch + {phase.index})\n"
                )
            continue

        generated = _generate_device_body(selection)
        if generated.__dataclass_fields__.get("scratch_elements") is not None:
            scratch_elements[phase.index] = generated.scratch_elements
        variant_key = (phase.body.provider, phase.body.name, generated.source)
        if variant_key not in variants:
            variant_id = len(variants)
            body_symbol = f"_body_{variant_id}"
            variants[variant_key] = (body_symbol, variant_id, generated)
            body_definitions.append(_nested_body_source(generated.source, body_symbol))
        body_symbol, variant_id, generated = variants[variant_key]

        iterator_index = len(work_iterators)
        iterator_name = f"work_iterator_{phase.index}"
        step = generated.work_step
        work_iterators.append(
            WorkIteratorBinding(
                iterator_name,
                phase.index,
                phase.region_id,
                "torch.int64",
                generated.tile_count,
                0,
                step,
                generated.work_iterator_encoding,
            )
        )

        program_input_ids = tuple(
            dict.fromkeys(load.value_id for load in selection.program.loads)
        )
        tensor_names = tuple(
            _tensor_name(
                tensor_arguments,
                value_id,
                phase.region_id,
                "input",
                prefer_conversion=any(
                    conversion.value_id == value_id
                    and conversion.target_region_id == phase.region_id
                    for conversion in phase.conversion_bodies
                ),
            )
            for value_id in program_input_ids
        ) + (
            _tensor_name(
                tensor_arguments,
                selection.outputs[0],
                phase.region_id,
                "output",
            ),
        )
        tensor_indices = tuple(
            _tensor_argument_index(tensor_arguments, tensor_name)
            for tensor_name in tensor_names
        )
        scratch_name = (
            f"_scratch_{phase.index}"
            if phase.body.scratch.size_bytes
            else "None"
        )
        runner_lines = [
            f"        {body_symbol}(\n"
            f"            work_iterators[{iterator_index}],\n"
            "            _block,\n"
            "            _thread,\n"
            "            workers,\n"
            f"            ({', '.join(f'tensors[{index}]' for index in tensor_indices)},),\n"
            f"            {scratch_name},\n"
            "            None,\n"
            "        )",
        ]

        conversion_symbols: list[str] = []
        for conversion_index, conversion in enumerate(phase.conversion_bodies):
            conversion_symbol = f"_conversion_{phase.index}_{conversion_index}"
            target_selection = normalized[conversion.target_region_id]
            conversion_generated = _generate_conversion_body(
                conversion, target_selection.body, target_selection
            )
            conversion_key = (
                "conversion",
                f"{conversion.source_layout}:{conversion.target_layout}",
                conversion_generated.source,
            )
            if conversion_key not in variants:
                variants[conversion_key] = (conversion_symbol, len(variants), conversion_generated)
                conversion_definitions.append(
                    _nested_body_source(conversion_generated.source, conversion_symbol)
                )
            else:
                conversion_symbol = variants[conversion_key][0]
            conversion_symbols.append(conversion_symbol)
            conversion_iterator_index = len(work_iterators)
            conversion_iterator_name = (
                f"work_iterator_{phase.index}_conversion_{conversion_index}"
            )
            work_iterators.append(
                WorkIteratorBinding(
                    conversion_iterator_name,
                    phase.index,
                    phase.region_id,
                    "torch.int64",
                    conversion_generated.tile_count,
                    0,
                    conversion_generated.threads_per_cta,
                    conversion_generated.work_iterator_encoding,
                )
            )
            source_tensor = _tensor_name(
                tensor_arguments,
                conversion.value_id,
                conversion.source_region_id,
                "output",
            )
            target_tensor = _tensor_name(
                tensor_arguments,
                conversion.value_id,
                conversion.target_region_id,
                "input",
                prefer_conversion=True,
            )
            runner_lines.append(
            f"        {conversion_symbol}(\n"
            f"            work_iterators[{conversion_iterator_index}],\n"
            "            _block,\n"
            "            _thread,\n"
            "            workers,\n"
            f"            (tensors[{_tensor_argument_index(tensor_arguments, source_tensor)}], tensors[{_tensor_argument_index(tensor_arguments, target_tensor)}],),\n"
                "            None,\n"
                "            None,\n"
                "        )"
            )

        if phase.barrier is not None:
            phase_publishers.append(
                f"    @cute.jit\n"
                f"    def _publish_phase_{phase.index}(epoch: cutlass.Int32):\n"
                f"        grid_barrier(ready, workers, epoch + {phase.index})\n"
            )
        phase_runner_symbols[phase.index] = f"_run_variant_{variant_id}"
        phase_runner_calls.setdefault(variant_id, []).append(
            (phase.index, runner_lines)
        )
        phase_records.append(
            PhaseEmission(
                phase.index,
                phase.region_id,
                body_symbol,
                variant_id,
                phase.body.provider,
                phase.body.name,
                len(phase.work_items),
                generated.tile_count,
                tuple(conversion_symbols),
                phase.barrier.id if phase.barrier else None,
            )
        )

    for variant_id, calls in phase_runner_calls.items():
        phase_runners.append(_phase_runner_source(variant_id, calls))

    if event_schedule is not None:
        event_by_region: dict[str, int] = {}
        for event in event_schedule.events:
            producer_regions = {
                event_schedule.task_for(task_id).region_id
                for task_id in event.producer_task_ids
            }
            if len(producer_regions) != 1:
                raise UnsupportedCodegen("event producers must share one region")
            event_by_region[producer_regions.pop()] = event.id
        phase_publishers.clear()
        for phase in plan.phases:
            wait_lines: list[str] = []
            for producer_region_id in phase.dependencies:
                if producer_region_id not in event_by_region:
                    raise UnsupportedCodegen(
                        f"region {phase.region_id} waits on an unmapped event"
                    )
                event_id = event_by_region[producer_region_id]
                wait_lines.extend(
                    (
                        "        wait_for_event(",
                        "            ready,",
                        f"            cutlass.Int32({event_id}),",
                        f"            cutlass.Int32({plan.launch.grid_size}),",
                        "            epoch,",
                        "        )",
                    )
                )
            if wait_lines:
                phase_publishers.append(
                    f"    @cute.jit\n"
                    f"    def _wait_phase_{phase.index}(epoch: cutlass.Int32):\n"
                    + "\n".join(wait_lines)
                    + "\n"
                )
            if phase.region_id in event_by_region:
                event_id = event_by_region[phase.region_id]
                phase_publishers.append(
                    f"    @cute.jit\n"
                    f"    def _publish_phase_{phase.index}():\n"
                    f"        publish_event(ready, cutlass.Int32({event_id}))\n"
                )
        phase_records = [
            replace(record, barrier=None) for record in phase_records
        ]

    shared_source = _shared_scratch_source(plan, scratch_elements)
    scratch_prelude = _scratch_prelude(plan, scratch_elements)
    shared_memory_bytes = (
        g0_shared_bytes + _generic_scratch_bytes(plan, scratch_elements)
        if has_g0
        else plan.launch.dynamic_shared_memory_bytes
    )
    target = plan.selection.target
    if target is not None and shared_memory_bytes > target.max_shared_memory_per_cta:
        raise UnsupportedCodegen(
            f"generated kernel requires {shared_memory_bytes} shared-memory bytes"
        )
    g0_prepare_lines = [
        f"        {body_symbol}.prepare(\n"
        f"            g0_tensors[{group_index}][0],\n"
        f"            g0_tensors[{group_index}][1],\n"
        f"            g0_tensors[{group_index}][2],\n"
        "            max_active_clusters,\n"
        "        ),"
        for group_index, (body_symbol, *rest) in enumerate(g0_prepared)
    ]
    g0_values_source = (
        "    g0_values = (\n" + "\n".join(g0_prepare_lines) + "\n    )\n"
        if g0_prepare_lines
        else ""
    )
    launch_smem = "" if has_g0 else f"        smem={shared_memory_bytes},\n"
    runtime_import = (
        "from megabake.runtime import cooperative_event_worker_loop, publish_event, trap_if, wait_for_event"
        if event_schedule is not None
        else "from megabake.runtime import cooperative_worker_loop, grid_barrier, trap_if"
    )
    kernel_arguments = [
        "ready: cute.Tensor",
        "workers: cutlass.Int32",
        "epoch: cutlass.Int32",
        "work_iterators: tuple[cute.Tensor, ...]",
        "tensors: tuple[cute.Tensor, ...]",
        *(("g0_values: tuple[tuple[cute.Tensor, cute.Tensor, cute.Tensor], ...]",) if has_g0 else ()),
    ]
    launch_arguments = [
        "ready: cute.Tensor",
        "workers: cutlass.Int32",
        "epoch: cutlass.Int32",
        "work_iterators: tuple[cute.Tensor, ...]",
        "tensors: tuple[cute.Tensor, ...]",
        *(("g0_tensors: tuple[tuple[cute.Tensor, cute.Tensor, cute.Tensor], ...]",) if has_g0 else ()),
        "stream: cuda.CUstream",
    ]
    if has_g0:
        launch_arguments.insert(-1, "max_active_clusters: cutlass.Constexpr")
    source = f'''import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass._mlir.dialects import llvm

{runtime_import}
from megabake.bodies.cuda.sm90.gemm import HopperWgmmaGemmPersistentKernel

{"\n\n".join(g0_definitions)}


{shared_source}

@cute.kernel
def megabake_kernel(
{("," + chr(10)).join(f"    {argument}" for argument in kernel_arguments)}
):
    _, _, _block = cute.arch.block_idx()
    _thread, _, _ = cute.arch.thread_idx()
{scratch_prelude}
{chr(10).join(g0_scratch)}
{"".join(body_definitions)}
{"".join(conversion_definitions)}
    @cute.jit
    def _no_op():
        pass

    @cute.jit
    def _ignore_epoch(epoch: cutlass.Int32):
        pass

    @cute.jit
    def _cta_sync(epoch: cutlass.Int32):
        cute.arch.sync_threads()

    @cute.jit
    def _run_terminal(phase_index: cutlass.Int32):
        pass

{"".join(phase_runners)}
{"".join(phase_publishers)}
{_event_worker_loop(plan, phase_runner_symbols, event_schedule)}


@cute.jit
def launch(
{("," + chr(10)).join(f"    {argument}" for argument in launch_arguments)}
):
{g0_values_source}    megabake_kernel(
        ready,
        workers,
        epoch,
        work_iterators,
        tensors{"," if has_g0 else ""}
{f"        g0_values{chr(10)}" if has_g0 else ""}    ).launch(
        grid=[1, 1, workers],
        block=[{plan.launch.threads_per_block}, 1, 1],
        cooperative=True,
{launch_smem}        stream=stream,
    )
'''
    launch_metadata = LaunchMetadata(
        "megabake_kernel",
        "launch",
        plan.launch.grid_size,
        plan.launch.threads_per_block,
        True,
        shared_memory_bytes,
        plan.launch.architecture,
        ("ready", "workers", "epoch", "work_iterators", "tensors")
        + (("g0_tensors",) if has_g0 else ())
        + (("max_active_clusters",) if has_g0 else ())
        + ("stream",),
        tuple(work_iterators),
        tuple((*tensor_arguments, *g0_tensor_arguments)),
        tuple(phase_records),
        launch_policy=(
            "megabake_cooperative_events"
            if event_schedule is not None
            else "megabake_cooperative"
        ),
        body_variant_count=len(g0_definitions) + len(body_definitions),
        phase_runner_count=len(phase_runner_calls),
        source_bytes=len(source.encode("utf-8")),
    )
    return source, launch_metadata, _source_map(source, plan, phase_records)


_DYNAMIC_BINDING_KINDS = frozenset({"input", "state", "output", "alias"})


def _emit_table_module(
    plan: KernelPlan,
) -> tuple[str, LaunchMetadata, tuple[SourceMapEntry, ...]]:
    """Emit one body per variant and drive phases from a device-side table.

    Every tensor becomes an address-table slot rebuilt in the kernel from its
    static layout, so kernel parameters and generated code stay bounded by the
    number of distinct bodies rather than by the number of layers.
    """
    has_gemm = any(phase.body.provider == "g0_range" for phase in plan.phases)
    selected = {
        selection.region_id: (
            _widened_selection(selection, plan.launch.threads_per_block)
            if has_gemm
            else selection
        )
        for selection in plan.selection.selections
    }
    tensor_arguments = tuple(
        TensorArgument(
            f"tensor_{index}",
            binding.value_id,
            binding.storage_id,
            binding.kind,
            binding.layout or _value_layout(plan, binding.value_id),
            binding.binding_name,
            binding.region_ids,
        )
        for index, binding in enumerate(plan.storage.bindings)
    )
    gemm_variants: dict[tuple[Any, ...], int] = {}
    epilogue_definitions: list[str] = []
    gemm_definitions: list[str] = []
    # Variants that are paired GEMMs, and each GEMM row's trailing columns:
    # its second A slot + 1 (0 when unpaired), then its epilogue tensor slots.
    paired_variants: set[int] = set()
    gemm_extra_columns: list[tuple[int, ...]] = []
    gemm_kernel_tiles: dict[int, tuple[int, int]] = {}
    # Each GEMM variant's fused epilogues as (source, slot layouts); a phase
    # picks epilogue kind index + 1 at run time (0: plain store), so phases of
    # one shape share one kernel body whatever they fuse.
    variant_epilogues: dict[int, list[tuple[str, tuple[Any, ...]]]] = {}
    # Per variant: what its definition and runner need once all phases are seen.
    gemm_variant_specs: dict[int, dict[str, Any]] = {}
    gemm_shapes: list[tuple[int, int, int, bool]] = []
    gemm_template_phases: list[int] = []
    gemm_rows: list[tuple[int, ...]] = []
    # GEMM phase -> its GEMM row.
    gemm_of_phase: dict[int, int] = {}
    # GEMM rows whose weight operand can be streamed into shared memory.
    streamable_gemms: set[int] = set()
    gemm_splits: list[int] = []
    # Phase index -> (FP32 elements, Int32 counters) of phase scratch.
    phase_scratch: dict[int, tuple[int, int]] = {}
    gemm_scratch_bytes: set[int] = set()
    variants: dict[tuple[Any, ...], tuple[str, int, Any]] = {}
    body_definitions: list[str] = []
    runner_variants: dict[tuple[Any, ...], tuple[int, int, list[str]]] = {}
    work_iterators: list[WorkIteratorBinding] = []
    phase_records: list[PhaseEmission] = []
    phase_rows: list[tuple[int, ...]] = []
    # Work tiles per phase, for spreading concurrent phases across CTAs.
    phase_tiles: list[int] = []
    # Work units each phase's body iterates, from its rotated CTA index in
    # steps of the grid size; a CTA whose rotated index reaches this count
    # has nothing to do in the phase and skips its body.
    phase_units: dict[int, int] = {}
    # Tensor columns of each non-GEMM phase's row, which get resolved addresses.
    resolved_columns: dict[int, int] = {}
    # GEMM phases: n slot columns from GEMM_PHASE_SLOT_COLUMN resolve
    # (columns 0 and 1 are the GEMM row index and the epilogue kind).
    gemm_resolved_columns: dict[int, int] = {}
    # Stateful attention phase -> address slot of its KV valid length.
    length_publishes: dict[int, int] = {}
    # Stateful attention phases whose interval ends in a grid barrier that no
    # other length publish uses.
    deferred_publishes: set[int] = set()
    closing_barriers: set[int] = set()
    for phase in plan.phases:
        if not selected[phase.region_id].program.state_accesses:
            continue
        closing = next(
            (
                position
                for position in range(phase.index, len(plan.phases))
                if plan.phases[position].barrier is not None
            ),
            None,
        )
        if closing is not None and closing not in closing_barriers:
            closing_barriers.add(closing)
            deferred_publishes.add(phase.index)
    representative_phases: list[Any] = []
    scratch_elements: dict[int, int] = {}

    def body_variant(
        generated: Any, key: tuple[Any, ...], step: int, symbol: str | None = None
    ) -> tuple[str, int]:
        if key not in variants:
            symbol = symbol or f"_body_{len(variants)}"
            variants[key] = (symbol, len(variants), generated)
            body_definitions.append(
                _affine_work_iterator(
                    _nested_body_source(generated.source, symbol),
                    step,
                    generated.tile_count,
                )
            )
        symbol, variant_id, _ = variants[key]
        return symbol, variant_id

    def iterator_index(name: str, phase: Any, generated: Any, step: int) -> int:
        work_iterators.append(
            WorkIteratorBinding(
                name,
                phase.index,
                phase.region_id,
                "torch.int64",
                generated.tile_count,
                0,
                step,
                generated.work_iterator_encoding,
            )
        )
        return len(work_iterators) - 1

    for phase in plan.phases:
        selection = selected[phase.region_id]
        if phase.body.provider == "g0_range":
            if phase.conversion_bodies:
                raise UnsupportedCodegen(
                    f"table GEMM phase {phase.region_id} needs a layout conversion"
                )
            program = selection.program
            rows, columns = program.store.shape
            reduction = next(
                domain.size
                for domain in program.iteration_domains
                if domain.axis in program.reduction_axes
            )
            tile = kernel_tile(phase.body)
            splits = split_k(phase.body)
            bias = has_bias(phase.body)
            fused = has_fused_epilogue(phase.body)
            paired = is_paired(phase.body)
            fused_loads = epilogue_loads(program) if fused else ()
            fused_layouts = tuple(
                tensor_arguments[
                    _tensor_argument_index(
                        tensor_arguments,
                        _tensor_name(tensor_arguments, load.value_id, phase.region_id, "input"),
                    )
                ].layout
                for load in fused_loads
            )
            gemm_scratch_bytes.add(phase.body.scratch.size_bytes)
            gemm_key = (
                tile, phase.body.scratch.size_bytes, rows, columns, reduction, splits, bias, paired
            )
            if gemm_key not in gemm_variants:
                gemm_variant = len(gemm_variants)
                gemm_variants[gemm_key] = gemm_variant
                # A paired tile stacks 64 rows of each A on CUTLASS M.
                gemm_kernel_tiles[gemm_variant] = (2 * tile[0], tile[1]) if paired else tile
                if paired:
                    paired_variants.add(gemm_variant)
                # CUTLASS problem shape; swapAB computes the transposed product.
                gemm_shapes.append(
                    (columns, rows, reduction, True)
                    if is_swap_ab(phase.body)
                    else (rows, columns, reduction, False)
                )
                gemm_splits.append(splits)
                gemm_template_phases.append(phase.index)
                symbol = f"_gemm_{gemm_variant}"
                variants[("g0_range", gemm_key)] = (symbol, len(variants), None)
                gemm_variant_specs[gemm_variant] = {
                    "symbol": symbol,
                    "scratch": phase.body.scratch.size_bytes,
                    "splits": splits,
                    "bias_features": columns if bias else 0,
                    "swap_ab": is_swap_ab(phase.body),
                    "bias": bias,
                    "paired": paired,
                }
            gemm_variant = gemm_variants[gemm_key]
            epilogue_kind = 0
            if fused:
                entry = (
                    epilogue_source(program, "_epilogue"),
                    tuple(_layout_key(layout) for layout in fused_layouts),
                )
                kinds = variant_epilogues.setdefault(gemm_variant, [])
                known = [(source, tuple(_layout_key(layout) for layout in layouts)) for source, layouts in kinds]
                if entry not in known:
                    kinds.append((entry[0], fused_layouts))
                    epilogue_definitions.append(
                        epilogue_source(program, f"_epilogue_{gemm_variant}_{len(kinds)}")
                    )
                    known.append(entry)
                epilogue_kind = known.index(entry) + 1
            body_symbol, body_variant_id, _ = variants[("g0_range", gemm_key)]
            gemm_slots = [
                _tensor_argument_index(
                    tensor_arguments,
                    _tensor_name(tensor_arguments, load.value_id, phase.region_id, "input"),
                )
                for load in program.loads
            ]
            fused_slots = gemm_slots[len(program.loads) - len(fused_loads) :]
            del gemm_slots[len(program.loads) - len(fused_loads) :]
            second_a_slot = gemm_slots.pop() + 1 if paired else 0
            # Row: variant, A, B, C, then the bias slot + 1 (0 without bias).
            bias_slot = gemm_slots.pop() + 1 if bias else 0
            if is_swap_ab(phase.body):
                # CUTLASS A is the weight and B the activation.
                gemm_slots = [gemm_slots[1], gemm_slots[0], *gemm_slots[2:]]
            gemm_slots.append(
                _tensor_argument_index(
                    tensor_arguments,
                    _tensor_name(
                        tensor_arguments, selection.outputs[0], phase.region_id, "output"
                    ),
                )
            )
            gemm_index = len(gemm_rows)
            gemm_of_phase[phase.index] = gemm_index
            # swapAB GEMMs load the weight as CUTLASS A, which streaming
            # loads ahead when no region of the launch writes it.
            if is_swap_ab(phase.body) and all(
                _read_only_input(plan, tensor_arguments[slot])
                for slot in (gemm_slots[0], *((second_a_slot - 1,) if paired else ()))
            ):
                streamable_gemms.add(gemm_index)
            gemm_rows.append((gemm_variant, *gemm_slots, bias_slot))
            gemm_extra_columns.append((second_a_slot, epilogue_kind, *fused_slots))
            if splits > 1:
                output_tiles = -(-gemm_shapes[gemm_variant][0] // tile[0]) * -(
                    -gemm_shapes[gemm_variant][1] // tile[1]
                )
                # FP32 partials of every split of every tile, one counter per tile.
                # A paired tile holds both products' accumulators.
                phase_scratch[phase.index] = (
                    output_tiles * splits * tile[0] * tile[1] * (2 if paired else 1),
                    output_tiles,
                )
            runner_key = ("g0_range", gemm_variant)
            if runner_key not in runner_variants:
                representative_phases.append(phase)
                # Lines are written once every phase's epilogue kind is known.
                runner_variants[runner_key] = (len(runner_variants), 0, [])
            runner_id = runner_variants[runner_key][0]
            # After the GEMM row index: the epilogue kind, then the slots of
            # C, the bias and the epilogue tensors (GEMM_PHASE_SLOT_COLUMN on).
            # The slots resolve like other phases' slots, so the phase loop's
            # L1 prefetch covers all of them: no dependent gemm_table load
            # between the barrier and the GEMM body or before its epilogue.
            gemm_resolved = (
                gemm_slots[-1],
                *((bias_slot - 1,) if bias_slot else ()),
                *fused_slots,
            )
            gemm_resolved_columns[phase.index] = len(gemm_resolved)
            phase_rows.append(
                (
                    runner_id, int(phase.barrier is not None), 0, 0, 0, 0,
                    gemm_index, epilogue_kind, *gemm_resolved,
                )
            )
            phase_tiles.append(len(phase.work_items) * splits)
            gemm_shape = gemm_shapes[gemm_variant]
            phase_units[phase.index] = (
                -(-gemm_shape[0] // tile[0]) * -(-gemm_shape[1] // tile[1]) * splits
            )
            phase_records.append(
                PhaseEmission(
                    phase.index,
                    phase.region_id,
                    f"{body_symbol}.run_table",
                    body_variant_id,
                    phase.body.provider,
                    phase.body.name,
                    len(phase.work_items),
                    len(phase.work_items),
                    (),
                    phase.barrier.id if phase.barrier else None,
                )
            )
            continue
        # The table kernel publishes KV valid lengths after the barrier that
        # ends the attention's interval (see the phase loop below); without
        # such a barrier the body publishes from a single CTA itself.
        generated = _generate_device_body(
            selection,
            defer_length_publish=phase.index in deferred_publishes,
            workers=plan.launch.grid_size,
        )
        scratch_floats = getattr(generated, "phase_floats", 0)
        scratch_counters = getattr(generated, "phase_counters", 0)
        if scratch_floats or scratch_counters:
            phase_scratch[phase.index] = (scratch_floats, scratch_counters)
        if generated.__dataclass_fields__.get("scratch_elements") is not None:
            scratch_elements[phase.index] = generated.scratch_elements
        step = generated.work_step
        body_symbol, body_variant_id = body_variant(
            generated, (phase.body.provider, phase.body.name, generated.source), step
        )
        program_input_ids = tuple(
            dict.fromkeys(load.value_id for load in selection.program.loads)
        )
        slots = [
            _tensor_argument_index(
                tensor_arguments,
                _tensor_name(
                    tensor_arguments,
                    value_id,
                    phase.region_id,
                    "input",
                    prefer_conversion=any(
                        conversion.value_id == value_id
                        and conversion.target_region_id == phase.region_id
                        for conversion in phase.conversion_bodies
                    ),
                ),
            )
            for value_id in program_input_ids
        ]
        slots.append(
            _tensor_argument_index(
                tensor_arguments,
                _tensor_name(
                    tensor_arguments, selection.outputs[0], phase.region_id, "output"
                ),
            )
        )
        if phase.index in deferred_publishes:
            length_value = next(
                load.value_id
                for load in selection.program.loads
                if load.id == "attention:valid_length"
            )
            length_publishes[phase.index] = slots[program_input_ids.index(length_value)]
        calls: list[tuple[str, tuple[int, ...], Any, int]] = [
            (body_symbol, tuple(slots), generated, step)
        ]
        conversion_symbols: list[str] = []
        for conversion_index, conversion in enumerate(phase.conversion_bodies):
            target_selection = selected[conversion.target_region_id]
            conversion_generated = _generate_conversion_body(
                conversion, target_selection.body, target_selection
            )
            conversion_symbol, _ = body_variant(
                conversion_generated,
                (
                    "conversion",
                    f"{conversion.source_layout}:{conversion.target_layout}",
                    conversion_generated.source,
                ),
                conversion_generated.threads_per_cta,
                f"_conversion_{phase.index}_{conversion_index}",
            )
            conversion_symbols.append(conversion_symbol)
            source_slot = _tensor_argument_index(
                tensor_arguments,
                _tensor_name(
                    tensor_arguments,
                    conversion.value_id,
                    conversion.source_region_id,
                    "output",
                ),
            )
            target_slot = _tensor_argument_index(
                tensor_arguments,
                _tensor_name(
                    tensor_arguments,
                    conversion.value_id,
                    conversion.target_region_id,
                    "input",
                    prefer_conversion=True,
                ),
            )
            calls.append(
                (
                    conversion_symbol,
                    (source_slot, target_slot),
                    conversion_generated,
                    conversion_generated.threads_per_cta,
                )
            )

        # Phases share a runner when their bodies, slot layouts, and scratch
        # agree; only the slot numbers in the phase table differ.
        runner_key = (
            tuple(
                (
                    symbol,
                    tuple(_layout_key(tensor_arguments[slot].layout) for slot in call_slots),
                    step,
                )
                for symbol, call_slots, _, step in calls
            ),
            phase.body.scratch.size_bytes,
            scratch_elements.get(phase.index),
        )
        if runner_key not in runner_variants:
            runner_id = len(runner_variants)
            representative_phases.append(phase)
            scratch_name = (
                f"_scratch_{phase.index}" if phase.body.scratch.size_bytes else "None"
            )
            lines: list[str] = []
            column = 0
            for call_index, (symbol, call_slots, call_generated, step) in enumerate(calls):
                iterator = iterator_index(
                    f"work_iterator_variant_{runner_id}_{call_index}",
                    phase,
                    call_generated,
                    step,
                )
                tensors = ", ".join(
                    _slot_tensor_code(column + offset, tensor_arguments[slot].layout)
                    for offset, slot in enumerate(call_slots)
                )
                column += len(call_slots)
                lines.extend(
                    (
                        f"            {symbol}(",
                        f"                work_iterators[{iterator}],",
                        "                _rotated,",
                        "                _thread,",
                        "                _workers,",
                        f"                ({tensors},),",
                        f"                {scratch_name if call_index == 0 else 'None'},",
                        (
                            "                (phase_floats, phase_counters, "
                            "phase_table[phase, 4], phase_table[phase, 5]),"
                            if getattr(call_generated, "phase_floats", 0)
                            or getattr(call_generated, "phase_counters", 0)
                            else "                None,"
                        ),
                        "            )",
                    )
                )
            runner_variants[runner_key] = (runner_id, column, lines)
        runner_id, _, _ = runner_variants[runner_key]
        resolved_columns[phase.index] = sum(len(call_slots) for _, call_slots, _, _ in calls)
        phase_rows.append(
            (
                runner_id,
                int(phase.barrier is not None),
                0,
                0,
                0,
                0,
                *(slot for _, call_slots, _, _ in calls for slot in call_slots),
            )
        )
        phase_tiles.append(max(call[2].tile_count for call in calls))
        phase_units[phase.index] = phase_tiles[-1]
        phase_records.append(
            PhaseEmission(
                phase.index,
                phase.region_id,
                body_symbol,
                body_variant_id,
                phase.body.provider,
                phase.body.name,
                len(phase.work_items),
                generated.tile_count,
                tuple(conversion_symbols),
                phase.barrier.id if phase.barrier else None,
            )
        )

    # Phases of one barrier interval run concurrently, so each takes its own
    # phase-scratch region; later intervals reuse the regions.
    # Counter slots 0 and 1 hold the stall region's address (X0.3).
    float_elements = 0
    counter_count = STALL_COUNTER_SLOTS
    float_offset, counter_offset = 0, STALL_COUNTER_SLOTS
    for index, phase in enumerate(plan.phases):
        floats, counters = phase_scratch.get(phase.index, (0, 0))
        if floats or counters:
            row = phase_rows[index]
            phase_rows[index] = (*row[:4], float_offset, counter_offset, *row[6:])
            float_offset += floats
            counter_offset += counters
            float_elements = max(float_elements, float_offset)
            counter_count = max(counter_count, counter_offset)
        if phase.barrier is not None:
            float_offset, counter_offset = 0, STALL_COUNTER_SLOTS

    # Every body starts its tiles at CTA 0. Phases in one barrier interval run
    # concurrently, so each starts where the previous phase's tiles ended.
    # Recompute phases run every tile on every CTA; rotation -1 marks them.
    offset = 0
    for index, phase in enumerate(plan.phases):
        if index == 0 or plan.phases[index - 1].barrier is not None:
            offset = 0
        row = phase_rows[index]
        if phase.recompute:
            phase_rows[index] = (*row[:2], -1, *row[3:])
            continue
        phase_rows[index] = (*row[:2], offset % plan.launch.grid_size, *row[3:])
        offset += phase_tiles[index]
    # Weight streaming: which GEMM's stages each CTA loads at which barrier.
    stream_rows, stream_issues = _weight_stream_schedule(
        plan,
        [row[2] for row in phase_rows],
        phase_units,
        gemm_of_phase,
        [row[0] for row in gemm_rows],
        streamable_gemms,
        WEIGHT_STREAM_STAGES,
    )
    streamed_variants = {
        gemm_rows[gemm][0]
        for row in stream_rows
        for gemm in (entry - 1 for entry in row[0::3] if entry)
    }
    streaming = bool(streamed_variants)
    # X1.1/X1.2: a producer warp loads every GEMM's weights into one
    # persistent page ring (HopperWgmmaGemmPersistentKernel.ring_pages). It
    # needs swapAB decode GEMMs whose stages are 64-row weight pages and
    # 16-row activation slots, without split-K; weight streaming is the
    # older, mutually exclusive way to load ahead. The producer is a whole
    # warpgroup: ptxas budgets registers per warpgroup, so one extra warp
    # would cap every thread at 168 registers; setmaxnreg moves them to the
    # body warpgroups instead (_split_roles).
    producer = (
        has_gemm
        and not streaming
        and producer_warp_enabled()
        and plan.launch.threads_per_block % 128 == 0
        and plan.launch.threads_per_block <= 256
        and all(
            spec["splits"] == 1
            and spec["swap_ab"]
            and gemm_kernel_tiles[variant] == ((128, 16) if spec["paired"] else (64, 16))
            for variant, spec in gemm_variant_specs.items()
        )
    )
    ring_pages = 0
    if producer:
        # As many pages as shared memory holds next to the generic bodies'
        # scratch, up to MEGABAKE_RING_PAGES (X1.0: the gain saturates at
        # 20-24 pages of 8 KiB); below 8 pages the producer cannot run ahead.
        target = plan.selection.target
        budget = target.max_shared_memory_per_cta if target is not None else 227 * 1024
        free = budget - _generic_scratch_bytes(
            _representative_scratch_plan(plan, representative_phases), scratch_elements
        ) - 1024 - 1024 - 4 * 64 * 16 * 2
        ring_pages = min(producer_ring_pages(), free // ((64 * 64 + 16 * 64) * 2)) // 2 * 2
        if ring_pages < 8:
            producer, ring_pages = False, 0
    launch_threads = plan.launch.threads_per_block + (128 if producer else 0)
    for gemm_variant, spec in gemm_variant_specs.items():
        kinds = variant_epilogues.get(gemm_variant, [])
        epilogues = "".join(
            f"_epilogue_{gemm_variant}_{kind}, " for kind in range(1, len(kinds) + 1)
        )
        gemm_definitions.append(
            f"{spec['symbol']} = HopperWgmmaGemmPersistentKernel(\n"
            "    cutlass.Float32,\n"
            f"    {gemm_kernel_tiles[gemm_variant]!r},\n"
            "    (1, 1),\n"
            "    1,\n"
            "    True,\n"
            f"    smem_capacity={spec['scratch']},\n"
            "    reallocate_registers=False,\n"
            f"    splits={spec['splits']},\n"
            f"    bias_features={spec['bias_features']},\n"
            f"    bias_along_m={spec['swap_ab']!r},\n"
            f"    direct_store={spec['swap_ab']!r},\n"
            + (f"    epilogues=({epilogues}),\n" if kinds else "")
            + ("    paired=True,\n" if spec["paired"] else "")
            + (f"    ring_pages={ring_pages},\n" if ring_pages else "")
            + (
                f"    stream_stages={WEIGHT_STREAM_STAGES},\n"
                if gemm_variant in streamed_variants
                else ""
            )
            + (
                f"    stamp_address={gemm_stamp_address()},\n"
                f"    stamp_tag={gemm_variant + 1},\n"
                if gemm_stamp_address()
                else ""
            )
            + ")"
        )
        split_state = (
            [
                "                (",
                "                    phase_floats,",
                "                    phase_counters,",
                "                    phase_table[phase, 4],",
                "                    phase_table[phase, 5],",
                "                ),",
            ]
            if spec["splits"] > 1
            else []
        )
        bias_state = (
            [
                "                None," if spec["splits"] == 1 else "",
                f"                _slot_address(addresses, phase, {GEMM_PHASE_SLOT_COLUMN + 1}),",
            ]
            if spec["bias"]
            else []
        )
        # One tuple of tensors per epilogue kind, all read from the row's
        # epilogue slot columns; only the phase's own kind dereferences them.
        epilogue_state = (
            [
                # A closure: the GEMM reads the kind and builds the tensors
                # after its mainloop, so they hold no registers across it.
                "                epilogue_state=lambda: (",
                f"                    phase_table[phase, {6 + GEMM_PHASE_KIND_COLUMN}],",
                "                    (",
                *(
                    "                        ("
                    + "".join(
                        _slot_tensor_code(
                            GEMM_PHASE_SLOT_COLUMN + (2 if spec["bias"] else 1) + position, layout
                        )
                        + ", "
                        for position, layout in enumerate(layouts)
                    )
                    + "),"
                    for _, layouts in kinds
                ),
                "                    ),",
                "                ),",
            ]
            if kinds
            else []
        )
        runner_id = runner_variants[("g0_range", gemm_variant)][0]
        runner_variants[("g0_range", gemm_variant)] = (
            runner_id,
            0,
            [
                f"            {spec['symbol']}.run_table(",
                f"                gemm_values[{gemm_variant}],",
                "                _gemm_scratch,",
                "                table_descriptors(tma_descriptors, phase_table[phase, 6]),",
                "                _rotated,",
                *split_state,
                *(line for line in bias_state if line),
                *(
                    [
                        f"                c_address=_slot_address(addresses, phase, {GEMM_PHASE_SLOT_COLUMN}),"
                    ]
                    if spec["swap_ab"]
                    else []
                ),
                *epilogue_state,
                *(
                    ["                stream_tag=phase_table[phase, 6] + 1,"]
                    if gemm_variant in streamed_variants
                    else []
                ),
                "            )",
            ],
        )

    for index, length_slot in length_publishes.items():
        closing = next(
            (
                position
                for position in range(index, len(plan.phases))
                if plan.phases[position].barrier is not None
            ),
            None,
        )
        if closing is None or phase_rows[closing][3] != 0:
            raise UnsupportedCodegen(
                f"KV length of phase {index} has no grid barrier of its own to follow"
            )
        row = phase_rows[closing]
        phase_rows[closing] = (*row[:3], length_slot + 1, *row[4:])
    width = max(len(row) for row in phase_rows)
    # Barriers of one call get dense ordinals: the address barrier, the GEMM
    # descriptor barrier, then each phase barrier in order. Column 1 of a
    # phase row holds its barrier's ordinal (0: no barrier).
    phase_epoch_offset = 2 if has_gemm else 1
    ordinal = phase_epoch_offset
    for index, phase in enumerate(plan.phases):
        if phase.barrier is not None:
            row = phase_rows[index]
            phase_rows[index] = (row[0], ordinal, *row[2:])
            ordinal += 1
    barrier_epochs = ordinal
    # Trailing columns: the phase's work units, then, with weight streaming,
    # the stream-table row its closing barrier streams weights from (0: none).
    phase_table = tuple(
        row
        + (0,) * (width - len(row))
        + (phase_units[plan.phases[index].index],)
        + ((stream_issues.get(index, 0),) if streaming else ())
        for index, row in enumerate(phase_rows)
    )
    # Resolved slot addresses: the slot behind each phase's tensor column
    # (GEMM rows hold their GEMM row there instead and resolve nothing).
    slot_width = max(width - 6, 1)
    def resolves(index: int, column: int) -> bool:
        if index in gemm_resolved_columns:
            first = GEMM_PHASE_SLOT_COLUMN
            return first <= column < first + gemm_resolved_columns[index]
        return column < resolved_columns.get(index, 0)

    resolved = [
        row[6 + column] if resolves(index, column) else None
        for index, row in enumerate(phase_rows)
        for column in range(slot_width)
    ]
    resolved_slots = tuple(0 if slot is None else slot for slot in resolved)
    gemm_rows = [
        (*row, *extra) for row, extra in zip(gemm_rows, gemm_extra_columns, strict=True)
    ]
    gemm_row_width = max((len(row) for row in gemm_rows), default=0)
    gemm_rows = [row + (0,) * (gemm_row_width - len(row)) for row in gemm_rows]
    boundary_slots = tuple(
        index
        for index, argument in enumerate(tensor_arguments)
        if argument.kind in _DYNAMIC_BINDING_KINDS and argument.layout is not None
    )
    # Resolved entries of per-call slots change with them, so the kernel
    # prologue writes them from the dynamic-address ring too.
    boundary = set(boundary_slots)
    dynamic_slots = boundary_slots + tuple(
        len(tensor_arguments) + position
        for position, slot in enumerate(resolved)
        if slot is not None and slot in boundary
    )


    scratch_plan = _representative_scratch_plan(plan, representative_phases)
    shared_source = _shared_scratch_source(scratch_plan, scratch_elements)
    scratch_prelude = _scratch_prelude(scratch_plan, scratch_elements)
    shared_memory_bytes = plan.launch.dynamic_shared_memory_bytes
    if has_gemm:
        if len(gemm_scratch_bytes) != 1 or len({key[0] for key in gemm_variants}) != 1:
            raise UnsupportedCodegen(
                "table-launched GEMMs must share one tile and shared-memory layout"
            )
        # The allocator sizes dynamic shared memory from the GEMM stages and
        # the generic union; the 1 KiB covers the GEMM buffers' alignment.
        # Paired variants stage larger A tiles, so the GEMM storage is the
        # union of every variant's buffers.
        shared_memory_bytes = (
            (
                # Ring: mbarriers in the first KiB, then the A pages, B slots
                # and the epilogue's four 64x16 BF16 stages.
                1024 + ring_pages * (64 * 64 + 16 * 64) * 2 + 4 * 64 * 16 * 2
                if ring_pages
                else table_gemm_shared_bytes(
                    tuple(gemm_kernel_tiles.values()), next(iter(gemm_scratch_bytes))
                )
                if paired_variants
                else next(iter(gemm_scratch_bytes))
            )
            + _generic_scratch_bytes(scratch_plan, scratch_elements)
            + 1024
        )
    target = plan.selection.target
    if target is not None and shared_memory_bytes > target.max_shared_memory_per_cta:
        raise UnsupportedCodegen(
            f"generated kernel requires {shared_memory_bytes} shared-memory bytes"
        )

    if CTA_TIMER and (
        len(plan.phases) > CTA_TIMER_PHASES or plan.launch.grid_size > CTA_TIMER_WORKERS
    ):
        raise UnsupportedCodegen(
            f"CTA timer holds {CTA_TIMER_PHASES} phases of {CTA_TIMER_WORKERS} CTAs"
        )
    dispatch = "\n".join(
        f"            if variant == {runner_id}:\n" + "\n".join("    " + line for line in lines)
        for runner_id, _, lines in runner_variants.values()
    )
    # Per-call boundary addresses arrive through a host-mapped ring of
    # buffers whose base never changes; the call counter selects this call's
    # entry, whose first word repeats the counter. The first CTA checks it and
    # copies the addresses into the device table before the first barrier.
    # A stamp of -1 marks a CUDA graph's ring, whose addresses fit every call.
    ring_words = len(dynamic_slots) + 1
    stall_timeout = stall_timeout_ns()
    stall_inject = stall_inject_phase()
    if stall_inject is not None:
        # The k-th phase barrier -> the phase that closes with it.
        stall_inject = [phase.index for phase in plan.phases if phase.barrier is not None][stall_inject]
    stall_publish = (
        "    if _block == 0:\n"
        "        if _thread == 0:\n"
        "            publish_stall_base(phase_counters, _stall_base)\n"
        if stall_timeout > 0
        else ""
    )
    dynamic_writes = (
        "        _dynamic = cute.make_tensor(\n"
        "            cute.make_ptr(\n"
        "                cutlass.Int64,\n"
        f"                dynamic_addresses + cutlass.Int64(_call % {DYNAMIC_ADDRESS_RING}) * {ring_words * 8},\n"
        "                cute.AddressSpace.gmem,\n"
        "                assumed_align=8,\n"
        "            ),\n"
        f"            cute.make_layout({ring_words}),\n"
        "        )\n"
        "        trap_if((_dynamic[0] != cutlass.Int64(_call)) & (_dynamic[0] != cutlass.Int64(-1)))\n"
        + (
            f"        for _position in cutlass.range(_thread, {len(dynamic_slots)}, {plan.launch.threads_per_block}, unroll=1):\n"
            "            addresses[dynamic_slot_table[_position]] = _dynamic[_position + 1]\n"
            if dynamic_slots
            else ""
        )

    )
    # Phase barriers follow the address barrier and, with GEMMs, the
    # descriptor barrier.

    stream_argument = "    stream_table: cute.Tensor,\n" if streaming else ""
    gemm_kernel_arguments = (
        "    gemm_table: cute.Tensor,\n"
        f"{stream_argument}"
        "    tma_descriptors: cute.Tensor,\n"
        "    gemm_values,\n"
        if has_gemm
        else ""
    )
    gemm_launch_arguments = (
        "    gemm_table: cute.Tensor,\n"
        f"{stream_argument}"
        "    tma_descriptors: cute.Tensor,\n"
        "    gemm_tensors: tuple[tuple[cute.Tensor, cute.Tensor, cute.Tensor], ...],\n"
        "    max_active_clusters: cutlass.Constexpr,\n"
        if has_gemm
        else ""
    )
    gemm_prepare = (
        "    gemm_values = (\n"
        + "".join(
            f"        _gemm_{variant}.prepare(\n"
            f"            gemm_tensors[{variant}][0],\n"
            f"            gemm_tensors[{variant}][1],\n"
            f"            gemm_tensors[{variant}][2],\n"
            "            max_active_clusters,\n"
            "        ),\n"
            for variant in range(len(gemm_shapes))
        )
        + "    )\n"
        if has_gemm
        else ""
    )
    gemm_scratch = (
        (
            "    _gemm_scratch = cutlass.utils.SmemAllocator().allocate(\n"
            "        union_shared_storage(("
            + "".join(f"_gemm_{variant}, " for variant in range(len(gemm_shapes)))
            + "))\n"
            "    )\n"
            if paired_variants or ring_pages
            else "    _gemm_scratch = cutlass.utils.SmemAllocator().allocate(_gemm_0.shared_storage)\n"
        )
        if has_gemm
        else ""
    )
    gemm_builds = "\n".join(
        f"            if gemm_variant == {variant}:\n"
        f"                build_table_descriptors(\n"
        f"                    gemm_values[{variant}],\n"
        "                    tma_descriptors,\n"
        "                    gemm,\n"
        "                    addresses[gemm_table[gemm, 1]],\n"
        "                    addresses[gemm_table[gemm, 2]],\n"
        "                    addresses[gemm_table[gemm, 3]],\n"
        f"                    {shape[:3]!r},\n"
        f"                    {shape[3]!r},\n"
        f"                    {gemm_splits[variant]!r},\n"
        + (
            f"                    addresses[gemm_table[gemm, {GEMM_SECOND_A_COLUMN}] - 1],\n"
            if variant in paired_variants
            else ""
        )
        + "                )"
        for variant, shape in enumerate(gemm_shapes)
    )
    # One warp per CTA writes the descriptors of the GEMM phases it owns from
    # this call's addresses, then a barrier publishes them to every TMA user.
    gemm_prologue = (
        "    _warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())\n"
        "    if _warp == 0:\n"
        f"        for gemm in cutlass.range(_block, {len(gemm_rows)}, workers, unroll=1):\n"
        "            gemm_variant = cute.arch.make_warp_uniform(gemm_table[gemm, 0])\n"
        f"{gemm_builds}\n"
        "        cute.nvgpu.cpasync.fence_tma_desc_release()\n"
        + (
            # Shared memory starts undefined: no CTA has streamed stages yet.
            "    if _thread == 0:\n"
            f"        _gemm_scratch.split_arrival[{STREAM_TAG_SLOT}] = cutlass.Int32(0)\n"
            if streaming
            else ""
        )
        + (
            "    if _thread == 0:\n        _gemm_0.ring_init(_gemm_scratch, _stall_base)\n" if ring_pages else ""
        )
        + f"    {_grid_barrier('_ordinal + cutlass.Int32(1)', 'cutlass.Int32(-1)', stall_timeout)}\n"
        + (
            # Descriptors are published: release the producer warp.
            f"    cute.arch.barrier_arrive(barrier_id={PRODUCER_START_BARRIER}, "
            f"number_of_threads={plan.launch.threads_per_block + 32})\n"
            if producer
            else ""
        )
        if has_gemm
        else ""
    )
    gemm_kernel_call = (
        "        gemm_table,\n"
        + ("        stream_table,\n" if streaming else "")
        + "        tma_descriptors,\n        gemm_values,\n"
        if has_gemm
        else ""
    )
    launch_smem = "" if has_gemm else f"        smem={shared_memory_bytes},\n"
    phase_barrier = (
        f"            {_grid_barrier('_ordinal + phase_table[phase, 1]', 'phase', stall_timeout, stall_inject)}\n"
        if not streaming
        else _weight_stream_source(
            f"phase_table[phase, {width + 1}]", sorted(streamed_variants), stall_timeout
        )
    )
    phase_table_width = len(phase_table[0])
    gemm_import = (
        "from megabake.bodies.cuda.sm90.gemm import (\n"
        "    HopperWgmmaGemmPersistentKernel,\n"
        "    build_table_descriptors,\n"
        "    table_descriptors,\n"
        "    union_shared_storage,\n"
        ")\n"
        if has_gemm
        else ""
    )
    source = f"""import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass._mlir.dialects import llvm
from cutlass.cute.tensor import _Tensor

from megabake.runtime import (
    checked_grid_barrier,
    checked_grid_wait,
    counting_grid_arrive,
    counting_grid_barrier,
    counting_grid_wait,
    poll_at_least,
    prefetch_l1,
    publish_stall_base,
    stall_base_from_counters,
    prefetch_tensormap,
    trap_if,
)
{gemm_import}
{chr(10).join(epilogue_definitions)}
{chr(10).join(gemm_definitions)}


@cute.jit
def _slot_address(addresses: cute.Tensor, phase: cutlass.Int32, column: cutlass.Constexpr):
    # A phase's resolved slot address (see _slot).
    return addresses[{len(tensor_arguments)} + phase * {slot_width} + column]


@cute.jit
def _slot(
    addresses: cute.Tensor,
    phase_table: cute.Tensor,
    phase: cutlass.Int32,
    column: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    layout: cutlass.Constexpr,
    alignment: cutlass.Constexpr,
):
    # The host appends each phase's resolved slot addresses to the table.
    address = addresses[{len(tensor_arguments)} + phase * {slot_width} + column]
    if cutlass.const_expr(dtype is cutlass.Boolean):
        # Boolean tensors are byte-backed, matching DLPack-bound tensors.
        storage = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int8, address, cute.AddressSpace.gmem, assumed_align=alignment
            ),
            layout,
        )
        return _Tensor(storage, dtype=cutlass.Boolean)
    return cute.make_tensor(
        cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=alignment),
        layout,
    )


{shared_source}

@cute.kernel
def megabake_kernel(
    ready: cute.Tensor,
    workers: cutlass.Int32,
    addresses: cute.Tensor,
    phase_table: cute.Tensor,
    work_iterators: tuple[cute.Tensor, ...],
    dynamic_addresses: cutlass.Int64,
    dynamic_slot_table: cute.Tensor,
    phase_floats: cute.Tensor,
    phase_counters: cute.Tensor,
{gemm_kernel_arguments}):
    _, _, _block = cute.arch.block_idx()
    _thread, _, _ = cute.arch.thread_idx()
    # Every CTA reads the call counter before the first barrier; block 0
    # advances it after that barrier, for the next launch on this stream.
    _call = ready[workers + {call_counter_offset()}]
    _ordinal = _call * cutlass.Int32({barrier_epochs})
    # Stall records (X0.3) go to the pinned tail of the dynamic-address ring.
    _stall_base = dynamic_addresses + cutlass.Int64({DYNAMIC_ADDRESS_RING * ring_words * 8})
{_prelude_allocations(scratch_prelude) + gemm_scratch + _ROLE_SPLIT + _prelude_rest(scratch_prelude) if producer else scratch_prelude + chr(10) + gemm_scratch}
{"".join(body_definitions)}
    # Publish this call's boundary addresses before any phase reads them.
    if _block == 0:
{dynamic_writes}
{stall_publish}    {_grid_barrier('_ordinal', 'cutlass.Int32(-1)', stall_timeout)}
    if _block == 0:
        if _thread == 0:
            ready[workers + {call_counter_offset()}] = _call + cutlass.Int32(1)
{gemm_prologue}{_phase_timer_stamp(len(plan.phases), "    ")}{_cta_timer_stamp("0", "    ")}
    for phase in cutlass.range({len(plan.phases)}, unroll=1):
        if _thread == 0:
            if phase + 1 < {len(plan.phases)}:
                prefetch_l1(phase_table.iterator + (phase + 1) * {phase_table_width})
                prefetch_l1(addresses.iterator + {len(tensor_arguments)} + (phase + 1) * {slot_width})
        # Table entries are the same on every thread; saying so lets ptxas
        # treat the dispatch branches around the GEMM bodies as warp-uniform.
        variant = cute.arch.make_warp_uniform(phase_table[phase, 0])
        _rotation = cute.arch.make_warp_uniform(phase_table[phase, 2])
        _units = cute.arch.make_warp_uniform(phase_table[phase, {width}])
{_cta_timer_stamp(f"{1 + 2 * len(plan.phases)} + phase", "        ") if 3 * len(plan.phases) + 1 <= CTA_TIMER_STRIDE else ""}        _rotated = (_block + workers - _rotation) % workers
        _workers = workers
        if _rotation < 0:
            # A recompute phase: this CTA runs every tile itself.
            _rotated = cutlass.Int32(0)
            _workers = cutlass.Int32(1)
        if _rotated < _units:
{dispatch}
        if phase_table[phase, 2] < 0:
            # Later TMA reads in this interval see the generic-proxy writes.
            cute.arch.fence_proxy("async.global")
{_descriptor_prefetch(runner_variants, paired_variants, len(plan.phases))}{_cta_timer_stamp("1 + 2 * phase", "        ", synchronize=True)}        if phase_table[phase, 1] != 0:
{phase_barrier}{_phase_timer_stamp("phase", "            ")}            # Every reader of this interval's KV valid length has finished.
            if phase_table[phase, 3] != 0:
                if _block == 0:
                    if _thread == 0:
                        _length = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Int32,
                                addresses[phase_table[phase, 3] - 1],
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(1),
                        )
                        _length[0] = _length[0] + cutlass.Int32(1)
        else:
            # Independent phases share a barrier interval; the CTA still
            # synchronizes before the next body reuses its shared scratch.
            cute.arch.sync_threads()
{_cta_timer_stamp("2 + 2 * phase", "        ")}

@cute.jit
def launch(
    ready: cute.Tensor,
    workers: cutlass.Int32,
    addresses: cute.Tensor,
    phase_table: cute.Tensor,
    work_iterators: tuple[cute.Tensor, ...],
    dynamic_addresses: cutlass.Int64,
    dynamic_slot_table: cute.Tensor,
    phase_floats: cute.Tensor,
    phase_counters: cute.Tensor,
{gemm_launch_arguments}    stream: cuda.CUstream,
):
{gemm_prepare}    megabake_kernel(
        ready,
        workers,
        addresses,
        phase_table,
        work_iterators,
        dynamic_addresses,
        dynamic_slot_table,
        phase_floats,
        phase_counters,
{gemm_kernel_call}    ).launch(
        grid=[1, 1, workers],
        block=[{launch_threads}, 1, 1],
        cooperative=True,
{launch_smem}        stream=stream,
    )
"""
    if producer:
        source = _split_roles(
            source,
            plan.launch.threads_per_block // 32,
            _producer_source(
                plan.launch.threads_per_block + 32,
                len(plan.phases),
                width,
                {
                    entry[0]: key[1]
                    for key, entry in runner_variants.items()
                    if len(key) == 2 and key[0] == "g0_range"
                },
            ),
        )
    launch_metadata = LaunchMetadata(
        "megabake_kernel",
        "launch",
        plan.launch.grid_size,
        plan.launch.threads_per_block,
        True,
        shared_memory_bytes,
        plan.launch.architecture,
        (
            "ready",
            "workers",
            "addresses",
            "phase_table",
            "work_iterators",
            "dynamic_addresses",
            "dynamic_slot_table",
            "phase_floats",
            "phase_counters",
            *(
                (
                    "gemm_table",
                    *(("stream_table",) if streaming else ()),
                    "tma_descriptors",
                    "gemm_tensors",
                    "max_active_clusters",
                )
                if has_gemm
                else ()
            ),
            "stream",
        ),
        tuple(work_iterators),
        tensor_arguments,
        tuple(phase_records),
        launch_policy="megabake_cooperative_table",
        body_variant_count=len(variants),
        phase_runner_count=len(runner_variants),
        source_bytes=len(source.encode("utf-8")),
        phase_table=phase_table,
        dynamic_slots=dynamic_slots,
        resolved_slots=resolved_slots,
        barrier_epochs=barrier_epochs,
        gemm_table=tuple(gemm_rows),
        stream_table=tuple(stream_rows) if streaming else (),
        gemm_template_phases=tuple(gemm_template_phases),
        gemm_splits=tuple(gemm_splits),
        phase_float_elements=float_elements,
        phase_counter_count=counter_count,
        producer_warp=producer,
    )
    return source, launch_metadata, _source_map(source, plan, phase_records)


# Table GEMM rows: variant, A, B, C, bias + 1, the second A slot + 1, the
# fused epilogue kind (0: plain), then the slots of that epilogue's tensors.
GEMM_SECOND_A_COLUMN = 5
GEMM_EPILOGUE_KIND_COLUMN = 6
GEMM_EPILOGUE_COLUMN = 7
# A GEMM phase's slot region (phase table columns from 6): the GEMM row index,
# the epilogue kind, then the resolved slots of C, the bias and the epilogue
# tensors.
GEMM_PHASE_KIND_COLUMN = 1
GEMM_PHASE_SLOT_COLUMN = 2




def _read_only_input(plan: KernelPlan, argument: TensorArgument) -> bool:
    """Whether ``argument``'s storage is a launch input that no region of
    the plan writes (weights), so it can be read before any barrier."""
    graph = plan.graph
    value = graph.values.get(argument.value_id)
    if value is None or argument.kind == "state":
        return False
    root_id = value.storage_relation.storage_id or argument.value_id
    root = graph.values.get(root_id)
    if root is None or root.producer is not None or root.id not in graph.inputs:
        return False
    return not any(
        (graph.values[output].storage_relation.storage_id or output) == root_id
        for selection in plan.selection.selections
        for output in selection.outputs
    )


def _weight_stream_schedule(
    plan: KernelPlan,
    rotations: Sequence[int],
    units: Mapping[int, int],
    gemm_of_phase: Mapping[int, int],
    gemm_variants: Sequence[int],
    streamable: set[int],
    stages: int,
) -> tuple[list[tuple[int, ...]], dict[int, int]]:
    """Which GEMM's pipeline stages each CTA loads at which grid barrier.

    At the barrier closing an interval, every CTA can load the weights of
    its first tile of the next interval: CTAs that finish early overlap the
    loads with the wait, and every CTA with the norms folded into the next
    interval. Only GEMM bodies use the GEMM's shared memory, so a CTA streams
    for the first GEMM phase of the interval it has tiles in, when that
    GEMM's weight is streamable. Rotations and units are the phase table's,
    so the CTA's tiles follow from its block index.

    Returns the stream table (row 0 all zero; see
    ``LaunchMetadata.stream_table``) and {closing phase: stream-table row}.
    """
    workers = plan.launch.grid_size
    rows: list[tuple[int, ...]] = [(0,) * (3 * workers)]
    issues: dict[int, int] = {}
    if stages <= 0 or not streamable:
        return rows, issues
    intervals: list[list[int]] = [[]]
    for phase in plan.phases:
        intervals[-1].append(phase.index)
        if phase.barrier is not None:
            intervals.append([])
    for previous, members in zip(intervals, intervals[1:]):
        entries = [0] * (3 * workers)
        for cta in range(workers):
            for index in members:
                if index not in gemm_of_phase:
                    continue
                rotation = rotations[index]
                if rotation < 0:
                    # Recomputed on every CTA, so it takes every CTA's stages.
                    break
                rotated = (cta + workers - rotation) % workers
                if rotated >= units[index]:
                    continue
                gemm = gemm_of_phase[index]
                if gemm in streamable:
                    entries[3 * cta : 3 * cta + 3] = (gemm + 1, gemm_variants[gemm], rotated)
                break
        if any(entries):
            issues[previous[-1]] = len(rows)
            rows.append(tuple(entries))
    return rows, issues


_ROLE_SPLIT = "    # MEGABAKE_ROLE_SPLIT\n"


# setmaxnreg budgets (X1.1): 256 body threads x 224 + 128 producer x 64 =
# all 65,536 registers of the SM (the producer starves at 40).
BODY_REGISTERS = 224
PRODUCER_REGISTERS = 64


def _prelude_allocations(prelude: str) -> str:
    """The shared-memory allocations of a scratch prelude. With a producer
    warp they run before the role split, so both roles see one layout."""
    return "".join(
        line + "\n" for line in prelude.splitlines()
        if line.startswith(("    _worker_scratch = ", "    _scratch_"))
    )


def _prelude_rest(prelude: str) -> str:
    return "".join(
        line + "\n" for line in prelude.splitlines()
        if line.strip() and not line.startswith(("    _worker_scratch = ", "    _scratch_"))
    )


def _producer_source(
    handshake_threads: int, phases: int, units_column: int, gemm_runners: Mapping[int, int]
) -> str:
    """The producer warp's program (X1.1): after the prologue, walk the phase
    table as the body warps do and, for every GEMM phase this CTA has tiles
    in, load their weights into the ring (produce_table). It waits only for
    free ring pages, never for a grid barrier."""
    dispatch = "".join(
        f"                        if variant == {runner_id}:\n"
        f"                            _page, _lap = _gemm_{gemm_variant}.produce_table(\n"
        f"                                gemm_values[{gemm_variant}],\n"
        "                                _gemm_scratch,\n"
        "                                table_descriptors(tma_descriptors, phase_table[phase, 6]),\n"
        "                                _rotated,\n"
        "                                _page,\n"
        "                                _lap,\n"
        "                            )\n"
        for runner_id, gemm_variant in sorted(gemm_runners.items())
    )
    return (
        f"            cute.arch.barrier(barrier_id={PRODUCER_START_BARRIER}, "
        f"number_of_threads={handshake_threads})\n"
        "            _page = cutlass.Int32(0)\n"
        "            _lap = cutlass.Int32(0)\n"
        f"            for phase in cutlass.range({phases}, unroll=1):\n"
        "                variant = cute.arch.make_warp_uniform(phase_table[phase, 0])\n"
        "                _rotation = cute.arch.make_warp_uniform(phase_table[phase, 2])\n"
        f"                _units = cute.arch.make_warp_uniform(phase_table[phase, {units_column}])\n"
        "                _rotated = (_block + workers - _rotation) % workers\n"
        "                if _rotation >= 0:\n"
        "                    if _rotated < _units:\n"
        + dispatch
    )


def _split_roles(source: str, producer_warp: int, producer: str) -> str:
    """Run everything after the role split in megabake_kernel on the body
    warpgroups only. The warpgroup after them lowers its register budget,
    and its first warp runs ``producer``; the other three warps idle out."""
    start = source.index(_ROLE_SPLIT)
    end = source.index("\n\n@cute.jit\ndef launch(")
    body = source[start + len(_ROLE_SPLIT) : end]
    return (
        source[:start]
        + "    _role = cute.arch.make_warp_uniform(cute.arch.warp_idx())\n"
        + f"    if _role >= {producer_warp}:\n"
        + f"        cute.arch.setmaxregister_decrease({PRODUCER_REGISTERS})\n"
        + f"        if _role == {producer_warp}:\n"
        + producer
        + "    else:\n"
        + f"        cute.arch.setmaxregister_increase({BODY_REGISTERS})\n"
        + "\n".join(("    " + line) if line.strip() else line for line in body.split("\n"))
        + source[end:]
    )


def _grid_barrier(
    ordinal: str, phase: str, stall_timeout: int, inject: int | None = None, wait_only: bool = False
) -> str:
    """One grid barrier (or its waiting half) as generated source. With stall
    diagnostics on (X0.3) its wait reports a timeout; ``inject`` names a phase
    whose barrier the last CTA does not arrive at (test hook)."""
    if stall_timeout <= 0:
        return (f"counting_grid_wait(ready, workers, {ordinal})" if wait_only
                else f"counting_grid_barrier(ready, workers, {ordinal})")
    common = f"ready, workers, {ordinal}, _stall_base, _call, {phase}, {stall_timeout}"
    if wait_only:
        return f"checked_grid_wait({common})"
    arrive = (
        f", (_block != workers - cutlass.Int32(1)) | ({phase} != cutlass.Int32({inject}))"
        if inject is not None
        else ""
    )
    return f"checked_grid_barrier({common}{arrive})"


def _weight_stream_source(stream_row: str, variants: Sequence[int], stall_timeout: int = 0) -> str:
    """A phase's closing grid barrier that streams weights (see
    ``_weight_stream_schedule``): after this CTA arrives, warp 0 loads the
    stages of the GEMM its stream-table entry names, then the CTA waits."""
    dispatch = "".join(
        f"                        if _target_variant == {variant}:\n"
        f"                            _gemm_{variant}.preload_table(\n"
        f"                                gemm_values[{variant}],\n"
        "                                _gemm_scratch,\n"
        "                                table_descriptors(tma_descriptors, _target - 1),\n"
        "                                _target_block,\n"
        "                                _target,\n"
        "                            )\n"
        for variant in variants
    )
    return (
        "            counting_grid_arrive(ready, workers)\n"
        f"            _stream = cute.arch.make_warp_uniform({stream_row})\n"
        "            if _stream != 0:\n"
        "                if _warp == 0:\n"
        "                    _target = cute.arch.make_warp_uniform(stream_table[_stream, 3 * _block])\n"
        "                    if _target != 0:\n"
        "                        _target_variant = cute.arch.make_warp_uniform(\n"
        "                            stream_table[_stream, 3 * _block + 1]\n"
        "                        )\n"
        "                        _target_block = cute.arch.make_warp_uniform(\n"
        "                            stream_table[_stream, 3 * _block + 2]\n"
        "                        )\n"
        f"{dispatch}"
        f"            {_grid_barrier('_ordinal + phase_table[phase, 1]', 'phase', stall_timeout, wait_only=True)}\n"
    )


def _descriptor_prefetch(runner_variants, paired_variants, phases: int) -> str:
    """Before this phase's barrier, thread 0 prefetches the next phase's TMA
    weight and activation descriptors when it is a table GEMM, so its first
    TMA load does not fetch them from global memory after the barrier."""
    gemm_runners = {
        runner_id: variant in paired_variants
        for key, (runner_id, _, _) in runner_variants.items()
        if key[0] == "g0_range"
        for variant in (key[1],)
    }
    if not gemm_runners:
        return ""
    stride = TABLE_DESCRIPTORS_PER_GEMM * 16
    lines = [
        "        if _thread == 0:",
        f"            if phase + 1 < {phases}:",
        "                _next = phase_table[phase + 1, 0]",
    ]
    for runner_id, paired in sorted(gemm_runners.items()):
        operands = (0, 1, 3) if paired else (0, 1)
        lines.append(f"                if _next == {runner_id}:")
        lines.append(
            "                    _descriptors = tma_descriptors.iterator.toint()"
            f" + cutlass.Int64(phase_table[phase + 1, 6]) * {stride * 8}"
        )
        for operand in operands:
            lines.append(
                f"                    prefetch_tensormap(_descriptors + {operand * 128})"
            )
    return "\n".join(lines) + "\n"


def _cta_timer_stamp(offset: str, indent: str, *, synchronize: bool = False) -> str:
    """This CTA records a stamp at ``offset`` within its own timer row.

    ``synchronize`` first waits for every thread of the CTA, so a stamp taken
    after a body covers all of its warps."""
    if not CTA_TIMER:
        return ""
    return (
        (f"{indent}cute.arch.sync_threads()\n" if synchronize else "")
        + f"{indent}if _thread == 0:\n"
        f"{indent}    ready[workers + {2 + GRID_TIMER_SLOTS} + _block * {CTA_TIMER_STRIDE} + {offset}] = (\n"
        f"{indent}        cute.arch.globaltimer_lo()\n"
        f"{indent}    )\n"
    )


def _phase_timer_stamp(slot: int | str, indent: str) -> str:
    """Block 0 records when every CTA has passed this barrier."""
    if not PHASE_TIMER_SLOTS:
        return ""
    if isinstance(slot, int) and slot >= GRID_TIMER_SLOTS:
        raise UnsupportedCodegen(
            f"phase timer holds {GRID_TIMER_SLOTS} stamps, plan needs {slot + 1}"
        )
    return (
        f"{indent}if _block == 0:\n"
        f"{indent}    if _thread == 0:\n"
        f"{indent}        ready[workers + 2 + {slot}] = cute.arch.globaltimer_lo()\n"
    )


def _value_layout(plan: KernelPlan, value_id: str) -> CuTeLayout | None:
    value = plan.graph.values[value_id]
    if value.shape is None or value.strides is None or value.dtype is None:
        if value.users:
            raise UnsupportedCodegen(
                f"table launch needs a static layout for {value_id}"
            )
        # Unused non-tensor inputs such as flags keep an empty slot.
        return None
    return CuTeLayout(
        value.dtype, value.shape, value.strides, DTYPE_BYTES[value.dtype]
    )


def _layout_key(layout: Any) -> tuple[Any, ...]:
    return (layout.dtype, tuple(layout.shape), tuple(layout.strides))


def _slot_tensor_code(column: int, layout: Any) -> str:
    dtype = _CUTLASS_TYPES[layout.dtype]
    shape = tuple(layout.shape)
    strides = tuple(layout.strides)
    layout_code = (
        "cute.make_layout(1)"
        if not shape
        else f"cute.make_layout({shape!r}, stride={strides!r})"
    )
    alignment = DTYPE_BYTES[layout.dtype]
    return (
        f"_slot(addresses, phase_table, phase, {column}, {dtype}, "
        f"{layout_code}, {alignment})"
    )


def _representative_scratch_plan(plan: KernelPlan, phases: Sequence[Any]) -> KernelPlan:
    indices = {phase.index for phase in phases}
    worker = plan.storage.worker_scratch[0]
    representative_worker = replace(
        worker,
        phases=tuple(
            layout
            for layout, phase in zip(worker.phases, plan.phases, strict=True)
            if phase.index in indices
        ),
    )
    return replace(
        plan,
        phases=tuple(phase for phase in plan.phases if phase.index in indices),
        storage=replace(
            plan.storage,
            worker_scratch=(representative_worker, *plan.storage.worker_scratch[1:]),
        ),
    )


def _phase_runner_source(
    variant_id: int, calls: list[tuple[int, list[str]]]
) -> str:
    ordered_calls = sorted(calls)

    def dispatch(
        entries: list[tuple[int, list[str]]], indent: str
    ) -> list[str]:
        if len(entries) == 1:
            return [f"{indent}if phase_index == {entries[0][0]}:", *(
                f"{indent}    {line.strip()}" for line in entries[0][1]
            )]
        midpoint = len(entries) // 2
        return [
            f"{indent}if phase_index < {entries[midpoint][0]}:",
            *dispatch(entries[:midpoint], indent + "    "),
            f"{indent}else:",
            *dispatch(entries[midpoint:], indent + "    "),
        ]

    lines = [
        "    @cute.jit",
        f"    def _run_variant_{variant_id}(phase_index: cutlass.Int32):",
        *(
            ordered_calls[0][1]
            if len(ordered_calls) == 1
            else dispatch(ordered_calls, "        ")
        ),
    ]
    return "\n".join(lines) + "\n"


def _event_worker_loop(
    plan: KernelPlan,
    phase_runner_symbols: Mapping[int, str],
    event_schedule: EventSchedule | None,
) -> str:
    if event_schedule is None:
        return f'''    cooperative_worker_loop(
        ready,
        workers,
        epoch,
        (
{("," + chr(10)).join([f"            ({phase.index}, {phase_runner_symbols[phase.index]}, _no_op, {f'_publish_phase_{phase.index}' if phase.barrier is not None else ('_cta_sync' if phase.index + 1 < len(plan.phases) else '_ignore_epoch')})" for phase in plan.phases] + [f"            ({len(plan.phases)}, _run_terminal, _no_op, _ignore_epoch)"])}
        ),
    )
'''
    event_regions = {
        event_schedule.task_for(task_id).region_id
        for event in event_schedule.events
        for task_id in event.producer_task_ids
    }
    entries = tuple(
        (
            phase.index,
            f"_wait_phase_{phase.index}" if phase.dependencies else "_ignore_epoch",
            phase_runner_symbols[phase.index],
            f"_publish_phase_{phase.index}"
            if phase.region_id in event_regions
            else "_no_op",
        )
        for phase in plan.phases
    )
    return f'''    cooperative_event_worker_loop(
        ready,
        workers,
        epoch,
        (
{("," + chr(10)).join(f"            ({entry[0]}, {entry[1]}, {entry[2]}, {entry[3]})" for entry in entries)}
        ),
    )
'''


def _widened_selection(
    selection: SelectedBody, threads: int
) -> SelectedBody:
    """``selection`` with its body run by CTAs of the launch's ``threads``."""
    body = selection.body
    if body.participants.participating_threads == threads:
        return selection
    return replace(
        selection, body=provider(body.provider).widen(selection.program, body, threads)
    )


def _generate_device_body(
    selection: SelectedBody,
    *,
    defer_length_publish: bool = False,
    workers: int | None = None,
):
    """``workers`` is given for table launches, whose bodies may split work
    across CTAs through phase scratch."""
    return provider(selection.body.provider).emit(
        selection.program,
        selection.body,
        defer_length_publish=defer_length_publish,
        workers=workers,
    )


def _generate_conversion_body(
    conversion: ConversionBody, base_body: BodyConfig, selection: SelectedBody
):
    axes = base_body.layouts.inputs[conversion.value_id].axes
    if any(isinstance(axis, int) for axis in axes):
        raise UnsupportedCodegen(
            "layout conversion with constant broadcast axes is not supported"
        )
    if conversion.source_layout.dtype != conversion.target_layout.dtype:
        raise UnsupportedCodegen("layout conversion cannot change dtype")

    domains = tuple(
        IterationDomain(axis, size)
        for axis, size in zip(
            axes,
            conversion.source_layout.shape,
            strict=True,
        )
    )
    load = IndexedLoad(
        "load:0",
        conversion.value_id,
        axes,
        conversion.source_layout.strides,
        conversion.source_layout.dtype,
    )
    converted_id = (
        f"{conversion.value_id}:converted:{conversion.target_region_id}"
    )
    program = TensorProgram(
        family="copy",
        source_target="megabake.layout_conversion",
        decomposition="explicit",
        iteration_domains=domains,
        reduction_axes=(),
        loads=(load,),
        init=None,
        combine=Load(load.id, load.dtype),
        post=None,
        accumulator_dtype=None,
        store=TensorStore(
            converted_id,
            conversion.target_layout.shape,
            conversion.target_layout.dtype,
            conversion.target_layout.strides,
            True,
        ),
        predicates=tuple(
            AxisPredicate(domain.axis, domain.size) for domain in domains
        ),
    )
    output_axes = tuple(domain.axis for domain in domains)
    threads = base_body.participants.participating_threads
    config = replace(
        base_body,
        semantic=replace(
            base_body.semantic,
            operation_family="copy",
            source_op_ids=(f"conversion:{conversion.value_id}",),
            access_effects={
                "reads": (conversion.value_id,),
                "writes": (converted_id,),
                "owns_storage": True,
            },
        ),
        layouts=LayoutContract(
            {
                conversion.value_id: TensorLayout(
                    conversion.source_layout.dtype,
                    axes,
                    conversion.source_layout.strides,
                    conversion.source_layout.alignment_bytes,
                    conversion.source_layout.shape,
                )
            },
            {
                converted_id: TensorLayout(
                    conversion.target_layout.dtype,
                    axes,
                    conversion.target_layout.strides,
                    conversion.target_layout.alignment_bytes,
                    conversion.target_layout.shape,
                )
            },
        ),
        tiles=replace(
            base_body.tiles,
            cta_shape=(threads,) + (1,) * max(len(output_axes) - 1, 0),
            work_partition="one output element per thread",
            reduction_tile=(),
        ),
        pipeline=replace(
            base_body.pipeline,
            barriers=(),
            async_operations=(),
        ),
        scratch=replace(
            base_body.scratch,
            owner="none",
            size_bytes=0,
        ),
        completion=replace(
            base_body.completion,
            outputs_ready="copied elements are stored when run returns",
            scratch_reusable="scratch is reusable when run returns",
        ),
    )
    return emit_map_body(program, config)


def _nested_body_source(source: str, symbol: str) -> str:
    parsed = ast.parse(source)
    function = next(
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run"
    )
    source_lines = source.splitlines(keepends=True)
    decorator_line = min(
        (decorator.lineno for decorator in function.decorator_list),
        default=function.lineno,
    )
    body_source = "".join(
        source_lines[decorator_line - 1 : function.end_lineno]
    )
    body_source = body_source.replace("@cute.kernel", "@cute.jit", 1)
    body_source = body_source.replace("def run(", f"def {symbol}(", 1)
    body_source = body_source.replace(
        "    work_iterator: cute.Tensor,",
        (
            "        work_iterator: cute.Tensor,\n"
            "        block: cutlass.Int32,\n"
            "        thread: cutlass.Int32,"
            "\n        workers: cutlass.Int32,"
        ),
        1,
    )
    body_source = body_source.replace(
        "    block, _, _ = cute.arch.block_idx()\n", "", 1
    )
    body_source = body_source.replace(
        "    thread, _, _ = cute.arch.thread_idx()\n", "", 1
    )
    body_source = body_source.replace(
        "    workers, _, _ = cute.arch.grid_dim()\n", "", 1
    )
    body_source = body_source.replace("    scratch: cute.Tensor,", "    scratch,", 1)
    body_source = body_source.replace("scratch[block,", "scratch[")
    return textwrap.indent(body_source, "    ") + "\n"


def _affine_work_iterator(body_source: str, step: int, tile_count: int) -> str:
    """Compute work-iterator entries instead of loading them.

    Table launches fill every work iterator with ``arange(0, tiles * step,
    step)``, and a body's first data access depends on its tile's entry: an
    L2 round trip per tile on the critical path. The tensor stays bound."""
    return body_source.replace(
        "work_iterator[tile]", f"(cutlass.Int64(tile) * {step})"
    ).replace("cute.size(work_iterator)", str(tile_count))


def _generic_scratch_bytes(
    plan: KernelPlan, scratch_elements: Mapping[int, int]
) -> int:
    sizes = []
    for phase in plan.phases:
        if phase.body.provider == "g0_range":
            continue
        offset = 0
        for _ in phase.body.pipeline.barriers:
            offset = _round_up(offset + 8, 8)
        for _ in phase.body.pipeline.descriptors:
            offset = _round_up(offset + 128, 128)
        alignment = max(1, phase.body.scratch.alignment_bytes)
        body_bytes = _phase_scratch_bytes(plan, phase, scratch_elements)
        offset = _round_up(offset, alignment) + body_bytes
        sizes.append(_round_up(offset, alignment))
    return max(sizes, default=0)


def _shared_scratch_source(
    plan: KernelPlan, scratch_elements: Mapping[int, int]
) -> str:
    worker = plan.storage.worker_scratch[0]
    phase_structs: list[str] = []
    scratch_phase_indices: list[int] = []
    generic_phases = tuple(
        phase for phase in plan.phases if phase.body.provider != "g0_range"
    )
    if not any(_phase_has_scratch_fields(phase) for phase in generic_phases):
        return ""
    for phase_layout, phase in zip(worker.phases, plan.phases, strict=True):
        if phase.body.provider == "g0_range":
            continue
        fields: list[str] = []
        for index, name in enumerate(phase.body.pipeline.barriers):
            fields.append(
                f"    barrier_{index}: cute.struct.Align[\n"
                f"        cute.struct.MemRange[cutlass.Int64, 1], 8\n"
                f"    ]"
            )
        descriptor_offsets = sorted(
            phase_layout.descriptor_offsets.items(), key=lambda item: item[1]
        )
        for index, (name, offset) in enumerate(descriptor_offsets):
            next_offset = (
                descriptor_offsets[index + 1][1]
                if index + 1 < len(descriptor_offsets)
                else phase_layout.body_data_offset_bytes
            )
            size = max(0, next_offset - offset)
            if size:
                fields.append(
                    f"    descriptor_{index}: cute.struct.Align[\n"
                    f"        cute.struct.MemRange[cutlass.UInt8, {size}], 128\n"
                    f"    ]"
                )
        if phase.index in scratch_elements or phase.body.scratch.size_bytes:
            body_scratch_elements = _phase_scratch_elements(
                plan, phase, scratch_elements
            )
        else:
            body_scratch_elements = 0
        if body_scratch_elements:
            accumulator_dtype = next(
                selection.program.accumulator_dtype
                for selection in plan.selection.selections
                if selection.region_id == phase.region_id
            )
            cutlass_type = _CUTLASS_TYPES[accumulator_dtype]
            fields.append(
                f"    body: cute.struct.Align[\n"
                f"        cute.struct.MemRange[{cutlass_type}, {body_scratch_elements}],\n"
                f"        {phase.body.scratch.alignment_bytes}\n"
                f"    ]"
            )
        if fields:
            scratch_phase_indices.append(phase.index)
            phase_structs.append(
                f"@cute.struct\nclass _Phase{phase.index}Scratch:\n"
                + "\n".join(fields)
            )
    if not phase_structs:
        return ""
    union_fields = "\n".join(
        f"    phase_{index}: _Phase{index}Scratch"
        for index in scratch_phase_indices
    )
    return (
        "\n\n".join(phase_structs)
        + "\n\n\n@cute.union\nclass _WorkerScratch:\n"
        + union_fields
    )


def _scratch_prelude(
    plan: KernelPlan, scratch_elements: Mapping[int, int]
) -> str:
    generic_phases = tuple(
        phase for phase in plan.phases if phase.body.provider != "g0_range"
    )
    if not any(_phase_has_scratch_fields(phase) for phase in generic_phases):
        return ""
    lines = [
        "    _worker_scratch = cutlass.utils.SmemAllocator().allocate(_WorkerScratch)"
    ]
    for phase in plan.phases:
        if phase.body.provider == "g0_range":
            continue
        if _phase_has_scratch_fields(phase):
            elements = _phase_scratch_elements(plan, phase, scratch_elements)
            lines.append(
                f"    _scratch_{phase.index} = _worker_scratch.phase_{phase.index}.body"
                f".get_tensor(cute.make_layout({elements}))"
            )
    for phase in plan.phases:
        if phase.body.provider == "g0_range":
            continue
        if _phase_has_scratch_fields(phase):
            for index in range(len(phase.body.pipeline.barriers)):
                lines.extend(
                    (
                        "    if _thread == 0:",
                        f"        _worker_scratch.phase_{phase.index}.barrier_{index}[0] = 0",
                        "    cute.arch.sync_threads()",
                    )
                )
    return "\n".join(lines) + "\n"


def _phase_scratch_elements(
    plan: KernelPlan, phase: Any, scratch_elements: Mapping[int, int]
) -> int:
    accumulator_dtype = next(
        selection.program.accumulator_dtype
        for selection in plan.selection.selections
        if selection.region_id == phase.region_id
    )
    return _phase_scratch_bytes(
        plan, phase, scratch_elements
    ) // DTYPE_BYTES[accumulator_dtype]


def _phase_scratch_bytes(
    plan: KernelPlan, phase: Any, scratch_elements: Mapping[int, int]
) -> int:
    if phase.index in scratch_elements:
        accumulator_dtype = next(
            selection.program.accumulator_dtype
            for selection in plan.selection.selections
            if selection.region_id == phase.region_id
        )
        return scratch_elements[phase.index] * DTYPE_BYTES[accumulator_dtype]
    return phase.body.scratch.size_bytes


def _phase_has_scratch_fields(phase: Any) -> bool:
    return bool(
        phase.body.pipeline.barriers
        or phase.body.pipeline.descriptors
        or phase.body.scratch.size_bytes
    )


def _tensor_name(
    arguments: tuple[TensorArgument, ...],
    value_id: str,
    region_id: str,
    role: str,
    prefer_conversion: bool = False,
) -> str:
    candidates = [
        argument
        for argument in arguments
        if argument.value_id == value_id and region_id in argument.region_ids
    ]
    if not candidates:
        raise UnsupportedCodegen(
            f"region {region_id} has no tensor argument for {value_id}"
        )
    if prefer_conversion:
        conversions = [argument for argument in candidates if argument.kind == "conversion"]
        if conversions:
            return conversions[0].name
    if role == "output":
        outputs = [
            argument
            for argument in candidates
            if argument.kind in {"workspace", "output"}
        ]
        if outputs:
            return outputs[0].name
    return candidates[0].name


def _tensor_argument_index(
    arguments: tuple[TensorArgument, ...], name: str
) -> int:
    return next(
        index for index, argument in enumerate(arguments) if argument.name == name
    )


def _source_map(
    source: str,
    plan: KernelPlan,
    phase_records: tuple[PhaseEmission, ...],
) -> tuple[SourceMapEntry, ...]:
    selected = {
        selection.region_id: selection for selection in plan.selection.selections
    }
    entries: list[SourceMapEntry] = []
    for record in phase_records:
        if record.provider == "g0_range":
            source_path = Path(inspect.getsourcefile(HopperWgmmaGemmPersistentKernel))
            lines, line_start = inspect.getsourcelines(
                HopperWgmmaGemmPersistentKernel.run
            )
            entries.append(
                SourceMapEntry(
                    record.body_symbol,
                    "device_body",
                    line_start,
                    line_start + len(lines) - 1,
                    record.index,
                    record.region_id,
                    record.provider,
                    record.body_name,
                    selected[record.region_id].source_op_ids,
                    str(source_path),
                )
            )
        else:
            lines = _symbol_lines(source, record.body_symbol)
            entries.append(
                SourceMapEntry(
                    record.body_symbol,
                    "device_body",
                    lines[0],
                    lines[1],
                    record.index,
                    record.region_id,
                    record.provider,
                    record.body_name,
                    selected[record.region_id].source_op_ids,
                )
            )
        for symbol in record.conversion_symbols:
            lines = _symbol_lines(source, symbol)
            entries.append(
                SourceMapEntry(
                    symbol,
                    "layout_conversion",
                    lines[0],
                    lines[1],
                    record.index,
                    record.region_id,
                    "generic",
                    "layout_conversion",
                    (f"conversion:{record.region_id}",),
                )
            )
        if record.barrier:
            source_lines = source.splitlines()
            line = next(
                (
                    index
                    for index, line in enumerate(source_lines, 1)
                    if "grid_barrier(ready, workers, epoch + " in line
                    and f"epoch + {record.index})" in line
                ),
                None,
            ) or next(
                index
                for index, line in enumerate(source_lines, 1)
                if "grid_barrier(ready, workers, epoch + phase + " in line
                or "counting_grid_barrier(ready, workers, _ordinal + phase_table[phase, 1])" in line
                or "checked_grid_barrier(ready, workers, _ordinal + phase_table[phase, 1]," in line
                or "counting_grid_arrive(ready, workers)" in line
            )
            entries.append(
                SourceMapEntry(
                    record.barrier,
                    "phase_barrier",
                    line,
                    line,
                    record.index,
                    record.region_id,
                    record.provider,
                    record.body_name,
                    selected[record.region_id].source_op_ids,
                )
            )
    kernel_lines = _symbol_lines(source, "megabake_kernel")
    entries.append(
        SourceMapEntry(
            "megabake_kernel",
            "enclosing_kernel",
            kernel_lines[0],
            kernel_lines[1],
            None,
            None,
            None,
            None,
        )
    )
    launch_lines = _symbol_lines(source, "launch")
    entries.append(
        SourceMapEntry(
            "launch",
            "host_launch_wrapper",
            launch_lines[0],
            launch_lines[1],
            None,
            None,
            None,
            None,
        )
    )
    return tuple(entries)


def _symbol_lines(source: str, symbol: str) -> tuple[int, int]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name == symbol:
            return node.lineno, node.end_lineno
    raise UnsupportedCodegen(f"generated symbol {symbol} is missing")


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        result = {}
        for field in value.__dataclass_fields__:
            result[field] = getattr(value, field)
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)
