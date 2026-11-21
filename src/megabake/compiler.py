from __future__ import annotations

import contextlib

import collections
import copy
import functools
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from torch.utils import _pytree as pytree

from .bodies import SelectionPolicy, TargetFacts
from .bodies.catalog import is_table_gemm, launch_options, table_gemm
from .device import device_tree
from .bodies.cuda.sm90.gemm import TABLE_DESCRIPTORS_PER_GEMM, cute_gemm_tensors, is_swap_ab
from .cache import PlanCache, make_plan_key
from .codegen import GEMM_SECOND_A_COLUMN, GeneratedCuTeModule, emit_cute
from .diagnostics import CompilerDiagnostic, CompilerError
from .frontend import CaptureContext, PreparedFX
from .graph import ComputeGraph, import_fx
from .ir import DataIndex, TensorProgram
from .lowering import canonicalize_semantics, semantic_coverage
from .memory import distinct_workspace, verify_plan
from .plan import (
    BodySelectionPlan,
    KernelPlan,
    PlanMeasurement,
    PlanSearchResult,
    SearchBudget,
    search_plans,
)
from .planner import enumerate_body_selections, with_emitted_staging
from .platform import cuda, cute, cutlass, from_dlpack
from .regions import form_regions
from .runtime import (
    DYNAMIC_ADDRESS_RING,
    consumer_cta_sync,
    register_stall_region,
    stall_region_words,
    EventRuntimeAdmission,
    RuntimeAdmission,
    _stream_is_capturing,
    admit_event_runtime,
    admit_runtime,
    execute_event_plan,
    execute_phase_plan,
    make_ready,
    plan_with_tree,
    reset_ready,
)
from .schedule import (
    EventSchedule,
    _recompute_assignments,
    _worker_assignments,
    make_region_event_schedule,
)


class UnsupportedCompiler(RuntimeError):
    pass


def _source_location(operation: Any) -> str:
    location = (operation.source_location or "").strip()
    lines = location.splitlines()
    return next(
        (line.strip() for line in reversed(lines) if line.lstrip().startswith("File ")),
        lines[-1] if lines else "unknown source location",
    )


def _semantic_diagnostic(graph: ComputeGraph) -> CompilerDiagnostic | None:
    unresolved = tuple(
        operation for operation in graph.ops.values() if operation.tensor_program is None
    )
    if not unresolved:
        return None
    messages = [
        (
            f"{operation.id} ({operation.attributes.get('target')}) "
            f"at {_source_location(operation)}: "
            f"{operation.numerical_rules.get('reason', 'missing lowering rule')}"
        )
        for operation in unresolved
    ]
    return CompilerDiagnostic(
        "semantic-lowering",
        "; ".join(messages),
        tuple(operation.id for operation in unresolved),
        tuple(region.id for region in graph.regions),
        "complete MegaBake semantic coverage",
    )


def _preflight_complete_plan(
    graph: ComputeGraph,
    target: TargetFacts,
    policy: SelectionPolicy | None,
) -> None:
    # One structurally complete plan proves existence; the measured search
    # that follows builds its own candidates.
    result = search_plans(graph, target, policy, SearchBudget(evaluator=None, max_plans=1))
    if any(
        candidate.plan is not None and candidate.status == "unmeasured"
        for candidate in result.candidates
    ):
        return
    semantic = _semantic_diagnostic(graph)
    if semantic is not None:
        raise CompilerError(semantic, safe_for_fallback=True)
    result = search_plans(graph, target, policy, SearchBudget(evaluator=None))
    reasons = [
        f"{candidate.description}: {candidate.reason or candidate.status}"
        for candidate in result.candidates
        if candidate.status != "measured"
    ] + [
        f"{rejection.region_id}/{rejection.provider}: {rejection.reason}"
        for rejection in result.rejections
    ]
    raise CompilerError(
        CompilerDiagnostic(
            "plan-search",
            "; ".join(dict.fromkeys(reasons)) or "no complete plan was produced",
            _source_op_ids_from_graph(graph),
            tuple(region.id for region in graph.regions),
            "one structurally complete plan",
        ),
        safe_for_fallback=True,
    )


@dataclass(frozen=True)
class CompiledResources:
    kernel_name: str
    registers_per_thread: int
    static_shared_memory_bytes: int
    local_memory_bytes_per_thread: int
    dynamic_shared_memory_bytes: int
    active_blocks_per_sm: int
    max_cooperative_workers: int
    workspace_bytes: int
    peak_live_workspace_bytes: int
    body_variant_count: int
    phase_runner_count: int
    compiled_code_bytes: int
    generated_code_bytes: int


@dataclass(frozen=True)
class CompiledKernel:
    plan: KernelPlan
    generated: GeneratedCuTeModule
    compiled: Any
    resources: CompiledResources
    admission: RuntimeAdmission | EventRuntimeAdmission
    event_schedule: EventSchedule | None
    input_order: tuple[str, ...]
    parameter_names: frozenset[str]
    work_iterators: Mapping[str, torch.Tensor]
    max_active_clusters: Any | None = None
    search: Any | None = None


@dataclass(frozen=True)
class CallableTrace:
    workload_kernel_launches: int
    recurring_gpu_operations: tuple[tuple[str, int], ...]


# Above this size, per-phase kernel parameters exceed CUDA's limit, so plans
# use the table-driven launch: tensors come from a device address table and
# TMA GEMM descriptors are written per phase in the kernel prologue.
_TABLE_LAUNCH_OPS = 1_000


# CUDA kernel parameters are limited to 32 KiB; launches passing one CuTe
# tensor per value switch to the table launch well before that.
_TABLE_LAUNCH_PARAMETER_BYTES = 24_000


def _uses_table_launch(graph: ComputeGraph) -> bool:
    """Large graphs, and graphs whose tensors would overflow the kernel
    parameter space even with few ops (wide models with few layers), use
    the table-driven launch. The estimate counts every shaped value as a
    pointer plus a dynamic shape and stride per axis."""
    if len(graph.ops) > _TABLE_LAUNCH_OPS:
        return True
    parameter_bytes = sum(
        8 + 16 * len(value.shape) for value in graph.values.values() if value.shape is not None
    )
    return parameter_bytes > _TABLE_LAUNCH_PARAMETER_BYTES


# A measured candidate must match the captured reference within this
# relative L2 error before it can be selected. Deep table-launched graphs
# accumulate bf16 rounding differences: on the 30-layer pinned SmolLM the
# generic bodies differ from eager bf16 by 1.6% while both sit equally far
# from an fp32 reference, so those graphs use the 2% model-level tolerance.
_PLAN_RELATIVE_L2_LIMIT = 0.01
_TABLE_PLAN_RELATIVE_L2_LIMIT = 0.02
# A candidate outside these limits can still be exact: deep models amplify
# bf16 rounding-order differences. On SmolLM-1.7B's first decode call every
# region matches PyTorch on its own inputs (GEMMs to 3e-4, everything else
# bit for bit), yet the logits end 4.2% from eager. Such a candidate passes
# teacher-forced verification: each region must reproduce PyTorch evaluated
# on the inputs the candidate itself produced.
_REGION_RELATIVE_L2_LIMIT = 0.01


@dataclass
class _TeacherForcedCheck:
    """Verify a candidate region by region, on inputs it computed itself."""

    prepared: PreparedFX
    snapshot: _StateSnapshot

    def failure(self, compiled: CompiledKernel, actual: Any) -> str | None:
        """None when the candidate is verified, otherwise the reason it is not."""
        plan = compiled.plan
        if plan.graph is None or not _uses_table_launch(plan.graph):
            # Only table launches read workspace addresses from device memory,
            # so only they can rerun the same kernel without reuse.
            return "needs a table launch"
        distinct = replace(compiled, plan=distinct_workspace(plan))
        bound = bind_callable(
            distinct,
            self.prepared.context,
            parameter_tensors=_current_parameter_tensors(
                distinct, self.prepared.context, self.snapshot.inputs
            ),
        )
        self.snapshot.reset()
        with torch.no_grad():
            rerun = bound(*self.snapshot.inputs)
        torch.cuda.synchronize()
        # Same code and arithmetic order; only workspace addresses differ.
        if not _outputs_identical(actual, rerun):
            return "outputs change when workspace is not reused"
        values = bound.megabake_runtime_state.values
        originals = {id(private): original for private, original in self.snapshot.pairs}
        nodes = list(self.prepared.graph.graph.nodes)
        placeholders = [node.name for node in nodes if node.op == "placeholder"]
        inputs = {
            name: originals.get(id(value), value)
            for name, value in zip(placeholders, self.snapshot.inputs, strict=True)
        }
        state = {
            name for name, value in inputs.items() if any(value is o for o in originals.values())
        }
        by_name = {node.name: node for node in nodes}
        order = {node.name: index for index, node in enumerate(nodes)}
        for selected in plan.selection.selections:
            reason = _region_failure(
                selected, values, inputs, state, by_name, order, self.prepared.graph
            )
            if reason is not None:
                return reason
        return None


def _region_failure(selected, values, inputs, state, by_name, order, module=None) -> str | None:
    members = sorted(
        (op_id.removeprefix("op:") for op_id in selected.source_op_ids), key=order.__getitem__
    )
    env: dict[str, Any] = {}

    def fetch(argument: Any) -> Any:
        if not isinstance(argument, torch.fx.Node):
            return argument
        name = argument.name
        if name in env:
            return env[name]
        if name in inputs:
            # Mutating sources (KV writes) must not touch the captured state.
            return inputs[name].clone() if name in state else inputs[name]
        value = values.get(f"value:{name}")
        if value is not None:
            return value
        parts = []
        while (part := values.get(f"value:{name}.{len(parts)}")) is not None:
            parts.append(part)
        if parts:
            return tuple(parts)
        # Not materialized (a view of a materialized value, say): recompute it
        # with PyTorch from what the candidate did materialize.
        if (
            argument.op == "placeholder"
            or getattr(argument.target, "_schema", None) is not None
            and argument.target._schema.is_mutable
        ):
            raise KeyError(f"{name} is not materialized")
        env[name] = evaluate(argument)
        return env[name]

    def evaluate(node: torch.fx.Node) -> Any:
        args = torch.fx.node.map_aggregate(node.args, fetch)
        kwargs = torch.fx.node.map_aggregate(node.kwargs, fetch)
        if node.op == "get_attr":
            return functools.reduce(getattr, node.target.split("."), module)
        if node.op == "call_method":
            return getattr(args[0], node.target)(*args[1:], **kwargs)
        return node.target(*args, **kwargs)

    try:
        for name in members:
            env[name] = evaluate(by_name[name])
    except Exception as error:  # noqa: BLE001 - report candidate evaluation failures.
        return f"cannot evaluate {selected.region_id}: {type(error).__name__}: {error}"
    compared = 0
    for value_id in selected.outputs:
        actual = values.get(value_id)
        name, _, index = value_id.removeprefix("value:").partition(".")
        expected = env.get(name)
        if index and isinstance(expected, (tuple, list)):
            expected = expected[int(index)]
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            continue
        compared += 1
        if actual.shape != expected.shape:
            return f"{value_id} has shape {tuple(actual.shape)}, PyTorch {tuple(expected.shape)}"
        if not expected.is_floating_point():
            if not torch.equal(actual, expected):
                return f"{value_id} differs from PyTorch on its own inputs"
            continue
        finite = torch.isfinite(expected)
        if not torch.equal(finite, torch.isfinite(actual)):
            return f"{value_id} is non-finite where PyTorch is finite, or the reverse"
        difference = (actual.float() - expected.float())[finite].norm()
        scale = expected.float()[finite].norm()
        error = (difference / scale).item() if scale > 0 else difference.item()
        if error > _REGION_RELATIVE_L2_LIMIT:
            return f"{value_id} is {error:.3g} from PyTorch on its own inputs"
    if compared == 0:
        return f"{selected.region_id} has no output to verify"
    return None


def _outputs_identical(expected: Any, actual: Any) -> bool:
    expected_values, _ = pytree.tree_flatten(expected)
    actual_values, _ = pytree.tree_flatten(actual)
    return len(expected_values) == len(actual_values) and all(
        torch.equal(left, right)
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
        else left == right
        for left, right in zip(expected_values, actual_values, strict=True)
    )


def _plan_relative_l2_limit(plan: KernelPlan) -> float:
    if plan.graph is not None and _uses_table_launch(plan.graph):
        return _TABLE_PLAN_RELATIVE_L2_LIMIT
    return _PLAN_RELATIVE_L2_LIMIT


def compile_post_grad(
    prepared: PreparedFX,
    *,
    target: TargetFacts | None = None,
    policy: SelectionPolicy | None = None,
    output_dir: Path | str | None = None,
    search_budget: SearchBudget | int | None = None,
    scheduler: str = "phase",
    cache: PlanCache | None = None,
) -> Callable[..., Any]:
    compile_started = time.perf_counter()
    if scheduler not in {"phase", "region-events", "auto"}:
        raise ValueError("scheduler must be 'phase', 'region-events', or 'auto'")
    graph, selected_target = _post_grad_graph(prepared, target)
    if policy is not None and policy.mode != prepared.context.mode:
        raise ValueError(
            f"selection mode {policy.mode!r} does not match backend mode {prepared.context.mode!r}"
        )
    if policy is None:
        table_launch = _uses_table_launch(graph)
        # The device tree the launch is planned with: measured on this GPU if
        # tools/calibrate.py has run, else the spec tree of its family.
        tree = device_tree(selected_target) if plan_with_tree() else None
        policy = SelectionPolicy(
            mode=prepared.context.mode,
            table_launch=table_launch,
            target_options=launch_options(graph, selected_target, table_launch, tree),
            device=tree,
        )
    if isinstance(search_budget, int):
        if search_budget < 1:
            raise ValueError("search budget must be positive")
        budget = SearchBudget(max_plans=search_budget)
    else:
        budget = search_budget or SearchBudget()
        if search_budget is None and _uses_table_launch(graph):
            # A full model compiles for minutes, so one plan is compiled.
            # With a device tree the cost model picks it among the legal
            # plans; without one it is the greedy seed.
            budget = SearchBudget(
                max_plans=3 if policy.device is not None else 1, max_compilations=1
            )

    plan_key: str | None = None
    if cache is not None:
        selections = enumerate_body_selections(graph, selected_target, policy, max_plans=1)
        key_selection = (
            selections[0]
            if selections
            else BodySelectionPlan((), (), (), (), graph, selected_target)
        )
        body_revisions = tuple(
            dict.fromkeys(
                (
                    selected.body.provider,
                    selected.body.evidence.revision,
                    selected.body.semantic.numerical_policy,
                )
                for selection in selections
                for selected in selection.selections
            )
        )
        plan_key = make_plan_key(
            key_selection,
            prepared.context,
            scheduler=scheduler,
            body_revisions=body_revisions,
            device=None if policy.device is None else policy.device.digest,
        )

    def compile_selected() -> CompiledKernel:
        _preflight_complete_plan(graph, selected_target, policy)
        state_snapshot = _StateSnapshot.create(graph, prepared.example_inputs)
        evaluation = replace(prepared, example_inputs=state_snapshot.inputs)
        with torch.no_grad():
            state_snapshot.reset()
            reference = pytree.tree_map_only(
                torch.Tensor,
                torch.clone,
                prepared.forward(*evaluation.example_inputs),
            )
        teacher_check = _TeacherForcedCheck(prepared, state_snapshot)
        if scheduler == "auto":
            phase_result = _search_scheduler(
                graph,
                selected_target,
                policy,
                budget,
                evaluation,
                reference,
                output_dir,
                "phase",
                state_snapshot.reset,
                teacher_check,
            )
            event_result = _search_scheduler(
                graph,
                selected_target,
                policy,
                budget,
                evaluation,
                reference,
                output_dir,
                "region-events",
                state_snapshot.reset,
                teacher_check,
            )
            choices = (phase_result, event_result)
            chosen = min(
                (result for result in choices if result.selected is not None),
                key=_selected_median,
                default=phase_result,
            )
            result = PlanSearchResult(
                chosen.selected,
                chosen.selected_compiled,
                tuple(candidate for labeled in choices for candidate in labeled.candidates),
                tuple(rejection for labeled in choices for rejection in labeled.rejections),
                any(labeled.budget_exhausted for labeled in choices),
                chosen.selected_description,
            )
        else:
            result = _search_scheduler(
                graph,
                selected_target,
                policy,
                budget,
                evaluation,
                reference,
                output_dir,
                scheduler,
                state_snapshot.reset,
                teacher_check,
            )
        if result.selected is None or result.selected_compiled is None:
            reasons = "; ".join(
                f"{candidate.description}: {candidate.reason or candidate.status}"
                for candidate in result.candidates
                if candidate.status != "measured"
            )
            raise CompilerError(
                CompilerDiagnostic(
                    "plan-search",
                    reasons or "no measured complete plan was produced",
                    _source_op_ids_from_graph(graph),
                    tuple(region.id for region in graph.regions),
                    "one correct measured complete plan",
                )
            )
        result = replace(result, selected=result.selected_compiled.plan)
        return replace(result.selected_compiled, search=result)

    cache_hit = False
    if cache is not None:
        assert plan_key is not None
        cache_result = cache.get_or_compile(plan_key, compile_selected)
        compiled = cache_result.compiled
        cache_hit = cache_result.hit
    else:
        compiled = compile_selected()
    measurement = next(
        (
            candidate.measurement
            for candidate in compiled.search.candidates
            if candidate.description == compiled.search.selected_description
            and candidate.measurement is not None
        ),
        None,
    )
    prepared.context.requirements["generic_coverage"] = {
        "semantic_support": semantic_coverage(graph),
        "body_names": tuple(
            dict.fromkeys(
                f"{phase.body.provider}/{phase.body.name}"
                for phase in compiled.plan.phases
                if phase.body.provider == "generic"
            )
        ),
        "workspace_bytes": compiled.plan.storage.workspace_bytes,
        "performance": {
            name: None if measurement is None else getattr(measurement, name)
            for name in (
                "median_isolated_launch_us",
                "median_device_event_span_us",
                "median_complete_call_us",
            )
        },
    }
    parameter_tensors = _current_parameter_tensors(
        compiled, prepared.context, prepared.example_inputs
    )
    forward = bind_callable(compiled, prepared.context, parameter_tensors=parameter_tensors)
    forward.megabake_compile_us = (time.perf_counter() - compile_started) * 1_000_000
    forward.megabake_cache_hit = cache_hit
    return forward


def compile_inductor_fallback(prepared: PreparedFX, reason: str) -> Callable[..., Any]:
    from torch._inductor.compile_fx import compile_fx

    graph = copy.deepcopy(prepared.graph)
    compiled = compile_fx(graph, prepared.example_inputs)
    context = prepared.context
    parameter_positions = {
        index for index, name in enumerate(context.input_order) if name in context.parameter_names
    }
    runtime_positions = [
        index for index in range(len(context.input_order)) if index not in parameter_positions
    ]
    current_parameters = {
        context.input_order[index]: prepared.example_inputs[index]
        for index in parameter_positions
        if index < len(prepared.example_inputs)
    }
    launch_count = 0

    def forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal launch_count
        if kwargs:
            raise CompilerError(
                CompilerDiagnostic(
                    "fallback-binding",
                    "keyword arguments are not supported by the fallback callable",
                    requirement="positional runtime arguments",
                )
            )
        runtime_values = pytree.tree_flatten(args)[0]
        if len(runtime_values) == len(context.input_order):
            full_inputs = list(runtime_values)
        elif len(runtime_values) == len(runtime_positions):
            full_inputs = [None] * len(context.input_order)
            for position, value in zip(runtime_positions, runtime_values, strict=True):
                full_inputs[position] = value
            for position in parameter_positions:
                name = context.input_order[position]
                value = current_parameters.get(name)
                if value is None:
                    raise CompilerError(
                        CompilerDiagnostic(
                            "fallback-binding",
                            f"missing current parameter {name}",
                            requirement="current parameter binding",
                        )
                    )
                full_inputs[position] = value
        else:
            raise CompilerError(
                CompilerDiagnostic(
                    "fallback-binding",
                    (
                        f"expected {len(runtime_positions)} runtime arguments "
                        f"or {len(context.input_order)} graph arguments, "
                        f"got {len(runtime_values)}"
                    ),
                    requirement="runtime argument count",
                )
            )

        output = compiled(*full_inputs)
        launch_count += 1
        forward.megabake_launch_count = launch_count
        context.requirements["fallback_launch_count"] = launch_count
        return pytree.tree_unflatten(
            output,
            context.runtime_output_tree_spec or context.output_tree_spec,
        )

    context.requirements["execution"] = "inductor_fallback"
    context.requirements["fallback_reason"] = reason
    context.requirements["fallback_launch_count"] = launch_count
    forward.megabake_execution = "inductor_fallback"
    forward.megabake_fallback_reason = reason
    forward.megabake_launch_count = launch_count
    forward.megabake_context = context
    return forward


def compile_plan(
    plan: KernelPlan,
    example_inputs: Sequence[Any],
    *,
    output_dir: Path | str | None = None,
    scheduler: str = "phase",
) -> CompiledKernel:
    if scheduler not in {"phase", "region-events"}:
        raise ValueError("compile_plan scheduler must be 'phase' or 'region-events'")
    _reject_invalid_plan(plan)
    if plan.graph is None:
        raise CompilerError(
            CompilerDiagnostic(
                "plan-validation",
                "kernel plan is missing its source graph",
                requirement="source graph",
            )
        )
    device = _common_cuda_device(example_inputs)
    selected_target = plan.target
    if selected_target is None:
        raise CompilerError(
            CompilerDiagnostic(
                "plan-validation",
                "kernel plan is missing target facts",
                requirement="target facts",
            )
        )
    with torch.cuda.device(device):
        artifact_dir = Path(output_dir) if output_dir is not None else Path("artifacts/generated")
        input_tensors = _input_tensors(plan.graph, example_inputs)
        event_schedule = (
            make_region_event_schedule(
                plan.selection,
                plan.dependencies,
                worker_count=plan.launch.grid_size,
            )
            if scheduler == "region-events"
            else None
        )
        current_plan = plan
        seen_workers = {current_plan.launch.grid_size}
        for _ in range(4):
            if (
                event_schedule is not None
                and event_schedule.worker_count != current_plan.launch.grid_size
            ):
                event_schedule = make_region_event_schedule(
                    current_plan.selection,
                    current_plan.dependencies,
                    worker_count=current_plan.launch.grid_size,
                )
            generated = emit_cute(
                current_plan,
                artifact_dir,
                event_schedule=event_schedule,
                table=_uses_table_launch(current_plan.graph),
            )
            module = _load_generated_module(generated)
            max_active_clusters = _max_active_clusters(generated)
            state = _RuntimeState.create(
                current_plan,
                generated,
                input_tensors,
                max_active_clusters=max_active_clusters,
                ready=(
                    torch.zeros(
                        len(event_schedule.events),
                        dtype=torch.int32,
                        device=device,
                    )
                    if event_schedule is not None
                    else None
                ),
            )
            with (
                consumer_cta_sync(generated.launch.threads_per_block)
                if generated.launch.producer_warp
                else contextlib.nullcontext()
            ):
                compiled = cute.compile(module.launch, *state.compile_arguments)
            resources = _query_resources(
                compiled,
                current_plan,
                generated,
                generated.launch.launch_threads,
                generated.launch.shared_memory_bytes,
            )
            max_workers = (
                resources.active_blocks_per_sm
                * torch.cuda.get_device_properties(device).multi_processor_count
            )
            legal_workers = min(current_plan.launch.grid_size, max_workers)
            if legal_workers < 1:
                raise CompilerError(
                    CompilerDiagnostic(
                        "resource-admission",
                        "compiled kernel cannot sustain a cooperative worker",
                        _source_op_ids(current_plan),
                        _region_ids(current_plan),
                        "at least one co-resident cooperative worker",
                    )
                )
            if legal_workers != current_plan.launch.grid_size:
                if legal_workers in seen_workers:
                    raise CompilerError(
                        CompilerDiagnostic(
                            "resource-admission",
                            "worker admission did not converge",
                            _source_op_ids(current_plan),
                            _region_ids(current_plan),
                            "stable legal worker count",
                        )
                    )
                seen_workers.add(legal_workers)
                current_plan = _with_worker_count(
                    current_plan, legal_workers, resources.active_blocks_per_sm
                )
                continue

            current_plan = replace(
                current_plan,
                launch=replace(
                    current_plan.launch,
                    registers_per_thread=resources.registers_per_thread,
                    max_active_blocks_per_sm=resources.active_blocks_per_sm,
                    required_active_blocks=legal_workers,
                ),
            )
            if current_plan.mapping is not None:
                # The mapping as emitted: codegen fits the producer's ring to
                # the shared memory it has, and may drop the producer.
                current_plan = with_emitted_staging(
                    current_plan, generated.launch.producer_warp, generated.launch.ring_pages
                )
            try:
                admission = (
                    admit_event_runtime(event_schedule)
                    if event_schedule is not None
                    else admit_runtime(current_plan)
                )
            except Exception as error:
                raise CompilerError(
                    CompilerDiagnostic(
                        "resource-admission",
                        str(error),
                        _source_op_ids(current_plan),
                        _region_ids(current_plan),
                        "legal cooperative launch",
                    )
                ) from error
            input_order = tuple(
                current_plan.graph.values[value_id].binding_name or value_id
                for value_id in current_plan.graph.inputs
            )
            parameter_names = frozenset(
                name
                for name, value in zip(input_order, example_inputs, strict=True)
                if isinstance(value, torch.nn.Parameter)
            )
            torch.cuda.synchronize(device)
            return CompiledKernel(
                current_plan,
                generated,
                compiled,
                resources,
                admission,
                event_schedule,
                input_order,
                parameter_names,
                state.work_iterators,
                max_active_clusters,
            )

    raise CompilerError(
        CompilerDiagnostic(
            "resource-admission",
            "worker admission exceeded its retry budget",
            _source_op_ids(plan),
            _region_ids(plan),
            "stable legal worker count",
        )
    )


def _current_parameter_tensors(
    compiled_kernel: CompiledKernel,
    context: CaptureContext,
    example_inputs: Sequence[Any],
) -> dict[str, torch.Tensor]:
    positions = {spec.name: position for position, spec in enumerate(context.input_specs)}
    values: dict[str, torch.Tensor] = {}
    for graph_name in compiled_kernel.input_order:
        if graph_name not in compiled_kernel.parameter_names:
            continue
        candidates = (graph_name, graph_name.removeprefix("l_").removesuffix("_"))
        position = next((positions[name] for name in candidates if name in positions), None)
        if position is None:
            raise CompilerError(
                CompilerDiagnostic(
                    "callable-binding",
                    f"cannot map parameter {graph_name!r} to a captured input",
                    requirement="current parameter binding",
                )
            )
        values[graph_name] = example_inputs[position]
    return values


def _initial_ready(compiled_kernel: CompiledKernel, device: torch.device) -> torch.Tensor | None:
    if compiled_kernel.event_schedule is None:
        return None
    return torch.zeros(
        len(compiled_kernel.event_schedule.events),
        dtype=torch.int32,
        device=device,
    )


def _max_active_clusters(generated: GeneratedCuTeModule) -> Any | None:
    if "max_active_clusters" not in generated.launch.arguments:
        return None
    return cutlass.utils.HardwareInfo().get_max_active_clusters(1)


def bind_callable(
    compiled_kernel: CompiledKernel,
    context: CaptureContext,
    *,
    parameter_tensors: Mapping[str, torch.Tensor] | None = None,
) -> Callable[..., Any]:
    if not context.input_order:
        raise CompilerError(
            CompilerDiagnostic(
                "callable-binding",
                "capture context has no input order",
                requirement="runtime argument order",
            )
        )
    graph = compiled_kernel.plan.graph
    if graph is None:
        raise CompilerError(
            CompilerDiagnostic(
                "callable-binding",
                "compiled plan is missing its source graph",
                requirement="source graph",
            )
        )
    parameter_positions = {
        index
        for index, name in enumerate(context.input_order)
        if name in compiled_kernel.parameter_names
    }
    runtime_positions = [
        index for index in range(len(context.input_order)) if index not in parameter_positions
    ]
    current_parameters = dict(parameter_tensors or {})
    # The most recent calls only; a decode loop would otherwise grow it forever.
    trace: collections.deque[CallableTrace] = collections.deque(maxlen=_TRACE_LENGTH)
    runtime_lock = threading.Lock()
    state: _RuntimeState | None = None
    next_epoch = 0
    epoch_stride = _epoch_stride(compiled_kernel)
    # Inputs of the previous call: tensors passed again as the same objects
    # (parameters, KV state) keep their validation.
    previous_inputs: list[Any] = []

    def forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal state, next_epoch, previous_inputs
        if kwargs:
            raise CompilerError(
                CompilerDiagnostic(
                    "callable-binding",
                    "keyword arguments are not supported by generated callables",
                    requirement="positional runtime arguments",
                )
            )
        # Dynamo passes flat tensors; only nested arguments need flattening.
        runtime_values = (
            args
            if all(isinstance(value, torch.Tensor) for value in args)
            else pytree.tree_flatten(args)[0]
        )
        if len(runtime_values) == len(context.input_order):
            full_inputs = list(runtime_values)
        elif len(runtime_values) == len(runtime_positions):
            full_inputs = [None] * len(context.input_order)
            for position, value in zip(runtime_positions, runtime_values, strict=True):
                full_inputs[position] = value
            for position in parameter_positions:
                name = context.input_order[position]
                value = current_parameters.get(name)
                if value is None:
                    raise CompilerError(
                        CompilerDiagnostic(
                            "callable-binding",
                            f"missing current parameter {name}",
                            requirement="current parameter binding",
                        )
                    )
                full_inputs[position] = value
        else:
            raise CompilerError(
                CompilerDiagnostic(
                    "callable-binding",
                    "expected "
                    f"{len(runtime_positions)} runtime arguments "
                    f"or {len(context.input_order)} graph arguments, "
                    f"got {len(runtime_values)}",
                    requirement="runtime argument count",
                )
            )

        with runtime_lock:
            input_tensors = _input_tensors(graph, full_inputs, previous_inputs)
            previous_inputs = full_inputs
            if state is None:
                # Structural: every graph input is bound on every call.
                _validate_data_indices(compiled_kernel.plan, input_tensors)
            active_stream = torch.cuda.current_stream(next(iter(input_tensors.values())).device)
            capturing = _stream_is_capturing(active_stream)
            if capturing and (
                _graph_capture is None
                or state is None
                or compiled_kernel.event_schedule is not None
                or compiled_kernel.generated.launch.launch_policy != "megabake_cooperative_table"
            ):
                raise CompilerError(
                    CompilerDiagnostic(
                        "callable-binding",
                        "CUDA graph capture needs megabake.graphed and a table launch",
                        requirement="graphed table launch",
                    )
                )

            if state is None:
                ready = _initial_ready(compiled_kernel, active_stream.device)
                state = _RuntimeState.create(
                    compiled_kernel.plan,
                    compiled_kernel.generated,
                    input_tensors,
                    max_active_clusters=compiled_kernel.max_active_clusters,
                    ready=ready,
                    work_iterators=compiled_kernel.work_iterators,
                )
                forward.megabake_runtime_state = state
            else:
                state.rebind(input_tensors, full_inputs)

            launches = 0
            uploads_before = state.address_table_uploads

            def kernel(ready: torch.Tensor, epoch: int, stream: torch.cuda.Stream) -> None:
                nonlocal launches
                state.launch(compiled_kernel.compiled, epoch, stream)
                launches += 1

            if compiled_kernel.event_schedule is None:
                execute_phase_plan(
                    compiled_kernel.plan,
                    kernel,
                    active_stream,
                    admission=compiled_kernel.admission,
                    ready=state.ready,
                    epoch=next_epoch,
                    synchronize=False,
                    capturable=capturing,
                )
            else:
                execute_event_plan(
                    compiled_kernel.event_schedule,
                    kernel,
                    active_stream,
                    admission=compiled_kernel.admission,
                    events=state.ready,
                    epoch=next_epoch,
                    validate_counters=False,
                    synchronize=False,
                )
            outputs = tuple(state.tensor_for_value(value_id) for value_id in graph.outputs)
            trace.append(
                CallableTrace(
                    workload_kernel_launches=launches,
                    recurring_gpu_operations=_recurring_operations(
                        launches, state.address_table_uploads - uploads_before
                    ),
                )
            )
            next_epoch += epoch_stride
            if next_epoch > 2**31 - 1 - epoch_stride:
                if compiled_kernel.event_schedule is None:
                    state.reset_ready()
                else:
                    state.ready.zero_()
                next_epoch = 0
            return pytree.tree_unflatten(
                outputs,
                context.runtime_output_tree_spec or context.output_tree_spec,
            )

    forward.megabake_compiled = compiled_kernel
    forward.megabake_trace = trace
    forward.megabake_context = context
    forward.megabake_runtime_state = state
    return forward


# Runtime states launched by the capture in progress; None outside ``graphed``.
_graph_capture: list[_RuntimeState] | None = None


class GraphedCall:
    """One MegaBake call captured in a CUDA graph; made by ``graphed``."""

    def __init__(self, function: Callable[..., Any], args: tuple[Any, ...], warmup: int):
        global _graph_capture
        # The graph retains kernel handles, so their compiled owner must stay alive.
        self._function = function
        self._inputs, self._spec = pytree.tree_flatten(args)
        saved = [leaf.clone() if isinstance(leaf, torch.Tensor) else None for leaf in self._inputs]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                function(*args)
            # Warm-up calls advance stateful arguments (KV caches); undo them.
            for leaf, original in zip(self._inputs, saved, strict=True):
                if original is not None:
                    leaf.copy_(original)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        del saved
        self._graph = torch.cuda.CUDAGraph()
        _graph_capture = []
        try:
            with torch.cuda.graph(self._graph):
                self.output = function(*args)
            self._states = tuple(dict.fromkeys(_graph_capture))
        finally:
            _graph_capture = None
        if not self._states:
            raise CompilerError(
                CompilerDiagnostic(
                    "callable-binding",
                    "the captured call launched no MegaBake kernel",
                    requirement="a MegaBake-compiled function",
                )
            )

    def __call__(self, *args: Any) -> Any:
        leaves, spec = pytree.tree_flatten(args)
        if spec != self._spec:
            raise ValueError("a graphed call takes the argument structure it was captured with")
        for given, captured in zip(leaves, self._inputs, strict=True):
            if given is captured:
                continue
            if not isinstance(captured, torch.Tensor):
                if given != captured:
                    raise ValueError(f"graphed call captured {captured!r}, got {given!r}")
            elif given.shape != captured.shape or given.dtype != captured.dtype:
                raise ValueError(
                    f"graphed call captured a {captured.dtype} {tuple(captured.shape)} tensor, "
                    f"got {given.dtype} {tuple(given.shape)}"
                )
            else:
                captured.copy_(given)
        for state in self._states:
            state.graph_replayed()
        self._graph.replay()
        return self.output


def graphed(function: Callable[..., Any], *args: Any, warmup: int = 2) -> GraphedCall:
    """Capture one call of a MegaBake-compiled ``function`` in a CUDA graph.

    Each later call copies tensor arguments passed as new objects into the
    captured ones (pass the captured tensors, updated in place, to skip even
    that), then replays the graph: no guards, binding or launch work on the
    host. It returns the captured outputs, which the next call overwrites.
    ``warmup`` calls run first; tensor arguments are restored afterwards.
    """
    return GraphedCall(function, args, max(warmup, 2))


def _post_grad_graph(
    prepared: PreparedFX, target: TargetFacts | None
) -> tuple[ComputeGraph, TargetFacts]:
    try:
        graph = import_fx(prepared.graph, prepared.context)
        graph = form_regions(canonicalize_semantics(graph))
        semantic = _semantic_diagnostic(graph)
        if semantic is not None:
            raise CompilerError(semantic, safe_for_fallback=True)
        selected_target = target or _target_facts(_common_cuda_device(prepared.example_inputs))
        return graph, selected_target
    except CompilerError:
        raise
    except Exception as error:
        raise CompilerError(
            CompilerDiagnostic(
                "planning",
                str(error),
                requirement="valid complete KernelPlan",
            )
        ) from error


@dataclass(frozen=True)
class _StateSnapshot:
    """Private copies of mutable state inputs used while searching plans."""

    inputs: tuple[Any, ...]
    pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    @classmethod
    def create(cls, graph: ComputeGraph, example_inputs: Sequence[Any]) -> _StateSnapshot:
        state_ids = {
            value_id
            for operation in graph.ops.values()
            if isinstance(operation.tensor_program, TensorProgram)
            for access in operation.tensor_program.state_accesses
            for value_id in (
                access.key_cache_value_id,
                access.value_cache_value_id,
                access.valid_length_value_id,
            )
        } | {
            # Inputs written by in-place copies (recurrent and convolution state).
            value_id
            for operation in graph.ops.values()
            if operation.effects.get("mutates_inputs")
            for value_id in operation.effects.get("mutated_operands", operation.operands[:1])
            if value_id in graph.inputs
        }
        inputs = list(example_inputs)
        pairs = []
        for position, value_id in enumerate(graph.inputs):
            if value_id in state_ids and isinstance(inputs[position], torch.Tensor):
                original = inputs[position].detach().clone()
                inputs[position] = original.clone()
                pairs.append((inputs[position], original))
        return cls(tuple(inputs), tuple(pairs))

    def reset(self) -> None:
        for private, original in self.pairs:
            private.copy_(original)


def _make_plan_evaluator(
    prepared: PreparedFX,
    reference: Any,
    output_dir: Path | str | None,
    scheduler: str = "phase",
    reset_state: Callable[[], None] | None = None,
    teacher_check: _TeacherForcedCheck | None = None,
):
    def evaluate(plan: KernelPlan, budget: SearchBudget) -> PlanMeasurement:
        try:
            compiled = compile_plan(
                plan,
                prepared.example_inputs,
                output_dir=output_dir,
                scheduler=scheduler,
            )
            bound = bind_callable(
                compiled,
                prepared.context,
                parameter_tensors=_current_parameter_tensors(
                    compiled, prepared.context, prepared.example_inputs
                ),
            )

            def executable(*arguments: Any) -> Any:
                # Stateful candidates start every call from the captured state.
                if reset_state is not None:
                    reset_state()
                return bound(*arguments)

            with torch.no_grad():
                actual = executable(*prepared.example_inputs)
            relative_error = _output_relative_l2(reference, actual)
            if relative_error >= _plan_relative_l2_limit(plan):
                failure = (
                    teacher_check.failure(compiled, actual)
                    if teacher_check is not None
                    else "no teacher-forced check"
                )
                if failure is not None:
                    return PlanMeasurement(
                        compiled=compiled,
                        resources=compiled.resources,
                        correct=False,
                        error=(
                            f"relative L2 error {relative_error:.6g}; "
                            f"teacher-forced verification: {failure}"
                        ),
                    )
            complete_samples, device_samples = _complete_call_samples(
                executable,
                prepared.example_inputs,
                budget.timing_samples,
            )
            verification_samples, _ = _complete_call_samples(executable, prepared.example_inputs, 1)
            isolated_samples = _isolated_kernel_samples(
                compiled,
                prepared.context,
                prepared.example_inputs,
                budget.timing_samples,
                reset_state,
            )
            return PlanMeasurement(
                compiled=compiled,
                resources=compiled.resources,
                correct=True,
                complete_call_samples_us=complete_samples,
                device_event_span_samples_us=device_samples,
                median_complete_call_us=median(complete_samples),
                median_device_event_span_us=median(device_samples),
                isolated_launch_samples_us=isolated_samples,
                median_isolated_launch_us=median(isolated_samples),
                verification_complete_call_us=verification_samples[0],
            )
        except Exception as error:  # noqa: BLE001 - convert measurement failures to results.
            return PlanMeasurement(error=f"{type(error).__name__}: {error}")

    return evaluate


def _search_scheduler(
    graph: ComputeGraph,
    target: TargetFacts,
    policy: Any | None,
    budget: SearchBudget,
    prepared: PreparedFX,
    reference: Any,
    output_dir: Path | str | None,
    scheduler: str,
    reset_state: Callable[[], None] | None = None,
    teacher_check: _TeacherForcedCheck | None = None,
):
    if policy is not None:
        policy = replace(policy, event_schedule=scheduler == "region-events")
    evaluator = (
        budget.evaluator
        if scheduler == "phase" and budget.evaluator is not None
        else _make_plan_evaluator(
            prepared, reference, output_dir, scheduler, reset_state, teacher_check
        )
    )
    scheduler_budget = replace(
        budget,
        evaluator=evaluator,
    )
    result = search_plans(graph, target, policy, scheduler_budget)
    prefix = f"scheduler={scheduler}; "
    return replace(
        result,
        candidates=tuple(
            replace(candidate, description=prefix + candidate.description)
            for candidate in result.candidates
        ),
        selected_description=(
            None if result.selected_description is None else prefix + result.selected_description
        ),
    )


def _selected_median(result: Any) -> float:
    for candidate in result.candidates:
        if candidate.plan is result.selected and candidate.measurement is not None:
            median_value = (
                candidate.measurement.median_isolated_launch_us
                if candidate.measurement.median_isolated_launch_us is not None
                else candidate.measurement.median_device_event_span_us
                if candidate.measurement.median_device_event_span_us is not None
                else candidate.measurement.median_complete_call_us
            )
            if median_value is not None:
                return median_value
    return float("inf")


def _isolated_kernel_samples(
    compiled_kernel: CompiledKernel,
    context: CaptureContext,
    example_inputs: Sequence[Any],
    count: int,
    reset_state: Callable[[], None] | None = None,
) -> tuple[float, ...]:
    graph_inputs = _graph_ordered_inputs(compiled_kernel, context, example_inputs)
    graph = compiled_kernel.plan.graph
    if graph is None:
        raise CompilerError(
            CompilerDiagnostic(
                "runtime-binding",
                "compiled plan is missing its source graph",
                requirement="source graph",
            )
        )
    state = _RuntimeState.create(
        compiled_kernel.plan,
        compiled_kernel.generated,
        _input_tensors(graph, graph_inputs),
        max_active_clusters=compiled_kernel.max_active_clusters,
        ready=_initial_ready(compiled_kernel, torch.cuda.current_stream().device),
        work_iterators=compiled_kernel.work_iterators,
    )
    if compiled_kernel.event_schedule is not None:
        state.ready.zero_()
    else:
        state.reset_ready()
    epoch_stride = _epoch_stride(compiled_kernel)
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    table_launch = compiled_kernel.generated.launch.launch_policy == "megabake_cooperative_table"
    fast_path = hasattr(compiled_kernel.compiled, "generate_execution_args") and hasattr(
        compiled_kernel.compiled, "run_compiled_program"
    )
    epoch = 0

    def prepare() -> tuple[Any, ...]:
        nonlocal epoch
        if table_launch:
            return state.prepare_launch(compiled_kernel.compiled, epoch, torch_stream)
        arguments = state.arguments(epoch, stream)
        epoch += epoch_stride
        if not fast_path:
            return arguments, arguments, arguments
        packed = compiled_kernel.compiled.generate_execution_args(*arguments)
        execution = packed[0] if isinstance(packed, tuple) and len(packed) == 2 else packed
        return execution, packed, arguments

    def launch(token: tuple[Any, ...]) -> None:
        if table_launch:
            state.run_launch(compiled_kernel.compiled, token, torch_stream)
        elif fast_path:
            compiled_kernel.compiled.run_compiled_program(token[0])
        else:
            compiled_kernel.compiled(*token[0])

    samples: list[float] = []
    # A stateful launch advances its state, so time one launch per reset.
    launches_per_sample = 1 if reset_state is not None else min(count, DYNAMIC_ADDRESS_RING)
    for _ in range(count):
        if reset_state is not None:
            reset_state()
        tokens = [prepare() for _ in range(launches_per_sample)]
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for token in tokens:
            launch(token)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000 / launches_per_sample)
    return tuple(samples)


def _graph_ordered_inputs(
    compiled_kernel: CompiledKernel,
    context: CaptureContext,
    example_inputs: Sequence[Any],
) -> tuple[Any, ...]:
    positions = {spec.name: position for position, spec in enumerate(context.input_specs)}
    ordered: list[Any] = [None] * len(example_inputs)
    for graph_name in compiled_kernel.input_order:
        candidates = (graph_name, graph_name.removeprefix("l_").removesuffix("_"))
        position = next((positions[name] for name in candidates if name in positions), None)
        if position is None:
            raise CompilerError(
                CompilerDiagnostic(
                    "runtime-binding",
                    f"cannot map graph input {graph_name!r} to a captured input",
                    requirement="complete runtime binding",
                )
            )
        ordered[position] = example_inputs[position]
    if any(value is None for value in ordered):
        raise CompilerError(
            CompilerDiagnostic(
                "runtime-binding",
                "could not map every captured input to the compiled graph",
                requirement="complete runtime binding",
            )
        )
    return tuple(ordered)


def _complete_call_samples(
    function: Callable[..., Any], arguments: Sequence[Any], count: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    host_samples: list[float] = []
    device_samples: list[float] = []
    with torch.no_grad():
        for _ in range(count):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_host = time.perf_counter()
            start_event.record()
            function(*arguments)
            end_event.record()
            end_event.synchronize()
            host_samples.append((time.perf_counter() - start_host) * 1_000_000)
            device_samples.append(start_event.elapsed_time(end_event) * 1_000)
    return tuple(host_samples), tuple(device_samples)


def _output_relative_l2(expected: Any, actual: Any) -> float:
    expected_values, _ = pytree.tree_flatten(expected)
    actual_values, _ = pytree.tree_flatten(actual)
    if len(expected_values) != len(actual_values):
        return float("inf")
    numerator = 0.0
    denominator = 0.0
    for expected_value, actual_value in zip(expected_values, actual_values, strict=True):
        if type(expected_value) is not type(actual_value):
            return float("inf")
        if not isinstance(expected_value, torch.Tensor):
            if expected_value != actual_value:
                return float("inf")
            continue
        if (
            not isinstance(actual_value, torch.Tensor)
            or expected_value.shape != actual_value.shape
            or expected_value.dtype != actual_value.dtype
        ):
            return float("inf")
        difference = (expected_value.float() - actual_value.float()).square().sum()
        denominator_value = expected_value.float().square().sum()
        numerator += float(difference.item())
        denominator += float(denominator_value.item())
    if denominator == 0:
        return numerator**0.5
    return (numerator / denominator) ** 0.5


def _source_op_ids_from_graph(graph: ComputeGraph) -> tuple[str, ...]:
    return tuple(graph.ops)


def _reject_invalid_plan(plan: KernelPlan) -> None:
    try:
        verify_plan(plan)
    except Exception as error:
        raise CompilerError(
            CompilerDiagnostic(
                "plan-validation",
                str(error),
                _source_op_ids(plan),
                _region_ids(plan),
                "valid complete KernelPlan",
            )
        ) from error


def _target_facts(device: torch.device) -> TargetFacts:
    if device.type != "cuda":
        raise UnsupportedCompiler("MegaBake generated kernels require CUDA")
    properties = torch.cuda.get_device_properties(device)
    architecture = f"sm{properties.major}{properties.minor}"
    if architecture not in {"sm90", "sm100", "sm103", "sm120", "sm121"}:
        raise UnsupportedCompiler(
            f"generated CuTe kernels require SM90 or Blackwell SM100/103/120/121; found {architecture}"
        )
    return TargetFacts(
        architecture="sm90a" if architecture == "sm90" else architecture,
        threads_per_warp=properties.warp_size,
        max_threads_per_cta=properties.max_threads_per_block,
        max_shared_memory_per_cta=properties.shared_memory_per_block_optin,
        max_registers_per_thread=255,
        multiprocessor_count=properties.multi_processor_count,
        supports_tma=True,
        supports_wgmma=architecture == "sm90",
    )


def _common_cuda_device(example_inputs: Sequence[Any]) -> torch.device:
    devices = {value.device for value in example_inputs if isinstance(value, torch.Tensor)}
    if not devices or any(device.type != "cuda" for device in devices):
        raise UnsupportedCompiler("all captured tensor inputs must be CUDA tensors")
    if len(devices) != 1:
        raise UnsupportedCompiler(f"captured inputs span multiple devices: {devices}")
    return next(iter(devices))


def _input_tensors(
    graph: ComputeGraph,
    example_inputs: Sequence[Any],
    previous: Sequence[Any] = (),
) -> dict[str, torch.Tensor]:
    """Bind graph inputs by value id, checking each against its guarded metadata.

    A tensor that is the same object as ``previous`` at its position was
    checked on an earlier call and is not checked again.
    """
    if len(example_inputs) != len(graph.inputs):
        raise CompilerError(
            CompilerDiagnostic(
                "input-binding",
                f"expected {len(graph.inputs)} graph inputs, got {len(example_inputs)}",
                requirement="graph argument count",
            )
        )
    result = dict(zip(graph.inputs, example_inputs))
    if len(previous) != len(example_inputs):
        previous = (None,) * len(example_inputs)
    for value_id, value, earlier in zip(graph.inputs, example_inputs, previous):
        if value is earlier:
            continue
        if not isinstance(value, torch.Tensor):
            raise CompilerError(
                CompilerDiagnostic(
                    "input-binding",
                    f"graph input {value_id} is not a tensor",
                    (value_id,),
                    requirement="tensor input",
                )
            )
        if value.device.type != "cuda":
            raise CompilerError(
                CompilerDiagnostic(
                    "input-binding",
                    f"graph input {value_id} is not on CUDA",
                    (value_id,),
                    requirement="CUDA input",
                )
            )
        _validate_input_metadata(value_id, graph.values[value_id], value)
    return result


def _validate_input_metadata(value_id: str, value: Any, tensor: torch.Tensor) -> None:
    expected = (value.shape, str(value.dtype), value.strides)
    actual = (tensor.shape, str(tensor.dtype), tuple(tensor.stride()))
    if actual != expected:
        raise CompilerError(
            CompilerDiagnostic(
                "input-binding",
                f"input {value_id} metadata changed: expected {expected}, got {actual}",
                (value_id,),
                requirement="guarded input shape, dtype, and strides",
            )
        )


def _gather_index_is_static(graph: ComputeGraph, value_id: str) -> bool:
    pending = [value_id]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if current in graph.inputs:
            return False
        value = graph.values.get(current)
        operation = graph.ops.get(value.producer) if value is not None else None
        if operation is None:
            return False
        target = operation.attributes.get("target")
        if target == "aten.add.Tensor":
            arguments = operation.attributes.get("fx_args", ())
            if len(arguments) != 2 or not isinstance(arguments[1], int):
                return False
        elif target not in {
            "prims.iota.default",
            "aten.unsqueeze.default",
            "aten.reshape.default",
            "aten.permute.default",
        }:
            return False
        pending.extend(operation.operands)
    return True


def _recurring_operations(launches: int, address_uploads: int) -> tuple[tuple[str, int], ...]:
    operations = [("workload_kernel", launches)]
    if address_uploads:
        operations.append(("address_table_upload", address_uploads))
    return tuple(operations)


def _epoch_stride(compiled_kernel: CompiledKernel) -> int:
    if compiled_kernel.event_schedule is not None:
        return 1
    if compiled_kernel.generated.launch.barrier_epochs:
        return compiled_kernel.generated.launch.barrier_epochs
    return max(
        max(
            (phase.index for phase in compiled_kernel.plan.phases if phase.barrier is not None),
            default=-1,
        )
        + 1,
        1,
    )


def _validate_data_indices(plan: KernelPlan, inputs: Mapping[str, torch.Tensor]) -> None:
    """Check gather-index bindings; index ranges trap inside the generated body."""
    for selected in plan.selection.selections:
        for load in selected.program.loads:
            for axis in load.axes:
                if not isinstance(axis, DataIndex):
                    continue
                index_load = next(
                    candidate
                    for candidate in selected.program.loads
                    if candidate.id == axis.load_id
                )
                index_tensor = inputs.get(index_load.value_id)
                if index_tensor is None:
                    if _gather_index_is_static(plan.graph, index_load.value_id):
                        continue
                    raise CompilerError(
                        CompilerDiagnostic(
                            "runtime-binding",
                            f"gather index {index_load.value_id} is not a runtime input",
                            selected.source_op_ids,
                            requirement="runtime gather index",
                        )
                    )


def _load_generated_module(generated: GeneratedCuTeModule) -> Any:
    spec = spec_from_file_location(generated.module_name, generated.source_path)
    if spec is None or spec.loader is None:
        raise CompilerError(
            CompilerDiagnostic(
                "codegen",
                f"cannot load generated module {generated.source_path}",
                requirement="loadable generated module",
            )
        )
    module = module_from_spec(spec)
    sys.modules[generated.module_name] = module
    spec.loader.exec_module(module)
    return module


def _query_resources(
    compiled: Any,
    plan: KernelPlan,
    generated: GeneratedCuTeModule,
    threads: int,
    dynamic_shared_bytes: int,
) -> CompiledResources:
    kernel_names = tuple(compiled.kernel_info)
    if len(kernel_names) != 1:
        raise CompilerError(
            CompilerDiagnostic(
                "resource-query",
                f"expected one complete kernel, found {len(kernel_names)}",
                requirement="one enclosing workload kernel",
            )
        )
    kernel_name = kernel_names[0]
    status, kernel = cuda.cuLibraryGetKernel(
        cuda.CUlibrary(int(compiled.library)), kernel_name.encode()
    )
    if int(status) != 0:
        raise CompilerError(
            CompilerDiagnostic(
                "resource-query",
                f"failed to open compiled kernel {kernel_name}: {status}",
                requirement="compiled CUDA kernel",
            )
        )
    attributes = {
        "registers": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
        "static_shared": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        "local": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
    }
    values = {}
    for label, attribute in attributes.items():
        status, value = cuda.cuKernelGetAttribute(attribute, kernel, 0)
        if int(status) != 0:
            raise CompilerError(
                CompilerDiagnostic(
                    "resource-query",
                    f"failed to query {label}: {status}",
                    requirement="compiled resource attributes",
                )
            )
        values[label] = int(value)
    status, function = cuda.cuKernelGetFunction(kernel)
    if int(status) != 0:
        raise CompilerError(
            CompilerDiagnostic(
                "resource-query",
                f"failed to get kernel function: {status}",
                requirement="CUDA function handle",
            )
        )
    status, active_blocks = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        function, threads, dynamic_shared_bytes
    )
    if int(status) != 0:
        raise CompilerError(
            CompilerDiagnostic(
                "resource-query",
                f"failed to query occupancy: {status}",
                requirement="compiled active-block limit",
            )
        )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return CompiledResources(
        kernel_name,
        values["registers"],
        values["static_shared"],
        values["local"],
        dynamic_shared_bytes,
        int(active_blocks),
        int(active_blocks) * properties.multi_processor_count,
        plan.storage.workspace_bytes,
        plan.storage.peak_live_workspace_bytes,
        generated.launch.body_variant_count,
        generated.launch.phase_runner_count,
        _compiled_code_bytes(compiled),
        generated.launch.source_bytes,
    )


def _compiled_code_bytes(compiled: Any) -> int:
    dump_to_object = getattr(compiled, "dump_to_object", None)
    if callable(dump_to_object):
        return len(dump_to_object("megabake_kernel"))
    cubin = getattr(compiled, "__cubin__", None)
    if isinstance(cubin, bytes):
        return len(cubin)
    if cubin is None:
        return 0
    cubin_path = Path(str(cubin))
    if cubin_path.is_file():
        return cubin_path.stat().st_size
    return len(str(cubin).encode("utf-8"))


def _with_worker_count(
    plan: KernelPlan, worker_count: int, active_blocks_per_sm: int
) -> KernelPlan:
    phases = tuple(
        replace(
            phase,
            workers=(
                _recompute_assignments(phase.region_id, phase.work_items, worker_count)
                if phase.recompute
                else _worker_assignments(phase.region_id, phase.work_items, worker_count)
            ),
            barrier=None
            if phase.barrier is None
            else replace(phase.barrier, participating_workers=tuple(range(worker_count))),
        )
        for phase in plan.phases
    )
    prototype = plan.storage.worker_scratch[0]
    worker_scratch = tuple(
        replace(prototype, worker_id=worker_id) for worker_id in range(worker_count)
    )
    storage = replace(plan.storage, worker_scratch=worker_scratch)
    launch = replace(
        plan.launch,
        grid_size=worker_count,
        max_active_blocks_per_sm=active_blocks_per_sm,
        required_active_blocks=worker_count,
    )
    return replace(plan, phases=phases, storage=storage, launch=launch)


# Barrier epochs stay below int32's maximum.
_MAX_EPOCH = 2**31 - 1
# Calls kept in a generated callable's ``megabake_trace``.
_TRACE_LENGTH = 64


@dataclass
class _TableRebind:
    """Slots and graph outputs as offsets from per-call root tensors."""

    inputs: list[torch.Tensor]
    output_layouts: list[Any]
    output_ids: list[str]
    slot_roots: np.ndarray
    slot_offsets: np.ndarray
    empty_slots: np.ndarray
    # (root, shape, strides, element offset, is the root itself) per output.
    graph_outputs: tuple[tuple[int, tuple[int, ...], tuple[int, ...], int, bool], ...]
    # Data pointer of every root for the current call.
    pointers: np.ndarray | None = None

    def addresses(self, roots: Sequence[torch.Tensor]) -> np.ndarray:
        self.pointers = np.fromiter(
            (tensor.data_ptr() for tensor in roots), dtype=np.int64, count=len(roots)
        )
        return self.slot_addresses()

    def slot_addresses(self) -> np.ndarray:
        addresses = self.pointers[self.slot_roots] + self.slot_offsets
        addresses[self.empty_slots] = 0
        return addresses


class _RuntimeState:
    def __init__(
        self,
        plan: KernelPlan,
        generated: GeneratedCuTeModule,
        input_tensors: Mapping[str, torch.Tensor],
        ready: torch.Tensor,
        work_iterators: Mapping[str, torch.Tensor],
        max_active_clusters: Any | None,
        workspace: Mapping[str, torch.Tensor],
        outputs: Mapping[str, torch.Tensor],
        values: Mapping[str, torch.Tensor],
        g0_tensors: tuple[tuple[torch.Tensor, ...], ...],
        workspace_base: torch.Tensor,
    ):
        self.plan = plan
        self.generated = generated
        self.input_tensors = input_tensors
        self.ready = ready
        self.work_iterators = work_iterators
        self.workspace = workspace
        self.workspace_base = workspace_base
        self.outputs = outputs
        self._values = values
        self._values_stale = False
        self.g0_tensors = g0_tensors
        self._normal_view_cache: dict[str, tuple[tuple[Any, ...], Any]] = {}
        self._work_iterator_view_cache: dict[str, tuple[tuple[Any, ...], Any]] = {}
        self._g0_view_cache: dict[int, tuple[tuple[tuple[Any, ...], ...], tuple[Any, ...]]] = {}
        self._ready_view: tuple[torch.Tensor, Any] | None = None
        self._max_active_clusters = max_active_clusters
        self._address_table: torch.Tensor | None = None
        self._address_mirror: np.ndarray | None = None
        self._phase_table: tuple[torch.Tensor, Any] | None = None
        self._gemm_table: tuple[torch.Tensor, Any] | None = None
        self._stream_table: tuple[torch.Tensor, Any] | None = None
        # Phase scratch of table launches: FP32 elements and Int32 counters.
        self._phase_scratch: tuple[tuple[torch.Tensor, Any], tuple[torch.Tensor, Any]] | None = None
        # Per-invocation TMA descriptors (three 128-byte maps per table GEMM),
        # rewritten by every call's kernel prologue.
        self._tma_descriptors: tuple[torch.Tensor, Any] | None = None
        self._checked_layouts: dict[int, tuple[Any, ...]] = {}
        self.address_table_uploads = 0
        # Outputs this state allocates per call, by value id.
        self._allocated_outputs: dict[str, Any] = {}
        # Table launches: per-call rebinding from root pointers (see
        # _prepare_table_rebind); None until the first full binding.
        self._table_rebind: _TableRebind | None = None
        self._slot_addresses: np.ndarray | None = None
        self._resolved_index: np.ndarray | None = None
        self._dynamic_slot_table: tuple[torch.Tensor, Any] | None = None
        self._dynamic_slot_indices: np.ndarray | None = None
        # Table launches: (pinned [ring, slots + 1] buffer, its NumPy view,
        # per-entry event of the launch that read it); see prepare_launch.
        self._dynamic_ring: tuple[torch.Tensor, np.ndarray, list[Any]] | None = None
        self._pending_dynamic_addresses: np.ndarray | None = None
        # Mirror of the device call counter; it is the next launch's number.
        self._calls = 0
        self._prepared_calls = 0
        self._calls_run = 0
        # (bound views, the argument tuple, its stream handle)
        self._table_argument_cache: tuple[tuple[Any, ...], tuple[Any, ...], int] | None = None
        # (arguments, packed execution arguments, what keeps them alive)
        self._packed_arguments: tuple[Any, Any, Any] | None = None
        self._work_iterator_views: tuple[Any, ...] | None = None
        # Address rings of CUDA graphs that captured this state; kept alive
        # with the state because the graphs read them on every replay.
        self._graph_rings: list[torch.Tensor] = []

    @classmethod
    def create(
        cls,
        plan: KernelPlan,
        generated: GeneratedCuTeModule,
        input_tensors: Mapping[str, torch.Tensor],
        max_active_clusters: Any | None = None,
        ready: torch.Tensor | None = None,
        work_iterators: Mapping[str, torch.Tensor] | None = None,
    ) -> _RuntimeState:
        if plan.graph is None:
            raise CompilerError(
                CompilerDiagnostic(
                    "runtime-binding",
                    "plan is missing its source graph",
                    requirement="source graph",
                )
            )
        _validate_state_capacities(plan, input_tensors)
        device = next(iter(input_tensors.values())).device
        if ready is None:
            ready = make_ready(plan.launch.grid_size, device)
        if work_iterators is None:
            work_iterators = {
                iterator.name: torch.arange(
                    0,
                    iterator.length * iterator.step,
                    iterator.step,
                    device=device,
                    dtype=torch.int64,
                )
                for iterator in generated.launch.work_iterators
            }
        workspace_base = torch.empty(plan.storage.workspace_bytes, dtype=torch.uint8, device=device)
        workspace: dict[str, torch.Tensor] = {
            assignment.storage_id: _workspace_view(workspace_base, assignment)
            for assignment in plan.storage.workspace
        }
        state = cls(
            plan,
            generated,
            input_tensors,
            ready,
            work_iterators,
            max_active_clusters,
            workspace,
            {},
            {},
            (),
            workspace_base,
        )
        state._refresh_values(input_tensors)
        return state

    def rebind(
        self,
        input_tensors: Mapping[str, torch.Tensor],
        ordered: Sequence[torch.Tensor] | None = None,
    ) -> None:
        if self._table_rebind is not None and self._fast_table_rebind(input_tensors, ordered):
            return
        self._refresh_values(input_tensors)

    def reset_ready(self) -> None:
        reset_ready(self.ready)
        self._calls = 0

    def _wrap_calls(self) -> None:
        """Restart the call counter before barrier epochs would pass int32."""
        # Counting barriers reach (ordinal + 1) * workers within a call.
        stride = max(self.generated.launch.barrier_epochs, 1)
        if (self._calls + 1) * stride * self.plan.launch.grid_size > _MAX_EPOCH:
            if self._prepared_calls != self._calls_run:
                raise CompilerError(
                    CompilerDiagnostic(
                        "runtime-binding",
                        "the call counter wraps while launches are still prepared",
                        requirement="ordered table launches",
                    )
                )
            self.reset_ready()

    def graph_replayed(self) -> None:
        """Account for one replay of a graph that captured this state's launch."""
        self._wrap_calls()
        self._calls += 1

    def poison_workspace(self, value: int = 0xE5) -> None:
        self.workspace_base.fill_(value)

    @property
    def values(self) -> dict[str, torch.Tensor]:
        """Tensor bound to each value for the current call.

        The table fast path updates only outputs; other bindings are derived
        again here, keeping this call's outputs, when something reads them."""
        if self._values_stale:
            self._values_stale = False
            outputs = {value_id: self._values[value_id] for value_id in self._allocated_outputs}
            self._refresh_values(self.input_tensors, reuse_outputs=outputs)
        return self._values

    def _refresh_values(
        self,
        input_tensors: Mapping[str, torch.Tensor],
        reuse_outputs: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        plan = self.plan
        if plan.graph is None:
            raise CompilerError(
                CompilerDiagnostic(
                    "runtime-binding",
                    "plan is missing its source graph",
                    requirement="source graph",
                )
            )
        device = next(iter(input_tensors.values())).device
        workspace = self.workspace
        values: dict[str, torch.Tensor] = {}
        self._allocated_outputs = {}
        for binding in plan.storage.bindings:
            if binding.storage_id in workspace:
                values[binding.value_id] = workspace[binding.storage_id]
                continue
            if binding.kind in {"input", "parameter", "state"}:
                values[binding.value_id] = input_tensors[binding.value_id]
                continue
            if binding.kind in {"alias", "output"}:
                internal_alias = _internal_alias(binding.value_id, plan.graph, values)
                if internal_alias is not None and binding.layout is not None:
                    alias_id, alias_tensor = internal_alias
                    values[binding.value_id] = alias_tensor.as_strided(
                        binding.layout.shape,
                        binding.layout.strides,
                        _alias_storage_offset(binding.value_id, alias_id, alias_tensor, plan.graph),
                    )
                    continue
                alias = _boundary_alias(binding.value_id, plan.graph, input_tensors)
                if alias is not None:
                    values[binding.value_id] = alias
                    continue
            if binding.kind == "output":
                if binding.layout is None:
                    raise CompilerError(
                        CompilerDiagnostic(
                            "runtime-binding",
                            f"output {binding.value_id} has no layout",
                            (binding.value_id,),
                            requirement="output layout",
                        )
                    )
                values[binding.value_id] = (
                    reuse_outputs[binding.value_id]
                    if reuse_outputs is not None and binding.value_id in reuse_outputs
                    else _allocate(binding.layout, device)
                )
                self._allocated_outputs[binding.value_id] = binding.layout
                continue
            if binding.kind == "alias":
                raise CompilerError(
                    CompilerDiagnostic(
                        "runtime-binding",
                        f"cannot resolve alias {binding.value_id}",
                        (binding.value_id,),
                        requirement="boundary alias",
                    )
                )
        # Table launches bind only one template group per GEMM variant; the
        # kernel writes every phase's descriptors from the address table.
        template_phases = set(self.generated.launch.gemm_template_phases)
        table_launch = self.generated.launch.launch_policy == "megabake_cooperative_table"
        g0_tensors = tuple(
            _g0_operands(phase.body, selected, plan.graph, input_tensors, values)
            for phase in plan.phases
            if is_table_gemm(phase.body.provider)
            and (not table_launch or phase.index in template_phases)
            for selected in plan.selection.selections
            if selected.region_id == phase.region_id
        )
        self.input_tensors = input_tensors
        self._values = values
        self._values_stale = False
        self.outputs = {value_id: values[value_id] for value_id in plan.graph.outputs}
        if reuse_outputs is not None:
            # Diagnostic re-derivation keeps the fast path and its templates.
            return
        self.g0_tensors = g0_tensors
        self._slot_addresses = None
        self._table_rebind = None

    def _prepare_table_rebind(self) -> None:
        """Express every slot and graph output as (root, byte offset).

        Roots are the graph inputs, the workspace, and the outputs allocated
        per call. After one full binding, later calls only re-check inputs
        whose tensor object changed and gather root pointers, instead of
        re-deriving thousands of bindings in Python. A value whose storage
        cannot be traced to a root keeps the full path.
        """
        graph = self.plan.graph
        launch = self.generated.launch
        inputs = [self.input_tensors[value_id] for value_id in graph.inputs]
        outputs = list(self._allocated_outputs)
        roots = [*inputs, self.workspace_base, *(self.values[v] for v in outputs)]
        index = {value_id: position for position, value_id in enumerate(graph.inputs)}
        index.update(
            {value_id: len(inputs) + 1 + position for position, value_id in enumerate(outputs)}
        )
        workspace_ids = {
            binding.value_id
            for binding in self.plan.storage.bindings
            if binding.storage_id in self.workspace
        }

        def locate(value_id: str) -> tuple[int, int] | None:
            tensor = self.values.get(value_id)
            if tensor is None:
                return None
            root_id = _storage_root(value_id, graph)
            if value_id in workspace_ids or root_id in workspace_ids:
                position = len(inputs)
            elif value_id in index:
                position = index[value_id]
            elif root_id in index:
                position = index[root_id]
            else:
                return None
            root = roots[position]
            if tensor.untyped_storage().data_ptr() != root.untyped_storage().data_ptr():
                return None
            return position, tensor.data_ptr() - root.data_ptr()

        slot_roots = np.full(len(launch.tensors), -1, dtype=np.int64)
        slot_offsets = np.zeros(len(launch.tensors), dtype=np.int64)
        for slot, argument in enumerate(launch.tensors):
            if argument.layout is None:
                continue
            located = locate(argument.value_id)
            if located is None:
                return
            slot_roots[slot], slot_offsets[slot] = located
        graph_outputs = []
        for value_id in graph.outputs:
            located = locate(value_id)
            if located is None:
                return
            tensor = self.values[value_id]
            position, offset = located
            element = tensor.element_size()
            if offset % element:
                return
            graph_outputs.append(
                (
                    position,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    offset // element,
                    tensor is roots[position],
                )
            )
        self._table_rebind = _TableRebind(
            inputs,
            [self._allocated_outputs[value_id] for value_id in outputs],
            outputs,
            np.where(slot_roots < 0, 0, slot_roots),
            np.where(slot_roots < 0, 0, slot_offsets),
            slot_roots < 0,
            tuple(graph_outputs),
        )
        self._slot_addresses = self._table_rebind.addresses(roots)

    def _fast_table_rebind(
        self,
        input_tensors: Mapping[str, torch.Tensor],
        ordered: Sequence[torch.Tensor] | None = None,
    ) -> bool:
        """Rebind from root pointers; ``ordered`` lists the inputs in graph order.

        Only inputs passed as new tensor objects are checked and have their
        pointer re-read; parameters and state passed again cost nothing.
        """
        rebind = self._table_rebind
        graph = self.plan.graph
        inputs = (
            list(ordered)
            if ordered is not None
            else [input_tensors[value_id] for value_id in graph.inputs]
        )
        pointers = rebind.pointers
        for position, (tensor, previous) in enumerate(zip(inputs, rebind.inputs)):
            if tensor is previous:
                continue
            if (
                tensor.shape != previous.shape
                or tensor.stride() != previous.stride()
                or tensor.dtype != previous.dtype
                or tensor.device != previous.device
            ):
                return False
            pointers[position] = tensor.data_ptr()
        device = self.workspace_base.device
        outputs = [_allocate(layout, device) for layout in rebind.output_layouts]
        first_output = len(inputs) + 1
        for index, tensor in enumerate(outputs):
            pointers[first_output + index] = tensor.data_ptr()
        roots = [*inputs, self.workspace_base, *outputs]
        self._slot_addresses = rebind.slot_addresses()
        rebind.inputs = inputs
        self.input_tensors = input_tensors
        values = self._values
        for value_id, tensor in zip(rebind.output_ids, outputs, strict=True):
            values[value_id] = tensor
        graph_outputs = {}
        for value_id, (position, shape, stride, offset, is_root) in zip(
            graph.outputs, rebind.graph_outputs, strict=True
        ):
            root = roots[position]
            graph_outputs[value_id] = (
                root if is_root else root.as_strided(shape, stride, root.storage_offset() + offset)
            )
        values.update(graph_outputs)
        self.outputs = graph_outputs
        self._values_stale = True
        return True

    def tensor_for_value(self, value_id: str) -> torch.Tensor:
        if value_id in self.outputs:
            return self.outputs[value_id]
        return self.values[value_id]

    @property
    def compile_arguments(self) -> tuple[Any, ...]:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return self.arguments(0, stream, include_constants=True)

    def arguments(
        self, epoch: int, stream: cuda.CUstream, *, include_constants: bool = False
    ) -> tuple[Any, ...]:
        if self.generated.launch.launch_policy == "megabake_cooperative_table":
            return self._table_arguments(stream, include_constants=include_constants)
        normal_views = {
            argument.name: self._cached_view(
                self._normal_view_cache,
                argument.name,
                self.values[argument.value_id],
            )
            for argument in self.generated.launch.tensors
            if not argument.kind.startswith("g0_")
        }
        g0_view_groups = tuple(
            self._cached_gemm_tensors(group_index, tensors)
            for group_index, tensors in enumerate(self.g0_tensors)
        )
        g0_views = tuple(view for group in g0_view_groups for view in group)
        g0_names = [
            argument.name
            for argument in self.generated.launch.tensors
            if argument.kind.startswith("g0_")
        ]
        g0_by_name = dict(zip(g0_names, g0_views, strict=True))
        max_active_clusters = self._max_active_clusters
        arguments = []
        for name in self.generated.launch.arguments:
            if name == "ready":
                if self._ready_view is None or self._ready_view[0] is not self.ready:
                    self._ready_view = (self.ready, _cute_view(self.ready))
                arguments.append(self._ready_view[1])
            elif name == "workers":
                arguments.append(cutlass.Int32(self.plan.launch.grid_size))
            elif name == "epoch":
                arguments.append(cutlass.Int32(epoch))
            elif name == "work_iterators":
                arguments.append(
                    tuple(
                        self._cached_view(
                            self._work_iterator_view_cache,
                            iterator.name,
                            self.work_iterators[iterator.name],
                        )
                        for iterator in self.generated.launch.work_iterators
                    )
                )
            elif name == "tensors":
                arguments.append((*normal_views.values(), *g0_views))
            elif name == "g0_tensors":
                arguments.append(g0_view_groups)
            elif name in normal_views:
                arguments.append(normal_views[name])
            elif name in g0_by_name:
                arguments.append(g0_by_name[name])
            elif name == "max_active_clusters":
                if include_constants:
                    arguments.append(max_active_clusters)
            elif name == "stream":
                arguments.append(stream)
            else:
                raise CompilerError(
                    CompilerDiagnostic(
                        "runtime-binding",
                        f"generated launch has unknown argument {name}",
                        requirement="complete launch binding",
                    )
                )
        return tuple(arguments)

    def _table_arguments(
        self,
        stream: cuda.CUstream,
        *,
        include_constants: bool = False,
        ring: torch.Tensor | None = None,
    ) -> tuple[Any, ...]:
        """Kernel arguments of a table launch.

        None of them changes between calls: per-call boundary addresses go
        through the dynamic-address ring and barrier epochs come from the
        device call counter. The same tuple is returned while the bound
        tables and views are unchanged, so its packed form can be reused.
        """
        launch = self.generated.launch
        device = self.ready.device
        if self._slot_addresses is None:
            for slot, argument in enumerate(launch.tensors):
                if argument.layout is not None:
                    self._check_slot_layout(slot, argument, self.values[argument.value_id])
            self._prepare_table_rebind()
        if self._slot_addresses is None:
            addresses = np.array(
                [
                    0 if argument.layout is None else self.values[argument.value_id].data_ptr()
                    for argument in launch.tensors
                ]
                or [0],
                dtype=np.int64,
            )
        else:
            addresses = self._slot_addresses
        if launch.gemm_table:
            self._check_gemm_alignment(addresses)
        if launch.resolved_slots:
            # Each phase column's slot address follows the slot addresses.
            if self._resolved_index is None:
                self._resolved_index = np.concatenate(
                    (
                        np.arange(len(addresses), dtype=np.int64),
                        np.asarray(launch.resolved_slots, dtype=np.int64),
                    )
                )
            addresses = addresses[self._resolved_index]
        if self._dynamic_slot_indices is None:
            self._dynamic_slot_indices = np.asarray(launch.dynamic_slots, dtype=np.int64)
        dynamic_slots = self._dynamic_slot_indices
        static_addresses = addresses.copy()
        static_addresses[dynamic_slots] = 0
        if self._address_table is None or not np.array_equal(
            self._address_mirror, static_addresses
        ):
            # Static slots (weights and workspace) change only on rebinding;
            # the kernel prologue writes the per-call boundary slots itself.
            table = torch.from_numpy(static_addresses)
            if self._address_table is None:
                self._address_table = table.to(device)
            else:
                self._address_table.copy_(table)
            self._address_mirror = static_addresses
            self.address_table_uploads += 1
        self._pending_dynamic_addresses = addresses[dynamic_slots]
        if self._phase_table is None:
            phase_table = torch.tensor(launch.phase_table, dtype=torch.int32, device=device)
            self._phase_table = (phase_table, _cute_view(phase_table))
        if self._ready_view is None or self._ready_view[0] is not self.ready:
            self._ready_view = (self.ready, _cute_view(self.ready))
        if self._work_iterator_views is None:
            self._work_iterator_views = tuple(
                _cute_view(self.work_iterators[iterator.name]) for iterator in launch.work_iterators
            )
        if self._dynamic_slot_table is None:
            slot_table = torch.tensor(
                list(launch.dynamic_slots) or [0], dtype=torch.int32, device=device
            )
            self._dynamic_slot_table = (slot_table, _cute_view(slot_table))

        if self._phase_scratch is None:
            # Counters start at zero, and every phase leaves its own at zero.
            floats = torch.empty(
                max(launch.phase_float_elements, 4), dtype=torch.float32, device=device
            )
            counters = torch.zeros(
                max(launch.phase_counter_count, 4), dtype=torch.int32, device=device
            )
            self._phase_scratch = (
                (floats, _cute_view(floats)),
                (counters, _cute_view(counters)),
            )
        if self._dynamic_ring is None:
            host = _pinned_address_ring(
                len(launch.dynamic_slots) + 1, launch.grid_size, _phase_names(self.generated)
            )
            self._dynamic_ring = (host, host.numpy(), [None] * DYNAMIC_ADDRESS_RING)
        if launch.gemm_table:
            if self._gemm_table is None:
                gemm_table = torch.tensor(launch.gemm_table, dtype=torch.int32, device=device)
                self._gemm_table = (gemm_table, _cute_view(gemm_table))
            if self._stream_table is None and launch.stream_table:
                stream_table = torch.tensor(launch.stream_table, dtype=torch.int32, device=device)
                self._stream_table = (stream_table, _cute_view(stream_table))
            if self._tma_descriptors is None:
                descriptors = torch.zeros(
                    len(launch.gemm_table) * TABLE_DESCRIPTORS_PER_GEMM * 16,
                    dtype=torch.int64,
                    device=device,
                )
                self._tma_descriptors = (
                    descriptors,
                    from_dlpack(descriptors, assumed_align=128),
                )
            gemm_views = tuple(
                self._cached_gemm_tensors(index, tensors)
                for index, tensors in enumerate(self.g0_tensors)
            )
        else:
            gemm_views = ()
        parts = (
            self._ready_view[1],
            self._cached_view(self._normal_view_cache, "addresses", self._address_table),
            self._phase_table[1],
            self._work_iterator_views,
            self._dynamic_slot_table[1],
            self._phase_scratch[0][1],
            self._phase_scratch[1][1],
            *(
                (
                    self._gemm_table[1],
                    *((self._stream_table[1],) if launch.stream_table else ()),
                    self._tma_descriptors[1],
                    *gemm_views,
                )
                if launch.gemm_table
                else ()
            ),
        )
        cached = self._table_argument_cache
        handle = int(stream)
        if (
            not include_constants
            and ring is None
            and cached is not None
            and cached[2] == handle
            and len(cached[0]) == len(parts)
            and all(old is new for old, new in zip(cached[0], parts))
        ):
            return cached[1]
        arguments: list[Any] = [
            self._ready_view[1],
            cutlass.Int32(self.plan.launch.grid_size),
            parts[1],
            self._phase_table[1],
            self._work_iterator_views,
            cutlass.Int64((self._dynamic_ring[0] if ring is None else ring).data_ptr()),
            self._dynamic_slot_table[1],
            self._phase_scratch[0][1],
            self._phase_scratch[1][1],
        ]
        if launch.gemm_table:
            arguments.extend(
                (
                    self._gemm_table[1],
                    *((self._stream_table[1],) if launch.stream_table else ()),
                    self._tma_descriptors[1],
                    gemm_views,
                )
            )
            if include_constants:
                arguments.append(self._max_active_clusters)
        arguments.append(stream)
        result = tuple(arguments)
        if not include_constants and ring is None:
            self._table_argument_cache = (parts, result, handle)
        return result

    def prepare_launch(
        self, compiled: Any, epoch: int, stream: torch.cuda.Stream
    ) -> tuple[Any, ...]:
        """Bind this launch's arguments; ``run_launch`` enqueues it.

        Table launches are numbered in order: launch ``n`` reads ring entry
        ``n % DYNAMIC_ADDRESS_RING``, which is stamped with ``n``, and the
        kernel traps if the stamp does not match its device call counter.
        Up to ``DYNAMIC_ADDRESS_RING`` launches can be prepared ahead, and
        they must run in the order they were prepared.
        """
        cu_stream = cuda.CUstream(stream.cuda_stream)
        if self.generated.launch.launch_policy != "megabake_cooperative_table":
            arguments = self.arguments(epoch, cu_stream)
            execution, keep = _pack_arguments(compiled, arguments)
            return (None, arguments, execution, keep)
        if _stream_is_capturing(stream):
            # A graph replays these addresses on every call, so every ring
            # entry holds them, stamped -1; the eager ring stays untouched.
            ring = _pinned_address_ring(
                len(self.generated.launch.dynamic_slots) + 1,
                self.generated.launch.grid_size,
                _phase_names(self.generated),
            )
            arguments = self._table_arguments(cu_stream, ring=ring)
            ring[:, 0] = -1
            ring[:, 1:] = torch.from_numpy(self._pending_dynamic_addresses)
            self._graph_rings.append(ring)
            _graph_capture.append(self)
            return (None, arguments, *_pack_arguments(compiled, arguments))
        self._wrap_calls()
        arguments = self._table_arguments(cu_stream)
        call = self._calls
        _host, rows, events = self._dynamic_ring
        entry = call % DYNAMIC_ADDRESS_RING
        if events[entry] is not None:
            # The launch that read this entry DYNAMIC_ADDRESS_RING calls ago.
            events[entry].synchronize()
        rows[entry, 0] = call
        rows[entry, 1 : 1 + len(self._pending_dynamic_addresses)] = self._pending_dynamic_addresses
        self._calls += 1
        self._prepared_calls += 1
        packed = self._packed_arguments
        if packed is None or packed[0] is not arguments:
            packed = (arguments, *_pack_arguments(compiled, arguments))
            self._packed_arguments = packed
        return (call, arguments, packed[1], packed[2])

    def run_launch(self, compiled: Any, token: tuple[Any, ...], stream: torch.cuda.Stream) -> None:
        call, arguments, execution, _ = token
        if execution is not None:
            compiled.run_compiled_program(execution)
        else:
            compiled(*arguments)
        if call is None:
            return
        self._calls_run += 1
        _, _, events = self._dynamic_ring
        entry = call % DYNAMIC_ADDRESS_RING
        event = events[entry] or torch.cuda.Event()
        event.record(stream)
        events[entry] = event

    def launch(self, compiled: Any, epoch: int, stream: torch.cuda.Stream) -> None:
        """Enqueue one launch of ``compiled`` on ``stream`` without blocking."""
        self.run_launch(compiled, self.prepare_launch(compiled, epoch, stream), stream)

    def _check_gemm_alignment(self, addresses: np.ndarray) -> None:
        """TMA descriptors need 16-byte-aligned operand base addresses."""
        gemm_slots = getattr(self, "_gemm_slots", None)
        if gemm_slots is None:
            table = np.asarray(self.generated.launch.gemm_table, dtype=np.int64)
            second = table[:, GEMM_SECOND_A_COLUMN] if table.shape[1] > GEMM_SECOND_A_COLUMN else []
            gemm_slots = np.unique(
                np.concatenate(
                    (
                        # Columns 1-3 are the A, B and C slots; paired GEMMs
                        # also load their second A with TMA.
                        table[:, 1:4].ravel(),
                        np.asarray([slot - 1 for slot in second if slot > 0], dtype=np.int64),
                    )
                )
            )
            self._gemm_slots = gemm_slots
        misaligned = gemm_slots[addresses[gemm_slots] % 16 != 0]
        if misaligned.size:
            argument = self.generated.launch.tensors[int(misaligned[0])]
            raise CompilerError(
                CompilerDiagnostic(
                    "runtime-binding",
                    f"{argument.value_id} is not 16-byte aligned for its TMA GEMM",
                    (argument.value_id,),
                    requirement="16-byte-aligned GEMM operand",
                )
            )

    def _check_slot_layout(self, slot: int, argument: Any, tensor: torch.Tensor) -> None:
        identity = _tensor_identity(tensor)
        if self._checked_layouts.get(slot) == identity:
            return
        layout = argument.layout
        shape = tuple(layout.shape)
        if tuple(tensor.shape) != shape or any(
            size > 1 and actual != expected
            for size, actual, expected in zip(shape, tensor.stride(), layout.strides, strict=True)
        ):
            raise CompilerError(
                CompilerDiagnostic(
                    "runtime-binding",
                    f"{argument.value_id} has shape {tuple(tensor.shape)} and strides "
                    f"{tuple(tensor.stride())}; the kernel was compiled for "
                    f"{shape} and {tuple(layout.strides)}",
                    (argument.value_id,),
                    requirement="compiled slot layout",
                )
            )
        self._checked_layouts[slot] = identity

    @staticmethod
    def _cached_view(
        cache: dict[Any, tuple[tuple[Any, ...], Any]], key: Any, tensor: torch.Tensor
    ) -> Any:
        entry = cache.get(key)
        identity = _tensor_identity(tensor)
        if entry is None or entry[0] != identity:
            entry = (identity, _cute_view(tensor))
            cache[key] = entry
        return entry[1]

    def _cached_gemm_tensors(
        self, group_index: int, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[Any, ...]:
        entry = self._g0_view_cache.get(group_index)
        identities = tuple(_tensor_identity(tensor) for tensor in tensors)
        if (
            entry is None
            or len(entry[0]) != len(tensors)
            or any(old != new for old, new in zip(entry[0], identities))
        ):
            splits = self.generated.launch.gemm_splits
            entry = (
                identities,
                cute_gemm_tensors(*tensors, splits=splits[group_index] if splits else 1),
            )
            self._g0_view_cache[group_index] = entry
        return entry[1]


def _phase_names(generated: Any) -> dict[int, str]:
    """Phase index -> "body (region)", for stall reports."""
    names: dict[int, str] = {}
    for entry in getattr(generated, "source_map", ()):
        if entry.phase_index is not None and entry.body_name and entry.phase_index not in names:
            names[entry.phase_index] = f"{entry.body_name} ({entry.region_id})"
    return names


def _pinned_address_ring(
    words: int, workers: int, phase_names: dict[int, str] | None = None
) -> torch.Tensor:
    """The host-mapped dynamic-address ring, (DYNAMIC_ADDRESS_RING, words),
    followed in the same pinned allocation by the stall region the kernel
    writes on a timeout (runtime.stall_report; X0.3)."""
    ring_words = DYNAMIC_ADDRESS_RING * words
    flat = torch.zeros(ring_words + stall_region_words(workers), dtype=torch.int64).pin_memory()
    ring = flat[:ring_words].view(DYNAMIC_ADDRESS_RING, words)
    register_stall_region(ring, ring_words, workers, phase_names)
    return ring


def _pack_arguments(compiled: Any, arguments: tuple[Any, ...]) -> tuple[Any, Any]:
    """CuTe execution arguments for ``arguments`` and what keeps them alive,
    or ``(None, None)`` when the executor cannot launch packed arguments."""
    if not (
        hasattr(compiled, "generate_execution_args") and hasattr(compiled, "run_compiled_program")
    ):
        return None, None
    generated = compiled.generate_execution_args(*arguments)
    execution = generated[0] if isinstance(generated, tuple) and len(generated) == 2 else generated
    return execution, generated


def _tensor_identity(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def _validate_state_capacities(plan: KernelPlan, input_tensors: Mapping[str, torch.Tensor]) -> None:
    for state in plan.storage.state:
        valid_length = input_tensors[state.valid_length_value_id]
        length = int(valid_length.item())
        if length < 0 or length + state.query_length > state.capacity:
            raise CompilerError(
                CompilerDiagnostic(
                    "state-capacity",
                    (
                        f"KV valid length {length} plus {state.query_length} exceeds "
                        f"capacity {state.capacity}"
                    ),
                    (state.valid_length_value_id,),
                    requirement="bounded KV state",
                )
            )


def _workspace_view(base: torch.Tensor, assignment: Any) -> torch.Tensor:
    dtype = getattr(torch, assignment.layout.dtype.removeprefix("torch."))
    element_size = dtype.itemsize
    if assignment.required_bytes % element_size:
        raise CompilerError(
            CompilerDiagnostic(
                "workspace-binding",
                f"workspace slice {assignment.storage_id} is not element-aligned",
                requirement="dtype-aligned workspace slice",
            )
        )
    storage = base.narrow(0, assignment.offset_bytes, assignment.required_bytes).view(dtype)
    return storage.as_strided(assignment.layout.shape, assignment.layout.strides)


def _allocate(layout: Any, device: torch.device) -> torch.Tensor:
    dtype = getattr(torch, layout.dtype.removeprefix("torch."))
    return torch.empty_strided(layout.shape, layout.strides, dtype=dtype, device=device)


def _boundary_alias(
    value_id: str,
    graph: ComputeGraph,
    input_tensors: Mapping[str, torch.Tensor],
    seen: set[str] | None = None,
) -> torch.Tensor | None:
    seen = seen or set()
    if value_id in seen:
        return None
    seen.add(value_id)
    if value_id in input_tensors:
        return input_tensors[value_id]
    value = graph.values.get(value_id)
    if value is None:
        return None
    for alias in value.storage_relation.aliases:
        base = _boundary_alias(alias, graph, input_tensors, seen)
        if base is not None:
            return base.as_strided(
                base.shape if value.shape is None else value.shape,
                base.stride() if value.strides is None else value.strides,
                _alias_storage_offset(value_id, alias, base, graph),
            )
    storage_id = value.storage_relation.storage_id
    if storage_id in input_tensors:
        base = input_tensors[storage_id]
        return base.as_strided(
            base.shape if value.shape is None else value.shape,
            base.stride() if value.strides is None else value.strides,
            _alias_storage_offset(value_id, storage_id, base, graph),
        )
    return None


def _alias_storage_offset(
    value_id: str, base_id: str, base: torch.Tensor, graph: ComputeGraph
) -> int:
    # FX offsets are relative to the shared source storage; rebase them onto
    # the runtime tensor that backs the aliased value.
    value = graph.values.get(value_id)
    base_value = graph.values.get(base_id)
    value_offset = 0 if value is None else value.storage_offset
    base_offset = 0 if base_value is None else base_value.storage_offset
    return base.storage_offset() + value_offset - base_offset


def _internal_alias(
    value_id: str,
    graph: ComputeGraph,
    values: Mapping[str, torch.Tensor],
) -> tuple[str, torch.Tensor] | None:
    candidates = [value_id]
    seen: set[str] = set()
    while candidates:
        candidate = candidates.pop()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate != value_id and candidate in values:
            return candidate, values[candidate]
        value = graph.values.get(candidate)
        if value is None:
            continue
        relation = value.storage_relation
        candidates.extend((*relation.aliases, relation.storage_id))
    return None


def _g0_operands(
    body: Any,
    selected: Any,
    graph: ComputeGraph,
    input_tensors: Mapping[str, torch.Tensor],
    values: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """CUTLASS ``(A, B, C)`` for a G0 GEMM: activation, weight and output, or
    weight, activation and the transposed output for swapAB."""
    # A bias (addmm) and fused epilogue operands are read by the GEMM
    # epilogue, not bound as operands; a paired GEMM's second weight shares
    # the first one's descriptor template.
    activation_load, *weight_loads = selected.program.loads[:2]
    activation = _g0_activation_view(
        activation_load.value_id,
        _g0_source_tensor(activation_load.value_id, graph, input_tensors, values),
        graph,
    )
    weights = tuple(
        _g0_source_tensor(load.value_id, graph, input_tensors, values) for load in weight_loads
    )
    output = values[selected.program.store.value_id]
    if output.dim() != 2:
        # A fused epilogue stores its chain's result, viewed as the GEMM output.
        output = output.reshape(selected.program.store.shape)
    gemm_impl = table_gemm(body.provider)
    if gemm_impl is not None:
        return gemm_impl.arrange_operands(activation, weights, output)
    if is_swap_ab(body):
        return (*weights, activation, output.t())
    return (activation, *weights, output)


def _g0_source_tensor(
    value_id: str,
    graph: ComputeGraph,
    input_tensors: Mapping[str, torch.Tensor],
    value_tensors: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if value_id in input_tensors:
        return input_tensors[value_id].detach()
    value = graph.values.get(value_id)
    if value is None:
        raise CompilerError(
            CompilerDiagnostic(
                "runtime-binding",
                f"GEMM input {value_id} is missing",
                (value_id,),
                requirement="GEMM input binding",
            )
        )
    for alias in value.storage_relation.aliases:
        if alias in input_tensors:
            return input_tensors[alias].detach()
        return _g0_source_tensor(alias, graph, input_tensors, value_tensors)
    return _g0_value_source_tensor(value_id, graph, value_tensors)


def _g0_activation_view(value_id: str, source: torch.Tensor, graph: ComputeGraph) -> torch.Tensor:
    # The G0 body flattens leading axes of A, which is only valid when the
    # source root already ends in the GEMM K axis; otherwise bind the 2-D view.
    value = graph.values.get(value_id)
    if (
        value is None
        or value.shape is None
        or value.strides is None
        or len(value.shape) != 2
        or tuple(source.shape[-1:]) == tuple(value.shape[-1:])
    ):
        return source
    root = graph.values.get(_storage_root(value_id, graph))
    root_offset = 0 if root is None else root.storage_offset
    return source.as_strided(
        value.shape,
        value.strides,
        source.storage_offset() + value.storage_offset - root_offset,
    )


def _storage_root(value_id: str, graph: ComputeGraph) -> str:
    seen: set[str] = set()
    while value_id not in seen:
        seen.add(value_id)
        value = graph.values.get(value_id)
        if value is None or not value.storage_relation.aliases:
            return value_id
        value_id = value.storage_relation.aliases[0]
    return value_id


def _g0_value_source_tensor(
    value_id: str,
    graph: ComputeGraph,
    value_tensors: Mapping[str, torch.Tensor],
    seen: set[str] | None = None,
) -> torch.Tensor:
    seen = seen or set()
    if value_id in seen:
        raise CompilerError(
            CompilerDiagnostic(
                "runtime-binding",
                f"GEMM input {value_id} has a cyclic alias",
                (value_id,),
                requirement="acyclic boundary alias",
            )
        )
    seen.add(value_id)
    if value_id in value_tensors:
        return value_tensors[value_id].detach()
    value = graph.values.get(value_id)
    if value is None:
        raise CompilerError(
            CompilerDiagnostic(
                "runtime-binding",
                f"GEMM input {value_id} is missing",
                (value_id,),
                requirement="GEMM input binding",
            )
        )
    for alias in value.storage_relation.aliases:
        if alias in value_tensors:
            return value_tensors[alias].detach()
        return _g0_value_source_tensor(alias, graph, value_tensors, seen)
    storage_id = value.storage_relation.storage_id
    if storage_id in value_tensors:
        return value_tensors[storage_id].detach()
    raise CompilerError(
        CompilerDiagnostic(
            "runtime-binding",
            f"GEMM input {value_id} does not alias a boundary tensor",
            (value_id,),
            requirement="GEMM boundary input",
        )
    )


def _cute_view(tensor: torch.Tensor) -> Any:
    if tensor.data_ptr() % 16:
        raise CompilerError(
            CompilerDiagnostic(
                "runtime-binding",
                f"tensor at {tensor.data_ptr():#x} is not 16-byte aligned",
                requirement="16-byte tensor alignment",
            )
        )
    result = from_dlpack(tensor.detach(), assumed_align=16)
    leading_dimension = next(
        (index for index, stride in enumerate(tensor.stride()) if stride == 1),
        None,
    )
    if leading_dimension is None:
        return result
    return result.mark_layout_dynamic(leading_dim=leading_dimension)


def _source_op_ids(plan: KernelPlan) -> tuple[str, ...]:
    if plan.graph is None:
        return ()
    return tuple(plan.graph.ops)


def _region_ids(plan: KernelPlan) -> tuple[str, ...]:
    if plan.graph is None:
        return ()
    return tuple(region.id for region in plan.graph.regions)
