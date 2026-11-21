"""Build the SmolLM prefill and decode megakernels and dump how they are populated.

Each mode is captured through the normal backend, compiled into one CuTe
workload kernel, run against eager execution, and written out for inspection:

    <out>/<mode>/model/...              FX graphs at each capture stage
    <out>/<mode>/generated/.../module.py and source_map.json
    <out>/<mode>/phases.txt             phase -> region, body, FX source ops
    <out>/<mode>/phase_table.txt        device phase table rows, slots decoded
    <out>/<mode>/address_table.txt      device address table checked per slot
    <out>/<mode>/workspace.txt          workspace slices and lifetimes
    <out>/<mode>/summary.json           correctness, launches, resources

    python tools/dump_megakernels.py --layers 2
    python tools/dump_megakernels.py --checkpoint --modes prefill decode
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megabake import make_backend
from megabake.compiler import _RuntimeState
from tests.fixtures import (
    SMOLLM_MODEL_NAME,
    SMOLLM_REVISION,
    SmolLMArchitecture,
    StatefulSmolLM,
)


def _load_model(args: argparse.Namespace) -> torch.nn.Module:
    if not args.checkpoint:
        torch.manual_seed(args.seed)
        return SmolLMArchitecture(layers=args.layers, device="cuda")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        SMOLLM_MODEL_NAME,
        revision=SMOLLM_REVISION,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    return model.to("cuda").eval()


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = max(expected.float().norm().item(), 1e-12)
    return (actual.float() - expected.float()).norm().item() / scale


def _runtime_state(compiled: Any) -> _RuntimeState:
    # The bound callable keeps its runtime state private; the most recent
    # instance bound to this compiled module is the one that just ran.
    states = [
        item
        for item in gc.get_objects()
        # type() rather than isinstance(): dead weak proxies raise on isinstance.
        if type(item) is _RuntimeState and getattr(item, "generated", None) is compiled.generated
    ]
    if not states:
        raise RuntimeError("no runtime state found for the compiled kernel")
    return states[-1]


def _tensor_label(argument: Any) -> str:
    name = argument.binding_name or argument.value_id
    return f"{name} [{argument.kind}]"


def _write_phases(path: Path, compiled: Any) -> None:
    plan = compiled.plan
    graph = plan.graph
    selections = {item.region_id: item for item in plan.selection.selections}
    emissions = {item.index: item for item in compiled.generated.launch.phases}
    lines = [
        (
            f"{len(plan.phases)} phases, "
            f"{compiled.resources.body_variant_count} body variants, "
            f"{len(graph.ops)} graph operations"
        ),
        "",
    ]
    for phase in plan.phases:
        emission = emissions.get(phase.index)
        selection = selections.get(phase.region_id)
        source_ops = selection.source_op_ids if selection else ()
        fx_nodes = [
            fx
            for op_id in source_ops
            if op_id in graph.ops
            for fx in graph.ops[op_id].source_fx_ids
        ]
        lines.append(
            f"[{phase.index:4d}] {phase.region_id}  "
            f"body={phase.body.provider}/{phase.body.name}"
            + (f" variant={emission.body_variant_id}" if emission else "")
            + f" work_items={len(phase.work_items)}"
            + f" barrier={'yes' if phase.barrier else 'no'}"
        )
        lines.append(f"        fx: {', '.join(fx_nodes) or '-'}")
        if selection:
            lines.append(f"        in:  {', '.join(selection.inputs) or '-'}")
            lines.append(f"        out: {', '.join(selection.outputs) or '-'}")
        if phase.dependencies:
            lines.append(f"        after: {', '.join(phase.dependencies)}")
    path.write_text("\n".join(lines) + "\n")


def _write_phase_table(path: Path, compiled: Any) -> None:
    launch = compiled.generated.launch
    tensors = launch.tensors
    lines = [
        "columns: runner_id, barrier_after, cta_offset, kv_length_slot + 1 (0: none), then the tensor slots of the",
        "phase's body calls (GEMM phases: their row in the GEMM table instead)",
        "",
    ]
    for index, row in enumerate(launch.phase_table):
        runner, barrier, offset, length_publish, *slots = row
        emission = launch.phases[index] if index < len(launch.phases) else None
        lines.append(
            f"[{index:4d}] runner={runner} barrier={barrier} cta_offset={offset} "
            f"length_publish={length_publish} "
            f"{emission.region_id if emission else ''} raw={list(row)}"
        )
        if emission is not None and emission.provider == "g0_range":
            variant, *slots = launch.gemm_table[slots[0]]
            lines.append(f"        gemm row {row[4]} variant={variant} (A, B, C slots)")
        for slot in slots:
            if slot == 0 and tensors and tensors[0].layout is None:
                continue
            lines.append(f"        slot {slot:4d} -> {_tensor_label(tensors[slot])}")
    path.write_text("\n".join(lines) + "\n")


def _write_address_table(path: Path, compiled: Any, state: _RuntimeState) -> int:
    launch = compiled.generated.launch
    dynamic = set(launch.dynamic_slots)
    device_table = state._address_table.cpu().tolist() if state._address_table is not None else []
    workspace = state.workspace_base.data_ptr() if state.workspace_base is not None else 0
    mismatches = 0
    lines = [
        "slot  source    table_value         expected            check  tensor",
        "",
    ]
    for slot, argument in enumerate(launch.tensors):
        if argument.layout is None:
            continue
        tensor = state.values.get(argument.value_id)
        expected = tensor.data_ptr() if tensor is not None else None
        actual = device_table[slot] if slot < len(device_table) else None
        # The kernel prologue writes per-call boundary addresses into the
        # device table, so after a call every slot holds its tensor address.
        source = "per-call" if slot in dynamic else "static"
        check = "ok" if actual == expected else "BAD"
        if check != "ok":
            mismatches += 1
        offset = ""
        if expected is not None and argument.kind == "workspace" and workspace:
            offset = f" (+{expected - workspace} in workspace)"
        lines.append(
            f"{slot:4d}  {source:8s}  {actual!s:18s}  {expected!s:18s}  "
            f"{check:5s}  {_tensor_label(argument)}{offset}"
        )
    lines.insert(1, f"address table uploads so far: {state.address_table_uploads}")
    path.write_text("\n".join(lines) + "\n")
    return mismatches


def _write_workspace(path: Path, compiled: Any) -> None:
    storage = compiled.plan.storage
    lines = [f"workspace bytes: {compiled.resources.workspace_bytes}", ""]
    for item in sorted(storage.workspace, key=lambda entry: entry.offset_bytes):
        lifetime = item.lifetime
        lines.append(
            f"offset {item.offset_bytes:10d} size {item.size_bytes:9d}  "
            f"{item.value_id}  producer={item.producer_region_id}  "
            f"lifetime={lifetime}"
        )
    for state in storage.state:
        lines.append(
            f"state K={state.key_cache_value_id} V={state.value_cache_value_id} "
            f"len={state.valid_length_value_id} capacity={state.capacity} "
            f"reads={state.read_range} writes={state.write_range}"
        )
    path.write_text("\n".join(lines) + "\n")


def _dump(mode_dir: Path, backend: Any, extra: dict[str, Any]) -> dict[str, Any]:
    compiled = backend.megabake_last_compiled
    state = _runtime_state(compiled)
    _write_phases(mode_dir / "phases.txt", compiled)
    _write_phase_table(mode_dir / "phase_table.txt", compiled)
    mismatches = _write_address_table(mode_dir / "address_table.txt", compiled, state)
    _write_workspace(mode_dir / "workspace.txt", compiled)
    trace = backend.megabake_last_trace[-1]
    summary = {
        **extra,
        "generated_source": str(compiled.generated.source_path),
        "source_map": str(compiled.generated.source_map_path),
        "phases": len(compiled.plan.phases),
        "graph_operations": len(compiled.plan.graph.ops),
        "address_table_mismatches": mismatches,
        "workload_kernel_launches": trace.workload_kernel_launches,
        "recurring_gpu_operations": trace.recurring_gpu_operations,
        "resources": dataclasses.asdict(compiled.resources),
    }
    (mode_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def _make_backend(mode: str, mode_dir: Path) -> Any:
    return make_backend(
        mode=mode,
        compile_kernel=True,
        cache=False,
        dump_dir=mode_dir,
        scheduler="phase",
    )


def run_prefill(model: torch.nn.Module, args: argparse.Namespace, out: Path) -> dict:
    mode_dir = out / "prefill"
    backend = _make_backend("prefill", mode_dir)
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    calls = [args.input_ids, [token + 7 for token in args.input_ids]]
    errors = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, ids in enumerate(calls):
            input_ids = torch.tensor([ids], device="cuda")
            mask = torch.ones_like(input_ids)
            actual = optimized(input_ids, attention_mask=mask).logits
            if index == 0:
                compile_seconds = time.perf_counter() - started
            expected = model(input_ids, attention_mask=mask).logits
            errors.append(
                {
                    "input_ids": ids,
                    "relative_error_vs_eager": _relative(actual, expected),
                    "argmax_matches": bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
                }
            )
    return _dump(
        mode_dir,
        backend,
        {"mode": "prefill", "compile_and_first_call_seconds": compile_seconds, "calls": errors},
    )


def run_decode(model: torch.nn.Module, args: argparse.Namespace, out: Path) -> dict:
    mode_dir = out / "decode"
    stateful = StatefulSmolLM(model).eval()
    config = model.config
    layers = config.num_hidden_layers
    head_dim = config.hidden_size // config.num_attention_heads
    shape = (1, args.capacity, config.num_key_value_heads, head_dim)

    def fresh_state() -> list[torch.Tensor]:
        caches = [
            torch.zeros(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2 * layers)
        ]
        lengths = [torch.zeros(1, device="cuda", dtype=torch.int32) for _ in range(layers)]
        return caches + lengths

    state = fresh_state()
    reference = [tensor.clone() for tensor in state]
    backend = _make_backend("decode", mode_dir)
    optimized = torch.compile(stateful, backend=backend, fullgraph=True, dynamic=False)
    steps = []
    started = time.perf_counter()
    with torch.no_grad():
        for position in range(args.capacity):
            token = torch.tensor([[args.input_ids[position % len(args.input_ids)]]], device="cuda")
            position_ids = torch.tensor([[position]], device="cuda")
            actual = optimized(token, position_ids, *state)
            if position == 0:
                compile_seconds = time.perf_counter() - started
            expected = stateful(token, position_ids, *reference)
            cache_errors = [
                _relative(state[index], reference[index]) for index in range(2 * layers)
            ]
            lengths = [int(tensor.item()) for tensor in state[2 * layers :]]
            steps.append(
                {
                    "position": position,
                    "relative_error_vs_eager": _relative(actual, expected),
                    "argmax_matches": bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
                    "worst_cache_relative_error": max(cache_errors),
                    "valid_lengths": sorted(set(lengths)),
                    "reference_valid_lengths": sorted(
                        {int(tensor.item()) for tensor in reference[2 * layers :]}
                    ),
                }
            )
    return _dump(
        mode_dir,
        backend,
        {
            "mode": "decode",
            "capacity": args.capacity,
            "compile_and_first_call_seconds": compile_seconds,
            "steps": steps,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--modes", nargs="+", choices=("prefill", "decode"), default=["prefill", "decode"]
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help=f"use {SMOLLM_MODEL_NAME}@{SMOLLM_REVISION[:8]} weights",
    )
    parser.add_argument(
        "--layers", type=int, default=30, help="layer count for the random-weight architecture"
    )
    parser.add_argument(
        "--input-ids",
        type=lambda value: [int(item) for item in value.split(",")],
        default=[1, 504, 3458, 314],
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=4,
        help="KV cache capacity; decode runs this many advancing calls",
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out", type=Path, default=Path("artifacts/megakernels"))
    args = parser.parse_args()

    model = _load_model(args)
    runners = {"prefill": run_prefill, "decode": run_decode}
    for mode in args.modes:
        summary = runners[mode](model, args, args.out)
        rows = summary.get("calls") or summary.get("steps")
        worst = max(row["relative_error_vs_eager"] for row in rows)
        print(
            f"{mode}: {summary['phases']} phases, "
            f"{summary['workload_kernel_launches']} launch/call, "
            f"worst error vs eager {worst:.4f}, "
            f"address mismatches {summary['address_table_mismatches']}, "
            f"{summary['compile_and_first_call_seconds']:.1f} s -> {args.out / mode}"
        )


if __name__ == "__main__":
    main()
