"""Locate the first generated region whose values diverge from eager FX.

The model is captured through the normal backend, compiled with every
intermediate in its own workspace slice, and run once. Each materialized value
is then compared with the same FX node executed eagerly, in phase order, so the
first divergent row names the responsible region and body.

    python tools/diff_intermediates.py --layers 2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.fx

from megabake import (
    assign_storage,
    derive_dependencies,
    make_backend,
    make_phase_schedule,
    select_bodies,
)
from megabake.bodies import SelectionPolicy
from megabake.bodies.cuda.sm90.gemm import HopperLaunchOptions
from megabake.compiler import (
    _TABLE_LAUNCH_OPS,
    _current_parameter_tensors,
    _post_grad_graph,
    bind_callable,
    compile_plan,
)


@dataclass(frozen=True)
class ValueDifference:
    phase: int
    value_id: str
    region_id: str
    relative_error: float


def diff_intermediates(
    model: torch.nn.Module, inputs: tuple, *, mode: str = "prefill"
) -> list[ValueDifference]:
    backend = make_backend(mode=mode, compile_kernel=False, cache=False)
    with torch.no_grad():
        torch.compile(model, backend=backend, fullgraph=True, dynamic=False)(
            *(_clone(value) for value in inputs)
        )
    prepared = backend.megabake_last_prepared

    eager: dict[str, torch.Tensor] = {}

    class Recorder(torch.fx.Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            if isinstance(result, torch.Tensor):
                eager[node.name] = result.clone()
            return result

    with torch.no_grad():
        Recorder(prepared.graph).run(*(_clone(value) for value in prepared.example_inputs))

    graph, target = _post_grad_graph(prepared, None)
    policy = SelectionPolicy(
        mode=mode, target_options=HopperLaunchOptions(tma_gemm=len(graph.ops) <= _TABLE_LAUNCH_OPS)
    )
    selection = select_bodies(graph, target, policy)
    plan = assign_storage(
        make_phase_schedule(selection, derive_dependencies(selection)),
        reuse_workspace=False,
    )
    compiled = compile_plan(plan, prepared.example_inputs)
    executable = bind_callable(
        compiled,
        prepared.context,
        parameter_tensors=_current_parameter_tensors(
            compiled, prepared.context, prepared.example_inputs
        ),
    )
    with torch.no_grad():
        executable(*(_clone(value) for value in prepared.example_inputs))
    torch.cuda.synchronize()

    phase_of = {phase.region_id: phase.index for phase in compiled.plan.phases}
    region_of = {
        operation_id: selected.region_id
        for selected in compiled.plan.selection.selections
        for operation_id in selected.source_op_ids
    }
    rows = []
    for value_id, actual in executable.megabake_runtime_state.values.items():
        expected = eager.get(value_id.removeprefix("value:"))
        value = graph.values.get(value_id)
        region_id = region_of.get(value.producer) if value is not None else None
        if expected is None or region_id is None:
            continue
        if actual.shape != expected.shape:
            error = float("inf")
        elif expected.is_floating_point():
            error = (
                (actual.float() - expected.float()).norm()
                / expected.float().norm().clamp_min(1e-12)
            ).item()
        else:
            error = (actual != expected).float().mean().item()
        rows.append(ValueDifference(phase_of[region_id], value_id, region_id, error))
    return sorted(rows, key=lambda row: (row.phase, row.value_id))


def _clone(value):
    return value.clone() if isinstance(value, torch.Tensor) else value


def main() -> None:
    from tests.fixtures import SmolLMArchitecture

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.02)
    arguments = parser.parse_args()

    torch.manual_seed(31)
    model = SmolLMArchitecture(layers=arguments.layers, device="cuda")
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    attention_mask = torch.ones_like(input_ids)
    rows = diff_intermediates(model, (input_ids, attention_mask))
    divergent = [row for row in rows if row.relative_error > arguments.tolerance]
    print(f"compared {len(rows)} values; {len(divergent)} exceed {arguments.tolerance}")
    for row in divergent[:20]:
        print(
            f"phase {row.phase:5d}  {row.region_id:45s} {row.value_id:35s} {row.relative_error:.4g}"
        )


if __name__ == "__main__":
    main()
