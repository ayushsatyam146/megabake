from __future__ import annotations

import ast
import re

import torch

from megabake import (
    assign_storage,
    derive_dependencies,
    emit_cute,
    make_phase_schedule,
    select_bodies,
    verify_plan,
)
from megabake.bodies import SelectionPolicy, TargetFacts
from megabake.compiler import _gather_index_is_static
from megabake.fx_handler.export import capture_exported_program
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions
from tests.fixtures import (
    SMOLLM_MODEL_NAME,
    SmolLMArchitecture,
    StatefulSmolLMArchitecture,
)


def _assert_key_accesses_use_loop_variable(source: str) -> None:
    assert not re.search(r"tensor_bindings\[[^\n]*\bk\b", source)


def _target() -> TargetFacts:
    return TargetFacts(
        architecture="sm90a",
        threads_per_warp=32,
        max_threads_per_cta=1024,
        max_shared_memory_per_cta=232_448,
        max_registers_per_thread=255,
        multiprocessor_count=60,
        supports_tma=True,
        supports_wgmma=True,
    )


def test_task31_builds_and_emits_complete_smollm_architecture_plan(tmp_path):
    torch.manual_seed(31)
    model = SmolLMArchitecture(layers=2, device="meta", dtype=torch.float32)
    input_ids = torch.empty(1, 4, dtype=torch.int64, device="meta")
    attention_mask = torch.ones(1, 4, dtype=torch.int64, device="meta")
    exported = torch.export.export(
        model,
        (input_ids,),
        {
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": False,
        },
    )
    capture = capture_exported_program(
        exported,
        args=(input_ids,),
        kwargs={
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": False,
        },
        model=model,
        model_name=SMOLLM_MODEL_NAME,
    )
    assert capture.context.requirements["preserve_output_wrappers"]

    graph = form_regions(canonicalize_semantics(import_fx(capture.graph, capture.context)))
    assert all(operation.tensor_program is not None for operation in graph.ops.values())
    # Both layers are captured; per-layer duplicates (the causal mask, RoPE
    # tables) merge into one copy each.
    assert len(graph.ops) > 220
    assert _gather_index_is_static(graph, "value:unsqueeze_3")
    assert not _gather_index_is_static(graph, graph.inputs[0])

    selection = select_bodies(graph, _target(), SelectionPolicy(mode="prefill", allow_tuned=False))
    assert selection.unsupported_regions == ()
    body_names = {selection.body.name for selection in selection.selections}
    assert {"generic_gather", "generic_reduction"} <= body_names
    # Hugging Face's RMSNorm and rotate_half RoPE each fuse into one region;
    # RoPE absorbs the concat.
    region_kinds = {selected.region_id.rsplit(":", 1)[-1] for selected in selection.selections}
    assert {"rms_norm", "rope"} <= region_kinds

    plan = assign_storage(make_phase_schedule(selection, derive_dependencies(selection)))
    verify_plan(plan)
    # At least one phase per Q/K/V/O, gate/up/down, and vocabulary projection.
    assert len(plan.phases) >= 7 * 2 + 1
    assert plan.storage.workspace_bytes > 0

    generated = emit_cute(plan, tmp_path)
    ast.parse(generated.source)
    _assert_key_accesses_use_loop_variable(generated.source)
    assert generated.source_path.exists()
    assert generated.source_map_path.exists()


def test_task31_builds_and_emits_stateful_smollm_decode_plan(tmp_path):
    torch.manual_seed(31)
    model = StatefulSmolLMArchitecture(layers=2, device="cpu", dtype=torch.bfloat16)
    input_ids = torch.empty(1, 1, dtype=torch.int64, device="cpu")
    position_ids = torch.empty(1, 1, dtype=torch.int64, device="cpu")
    key_caches = tuple(
        torch.empty(1, 4, 3, 64, dtype=torch.bfloat16, device="cpu") for _ in range(2)
    )
    value_caches = tuple(
        torch.empty(1, 4, 3, 64, dtype=torch.bfloat16, device="cpu") for _ in range(2)
    )
    valid_lengths = tuple(torch.empty(1, dtype=torch.int32, device="cpu") for _ in range(2))
    args = (input_ids, position_ids, *key_caches, *value_caches, *valid_lengths)
    exported = torch.export.export(model, args)
    capture = capture_exported_program(
        exported,
        args=args,
        model=model,
        model_name=SMOLLM_MODEL_NAME,
    )

    graph = form_regions(canonicalize_semantics(import_fx(capture.graph, capture.context)))

    selection = select_bodies(graph, _target(), SelectionPolicy(mode="decode"))
    assert selection.unsupported_regions == ()
    plan = assign_storage(make_phase_schedule(selection, derive_dependencies(selection)))
    verify_plan(plan)
    assert len(plan.storage.state) == 2
    assert all(state.capacity == 4 for state in plan.storage.state)

    generated = emit_cute(plan, tmp_path)
    ast.parse(generated.source)
    _assert_key_accesses_use_loop_variable(generated.source)
    assert generated.source_path.exists()


def _sliced_projection_graph():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            heads = (source * 2).view(1, 4, 9, 64)
            flattened = heads.reshape(4, 576)
            return flattened @ weight.T, heads[..., 32:]

    source = torch.randn(1, 4, 576)
    weight = torch.randn(576, 576)
    exported = torch.export.export(Model(), (source, weight))
    capture = capture_exported_program(exported, args=(source, weight))
    return import_fx(capture.graph, capture.context)


def test_task31_gemm_activation_binds_its_two_dimensional_view():
    from megabake.compiler import _g0_activation_view

    graph = _sliced_projection_graph()
    flattened = next(
        value
        for value in graph.values.values()
        if value.shape == (4, 576) and value.storage_relation.aliases
    )
    root = torch.randn(1, 4, 9, 64)
    view = _g0_activation_view(flattened.id, root, graph)
    assert view.shape == (4, 576)
    torch.testing.assert_close(view, root.reshape(4, 576))


def test_task31_alias_views_keep_source_storage_offsets():
    from megabake.compiler import _alias_storage_offset

    graph = _sliced_projection_graph()
    sliced = next(value for value in graph.values.values() if value.storage_offset == 32)
    base_id = sliced.storage_relation.storage_id
    base = torch.empty(1024)[16:]
    assert _alias_storage_offset(sliced.id, base_id, base, graph) == 48


def test_task31_plan_search_restores_state_between_runs():
    from megabake.compiler import _StateSnapshot

    model = StatefulSmolLMArchitecture(layers=1, device="cpu", dtype=torch.bfloat16)
    args = (
        torch.zeros(1, 1, dtype=torch.int64),
        torch.zeros(1, 1, dtype=torch.int64),
        torch.zeros(1, 4, 3, 64, dtype=torch.bfloat16),
        torch.zeros(1, 4, 3, 64, dtype=torch.bfloat16),
        torch.zeros(1, dtype=torch.int32),
    )
    capture = capture_exported_program(torch.export.export(model, args), args=args, model=model)
    graph = canonicalize_semantics(import_fx(capture.graph, capture.context))
    graph_inputs = capture.binding_values(args)
    snapshot = _StateSnapshot.create(graph, graph_inputs)

    assert len(snapshot.pairs) == 3
    for private, _ in snapshot.pairs:
        assert not any(private is value for value in graph_inputs)
        private.add_(1)
    snapshot.reset()
    assert all(torch.equal(private, original) for private, original in snapshot.pairs)
    assert all(int(value.abs().sum()) == 0 for value in args[2:])


def _table_decode_plan(monkeypatch, counted: str):
    """The 2-layer SmolLM-architecture decode plan as a table launch."""
    from megabake.bodies.catalog import launch_options

    monkeypatch.setenv("MEGABAKE_COUNTED_GEMM", counted)
    torch.manual_seed(31)
    model = StatefulSmolLMArchitecture(layers=2, device="cpu", dtype=torch.bfloat16)
    args = (
        torch.empty(1, 1, dtype=torch.int64),
        torch.empty(1, 1, dtype=torch.int64),
        *(torch.empty(1, 4, 3, 64, dtype=torch.bfloat16) for _ in range(4)),
        *(torch.empty(1, dtype=torch.int32) for _ in range(2)),
    )
    capture = capture_exported_program(
        torch.export.export(model, args), args=args, model=model, model_name=SMOLLM_MODEL_NAME
    )
    graph = form_regions(canonicalize_semantics(import_fx(capture.graph, capture.context)))
    policy = SelectionPolicy(
        mode="decode",
        table_launch=True,
        target_options=launch_options(graph, _target(), True),
    )
    selection = select_bodies(graph, _target(), policy)
    plan = assign_storage(make_phase_schedule(selection, derive_dependencies(selection)))
    verify_plan(plan)
    return plan


def test_task31_down_projection_counts_on_its_gate_up(monkeypatch):
    # Counted GEMMs join two WGMMA bodies; a GEMV down projection has none.
    monkeypatch.setenv("MEGABAKE_GEMV", "0")
    plan = _table_decode_plan(monkeypatch, "1")
    counted = [phase for phase in plan.phases if phase.counted]
    # One down projection per layer, each counting on its paired gate.up GEMM.
    assert len(counted) == 2
    by_region = {phase.region_id: phase for phase in plan.phases}
    for phase in counted:
        (producer,) = phase.counted
        assert by_region[producer].body.name.startswith("g0_range_swapab_")
        assert "_paired" in by_region[producer].body.name
        # No barrier between the producer's interval and the consumer.
        assert plan.phases[phase.index - 1].barrier is None
    baseline = _table_decode_plan(monkeypatch, "0")
    assert not any(phase.counted for phase in baseline.phases)
    barriers = lambda p: sum(phase.barrier is not None for phase in p.phases)  # noqa: E731
    assert barriers(baseline) == barriers(plan) + 2
