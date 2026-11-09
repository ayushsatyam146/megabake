from __future__ import annotations

import ast
from dataclasses import replace
import json

import torch

from megabake import (
    assign_storage,
    derive_dependencies,
    emit_cute,
    KernelPlan,
    make_phase_schedule,
    select_bodies,
)
from megabake.schedule import make_region_event_schedule
from megabake.plan import ConversionBody, CuTeLayout
from megabake.bodies import SelectionPolicy, TargetFacts
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions


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


def _plan(model, *inputs):
    prepared = prepare_fx_graph(
        torch.export.export(model, inputs).graph_module, inputs
    )
    graph = form_regions(
        canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    )
    selection = select_bodies(
        graph, _target(), SelectionPolicy(allow_tuned=False)
    )
    return assign_storage(
        make_phase_schedule(selection, derive_dependencies(selection))
    )


def test_emit_cute_deduplicates_equivalent_bodies_and_saves_diagnostics(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, left, right):
            return left.sum(dim=-1) + right.sum(dim=-1)

    plan = _plan(Model(), torch.randn(5, 70), torch.randn(5, 70))
    generated = emit_cute(plan, tmp_path)

    ast.parse(generated.source)
    assert generated.source.count("def _body_0(") == 1
    assert generated.source.count("def _body_1(") == 1
    # Both reductions share one body and one table-dispatched call site.
    assert generated.source.count("_body_0(") == 2
    assert "for phase in cutlass.range(3, unroll=1):" in generated.source
    assert "variant = cute.arch.make_warp_uniform(phase_table[phase, 0])" in generated.source
    # The independent reductions share a barrier interval before the join.
    assert [row[:2] for row in generated.launch.phase_table] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert generated.launch.phase_table[0][2:] != generated.launch.phase_table[1][2:]
    # Dense barrier ordinals per call: the address barrier and one phase barrier.
    assert generated.launch.barrier_epochs == 2
    assert generated.source_path.exists()
    assert generated.source_map_path.exists()

    diagnostics = json.loads(generated.source_map_path.read_text())
    kinds = {entry["kind"] for entry in diagnostics["entries"]}
    assert {"device_body", "phase_barrier", "enclosing_kernel", "host_launch_wrapper"} <= kinds
    assert all(generated.launch.work_iterators)
    assert all(iterator.dtype == "torch.int64" for iterator in generated.launch.work_iterators)
    assert generated.launch.body_variant_count == 2
    assert generated.launch.phase_runner_count == 2
    assert generated.launch.source_bytes == len(generated.source.encode())

    source_lines = generated.source.splitlines()
    for symbol, expected in (
        ("_body_0", "sync_threads"),
        # Stall checks (X0.3) are on by default: the checked barrier.
        ("megabake_kernel", "checked_grid_barrier(ready, workers, _ordinal + phase_table[phase, 1],"),
        ("launch", ".launch("),
    ):
        entry = next(
            entry for entry in generated.source_map if entry.symbol == symbol
        )
        mapped_source = "\n".join(
            source_lines[entry.line_start - 1 : entry.line_end]
        )
        assert expected in mapped_source


def test_emit_cute_reuses_phase_runners_with_event_scheduling(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, left, right):
            return left.sum(dim=-1) + right.sum(dim=-1)

    plan = _plan(Model(), torch.randn(5, 70), torch.randn(5, 70))
    event_schedule = make_region_event_schedule(
        plan.selection,
        plan.dependencies,
        worker_count=plan.launch.grid_size,
    )

    generated = emit_cute(plan, tmp_path, event_schedule=event_schedule)

    ast.parse(generated.source)
    assert "cooperative_event_worker_loop(" in generated.source
    assert generated.source.count("def _run_variant_0(") == 1
    assert "(0, _ignore_epoch, _run_variant_0, _publish_phase_0)" in generated.source


def test_emit_cute_puts_g0_inside_one_cooperative_kernel(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(4, 576, dtype=torch.bfloat16)
    weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    prepared = prepare_fx_graph(
        torch.export.export(Model(), (source, weight)).graph_module,
        (source, weight),
    )
    graph = form_regions(
        canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    )
    selection = select_bodies(graph, _target(), SelectionPolicy(allow_tuned=True))
    plan = assign_storage(
        make_phase_schedule(selection, derive_dependencies(selection))
    )

    generated = emit_cute(plan, tmp_path)

    ast.parse(generated.source)
    assert generated.launch.cooperative
    assert generated.launch.kernel_name == "megabake_kernel"
    assert generated.source.count("@cute.kernel") == 1
    assert ".prepare(" in generated.source
    assert ".run(" in generated.source
    assert "launch = body" not in generated.source


def test_emit_cute_reuses_g0_scratch_for_duplicate_body_variants(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, source, weight, other_weight):
            return source @ weight.T, source @ other_weight.T

    source = torch.randn(4, 576, dtype=torch.bfloat16)
    weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    other_weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    prepared = prepare_fx_graph(
        torch.export.export(Model(), (source, weight, other_weight)).graph_module,
        (source, weight, other_weight),
    )
    graph = form_regions(
        canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    )
    selection = select_bodies(graph, _target(), SelectionPolicy(allow_tuned=True))
    plan = assign_storage(
        make_phase_schedule(selection, derive_dependencies(selection))
    )

    generated = emit_cute(plan, tmp_path)

    ast.parse(generated.source)
    assert [phase.provider for phase in generated.launch.phases] == [
        "g0_range",
        "g0_range",
    ]
    assert generated.source.count("HopperWgmmaGemmPersistentKernel(") == 1
    assert generated.source.count("SmemAllocator().allocate(_g0_body_") == 1
    assert generated.launch.shared_memory_bytes == 230_400


def test_emit_cute_keeps_g0_variants_with_different_configurations(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, source, weight, other_weight):
            return source @ weight.T, source @ other_weight.T

    source = torch.randn(4, 576, dtype=torch.bfloat16)
    weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    other_weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    prepared = prepare_fx_graph(
        torch.export.export(Model(), (source, weight, other_weight)).graph_module,
        (source, weight, other_weight),
    )
    graph = form_regions(
        canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    )
    selection = select_bodies(graph, _target(), SelectionPolicy(allow_tuned=True))
    plan = assign_storage(
        make_phase_schedule(selection, derive_dependencies(selection))
    )
    first, second = plan.phases
    first_body = replace(
        first.body,
        tiles=replace(first.body.tiles, cta_shape=(64, 64)),
        scratch=replace(first.body.scratch, size_bytes=230_400),
    )
    second_body = replace(
        second.body,
        name=first.body.name,
        tiles=replace(second.body.tiles, cta_shape=(64, 128)),
        scratch=replace(second.body.scratch, size_bytes=214_016),
    )
    plan = replace(
        plan,
        phases=(replace(first, body=first_body), replace(second, body=second_body)),
        selection=replace(
            plan.selection,
            target=replace(
                plan.selection.target,
                max_shared_memory_per_cta=1_000_000,
            ),
        ),
    )

    generated = emit_cute(plan, tmp_path)

    assert generated.source.count("HopperWgmmaGemmPersistentKernel(") == 2
    assert generated.source.count("SmemAllocator().allocate(_g0_body_") == 2
    assert "(64, 64)" in generated.source
    assert "(64, 128)" in generated.source
    assert generated.launch.shared_memory_bytes == 444_416


def test_emit_cute_emits_explicit_layout_conversion(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, value):
            return value.relu().sum(dim=-1)

    value = torch.randn(2, 3)
    prepared = prepare_fx_graph(
        torch.export.export(Model(), (value,)).graph_module, (value,)
    )
    graph = form_regions(
        canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    )
    selection = select_bodies(
        graph, _target(), SelectionPolicy(allow_tuned=False)
    )
    producer = next(
        selected
        for selected in selection.selections
        if selected.program.family == "elementwise"
    )
    consumer = next(
        selected
        for selected in selection.selections
        if selected.program.family == "reduction"
    )
    value_id = producer.outputs[0]
    source_layout = CuTeLayout.from_contract(
        producer.body.layouts.outputs[value_id]
    )
    target_layout = replace(source_layout, strides=(1, 3))
    load = consumer.program.loads[0]
    consumer_program = replace(
        consumer.program,
        loads=(replace(load, strides=target_layout.strides),),
    )
    consumer_body = replace(
        consumer.body,
        layouts=replace(
            consumer.body.layouts,
            inputs={
                value_id: replace(
                    consumer.body.layouts.inputs[value_id],
                    strides=target_layout.strides,
                )
            },
        ),
    )
    conversion = ConversionBody(
        value_id,
        producer.region_id,
        consumer.region_id,
        source_layout,
        target_layout,
        24,
        (producer.region_id,),
    )
    converted_selection = replace(
        selection,
        selections=(
            producer,
            replace(
                consumer, program=consumer_program, body=consumer_body
            ),
        ),
        conversions=(conversion,),
    )
    plan = assign_storage(
        make_phase_schedule(
            converted_selection, derive_dependencies(converted_selection)
        )
    )

    generated = emit_cute(plan, tmp_path)
    ast.parse(generated.source)
    assert "_conversion_0_0(" in generated.source
    assert generated.launch.phases[0].conversion_symbols == ("_conversion_0_0",)
    assert any(
        entry.kind == "layout_conversion"
        for entry in generated.source_map
    )


def test_weight_stream_schedule_streams_each_ctas_first_gemm_of_the_next_interval():
    from types import SimpleNamespace

    from megabake.codegen import _weight_stream_schedule

    # Phase 0 closes the first interval; the second runs a norm recomputed on
    # every CTA, then q, k and v GEMMs side by side on 4 CTAs (v wraps
    # around); the third runs o, whose weight cannot be streamed.
    barriers = (True, False, False, False, True, True)
    plan = SimpleNamespace(
        phases=tuple(
            SimpleNamespace(index=index, barrier=object() if barrier else None)
            for index, barrier in enumerate(barriers)
        ),
        launch=SimpleNamespace(grid_size=4),
    )
    rotations = (0, -1, 0, 2, 3, 0)
    units = {0: 1, 1: 1, 2: 2, 3: 1, 4: 3, 5: 4}
    gemm_of_phase = {2: 0, 3: 1, 4: 2, 5: 3}
    variants = (0, 1, 1, 0)

    rows, issues = _weight_stream_schedule(
        plan, rotations, units, gemm_of_phase, variants, {0, 1, 2}, 8
    )

    # (GEMM row + 1, variant, rotated block) per CTA: CTAs 0-1 run q tiles
    # 0-1, CTA 2 the k tile, and CTA 3 v tile 0 (v tiles 1-2 come after q
    # on CTAs 0-1, which stream q).
    assert rows == [(0,) * 12, (1, 0, 0, 1, 0, 1, 2, 1, 0, 3, 1, 0)]
    assert issues == {0: 1}

    rows, issues = _weight_stream_schedule(
        plan, rotations, units, gemm_of_phase, variants, {0, 1, 2}, 0
    )
    assert rows == [(0,) * 12] and issues == {}

