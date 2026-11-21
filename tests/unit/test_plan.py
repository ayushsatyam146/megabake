from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import megabake.planner as planner_module
from megabake import make_tile_maps, select_bodies
from megabake.bodies import SelectionPolicy, TargetFacts
from megabake.compiler import _graph_ordered_inputs
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.ir import IndexedLoad
from megabake.lowering import canonicalize_semantics
from megabake.ops import HF_EAGER_NUMERICS, Attention, CacheEffects, HeadGrouping, Mask
from megabake.plan import (
    CuTeLayout,
    CuTeSwizzle,
    InvalidTileMap,
    PlanMeasurement,
    SearchBudget,
    search_plans,
)
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


def _graph(model, *inputs):
    exported = torch.export.export(model, inputs)
    prepared = prepare_fx_graph(exported.graph_module, inputs)
    return form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))


def test_select_bodies_prefers_measured_gemm_and_maps_complete_reduction_domain():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(128, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, 128, dtype=torch.bfloat16)
    selection = select_bodies(_graph(Model(), source, weight), _target())
    selected = next(
        selected for selected in selection.selections if selected.program.family == "contraction"
    )

    assert selected.body.provider == "g0_range"
    assert selected.body.evidence.measured
    assert selection.report.complete

    tile_map = make_tile_maps(selection)[0]
    assert tile_map.output_grid.tile_shape == (64, 128)
    assert tile_map.output_grid.counts == (2, 1)
    assert len(tile_map.accesses) == 2
    assert tile_map.work_items[0].reduction_range == (0, 128)

    access = next(access for access in tile_map.accesses if access.output.origin == (64, 0))
    left, right = access.inputs
    assert left.shape == (64, 128)
    assert right.shape == (128, 128)
    assert left.element_offset((64, 0)) == 64 * 128
    assert right.element_offset((0, 0)) == 0


def test_select_bodies_uses_legal_vector_chain_and_preserves_fallback():
    class Model(torch.nn.Module):
        def forward(self, value):
            return (torch.relu(value) + 1) * 2

    class ExternalIntermediate(torch.nn.Module):
        def forward(self, value):
            activated = torch.relu(value)
            return activated, activated + 1

    torch.manual_seed(71)
    value = torch.randn(2, 3)
    selection = select_bodies(_graph(Model(), value), _target())

    assert [selected.region_id for selected in selection.selections] == ["region:mul:fused"]
    selected = selection.selections[0]
    assert selected.source_op_ids == ("op:relu", "op:add", "op:mul")
    assert selected.inputs == ("value:arg0_1",)
    assert selected.outputs == ("value:mul",)
    assert selected.body.provider == "generic"
    assert selected.body.name == "generic_elementwise_fused"
    assert selected.body.semantic.expression_hooks == ("vector_chain",)
    assert selected.body.evidence.measurements["fusion_saved_bytes"] == 2 * 3 * 4 * 4
    assert selection.report.complete

    fallback = select_bodies(_graph(ExternalIntermediate(), value), _target())
    assert [selected.region_id for selected in fallback.selections] == [
        "region:relu",
        "region:add",
    ]
    # The externally used relu stays outside the chain: no fused candidate.
    assert not any(selected.region_id.endswith(":fused") for selected in fallback.selections)


def test_search_plans_measures_provider_and_worker_candidates():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(128, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, 128, dtype=torch.bfloat16)
    calls = []

    def evaluate(plan, budget):
        body_name = plan.selection.selections[0].body.name
        calls.append((body_name, plan.launch.grid_size))
        median = 30.0 if body_name.startswith("generic") else 20.0 if "64x64" in body_name else 10.0
        return PlanMeasurement(
            compiled=object(),
            resources={"registers_per_thread": 1},
            correct=True,
            complete_call_samples_us=(median - 1.0, median, median + 1.0),
            median_complete_call_us=median,
            verification_complete_call_us=median,
        )

    result = search_plans(
        _graph(Model(), source, weight),
        _target(),
        budget=SearchBudget(
            max_plans=8,
            max_compilations=8,
            timing_samples=3,
            evaluator=evaluate,
        ),
    )

    assert len(calls) == 6
    assert {name for name, _ in calls} == {
        "generic_contraction",
        "g0_range_64x64",
        "g0_range_64x128",
    }
    assert all(candidate.plan is not None for candidate in result.candidates)
    assert len(result.measured_candidates) == 6
    assert result.selected is not None
    assert result.selected.selection.selections[0].body.name == "g0_range_64x128"
    assert result.selected_compiled is not None
    assert result.report()["selected"].startswith("workers=")


def test_search_plans_prefers_isolated_time_over_host_binding_time():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(128, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, 128, dtype=torch.bfloat16)

    def evaluate(plan, budget):
        body_name = plan.selection.selections[0].body.name
        is_generic = body_name.startswith("generic")
        return PlanMeasurement(
            compiled=object(),
            resources={"registers_per_thread": 1},
            correct=True,
            complete_call_samples_us=(10.0 if is_generic else 20.0,),
            median_complete_call_us=10.0 if is_generic else 20.0,
            device_event_span_samples_us=(30.0 if is_generic else 5.0,),
            median_device_event_span_us=30.0 if is_generic else 5.0,
            isolated_launch_samples_us=(30.0 if is_generic else 5.0,),
            median_isolated_launch_us=30.0 if is_generic else 5.0,
        )

    result = search_plans(
        _graph(Model(), source, weight),
        _target(),
        budget=SearchBudget(
            max_plans=8,
            max_compilations=8,
            timing_samples=1,
            evaluator=evaluate,
        ),
    )

    assert result.selected is not None
    assert not result.selected.selection.selections[0].body.name.startswith("generic")


def test_graph_ordered_inputs_maps_aot_lifted_parameters():
    context = SimpleNamespace(
        input_specs=[
            SimpleNamespace(name="source"),
            SimpleNamespace(name="weight"),
        ]
    )
    compiled = SimpleNamespace(input_order=("l_weight_", "l_source_"))
    source, weight = object(), object()

    assert _graph_ordered_inputs(compiled, context, (source, weight)) == (
        source,
        weight,
    )


def test_search_plans_records_unmeasured_candidates_when_budget_is_exhausted():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(128, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, 128, dtype=torch.bfloat16)

    def evaluate(plan, budget):
        return PlanMeasurement(
            compiled=object(),
            correct=True,
            complete_call_samples_us=(10.0,),
            median_complete_call_us=10.0,
        )

    result = search_plans(
        _graph(Model(), source, weight),
        _target(),
        budget=SearchBudget(
            max_plans=3,
            max_compilations=1,
            timing_samples=1,
            evaluator=evaluate,
        ),
    )

    assert len(result.measured_candidates) == 1
    assert sum(candidate.status == "unmeasured" for candidate in result.candidates) == 2
    assert result.budget_exhausted
    assert all(
        candidate.reason == "compilation budget exhausted"
        for candidate in result.candidates
        if candidate.status == "unmeasured"
    )


def test_search_plans_prunes_excessive_mixed_g0_storage_before_compilation():
    class Model(torch.nn.Module):
        def forward(self, source, weight, other_weight):
            return source @ weight.T, source @ other_weight.T

    source = torch.randn(4, 576, dtype=torch.bfloat16)
    weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    other_weight = torch.randn_like(weight)
    evaluations = 0

    def evaluate(plan, budget):
        nonlocal evaluations
        evaluations += 1
        return PlanMeasurement(
            compiled=object(),
            correct=True,
            complete_call_samples_us=(10.0,),
            median_complete_call_us=10.0,
        )

    result = search_plans(
        _graph(Model(), source, weight, other_weight),
        _target(),
        budget=SearchBudget(
            max_plans=4,
            max_compilations=1,
            timing_samples=1,
            evaluator=evaluate,
        ),
    )

    rejected = [
        candidate
        for candidate in result.candidates
        if candidate.status == "rejected"
        and candidate.reason is not None
        and "projected shared memory 444416" in candidate.reason
    ]
    assert rejected
    assert all(candidate.plan is None for candidate in rejected)
    assert evaluations == 1


def test_make_tile_maps_clips_partial_output_tiles():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(65, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, 128, dtype=torch.bfloat16)
    selection = select_bodies(_graph(Model(), source, weight), _target())
    tile_map = make_tile_maps(selection)[0]
    partial = next(access for access in tile_map.accesses if access.output.origin[0] == 64)
    left = partial.inputs[0]

    assert partial.output.shape == (1, 128)
    assert left.shape == (1, 128)
    assert left.element_offset((64, 0)) == 64 * 128


def test_make_tile_maps_handles_unit_dimensions_and_boundary_strides():
    class TransposeReLU(torch.nn.Module):
        def forward(self, value):
            return torch.relu(value.t())

    class Sum(torch.nn.Module):
        def forward(self, value):
            return value.sum(dim=-1)

    transpose_graph = _graph(TransposeReLU(), torch.randn(3, 5))
    transpose_selection = select_bodies(transpose_graph, _target())
    transpose_map = make_tile_maps(transpose_selection)[0]
    input_tile = transpose_map.accesses[0].inputs[0]

    assert transpose_map.output_grid.tile_shape == (5, 3)
    assert input_tile.element_offset((2, 1)) == 2 * 1 + 1 * 5
    assert transpose_selection.report.rejections

    sum_graph = _graph(Sum(), torch.randn(1, 7))
    sum_map = make_tile_maps(select_bodies(sum_graph, _target()))[0]
    assert sum_map.output_grid.tile_shape == (1,)
    assert sum_map.accesses[0].inputs[0].shape == (1, 7)
    assert sum_map.accesses[0].reduction_range == (0, 7)

    row = CuTeLayout(
        dtype="torch.float32",
        shape=(1, 3),
        strides=(99, 1),
        alignment_bytes=4,
    )
    equivalent = replace(row, strides=(7, 1))
    assert not row.needs_conversion_to(equivalent)

    with pytest.raises(InvalidTileMap, match="swizzle"):
        CuTeLayout(
            dtype="torch.float32",
            shape=(2, 2),
            strides=(2, 1),
            alignment_bytes=4,
            swizzle=CuTeSwizzle(mask=3, shift=1),
        )


def test_make_tile_maps_inserts_layout_conversion_with_cost_and_dependency(
    monkeypatch,
):
    # relu -> sum would fuse as a reduction prologue, leaving no conversion.
    monkeypatch.setenv("MEGABAKE_REDUCTION_PROLOGUE", "0")
    class Model(torch.nn.Module):
        def forward(self, value):
            return value.relu().sum(dim=-1)

    real_get_candidates = planner_module.get_candidates

    def changed_input_layout(region, target, policy):
        result = real_get_candidates(region, target, policy)
        program = region.tensor_programs[0]
        if program.family != "reduction":
            return result
        config = result.configurations[0]
        value_id = program.loads[0].value_id
        layouts = replace(
            config.layouts,
            inputs={value_id: replace(config.layouts.inputs[value_id], strides=(1, 3))},
        )
        return replace(result, configurations=(replace(config, layouts=layouts),))

    monkeypatch.setattr(planner_module, "get_candidates", changed_input_layout)
    selection = select_bodies(_graph(Model(), torch.randn(2, 3)), _target())
    relu = next(
        selected
        for selected in selection.selections
        if selected.program.family in {"elementwise", "copy"}
    )
    reduction = next(
        selected for selected in selection.selections if selected.program.family == "reduction"
    )

    assert len(selection.conversions) == 1
    conversion = selection.conversions[0]
    assert conversion.value_id == relu.outputs[0]
    assert conversion.source_region_id == relu.region_id
    assert conversion.target_region_id == reduction.region_id
    assert conversion.cost_bytes == 2 * 3 * 4 * 2
    assert conversion.dependencies == (relu.region_id,)
    reduction_map = make_tile_maps(selection)[-1]
    assert reduction_map.conversion_bodies == (conversion,)


def test_mode_policy_selects_gemm_family_from_actual_dimensions():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    short_prefill = torch.randn(4, 128, dtype=torch.bfloat16)
    large_decode = torch.randn(128, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, 128, dtype=torch.bfloat16)

    short_prefill_body = select_bodies(
        _graph(Model(), short_prefill, weight),
        _target(),
        SelectionPolicy(mode="prefill"),
    ).selections[0]
    short_decode_body = select_bodies(
        _graph(Model(), short_prefill, weight),
        _target(),
        SelectionPolicy(mode="decode"),
    ).selections[0]
    large_decode_body = select_bodies(
        _graph(Model(), large_decode, weight),
        _target(),
        SelectionPolicy(mode="decode"),
    ).selections[0]

    assert short_prefill_body.body.provider == "g0_range"
    assert short_prefill_body.body.name == "g0_range_64x64"
    assert short_decode_body.body.provider == "g0_range"
    assert short_decode_body.body.name == "g0_range_64x64"
    assert large_decode_body.body.provider == "g0_range"
    assert large_decode_body.body.name == "g0_range_64x128"


def test_mode_policy_selects_attention_body_from_actual_query_work():
    def grouped_attention(source, key, value, mask):
        query = source.permute(0, 2, 1, 3)
        grouped_key = key.permute(0, 2, 1, 3).repeat_interleave(
            source.shape[2] // key.shape[2], dim=1
        )
        grouped_value = value.permute(0, 2, 1, 3).repeat_interleave(
            source.shape[2] // value.shape[2], dim=1
        )
        scores = (query @ grouped_key.transpose(-2, -1)) * source.shape[-1] ** -0.5 + mask
        probabilities = torch.softmax(scores.float(), dim=-1).to(source.dtype)
        return probabilities @ grouped_value

    class Model(torch.nn.Module):
        def forward(self, source, key, value, mask):
            return grouped_attention(source, key, value, mask)

    def selected_body(mode, query_length):
        source = torch.randn(2, query_length, 4, 8, dtype=torch.bfloat16)
        key = torch.randn(2, 7, 2, 8, dtype=torch.bfloat16)
        value = torch.randn(2, 7, 2, 8, dtype=torch.bfloat16)
        mask = torch.randn(2, 1, query_length, 7, dtype=torch.bfloat16)
        return (
            select_bodies(
                _graph(Model(), source, key, value, mask),
                _target(),
                SelectionPolicy(mode=mode),
            )
            .selections[0]
            .body.name
        )

    assert selected_body("prefill", 5) == "attention_prefill_rows"
    assert selected_body("decode", 1) == "attention_decode_rows"


def test_mode_policy_orders_bounded_worker_search():
    class Model(torch.nn.Module):
        def forward(self, value):
            return value * 2

    grid_sizes = []

    def evaluate(plan, budget):
        grid_sizes.append(plan.launch.grid_size)
        return PlanMeasurement(
            compiled=object(),
            correct=True,
            median_complete_call_us=1.0,
        )

    search_plans(
        _graph(Model(), torch.randn(32, 32)),
        _target(),
        SelectionPolicy(mode="decode"),
        SearchBudget(
            max_plans=3,
            max_compilations=3,
            timing_samples=1,
            evaluator=evaluate,
        ),
    )

    assert grid_sizes == [2, 4, 8]


def test_mode_policy_rejects_prefill_for_stateful_cache():
    class Model(torch.nn.Module):
        def forward(self, value):
            return value * 2

    graph = _graph(Model(), torch.randn(2, 3))
    access = IndexedLoad("attention:q", "value:arg0_1", (0, 0), (3, 1), "torch.bfloat16")
    stateful = Attention(
        algorithm="stable_softmax",
        numerics=HF_EAGER_NUMERICS,
        q=access,
        k=access,
        v=access,
        grouping=HeadGrouping(batch=1, query_heads=1, kv_heads=1, head_dim=3),
        scale=1.0,
        mask=Mask("stateful_causal", causal_offset=0),
        outputs=(),
        boundary_output=None,
        cache=CacheEffects("stateful"),
    )
    graph = replace(
        graph, regions=(replace(graph.regions[0], semantics=stateful), *graph.regions[1:])
    )

    with pytest.raises(ValueError, match="prefill mode cannot compile"):
        select_bodies(graph, _target(), SelectionPolicy(mode="prefill"))


def _region_check_fixture():
    from types import SimpleNamespace

    def step(value, weight, cache):
        projected = torch.mm(value, weight)
        cache.add_(projected)
        return (torch.relu(projected) * 2,)

    traced = torch.fx.symbolic_trace(step)
    nodes = list(traced.graph.nodes)
    by_name = {node.name: node for node in nodes}
    order = {node.name: index for index, node in enumerate(nodes)}
    value = torch.randn(4, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, 16, dtype=torch.bfloat16)
    cache = torch.zeros(4, 16, dtype=torch.bfloat16)
    projected = torch.mm(value, weight)
    values = {"value:mm": projected, "value:mul": torch.relu(projected) * 2}
    inputs = {"value": value, "weight": weight, "cache": cache}
    region = SimpleNamespace(
        region_id="region:relu:fused",
        source_op_ids=("op:relu", "op:mul"),
        outputs=("value:mul",),
    )
    return region, values, inputs, by_name, order


def test_region_check_accepts_rounding_and_rejects_wrong_values():
    from megabake.compiler import _region_failure

    region, values, inputs, by_name, order = _region_check_fixture()
    assert _region_failure(region, values, inputs, {"cache"}, by_name, order) is None
    # Rounding-order noise in an input does not matter: the reference uses the
    # candidate's own input.
    values["value:mm"] = values["value:mm"] * 1.01
    values["value:mul"] = torch.relu(values["value:mm"]) * 2
    assert _region_failure(region, values, inputs, {"cache"}, by_name, order) is None
    values["value:mul"] = values["value:mul"] * 1.05
    assert "from PyTorch on its own inputs" in _region_failure(
        region, values, inputs, {"cache"}, by_name, order
    )


def test_region_check_rejects_non_finite_and_unverifiable_regions():
    from types import SimpleNamespace

    from megabake.compiler import _region_failure

    region, values, inputs, by_name, order = _region_check_fixture()
    values["value:mul"] = values["value:mul"].clone()
    values["value:mul"][0, 0] = float("nan")
    assert "non-finite" in _region_failure(region, values, inputs, {"cache"}, by_name, order)
    hidden = SimpleNamespace(
        region_id="region:relu:fused", source_op_ids=("op:relu",), outputs=("value:relu",)
    )
    assert "no output to verify" in _region_failure(
        hidden, values, inputs, {"cache"}, by_name, order
    )


def test_region_check_leaves_captured_state_untouched():
    from types import SimpleNamespace

    from megabake.compiler import _region_failure

    _, values, inputs, by_name, order = _region_check_fixture()
    writer = SimpleNamespace(
        region_id="region:add_:state", source_op_ids=("op:mm", "op:add_"), outputs=("value:mm",)
    )
    assert _region_failure(writer, values, inputs, {"cache"}, by_name, order) is None
    assert torch.equal(inputs["cache"], torch.zeros(4, 16, dtype=torch.bfloat16))


def test_region_check_reads_graph_constants():
    from types import SimpleNamespace

    from megabake.compiler import _region_failure

    class Scaled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.full((4,), 3.0))

        def forward(self, value):
            return (value * self.scale,)

    module = torch.fx.symbolic_trace(Scaled())
    nodes = list(module.graph.nodes)
    value = torch.randn(4)
    region = SimpleNamespace(
        region_id="region:mul", source_op_ids=("op:scale", "op:mul"), outputs=("value:mul",)
    )
    assert (
        _region_failure(
            region,
            {"value:mul": value * 3.0},
            {"value": value},
            set(),
            {node.name: node for node in nodes},
            {node.name: index for index, node in enumerate(nodes)},
            module,
        )
        is None
    )


def test_region_check_recomputes_unmaterialized_views():
    from types import SimpleNamespace

    from megabake.compiler import _region_failure

    def step(value):
        return ((value[:, :4] * 3),)

    traced = torch.fx.symbolic_trace(step)
    nodes = list(traced.graph.nodes)
    value = torch.randn(2, 8)
    names = [node.name for node in nodes]
    slice_name = next(name for name in names if "getitem" in name)
    multiply = next(name for name in names if "mul" in name)
    region = SimpleNamespace(
        region_id="region:mul", source_op_ids=(f"op:{multiply}",), outputs=(f"value:{multiply}",)
    )
    assert slice_name not in ("value",)
    assert (
        _region_failure(
            region,
            {f"value:{multiply}": value[:, :4] * 3},
            {"value": value},
            set(),
            {node.name: node for node in nodes},
            {node.name: index for index, node in enumerate(nodes)},
        )
        is None
    )
