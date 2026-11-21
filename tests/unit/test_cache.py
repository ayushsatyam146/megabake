from __future__ import annotations

from types import SimpleNamespace

import torch

from megabake import PlanCache, make_plan_key, prepare_fx_graph
from megabake.bodies import TargetFacts
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.plan import PlanMeasurement, SearchBudget, search_plans
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


def _prepared(model: torch.nn.Module, *inputs: torch.Tensor):
    exported = torch.export.export(model, inputs)
    return prepare_fx_graph(
        exported.graph_module,
        inputs,
        output_tree_spec=exported.call_spec.out_spec,
    )


def _selected_plan(model: torch.nn.Module, *inputs: torch.Tensor):
    prepared = _prepared(model, *inputs)
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))

    def evaluate(plan, budget):
        return PlanMeasurement(
            compiled=object(),
            resources={"registers_per_thread": 1},
            correct=True,
            complete_call_samples_us=(1.0,),
            median_complete_call_us=1.0,
        )

    result = search_plans(
        graph,
        _target(),
        budget=SearchBudget(
            max_plans=8,
            max_compilations=8,
            timing_samples=1,
            evaluator=evaluate,
        ),
    )
    assert result.selected is not None
    return result.selected, prepared.context


def test_plan_key_covers_semantics_shapes_strides_mode_and_toolchain():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(4, 4, dtype=torch.bfloat16)
    weight = torch.randn(4, 4, dtype=torch.bfloat16)
    plan, context = _selected_plan(Model(), source, weight)
    key = make_plan_key(plan, context)

    assert key == make_plan_key(plan, context)
    assert key != make_plan_key(plan, context, scheduler="region-events")
    assert key != make_plan_key(plan, context, toolchain={"torch": "changed"})

    decode_context = type(context)(**{**context.__dict__, "mode": "decode"})
    assert key != make_plan_key(plan, decode_context)

    class OtherConstant(torch.nn.Module):
        def forward(self, source, weight):
            return (source @ weight.T) * 3

    other_plan, other_context = _selected_plan(OtherConstant(), source, weight)
    assert key != make_plan_key(other_plan, other_context)

    class Map(torch.nn.Module):
        def forward(self, value):
            return value * 2

    contiguous = torch.randn(4, 4)
    strided_base = torch.randn(4, 8)
    strided = strided_base.t()[:4, :]
    assert contiguous.shape == strided.shape
    assert contiguous.stride() != strided.stride()

    contiguous_plan, contiguous_context = _selected_plan(Map(), contiguous)
    strided_plan, strided_context = _selected_plan(Map(), strided)
    assert make_plan_key(contiguous_plan, contiguous_context) != make_plan_key(
        strided_plan, strided_context
    )

    smaller_plan, smaller_context = _selected_plan(Map(), torch.randn(3, 4))
    assert make_plan_key(contiguous_plan, contiguous_context) != make_plan_key(
        smaller_plan, smaller_context
    )


def test_plan_key_ignores_parameter_contents_and_addresses():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(4, 4, dtype=torch.bfloat16)
    first_weight = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
    second_weight = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
    plan, first_context = _selected_plan(Model(), source, first_weight)
    second_context = _prepared(Model(), source, second_weight).context

    assert make_plan_key(plan, first_context) == make_plan_key(plan, second_context)


def test_plan_cache_reuses_compiled_artifact_and_stores_diagnostics():
    cache = PlanCache()
    compiled = SimpleNamespace(search=SimpleNamespace(report=lambda: {"selected": "test-plan"}))
    compilations = 0

    def compiler():
        nonlocal compilations
        compilations += 1
        return compiled

    first = cache.get_or_compile("key", compiler)
    second = cache.get_or_compile("key", compiler)

    assert first.compiled is compiled and not first.hit
    assert second.compiled is compiled and second.hit
    assert first.diagnostics == {"selected": "test-plan"}
    assert compilations == 1
    assert cache.hits == 1 and cache.misses == 1
