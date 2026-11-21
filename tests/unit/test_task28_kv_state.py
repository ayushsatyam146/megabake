from __future__ import annotations

import pytest
import torch

from megabake import (
    assign_storage,
    derive_dependencies,
    make_backend,
    make_phase_schedule,
    select_bodies,
    verify_plan,
)
from megabake.bodies import SelectionPolicy, TargetFacts
from megabake.bodies.cuda.sm90.attention import emit_attention_body
from megabake.compiler import CompilerError, _validate_state_capacities
from megabake.frontend import prepare_fx_graph
from megabake.fx_handler.transformers import (
    KVCache,
    StatefulAttention,
    bind_kv_cache,
)
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.ops import Attention
from megabake.ops.attention import kv_attention
from megabake.regions import form_regions
from megabake.schedule import DependencyKind


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


def _inputs(capacity: int = 4, valid_length: int = 1):
    torch.manual_seed(28)
    return (
        torch.randn(1, 1, 2, 8, dtype=torch.bfloat16),
        torch.randn(1, 1, 2, 8, dtype=torch.bfloat16),
        torch.randn(1, 1, 2, 8, dtype=torch.bfloat16),
        torch.randn(1, capacity, 2, 8, dtype=torch.bfloat16),
        torch.randn(1, capacity, 2, 8, dtype=torch.bfloat16),
        torch.tensor([valid_length], dtype=torch.int32),
    )


def _graph(model, inputs):
    exported = torch.export.export(model.eval(), inputs)
    prepared = prepare_fx_graph(
        exported.graph_module,
        inputs,
        output_tree_spec=exported.call_spec.out_spec,
    )
    return form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))


def _plan(graph):
    selection = select_bodies(graph, _target(), SelectionPolicy(mode="decode"))
    dependencies = derive_dependencies(selection)
    plan = assign_storage(make_phase_schedule(selection, dependencies))
    verify_plan(plan)
    return plan


def test_task28_lowers_dynamic_stateful_decode_attention():
    graph = _graph(StatefulAttention(), _inputs())
    candidate = next(
        candidate
        for region in graph.regions
        for candidate in region.fusion_candidates
        if isinstance(candidate.semantics, Attention)
    )
    state = candidate.semantics.kv_state

    assert candidate.semantics.cache.policy == "stateful"
    assert state.capacity == 4
    assert state.query_length == 1
    assert state.read_range == (0, "valid_length")
    assert state.write_range == ("valid_length", "valid_length + query_length")
    assert state.valid_length_specialized is False
    assert state.storage_class == "bounded_contiguous"

    plan = _plan(graph)
    assert plan.selection.selections[0].body.name == "attention_stateful_decode_rows"
    assert len(plan.storage.state) == 1
    assert plan.storage.state[0].lifetime == "boundary_state"
    assert plan.storage.state[0].capacity == 4
    assert plan.storage.state[0].storage_identity not in {
        assignment.storage_id for assignment in plan.storage.workspace
    }


def test_task28_rejects_stateful_decode_in_prefill_mode():
    graph = _graph(StatefulAttention(), _inputs())
    with pytest.raises(ValueError, match="prefill mode cannot compile"):
        select_bodies(graph, _target(), SelectionPolicy(mode="prefill"))


def test_task28_rejects_runtime_length_beyond_capacity():
    graph = _graph(StatefulAttention(), _inputs(capacity=4))
    plan = _plan(graph)
    state = plan.storage.state[0]
    invalid_inputs = {state.valid_length_value_id: torch.tensor([4])}

    with pytest.raises(CompilerError, match="exceeds capacity"):
        _validate_state_capacities(plan, invalid_inputs)


def test_task28_adds_dependency_for_overlapping_state_reads_and_writes():
    class TwoStatefulCalls(torch.nn.Module):
        def forward(
            self,
            source,
            key,
            value,
            key_cache,
            value_cache,
            valid_length,
        ):
            first = kv_attention(source, key, value, key_cache, value_cache, valid_length)
            second = kv_attention(source, key, value, key_cache, value_cache, valid_length)
            return first, second

    graph = _graph(TwoStatefulCalls(), _inputs())
    plan = _plan(graph)
    state_dependencies = [
        dependency
        for dependency in plan.dependencies.dependencies
        if dependency.kind is DependencyKind.STATE_ACCESS
    ]

    assert len(state_dependencies) == 1
    assert state_dependencies[0].conservative is True
    assert len(plan.storage.state[0].read_regions) == 2
    assert len(plan.storage.state[0].write_regions) == 2


def test_task28_emits_runtime_length_reads_and_cache_writes():
    graph = _graph(StatefulAttention(), _inputs())
    selection = select_bodies(graph, _target(), SelectionPolicy(mode="decode"))
    generated = emit_attention_body(selection.selections[0].program, selection.selections[0].body)

    assert "current_length" in generated.source
    assert "attention:valid_length" not in generated.source
    assert generated.source.count("tensor_bindings[3][b, current_length, kv_head, d]") == 1
    assert generated.source.count("tensor_bindings[4][b, current_length, kv_head, d]") == 1
    assert "tensor_bindings[5][0] = current_length + cutlass.Int32(1)" in generated.source


def test_task28_wrapper_binds_current_cache_and_eager_state_advances():
    source, key, value, key_cache, value_cache, valid_length = _inputs(capacity=3)
    key_cache[:, :1] = torch.randn(1, 1, 2, 8, dtype=torch.bfloat16)
    value_cache[:, :1] = torch.randn(1, 1, 2, 8, dtype=torch.bfloat16)
    cache = KVCache(key_cache, value_cache, valid_length)
    bound = bind_kv_cache(StatefulAttention(), cache)

    exported = torch.export.export(bound, (source, key, value))
    assert any(spec.target == "key_cache" for spec in exported.graph_signature.input_specs)
    assert any(spec.target == "valid_length" for spec in exported.graph_signature.input_specs)

    expected_key = key_cache.clone()
    expected_value = value_cache.clone()
    expected_length = 1
    for new_key, new_value in ((key, value), (key * 0.5, value * 0.5)):
        expected_key[:, expected_length : expected_length + 1] = new_key
        expected_value[:, expected_length : expected_length + 1] = new_value
        expected_length += 1
        end = expected_length
        query = source.permute(0, 2, 1, 3)
        keys = expected_key[:, :end].permute(0, 2, 1, 3)
        values = expected_value[:, :end].permute(0, 2, 1, 3)
        scores = (query @ keys.transpose(-2, -1)).float() * 8**-0.5
        expected = (
            (torch.softmax(scores, dim=-1).to(torch.bfloat16) @ values)
            .permute(0, 2, 1, 3)
            .reshape(2, 1, 8)
        )
        actual = bound(source, new_key, new_value)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    assert int(valid_length.item()) == 3
    torch.testing.assert_close(key_cache, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache, expected_value, rtol=0, atol=0)


def test_task28_normal_torch_compile_frontend_preserves_state_mutation():
    inputs = _inputs(capacity=3)
    expected_inputs = tuple(
        value.clone() if isinstance(value, torch.Tensor) else value for value in inputs
    )
    backend = make_backend(mode="decode", compile_kernel=False, cache=False)
    optimized = torch.compile(StatefulAttention(), backend=backend, fullgraph=True, dynamic=False)

    with torch.no_grad():
        actual = optimized(*inputs)
        expected = StatefulAttention()(*expected_inputs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs[3], expected_inputs[3], rtol=0, atol=0)
    torch.testing.assert_close(inputs[4], expected_inputs[4], rtol=0, atol=0)
    torch.testing.assert_close(inputs[5], expected_inputs[5], rtol=0, atol=0)


def test_task28_backend_graph_drops_reinplaced_self_copies():
    inputs = _inputs(capacity=3)
    backend = make_backend(mode="decode", compile_kernel=False, cache=False)
    optimized = torch.compile(StatefulAttention(), backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        optimized(*inputs)

    prepared = backend.megabake_last_prepared
    self_copies = [
        node
        for node in prepared.graph.graph.nodes
        if node.target is torch.ops.aten.copy_.default and node.args[0] is node.args[1]
    ]
    assert self_copies
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    assert not any(
        operation.attributes.get("target") == "aten.copy_.default"
        for operation in graph.ops.values()
    )
    _plan(graph)
