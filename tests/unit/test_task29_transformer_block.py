from __future__ import annotations

import pytest
import torch

from megabake import (
    assign_storage,
    derive_dependencies,
    make_phase_schedule,
    select_bodies,
    verify_plan,
)
from megabake.bodies import SelectionPolicy, TargetFacts
from megabake.fx_handler.export import capture_exported_program
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.ops import Attention
from megabake.regions import form_regions
from tests.fixtures import StatefulTransformerBlock, TransformerBlock


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


def _plan(model, inputs, mode):
    exported = torch.export.export(model.eval(), inputs)
    prepared = capture_exported_program(exported, inputs, model=model, mode=mode)
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    selection = select_bodies(graph, _target(), SelectionPolicy(mode=mode))
    dependencies = derive_dependencies(selection)
    plan = assign_storage(make_phase_schedule(selection, dependencies))
    verify_plan(plan)
    return plan


def _prefill_inputs():
    torch.manual_seed(29)
    source = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    residual = torch.randn_like(source)
    query_cosine = torch.randn(2, 5, 2, 8, dtype=torch.bfloat16)
    query_sine = torch.randn_like(query_cosine)
    key_cosine = torch.randn(2, 5, 1, 8, dtype=torch.bfloat16)
    key_sine = torch.randn_like(key_cosine)
    mask = torch.randn(2, 1, 5, 5, dtype=torch.bfloat16)
    return source, residual, query_cosine, query_sine, key_cosine, key_sine, mask


def _decode_inputs():
    torch.manual_seed(29)
    source = torch.randn(2, 1, 16, dtype=torch.bfloat16)
    residual = torch.randn_like(source)
    query_cosine = torch.randn(2, 1, 2, 8, dtype=torch.bfloat16)
    query_sine = torch.randn_like(query_cosine)
    key_cosine = torch.randn(2, 1, 1, 8, dtype=torch.bfloat16)
    key_sine = torch.randn_like(key_cosine)
    key_cache = torch.randn(2, 4, 1, 8, dtype=torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    valid_length = torch.tensor([1], dtype=torch.int32)
    return (
        source,
        residual,
        query_cosine,
        query_sine,
        key_cosine,
        key_sine,
        key_cache,
        value_cache,
        valid_length,
    )


@pytest.mark.parametrize(
    ("model", "inputs", "mode"),
    [
        (TransformerBlock(device="cpu"), _prefill_inputs(), "prefill"),
        (StatefulTransformerBlock(device="cpu"), _decode_inputs(), "decode"),
    ],
)
def test_task29_forms_a_complete_block_plan(model, inputs, mode):
    plan = _plan(model, inputs, mode)

    assert plan.selection.report.complete
    source_targets = [selection.program.source_target for selection in plan.selection.selections]
    assert source_targets.count("rms_norm") == 2
    assert source_targets.count("rope") == 2
    if mode == "prefill":
        assert any(
            isinstance(candidate.semantics, Attention) and candidate.executable
            for region in plan.selection.graph.regions
            for candidate in region.fusion_candidates
        )
    else:
        assert "megabake.kv_attention" in source_targets
    assert all(phase.work_items for phase in plan.phases)
    assert all(phase.workers for phase in plan.phases)
    assert any(phase.barrier is not None for phase in plan.phases)

    if mode == "decode":
        assert plan.storage.state[0].capacity == 4
        assert plan.storage.state[0].lifetime == "boundary_state"
