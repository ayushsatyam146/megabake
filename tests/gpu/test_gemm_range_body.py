from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import pytest
import torch

from megabake.bodies import SelectionPolicy, TargetFacts, gemm_candidates
from megabake.bodies.cuda.sm90.gemm import (
    compile_g0_range_gemm,
    g0_range_body,
    g0_range_resources,
)
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="Hopper WGMMA range body requires SM90",
)


class Model(torch.nn.Module):
    def forward(self, source, weight):
        return source @ weight.T


def _region(source, weight):
    exported = torch.export.export(Model(), (source, weight))
    prepared = prepare_fx_graph(exported.graph_module, (source, weight))
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    return next(
        region
        for region in graph.regions
        if region.tensor_programs and region.tensor_programs[0].family == "contraction"
    )


def _target():
    properties = torch.cuda.get_device_properties(0)
    assert (properties.major, properties.minor) == (9, 0)
    return TargetFacts(
        architecture="sm90a",
        threads_per_warp=32,
        max_threads_per_cta=1024,
        max_shared_memory_per_cta=232_448,
        max_registers_per_thread=255,
        multiprocessor_count=properties.multi_processor_count,
        supports_tma=True,
        supports_wgmma=True,
    )


def test_g0_range_body_rebinds_changed_tensors_and_tile_count():
    source = torch.randn(4, 576, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 576, device="cuda", dtype=torch.bfloat16)
    region = _region(source, weight)
    configurations = gemm_candidates(region, _target(), SelectionPolicy()).configurations
    config = next(config for config in configurations if config.provider == "g0_range")
    body = g0_range_body(config)
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    output = torch.empty(4, 1536, device="cuda", dtype=torch.bfloat16)
    compiled = compile_g0_range_gemm(body, source, weight, output, max_clusters, stream)
    resources = g0_range_resources(
        compiled,
        config.participants.participating_threads,
        config.scratch.size_bytes,
    )
    assert resources["registers_per_thread"] == 232
    assert resources["shared_storage_bytes"] == 230_400
    assert resources["active_blocks_per_sm"] == 1

    compiled(source, weight, output, stream)
    torch.cuda.synchronize()
    reference = (source.float() @ weight.float().T).bfloat16()
    first_error = ((output.float() - reference.float()).norm() / reference.float().norm()).item()

    source = torch.randn(256, 576, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(256, 1536, device="cuda", dtype=torch.bfloat16)
    compiled(source, weight, output, stream)
    torch.cuda.synchronize()
    reference = (source.float() @ weight.float().T).bfloat16()
    second_error = ((output.float() - reference.float()).norm() / reference.float().norm()).item()

    assert first_error < 0.005
    assert second_error < 0.005
    assert torch.isfinite(output).all().item()
