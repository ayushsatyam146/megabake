from __future__ import annotations

import importlib.util
import sys

import cuda.bindings.driver as cuda
import pytest
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from megabake.bodies import emit_contraction_body
from megabake.bodies.catalog import get_candidates
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="contraction-body launcher requires CUDA"
)


def _target_and_policy():
    from megabake.bodies import SelectionPolicy, TargetFacts

    return (
        TargetFacts(
            architecture="sm90a",
            threads_per_warp=32,
            max_threads_per_cta=1024,
            max_shared_memory_per_cta=232_448,
            max_registers_per_thread=255,
            multiprocessor_count=60,
            supports_tma=True,
            supports_wgmma=True,
        ),
        SelectionPolicy(),
    )


def _program(model, *inputs):
    exported = torch.export.export(model, inputs)
    prepared = prepare_fx_graph(exported.graph_module, inputs)
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    region = next(
        region
        for region in graph.regions
        if region.tensor_programs and region.tensor_programs[0].family == "contraction"
    )
    target, policy = _target_and_policy()
    config = get_candidates(region, target, policy).configurations[0]
    return region.tensor_programs[0], config


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_body(program, config, name, tmp_path):
    body = emit_contraction_body(program, config)
    path = tmp_path / f"{name}_body.py"
    path.write_text(body.source, encoding="utf-8")
    return _load_module(path, f"megabake_{name}_body"), body


def _load_launcher(body_name, tensor_count, tmp_path):
    source = f"""from {body_name} import run
import cutlass.cute as cute
import cuda.bindings.driver as cuda


@cute.jit
def launch(
    work_iterator: cute.Tensor,
{"".join(f"    tensor_{index}: cute.Tensor,\n" for index in range(tensor_count))}
    stream: cuda.CUstream,
):
    run(
        work_iterator,
        ({", ".join(f"tensor_{index}" for index in range(tensor_count))},),
        None,
        None,
    ).launch(
        grid=[cute.size(work_iterator), 1, 1],
        block=[128, 1, 1],
        stream=stream,
    )
"""
    module_name = f"megabake_{body_name}_launcher"
    path = tmp_path / f"{module_name}.py"
    path.write_text(source, encoding="utf-8")
    return _load_module(path, module_name).launch


def _view(tensor):
    result = from_dlpack(tensor, assumed_align=16)
    leading_dimension = next(
        (index for index, stride in enumerate(tensor.stride()) if stride == 1),
        None,
    )
    if leading_dimension is None:
        return result
    return result.mark_layout_dynamic(leading_dim=leading_dimension)


def _run_body(name, program, config, sources, output, tmp_path):
    _body_module, body = _load_body(program, config, name, tmp_path)
    launcher = _load_launcher(f"megabake_{name}_body", len(program.loads) + 1, tmp_path)
    threads = body.threads_per_cta
    tile_count = (output.numel() + threads - 1) // threads
    work = torch.arange(tile_count, device="cuda", dtype=torch.int64) * threads
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    views = [_view(work), *(_view(source) for source in sources), _view(output)]
    compiled = cute.compile(launcher, *views, stream)
    compiled(*views, stream)
    torch.cuda.synchronize()
    return output


def test_contraction_body_launcher_handles_gemm_strides_tails_and_bias(tmp_path):
    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    class Biased(torch.nn.Module):
        def forward(self, source, weight, bias):
            return torch.nn.functional.linear(source, weight, bias)

    rows, inner, columns = 5, 70, 33
    source_base = torch.randn(inner, rows, dtype=torch.bfloat16)
    source = source_base.t()
    weight = torch.randn(columns, inner, dtype=torch.bfloat16)
    bias = torch.randn(columns, dtype=torch.bfloat16)

    gemm_program, gemm_config = _program(Gemm(), source, weight)
    biased_program, biased_config = _program(Biased(), source, weight, bias)

    gpu_source = source.cuda()
    gpu_weight = weight.cuda()
    gpu_bias = bias.cuda()
    gemm_output = torch.full((rows, columns), float("nan"), device="cuda", dtype=torch.bfloat16)
    biased_output = torch.full((rows, columns), float("nan"), device="cuda", dtype=torch.bfloat16)
    _run_body(
        "gemm",
        gemm_program,
        gemm_config,
        (gpu_source, gpu_weight.t()),
        gemm_output,
        tmp_path,
    )
    _run_body(
        "biased",
        biased_program,
        biased_config,
        (gpu_source, gpu_weight.t(), gpu_bias),
        biased_output,
        tmp_path,
    )

    torch.testing.assert_close(
        gemm_output.cpu(),
        source @ weight.T,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        biased_output.cpu(),
        torch.nn.functional.linear(source, weight, bias),
        rtol=2e-2,
        atol=2e-2,
    )
