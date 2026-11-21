from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace

import cuda.bindings.driver as cuda
import pytest
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from megabake.bodies import emit_reduction_body
from megabake.bodies.catalog import get_candidates
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.ir import Accumulator, Binary, Cast, Constant, Load, Unary
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="reduction-body launcher requires CUDA"
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


def _program(model, source):
    exported = torch.export.export(model, (source,))
    prepared = prepare_fx_graph(exported.graph_module, (source,))
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    region = graph.regions[0]
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
    body = emit_reduction_body(program, config)
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
    scratch: cute.Tensor,
    stream: cuda.CUstream,
):
    run(
        work_iterator,
        ({", ".join(f"tensor_{index}" for index in range(tensor_count))},),
        scratch,
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


def _run_body(name, program, config, source, output, tmp_path):
    _body_module, body = _load_body(program, config, name, tmp_path)
    launcher = _load_launcher(f"megabake_{name}_body", len(program.loads) + 1, tmp_path)
    work = torch.arange(output.numel(), device="cuda", dtype=torch.int64)
    scratch = torch.zeros(
        output.numel(),
        body.scratch_elements,
        device="cuda",
        dtype=torch.float32,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    views = [_view(work), _view(source), _view(output), _view(scratch)]
    compiled = cute.compile(launcher, *views, stream)
    compiled(*views, stream)
    torch.cuda.synchronize()
    return output


def _l2_program(program, config):
    program, config = _square_program(program, config)
    post = Unary(
        "sqrt",
        Cast(Accumulator(program.accumulator_dtype), program.accumulator_dtype, "none"),
        program.accumulator_dtype,
    )
    return replace(program, post=post), config


def _rms_program(program, config, epsilon, output_dtype):
    program, config = _square_program(program, config)
    reduction_size = next(
        domain.size for domain in program.iteration_domains if domain.axis in program.reduction_axes
    )
    post = Cast(
        Unary(
            "rsqrt",
            Binary(
                "add",
                Binary(
                    "div",
                    Cast(
                        Accumulator(program.accumulator_dtype),
                        program.accumulator_dtype,
                        "none",
                    ),
                    Constant(reduction_size, program.accumulator_dtype),
                    program.accumulator_dtype,
                ),
                Constant(epsilon, program.accumulator_dtype),
                program.accumulator_dtype,
            ),
            program.accumulator_dtype,
        ),
        output_dtype,
        "none",
    )
    program = replace(
        program,
        post=post,
        store=replace(program.store, dtype=output_dtype),
    )
    output_layout = replace(config.layouts.outputs[program.store.value_id], dtype=output_dtype)
    config = replace(
        config,
        layouts=replace(config.layouts, outputs={program.store.value_id: output_layout}),
    )
    return program, config


def _square_program(program, config):
    load = program.loads[0]
    squared = Binary(
        "mul",
        Load(load.id, load.dtype),
        Load(load.id, load.dtype),
        load.dtype,
    )
    combine = replace(program.combine, rhs=replace(program.combine.rhs, operand=squared))
    return replace(program, combine=combine), config


def test_reduction_body_launcher_handles_sum_mean_and_norms(tmp_path):
    class Sum(torch.nn.Module):
        def forward(self, value):
            return value.sum(dim=-1)

    class Mean(torch.nn.Module):
        def forward(self, value):
            return value.mean(dim=-1)

    rows, columns = 5, 1003
    source = torch.randn(rows, columns)
    gpu_source = source.cuda()
    sum_program, sum_config = _program(Sum(), source)
    mean_program, mean_config = _program(Mean(), source)
    l2_program, l2_config = _l2_program(sum_program, sum_config)
    rms_program, rms_config = _rms_program(sum_program, sum_config, 1e-6, "torch.bfloat16")

    outputs = {}
    for name, program, config, dtype in (
        ("sum", sum_program, sum_config, torch.float32),
        ("mean", mean_program, mean_config, torch.float32),
        ("l2", l2_program, l2_config, torch.float32),
        ("rms", rms_program, rms_config, torch.bfloat16),
    ):
        output = torch.full((rows,), float("nan"), device="cuda", dtype=dtype)
        outputs[name] = _run_body(name, program, config, gpu_source, output, tmp_path)

    torch.testing.assert_close(outputs["sum"].cpu(), source.sum(dim=-1))
    torch.testing.assert_close(outputs["mean"].cpu(), source.mean(dim=-1))
    torch.testing.assert_close(outputs["l2"].cpu(), source.pow(2).sum(dim=-1).sqrt())
    expected_rms = torch.rsqrt(source.pow(2).mean(dim=-1) + 1e-6)
    torch.testing.assert_close(
        outputs["rms"].cpu(),
        expected_rms.to(torch.bfloat16),
        rtol=2e-2,
        atol=1e-2,
    )


def test_reduction_body_launcher_handles_softmax_maximum(tmp_path):
    class Amax(torch.nn.Module):
        def forward(self, value):
            return value.amax(dim=-1, keepdim=True)

    rows, columns = 5, 1003
    source = torch.randn(rows, columns)
    source[1, 2] = float("nan")
    program, config = _program(Amax(), source)
    output = torch.full((rows, 1), float("nan"), device="cuda")
    result = _run_body("amax", program, config, source.cuda(), output, tmp_path)

    torch.testing.assert_close(result.cpu(), source.amax(dim=-1, keepdim=True), equal_nan=True)
