from __future__ import annotations

import pytest
import torch
from cuda.bindings import driver as cuda
from cutlass import cute

from megabake.bodies import SelectionPolicy
from megabake.bodies.cuda.sm100.gemm import BlackwellGemmProvider, emit_blackwell_gemm
from megabake.compiler import _target_facts
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions
from tests.gpu.test_generic_contraction_body import _load_launcher, _load_module, _view

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in {10, 12},
    reason="Blackwell CUDA device required",
)


@pytest.mark.parametrize("strided_input", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 32, 64), (5, 33, 70), (32, 64, 128)])
def test_blackwell_mma_tails_and_rebinding(tmp_path, dtype, shape, strided_input):
    m, n, k = shape

    class Gemm(torch.nn.Module):
        def forward(self, a, b):
            return a @ b if strided_input else a @ b.T

    torch.manual_seed(42)
    a = (
        torch.randn(k, m, device="cuda", dtype=dtype).T
        if strided_input
        else torch.randn(m, k, device="cuda", dtype=dtype)
    )
    b = torch.randn((k, n) if strided_input else (n, k), device="cuda", dtype=dtype)
    prepared = prepare_fx_graph(torch.export.export(Gemm(), (a, b)).graph_module, (a, b))
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    region = next(r for r in graph.regions if r.tensor_programs[0].family == "contraction")
    candidates = BlackwellGemmProvider().get_candidates(
        region, _target_facts(a.device), SelectionPolicy()
    )
    assert candidates.configurations, candidates.rejections
    body = emit_blackwell_gemm(region.tensor_programs[0], candidates.configurations[0])
    path = tmp_path / "blackwell_body.py"
    path.write_text(body.source)
    _load_module(path, "megabake_blackwell_body")
    launch = _load_launcher("megabake_blackwell_body", 3, tmp_path)
    work = torch.arange(body.tile_count, device="cuda", dtype=torch.int64)
    c = torch.full((m, n), float("nan"), device="cuda", dtype=dtype)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (_view(work), _view(a), _view(b if strided_input else b.T), _view(c), stream)
    compiled = cute.compile(launch, *args)
    for _ in range(2):
        compiled(*args)
        torch.cuda.synchronize()
        # Use FP64 accumulation for the reference.  PyTorch's default BF16
        # GEMM may use reduced-precision reductions and lose accuracy around
        # cancellation on Blackwell.
        expected = (a.double() @ (b.double() if strided_input else b.double().T)).to(dtype)
        torch.testing.assert_close(
            c, expected, atol=0.04 if dtype == torch.bfloat16 else 0.004, rtol=0.01
        )
        a.add_(0.125)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,k", [(129, 70), (257, 2048)])
def test_sm100_matvec_tails_and_rebinding(tmp_path, dtype, n, k, monkeypatch):
    monkeypatch.setenv("MEGABAKE_SM100_MATVEC", "1")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100-specific matvec")

    class Gemm(torch.nn.Module):
        def forward(self, a, b):
            return a @ b.T

    torch.manual_seed(2026)
    a = torch.randn(1, k, device="cuda", dtype=dtype) / k**0.5
    b = torch.randn(n, k, device="cuda", dtype=dtype) / k**0.5
    prepared = prepare_fx_graph(torch.export.export(Gemm(), (a, b)).graph_module, (a, b))
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    region = next(r for r in graph.regions if r.tensor_programs[0].family == "contraction")
    candidates = BlackwellGemmProvider().get_candidates(
        region, _target_facts(a.device), SelectionPolicy()
    )
    config = next(c for c in candidates.configurations if c.name == "blackwell_matvec_1x16")
    body = emit_blackwell_gemm(region.tensor_programs[0], config)
    path = tmp_path / "sm100_matvec_body.py"
    path.write_text(body.source)
    _load_module(path, "megabake_sm100_matvec_body")
    launch = _load_launcher("megabake_sm100_matvec_body", 3, tmp_path)
    work = torch.arange(body.tile_count, device="cuda", dtype=torch.int64)
    result = torch.full((1, n), float("nan"), device="cuda", dtype=dtype)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (_view(work), _view(a), _view(b.T), _view(result), stream)
    compiled = cute.compile(launch, *args)
    for _ in range(2):
        compiled(*args)
        torch.cuda.synchronize()
        expected = (a.double() @ b.double().T).to(dtype)
        torch.testing.assert_close(result, expected, atol=0.001, rtol=0.01)
        a.add_(0.125)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [2, 3, 4])
def test_sm100_smallrows_tails_and_rebinding(tmp_path, dtype, rows, monkeypatch):
    monkeypatch.setenv("MEGABAKE_SM100_MATVEC", "1")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100-specific small-row contraction")

    class Gemm(torch.nn.Module):
        def forward(self, a, b):
            return a @ b.T

    n, k = 129, 70
    torch.manual_seed(2028)
    a = torch.randn(rows, k, device="cuda", dtype=dtype) / k**0.5
    b = torch.randn(n, k, device="cuda", dtype=dtype) / k**0.5
    prepared = prepare_fx_graph(torch.export.export(Gemm(), (a, b)).graph_module, (a, b))
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    region = next(r for r in graph.regions if r.tensor_programs[0].family == "contraction")
    candidates = BlackwellGemmProvider().get_candidates(
        region, _target_facts(a.device), SelectionPolicy()
    )
    config = next(
        c for c in candidates.configurations if c.name == f"blackwell_smallrows_{rows}x16"
    )
    body = emit_blackwell_gemm(region.tensor_programs[0], config)
    path = tmp_path / "sm100_smallrows_body.py"
    path.write_text(body.source)
    _load_module(path, "megabake_sm100_smallrows_body")
    launch = _load_launcher("megabake_sm100_smallrows_body", 3, tmp_path)
    work = torch.arange(body.tile_count, device="cuda", dtype=torch.int64)
    result = torch.full((rows, n), float("nan"), device="cuda", dtype=dtype)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (_view(work), _view(a), _view(b.T), _view(result), stream)
    compiled = cute.compile(launch, *args)
    for _ in range(2):
        compiled(*args)
        torch.cuda.synchronize()
        expected = (a.double() @ b.double().T).to(dtype)
        torch.testing.assert_close(result, expected, atol=0.001, rtol=0.01)
        a.add_(0.125)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("input_rank", [2, 3])
def test_sm100_paired_matvec_fuses_gelu_and_rebinds(dtype, input_rank, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100-specific matvec")
    from megabake import make_backend

    monkeypatch.setenv("MEGABAKE_SM100_PAIRED_MATVEC", "1")

    class Paired(torch.nn.Module):
        def forward(self, source, gate, up):
            return torch.nn.functional.gelu(source @ gate.T, approximate="tanh") * (source @ up.T)

    torch.manual_seed(2027)
    args = (
        torch.randn((1, 70) if input_rank == 2 else (1, 1, 70), device="cuda", dtype=dtype) / 8,
        torch.randn(129, 70, device="cuda", dtype=dtype) / 8,
        torch.randn(129, 70, device="cuda", dtype=dtype) / 8,
    )
    model = Paired().eval()
    backend = make_backend(mode="decode", compile_kernel=True, scheduler="phase")
    compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for current in (args, tuple(value + 0.125 for value in args)):
            result = compiled_model(*current)
            reference = model(*current)
            relative = (
                (result.float() - reference.float()).norm() / reference.float().norm()
            ).item()
            assert relative < 0.02
        assert any(
            s.body.name == "blackwell_matvec_pair_1x16"
            for s in backend.megabake_last_compiled.plan.selection.selections
        )


@pytest.mark.parametrize("scheduler", ["phase", "region-events"])
def test_blackwell_complete_mlp_one_kernel_and_private_isolated_state(tmp_path, scheduler):
    from benchmarks.isolated import _graph_ordered_inputs, _MegaBakeKernel
    from benchmarks.measure import kernel_trace
    from benchmarks.workloads import mlp
    from megabake import make_backend

    torch.manual_seed(123)
    args = (
        torch.randn(5, 17, device="cuda", dtype=torch.bfloat16) / 4,
        torch.randn(33, 17, device="cuda", dtype=torch.bfloat16) / 4,
        torch.randn(33, 17, device="cuda", dtype=torch.bfloat16) / 4,
        torch.randn(17, 33, device="cuda", dtype=torch.bfloat16) / 6,
    )
    backend = make_backend(compile_kernel=True, scheduler=scheduler, dump_dir=tmp_path)
    bound = torch.compile(mlp, backend=backend, fullgraph=True)
    with torch.no_grad():
        result = bound(*args)
        reference = mlp(*args)
        assert (
            (result.float() - reference.float()).norm() / reference.float().norm()
        ).item() < 0.01
        compiled = backend.megabake_last_compiled
        assert any(s.body.provider == "blackwell_mma" for s in compiled.plan.selection.selections)
        # Search may legitimately mix a scalar tail GEMM with MMA at tiny shapes.
        assert any(
            c.description.count("blackwell_mma/") == 3 and c.measurement is not None
            for c in compiled.search.candidates
        )
        trace = kernel_trace(lambda: bound(*args), tmp_path / "trace.json")
        assert trace["launch_count"] == 1
        # Bind entirely new tensors, not just new contents at existing addresses.
        changed = tuple(t + 0.125 for t in args)
        changed_result = bound(*changed)
        changed_reference = mlp(*changed)
        assert (
            (changed_result.float() - changed_reference.float()).norm()
            / changed_reference.float().norm()
        ).item() < 0.01
        isolated = _MegaBakeKernel(compiled, args, _graph_ordered_inputs(compiled, args))
        isolated.launch_call(isolated.arguments())
        torch.cuda.synchronize()
        torch.testing.assert_close(isolated.output, result, atol=0.01, rtol=0.01)
        # Isolated epochs/buffers must not alter the public callable's state.
        torch.testing.assert_close(bound(*args), result, atol=0.01, rtol=0.01)
        if isolated.fast_path:
            token = isolated.packed()
            isolated.launch_fast(token)
            torch.cuda.synchronize()
            torch.testing.assert_close(isolated.output, result, atol=0.01, rtol=0.01)
