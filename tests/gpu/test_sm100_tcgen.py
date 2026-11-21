"""SM100 TMA/tcgen05 inside a cooperative table launch."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="SM100 CUDA device required",
)


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()


def test_shape_tuning_file_accepts_original_settings(monkeypatch, tmp_path):
    from argparse import Namespace

    from megabake import compiler
    from megabake.bodies.cuda.sm100 import tuning
    from megabake.compiler import _target_facts

    for name in (
        "MEGABAKE_SM100_TUNING_FILE",
        "MEGABAKE_SM100_TCGEN",
        "MEGABAKE_SM100_TCGEN_DECODE",
        "MEGABAKE_SM100_TCGEN_MAX_ROWS",
        "MEGABAKE_SM100_TCGEN_TILE",
        "MEGABAKE_SM100_SPLIT_K",
        "MEGABAKE_SM100_EPILOGUE",
    ):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEGABAKE_SM100_EPILOGUE", "1")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", compiler._TABLE_LAUNCH_OPS)
    output = tmp_path / "tuning.json"
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        tuning.tune_gemm(
            Namespace(
                tune_shapes=["128x512"],
                tune_rows=1,
                tune_cache_mib=0,
                tune_split_limits=["1"],
                samples=5,
                output=output,
            )
        )
    monkeypatch.setenv("MEGABAKE_SM100_EPILOGUE", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TUNING_FILE", str(output))
    target = _target_facts(torch.device("cuda:0"))
    assert tuning.choice_for_shape(target, 1, 128, 512) in ((64, 1), (128, 1))


def test_graphed_baseline_copies_new_tokens():
    from benchmarks.full_model import GraphedStep

    token = torch.tensor([[1]], device="cuda")
    step = GraphedStep(lambda token, position: token + position, token, [])
    for token_value, position_value in ((3, 2), (8, 5)):
        result = step(
            torch.tensor([[token_value]], device="cuda"),
            torch.tensor([[position_value]], device="cuda"),
        )
        assert result.item() == token_value + position_value


def test_graphed_prefill_copies_new_inputs(monkeypatch, tmp_path):
    from benchmarks.full_model import GraphedPrefill
    from megabake import compiler, make_backend

    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")

    class Model(torch.nn.Module):
        def forward(self, tokens, attention_mask):
            return tokens @ attention_mask.T

    torch.manual_seed(821)
    source = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16) / 16
    mask = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) / 16
    model = Model().eval()
    backend = make_backend(
        mode="prefill", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    function = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        graph = GraphedPrefill(function, source, mask)
        for x, y in ((source, mask), (source + 0.125, mask - 0.125)):
            torch.testing.assert_close(
                graph(x, attention_mask=y), model(x, attention_mask=y), atol=0.01, rtol=0.03
            )


def test_sm100_auto_tile_mixes_tiles_in_one_kernel(monkeypatch, tmp_path):
    from megabake import compiler, make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", "auto")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class MixedGemm(torch.nn.Module):
        def forward(self, source, first, second):
            return (source @ first.T) @ second.T

    torch.manual_seed(407)
    args = (
        torch.randn(1, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(192, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 192, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = MixedGemm().eval()
    backend = make_backend(
        mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for current in (args, tuple(tensor + 0.125 for tensor in args)):
            actual = compiled(*current)
            expected = model(*current)
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.05)
    phases = [
        phase
        for phase in backend.megabake_last_compiled.plan.phases
        if phase.body.provider == "sm100_tma"
    ]
    assert [phase.body.tiles.cta_shape for phase in phases] == [(8, 64), (8, 128)]
    assert backend.megabake_last_trace[-1].workload_kernel_launches == 1


@pytest.mark.parametrize(
    "tile,splits,rows,columns",
    [
        (64, 2, 1, 192),
        (64, 8, 1, 192),
        (128, 4, 1, 256),
        (128, 2, 9, 256),
    ],
)
def test_sm100_split_k_repeats_and_rebinds(monkeypatch, tmp_path, tile, splits, rows, columns):
    from megabake import compiler, make_backend
    from megabake.bodies.cuda.sm100.gemm import split_k

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", str(tile))
    monkeypatch.setenv("MEGABAKE_SM100_SPLIT_K", str(splits))
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    torch.manual_seed(678)
    args = (
        torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(columns, 512, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = Gemm().eval()
    backend = make_backend(
        mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        rebound = tuple(tensor + 0.125 for tensor in args)
        for index in range(20):
            current = args if index % 2 == 0 else rebound
            actual = compiled(*current)
            expected = model(*current)
            torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.03)
    phases = [
        phase
        for phase in backend.megabake_last_compiled.plan.phases
        if phase.body.provider == "sm100_tma"
    ]
    assert [split_k(phase.body) for phase in phases] == [splits]


@pytest.mark.parametrize("splits", [1, 4])
def test_sm100_fused_residual_and_plain_share_variant(monkeypatch, tmp_path, splits):
    from megabake import compiler, make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", "128")
    monkeypatch.setenv("MEGABAKE_SM100_SPLIT_K", str(splits))
    monkeypatch.setenv("MEGABAKE_SM100_EPILOGUE", "1")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Residual(torch.nn.Module):
        def forward(self, source, first, second, residual):
            return (source @ first.T + residual) @ second.T

    torch.manual_seed(193)
    args = (
        torch.randn(1, 256, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(1, 256, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = Residual().eval()
    backend = make_backend(
        mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    cases = (args, tuple(tensor + 0.125 for tensor in args), args)
    fused_outputs = []
    with torch.no_grad():
        for current in cases:
            actual = compiled(*current)
            fused_outputs.append(actual.clone())
            torch.testing.assert_close(actual, model(*current), atol=0.02, rtol=0.05)
    names = [phase.body.name for phase in backend.megabake_last_compiled.plan.phases]
    assert sum(name.endswith("_epilogue") for name in names) == 1
    assert sum(name.startswith("sm100_tma") for name in names) == 2
    monkeypatch.setenv("MEGABAKE_SM100_EPILOGUE", "0")
    torch._dynamo.reset()
    reference_backend = make_backend(
        mode="decode",
        compile_kernel=True,
        cache=False,
        dump_dir=tmp_path / "unfused",
        scheduler="phase",
    )
    unfused = torch.compile(model, backend=reference_backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for current, expected in zip(cases, fused_outputs, strict=True):
            torch.testing.assert_close(unfused(*current), expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [64, 128])
def test_sm100_tcgen_extended_prefill_rows(monkeypatch, tmp_path, rows):
    from megabake import compiler, make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_MAX_ROWS", "128")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    torch.manual_seed(472)
    source = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16) / 16
    weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) / 16
    backend = make_backend(
        mode="prefill", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    model = Gemm().eval()
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        torch.testing.assert_close(
            compiled(source, weight), model(source, weight), atol=0.01, rtol=0.03
        )
    assert any(
        phase.body.provider == "sm100_tma" for phase in backend.megabake_last_compiled.plan.phases
    )


@pytest.mark.parametrize(
    "rows,token_tile,feature_tile,splits",
    [
        (9, 16, 64, 1),
        (17, 16, 128, 2),
        (9, 32, 64, 8),
        (17, 32, 128, 2),
        (32, 32, 64, 1),
        (128, 32, 128, 1),
    ],
)
def test_sm100_wider_token_tiles(monkeypatch, tmp_path, rows, token_tile, feature_tile, splits):
    from megabake import compiler, make_backend
    from megabake.bodies.cuda.sm100.gemm import split_k

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", str(feature_tile))
    monkeypatch.setenv("MEGABAKE_SM100_TOKEN_TILE", str(token_tile))
    monkeypatch.setenv("MEGABAKE_SM100_SPLIT_K", str(splits))
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_MAX_ROWS", "128")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    torch.manual_seed(551)
    args = (
        torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = Gemm().eval()
    backend = make_backend(
        mode="prefill", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for current in (args, tuple(tensor + 0.125 for tensor in args), args):
            torch.testing.assert_close(compiled(*current), model(*current), atol=0.02, rtol=0.04)
    phase = next(
        phase
        for phase in backend.megabake_last_compiled.plan.phases
        if phase.body.provider == "sm100_tma"
    )
    assert phase.body.tiles.cta_shape == (token_tile, feature_tile)
    assert split_k(phase.body) == splits


@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("features", [256, 19456])
def test_sm100_tcgen_two_dependent_gemms_and_rebinding(monkeypatch, tmp_path, tile, features):
    from megabake import compiler, make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", str(tile))
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class TwoGemm(torch.nn.Module):
        def forward(self, source, first, second):
            return (source @ first.T) @ second.T

    torch.manual_seed(143)
    args = (
        torch.randn(4, 256, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(features, 256, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(128, features, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = TwoGemm().eval()
    backend = make_backend(
        mode="prefill", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for current in (args, tuple(tensor + 0.125 for tensor in args)):
            actual = compiled_model(*current)
            expected = model(*current)
            relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
            assert relative < 0.02
        compiled = backend.megabake_last_compiled
        assert compiled.generated.launch.launch_policy == "megabake_cooperative_table"
        assert sum(phase.body.provider == "sm100_tma" for phase in compiled.plan.phases) == 2
        assert backend.megabake_last_trace[-1].workload_kernel_launches == 1
        if features == 256 and tile == 128:
            unaligned = torch.empty(4 * 256 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(
                4, 256
            )
            unaligned.copy_(args[0])
            assert unaligned.data_ptr() % 16 != 0
            with pytest.raises(Exception, match="16-byte aligned"):
                compiled_model(unaligned, args[1], args[2])


@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("rows", [1, 6, 9, 17, 32])
def test_sm100_tcgen_multiple_token_tiles(monkeypatch, tmp_path, tile, rows):
    from megabake import compiler, make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", str(tile))
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class TwoGemm(torch.nn.Module):
        def forward(self, source, first, second):
            return (source @ first.T) @ second.T

    torch.manual_seed(243)
    args = (
        torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = TwoGemm().eval()
    backend = make_backend(
        mode="prefill", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        actual = compiled_model(*args)
        expected = model(*args)
    relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative < 0.02
    assert (
        sum(
            phase.body.provider == "sm100_tma"
            for phase in backend.megabake_last_compiled.plan.phases
        )
        == 2
    )
    assert backend.megabake_last_trace[-1].workload_kernel_launches == 1


def test_sm100_tcgen_one_token_decode(monkeypatch, tmp_path):
    from megabake import compiler, make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_DECODE", "1")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    args = (
        torch.randn(1, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = Gemm().eval()
    backend = make_backend(
        mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        actual, expected = compiled_model(*args), model(*args)
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
    assert any(
        phase.body.provider == "sm100_tma" for phase in backend.megabake_last_compiled.plan.phases
    )


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("splits", [1, 8])
@pytest.mark.parametrize(
    ("persistent", "acc_stages", "width", "fork"),
    [
        (False, 1, 128, False),
        (True, 1, 256, False),
        (True, 2, 256, False),
        (True, 1, 128, False),
        (True, 1, 256, True),
        (True, 2, 256, True),
    ],
)
def test_sm100_tcgen_multiple_tiles_per_worker(
    monkeypatch, tmp_path, fused, splits, persistent, acc_stages, width, fork
):
    from megabake import compiler, graphed, make_backend
    from megabake.bodies.cuda.sm100 import tuning
    from megabake.bodies.cuda.sm100.gemm import split_k

    monkeypatch.delenv("MEGABAKE_SM100_TUNING_FILE", raising=False)
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", "128")
    monkeypatch.setenv("MEGABAKE_SM100_SPLIT_K", str(splits))
    monkeypatch.setenv("MEGABAKE_SM100_EPILOGUE", "1" if fused else "0")
    monkeypatch.setenv("MEGABAKE_SM100_PERSISTENT", "1" if persistent else "0")
    monkeypatch.setenv("MEGABAKE_SM100_PERSISTENT_STAGES", str(acc_stages))
    monkeypatch.setenv("MEGABAKE_SM100_PERSISTENT_WIDTH", str(width))
    monkeypatch.setenv("MEGABAKE_SM100_CUTLASS_FORK", "1" if fork else "0")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)
    # Model tuning can override the default one-wave split heuristic.
    if splits > 1:
        monkeypatch.setattr(tuning, "choice_for_shape", lambda *args: (128, splits))

    class WideGemm(torch.nn.Module):
        def forward(self, source, weight, residual):
            result = source @ weight.T
            return result + residual if fused else result

    torch.manual_seed(812)
    args = (
        torch.randn(1, 512, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(32768, 512, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(1, 32768, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = WideGemm().eval()
    backend = make_backend(
        mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        actual = compiled(*args)
        torch.testing.assert_close(actual, model(*args), atol=0.02, rtol=0.05)
        generated = backend.megabake_last_compiled.generated
        phase = next(p for p in generated.launch.phases if p.provider == "sm100_tma")
        body = next(
            p.body
            for p in backend.megabake_last_compiled.plan.phases
            if p.body.provider == "sm100_tma"
        )
        assert split_k(body) == splits
        assert body.participants.participating_threads == width
        assert phase.device_tile_count * splits > generated.launch.grid_size
        assert phase.body_name.endswith("_epilogue") is fused
        original = tuple(tensor.clone() for tensor in args)
        rebound = tuple(tensor + 0.125 for tensor in original)
        graph = graphed(compiled, *args)
        for current in (original, rebound, original):
            actual = graph(*current)
            torch.testing.assert_close(actual, model(*current), atol=0.02, rtol=0.05)


@pytest.mark.parametrize("rows,matvec", [(1, True), (6, False)])
@pytest.mark.parametrize("fork", [False, True])
def test_persistent_gemm_with_warp_mma_fallback(monkeypatch, tmp_path, rows, matvec, fork):
    from megabake import compiler, graphed, make_backend

    monkeypatch.delenv("MEGABAKE_SM100_TUNING_FILE", raising=False)
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", "128")
    monkeypatch.setenv("MEGABAKE_SM100_PERSISTENT", "1")
    monkeypatch.setenv("MEGABAKE_SM100_PERSISTENT_WIDTH", "256")
    monkeypatch.setenv("MEGABAKE_SM100_CUTLASS_FORK", "1" if fork else "0")
    monkeypatch.setenv("MEGABAKE_SM100_MATVEC", "1" if matvec else "0")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Mixed(torch.nn.Module):
        def forward(self, source, first, second):
            return (source @ first.T) @ second.T

    torch.manual_seed(991)
    args = (
        torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(160, 256, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = Mixed().eval()
    backend = make_backend(
        mode="decode" if rows == 1 else "prefill",
        compile_kernel=True,
        cache=False,
        dump_dir=tmp_path,
        scheduler="phase",
    )
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        output = compiled(*args)
        torch.testing.assert_close(output, model(*args), atol=0.02, rtol=0.05)
        bodies = [phase.body.provider for phase in backend.megabake_last_compiled.plan.phases]
        assert "sm100_tma" in bodies and "blackwell_mma" in bodies
        assert backend.megabake_last_compiled.plan.launch.threads_per_block == 256
        graph = graphed(compiled, *args)
        for current in (args, tuple(value + 0.125 for value in args), args):
            torch.testing.assert_close(graph(*current), model(*current), atol=0.02, rtol=0.05)


@pytest.mark.parametrize("rows", [1, 32])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("splits", [1, 8])
@pytest.mark.parametrize(
    ("width", "acc_stages", "fork"),
    [(256, 1, False), (256, 2, False), (128, 1, False), (256, 1, True), (256, 2, True)],
)
def test_persistent_gemm_matches_existing_body(
    monkeypatch, tmp_path, rows, fused, splits, width, acc_stages, fork
):
    from megabake import compiler, make_backend
    from megabake.bodies.cuda.sm100.gemm import split_k

    monkeypatch.delenv("MEGABAKE_SM100_TUNING_FILE", raising=False)
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", "64")
    monkeypatch.setenv("MEGABAKE_SM100_TOKEN_TILE", "8" if rows == 1 else "32")
    monkeypatch.setenv("MEGABAKE_SM100_SPLIT_K", str(splits))
    monkeypatch.setenv("MEGABAKE_SM100_EPILOGUE", "1" if fused else "0")
    monkeypatch.setenv("MEGABAKE_SM100_PERSISTENT_STAGES", str(acc_stages))
    monkeypatch.setenv("MEGABAKE_SM100_PERSISTENT_WIDTH", str(width))
    monkeypatch.setenv("MEGABAKE_SM100_CUTLASS_FORK", "1" if fork else "0")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Gemm(torch.nn.Module):
        def forward(self, source, weight, residual):
            result = source @ weight.T
            return result + residual if fused else result

    torch.manual_seed(732)
    args = (
        torch.randn(rows, 1536, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(576, 1536, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(rows, 576, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = Gemm().eval()
    results = []
    for persistent in (False, True):
        monkeypatch.setenv("MEGABAKE_SM100_PERSISTENT", "1" if persistent else "0")
        torch._dynamo.reset()
        backend = make_backend(
            mode="decode" if rows == 1 else "prefill",
            compile_kernel=True,
            cache=False,
            dump_dir=tmp_path / str(persistent),
            scheduler="phase",
        )
        compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
        with torch.no_grad():
            results.append(compiled(*args).clone())
        body = next(
            p.body
            for p in backend.megabake_last_compiled.plan.phases
            if p.body.provider == "sm100_tma"
        )
        assert split_k(body) == splits
    torch.testing.assert_close(results[1], results[0], atol=0, rtol=0)


def test_sm100_persistent_default_runs_cutlass_fork(monkeypatch, tmp_path):
    from megabake import compiler, make_backend

    monkeypatch.delenv("MEGABAKE_SM100_TCGEN", raising=False)
    monkeypatch.delenv("MEGABAKE_SM100_PERSISTENT", raising=False)
    monkeypatch.delenv("MEGABAKE_SM100_CUTLASS_FORK", raising=False)
    monkeypatch.delenv("MEGABAKE_SM100_PERSISTENT_WIDTH", raising=False)
    monkeypatch.delenv("MEGABAKE_SM100_PERSISTENT_STAGES", raising=False)
    monkeypatch.delenv("MEGABAKE_SM100_TUNING_FILE", raising=False)
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    torch.manual_seed(411)
    args = (
        torch.randn(1, 512, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) / 16,
    )
    backend = make_backend(
        mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
    )
    model = Gemm().eval()
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        torch.testing.assert_close(compiled(*args), model(*args), atol=0.01, rtol=0.03)
    plan = backend.megabake_last_compiled.plan
    body = next(phase.body for phase in plan.phases if phase.body.provider == "sm100_tma")
    assert body.participants.participating_threads == 256
    assert "_persistent1_fork" in body.name
    assert "Sm100CutlassForkGemm" in backend.megabake_last_compiled.generated.source


def test_sm100_event_scheduler_uses_fallback_gemm(monkeypatch, tmp_path):
    from megabake import compiler, make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    # Two GEMMs, so the event schedule has more than one phase.
    class Gemms(torch.nn.Module):
        def forward(self, source, first, second):
            return (source @ first.T) @ second.T

    args = (
        torch.randn(4, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(192, 256, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = Gemms().eval()
    backend = make_backend(
        mode="prefill",
        compile_kernel=True,
        cache=False,
        dump_dir=tmp_path,
        scheduler="region-events",
    )
    compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        actual, expected = compiled_model(*args), model(*args)
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
    assert all(
        phase.body.provider != "sm100_tma" for phase in backend.megabake_last_compiled.plan.phases
    )
