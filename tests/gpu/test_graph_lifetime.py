"""Captured graphs retain the compiled callable that owns their kernels."""

import gc
import weakref

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {(9, 0), (10, 0)},
    reason="a supported SM90 or SM100 CUDA device is required",
)


def test_graphed_keeps_compiled_callable_alive(monkeypatch, tmp_path):
    from megabake import compiler, graphed, make_backend

    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    torch.manual_seed(345)

    def capture(features):
        args = (
            torch.randn(1, 512, device="cuda", dtype=torch.bfloat16) / 16,
            torch.randn(features, 512, device="cuda", dtype=torch.bfloat16) / 16,
        )
        backend = make_backend(
            mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase"
        )
        function = torch.compile(Gemm().eval(), backend=backend, fullgraph=True, dynamic=False)
        reference = weakref.ref(function)
        with torch.no_grad():
            graph = graphed(function, *args)
        return graph, reference, args

    first = capture(256)
    torch._dynamo.reset()
    second = capture(384)
    torch._dynamo.reset()
    gc.collect()
    for graph, reference, args in (first, second, first, second):
        assert reference() is not None
        with torch.no_grad():
            torch.testing.assert_close(graph(*args), args[0] @ args[1].T, atol=0.01, rtol=0.03)
