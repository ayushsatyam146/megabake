from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from megabake.frontend import make_backend

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="task 21 requires an SM90 CUDA device",
)


class Linear(torch.nn.Module):
    def forward(self, source, weight):
        return torch.nn.functional.linear(source, weight)


def test_task_21_reuses_compilation_with_private_runtime_state():
    torch.manual_seed(21)
    model = Linear().eval()
    source = torch.randn(4, 576, dtype=torch.bfloat16, device="cuda")
    old_weight = torch.nn.Parameter(torch.randn(1536, 576, dtype=torch.bfloat16, device="cuda"))
    new_weight = torch.nn.Parameter(torch.randn_like(old_weight))
    exported = torch.export.export(model, (source, old_weight))
    backend = make_backend(scheduler="phase")

    with torch.no_grad():
        first = backend(exported.graph_module, (source, old_weight))
        second = backend(exported.graph_module, (source, new_weight))

        assert second.megabake_cache_hit
        assert second.megabake_compiled is first.megabake_compiled
        assert not hasattr(second.megabake_compiled, "parameter_tensors")
        assert not hasattr(second.megabake_compiled, "ready")
        assert backend.megabake_plan_cache is not None
        assert backend.megabake_plan_cache.hits == 1

        barrier = threading.Barrier(2)

        def invoke(forward):
            barrier.wait()
            return forward(source)

        with ThreadPoolExecutor(max_workers=2) as executor:
            old_future = executor.submit(invoke, first)
            new_future = executor.submit(invoke, second)
            old_output = old_future.result()
            new_output = new_future.result()

        torch.testing.assert_close(old_output[0], model(source, old_weight), rtol=0.03, atol=8)
        torch.testing.assert_close(new_output[0], model(source, new_weight), rtol=0.03, atol=8)

        old_weight.add_(1)
        torch.testing.assert_close(first(source)[0], model(source, old_weight), rtol=0.03, atol=8)
