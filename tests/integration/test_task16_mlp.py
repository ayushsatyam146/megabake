from __future__ import annotations

import pytest
import torch
from torch.autograd import DeviceType

from megabake import make_backend
from tests.fixtures import CompleteMLP

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="task 16 requires an SM90 CUDA device",
)


def _relative_errors(actual, expected):
    return tuple(
        ((result.float() - reference.float()).norm() / reference.float().norm()).item()
        for result, reference in zip(actual, expected, strict=True)
    )


@pytest.mark.parametrize("batch_size", [4, 256])
def test_task16_compiles_complete_mlp_and_rebinds_weights(tmp_path, batch_size):
    torch.manual_seed(16)
    model = CompleteMLP().eval()
    backend = make_backend(dump_dir=tmp_path)
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    source = torch.randn(batch_size, 576, dtype=torch.bfloat16, device="cuda") / (576**0.5)

    with torch.no_grad():
        first = optimized(source)
        expected = model(source)
        first_errors = _relative_errors(first, expected)

        model.gate_weight.add_(1)
        changed = optimized(source)
        changed_expected = model(source)
        changed_errors = _relative_errors(changed, changed_expected)

    assert all(error < 0.01 for error in first_errors)
    assert all(error < 0.01 for error in changed_errors)
    assert not torch.equal(first[0], changed[0])

    context = backend.megabake_contexts[-1]
    compiled = backend.megabake_last_compiled
    source_ops = {
        operation_id
        for operation in compiled.plan.graph.ops.values()
        for operation_id in operation.source_fx_ids
    }
    assert context.requirements["execution"] == "megakernel"
    assert context.parameter_names == {
        "l_self_parameters_gate_weight_",
        "l_self_parameters_up_weight_",
        "l_self_parameters_down_weight_",
    }
    assert (
        sum(
            selection.body.provider == "g0_range"
            for selection in compiled.plan.selection.selections
        )
        == 3
    )
    assert any("convert_element_type" in operation_id for operation_id in source_ops)
    assert any(operation_id.endswith(":mul") for operation_id in source_ops)
    assert compiled.generated.source_path.exists()
    assert len(backend.megabake_last_trace) == 2


def test_task16_warmed_megabake_launches_one_kernel(tmp_path):
    torch.manual_seed(16)
    model = CompleteMLP().eval()
    backend = make_backend(dump_dir=tmp_path)
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    source = torch.randn(4, 576, dtype=torch.bfloat16, device="cuda") / (576**0.5)

    with torch.no_grad():
        optimized(source)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            optimized(source)
            torch.cuda.synchronize()

    kernel_names = [
        event.name for event in profiler.events() if event.device_type == DeviceType.CUDA
    ]
    assert len(kernel_names) == 1
