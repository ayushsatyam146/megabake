from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from megabake import make_backend

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="task 15 requires an SM90 CUDA device",
)


class ProjectionReduction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(1536, 576, dtype=torch.bfloat16, device="cuda")
        )

    def forward(self, source):
        return torch.relu(F.linear(source, self.weight)).sum(dim=-1)


def test_task_15_compiles_binds_and_reexecutes_through_torch_compile(tmp_path):
    torch.manual_seed(15)
    model = ProjectionReduction().eval()
    backend = make_backend(dump_dir=tmp_path)
    optimized = torch.compile(model, backend=backend, fullgraph=True)
    first = torch.randn(4, 576, dtype=torch.bfloat16, device="cuda")
    second = torch.randn_like(first)

    with torch.no_grad():
        first_output = optimized(first)
        torch.testing.assert_close(first_output, model(first), rtol=0.03, atol=8)

        second_output = optimized(second)
        torch.testing.assert_close(second_output, model(second), rtol=0.03, atol=8)

        model.weight.add_(1)
        changed_parameter_output = optimized(second)
        torch.testing.assert_close(changed_parameter_output, model(second), rtol=0.03, atol=8)

    context = backend.megabake_contexts[-1]
    compiled = backend.megabake_last_compiled
    assert context.requirements["execution"] == "megakernel"
    assert context.parameter_names == {"l_self_parameters_weight_"}
    assert compiled.resources.registers_per_thread > 0
    assert compiled.resources.local_memory_bytes_per_thread == 0
    assert compiled.resources.active_blocks_per_sm >= 1
    assert compiled.resources.max_cooperative_workers >= compiled.plan.launch.grid_size
    assert compiled.resources.workspace_bytes == compiled.plan.storage.workspace_bytes
    assert (
        compiled.resources.peak_live_workspace_bytes
        == compiled.plan.storage.peak_live_workspace_bytes
    )
    assert compiled.resources.body_variant_count > 0
    assert compiled.resources.compiled_code_bytes > 0
    assert compiled.resources.generated_code_bytes > 0
    assert compiled.generated.source_path.exists()
    assert compiled.generated.source_map_path.exists()
    assert len(backend.megabake_last_trace) == 3
