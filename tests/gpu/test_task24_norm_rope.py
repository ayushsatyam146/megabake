from __future__ import annotations

from collections import Counter

import pytest
import torch

from megabake import make_backend
from tests.fixtures import CompleteMLP

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="task 24 requires an SM90 CUDA device",
)


class RMSNorm(torch.nn.Module):
    def forward(self, source, weight, residual):
        widened = source.float()
        variance = torch.pow(widened, 2).mean(dim=-1, keepdim=True)
        normalized = torch.rsqrt(variance + 1e-5) * widened
        affine = normalized * weight
        return affine.to(torch.bfloat16) + residual


class RoPE(torch.nn.Module):
    def forward(self, source, cosine, sine):
        half = source.shape[-1] // 2
        first = source[..., :half] * cosine[..., :half] - source[..., half:] * sine[..., :half]
        second = source[..., half:] * sine[..., half:] + source[..., :half] * cosine[..., half:]
        return torch.cat((first, second), dim=-1)


class NormRoPEMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_weight = torch.nn.Parameter(torch.randn(576, device="cuda", dtype=torch.bfloat16))
        self.mlp = CompleteMLP()

    def forward(self, source, residual, cosine, sine):
        normalized = RMSNorm()(source, self.norm_weight, residual)
        projected, gate, activated, fused = self.mlp(normalized)
        rotated = RoPE()(projected, cosine, sine)
        return rotated, gate, activated, fused


def _compile_and_compare(model, tmp_path, inputs):
    backend = make_backend(
        compile_kernel=True,
        cache=False,
        dump_dir=tmp_path,
        scheduler="phase",
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        actual = optimized(*inputs)
        expected = model(*inputs)
    return backend, actual, expected


def test_task24_compiles_exact_rms_norm_with_residual(tmp_path):
    torch.manual_seed(24)
    model = RMSNorm().eval()
    source = torch.randn(4, 7, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(7, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(4, 7, device="cuda", dtype=torch.bfloat16)

    backend, actual, expected = _compile_and_compare(model, tmp_path, (source, weight, residual))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    selections = backend.megabake_last_compiled.plan.selection.selections
    assert [(selection.program.source_target, selection.body.name) for selection in selections] == [
        ("rms_norm", "generic_reduction")
    ]
    assert backend.megabake_contexts[-1].requirements["execution"] == "megakernel"


def test_task24_compiles_exact_rope_with_supported_stride(tmp_path):
    torch.manual_seed(24)
    model = RoPE().eval()
    source = torch.randn(6, 4, device="cuda", dtype=torch.bfloat16).t()
    cosine = torch.randn(4, 6, device="cuda", dtype=torch.bfloat16)
    sine = torch.randn(4, 6, device="cuda", dtype=torch.bfloat16)

    backend, actual, expected = _compile_and_compare(model, tmp_path, (source, cosine, sine))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    selections = backend.megabake_last_compiled.plan.selection.selections
    assert [(selection.program.source_target, selection.body.name) for selection in selections] == [
        ("rope", "generic_elementwise")
    ]
    assert backend.megabake_contexts[-1].requirements["execution"] == "megakernel"


def test_task24_composes_norm_rope_and_mlp_with_resource_checks(tmp_path):
    torch.manual_seed(24)
    model = NormRoPEMLP().eval()
    source = torch.randn(4, 576, device="cuda", dtype=torch.bfloat16) / 24
    residual = torch.randn(4, 576, device="cuda", dtype=torch.bfloat16)
    cosine = torch.randn(4, 576, device="cuda", dtype=torch.bfloat16)
    sine = torch.randn(4, 576, device="cuda", dtype=torch.bfloat16)

    backend, actual, expected = _compile_and_compare(
        model, tmp_path, (source, residual, cosine, sine)
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert all(
        (result.float() - reference.float()).norm().item() / reference.float().norm().item() < 0.01
        for result, reference in zip(actual[1:], expected[1:], strict=True)
    )

    compiled = backend.megabake_last_compiled
    selections = compiled.plan.selection.selections
    source_targets = [selection.program.source_target for selection in selections]
    assert source_targets.count("rms_norm") == 1
    assert source_targets.count("rope") == 1

    providers = Counter(phase.body.provider for phase in compiled.plan.phases)
    assert providers["g0_range"] == 3
    assert providers["generic"] >= 2
    assert {"generic_reduction", "generic_elementwise"}.issubset(
        phase.body.name for phase in compiled.plan.phases
    )
    assert all(phase.work_items for phase in compiled.plan.phases)

    phase_index = {phase.region_id: phase.index for phase in compiled.plan.phases}
    assert all(
        phase_index[dependency] < phase.index
        for phase in compiled.plan.phases
        for dependency in phase.dependencies
    )
    assert all(phase.workers for phase in compiled.plan.phases)
    assert any(phase.barrier is not None for phase in compiled.plan.phases)
    assert (
        max(phase.body.scratch.size_bytes for phase in compiled.plan.phases)
        <= compiled.generated.launch.shared_memory_bytes
    )
    assert compiled.resources.active_blocks_per_sm >= 1
    assert compiled.resources.max_cooperative_workers >= compiled.plan.launch.grid_size
    assert backend.megabake_contexts[-1].requirements["generic_coverage"]["workspace_bytes"] >= 0
