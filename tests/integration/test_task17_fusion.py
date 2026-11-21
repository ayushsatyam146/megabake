from __future__ import annotations

import pytest
import torch

from megabake import make_backend

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="task 17 requires an SM90 CUDA device",
)


def test_task17_executes_vector_chain_and_preserves_external_output_fallback(tmp_path):
    class FusedChain(torch.nn.Module):
        def forward(self, source):
            return (torch.relu(source) + 1) * 2

    class ExternalIntermediate(torch.nn.Module):
        def forward(self, source):
            activated = torch.relu(source)
            return activated, (activated + 1) * 2

    torch.manual_seed(17)
    source = torch.randn(4, 576, device="cuda")

    fused_backend = make_backend(dump_dir=tmp_path / "fused")
    fused = torch.compile(FusedChain().eval(), backend=fused_backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        torch.testing.assert_close(fused(source), (torch.relu(source) + 1) * 2)

    fused_compiled = fused_backend.megabake_last_compiled
    search = fused_compiled.search
    assert search is not None
    assert any(
        "region:mul:fused" in candidate.description and candidate.status == "measured"
        for candidate in search.candidates
    )
    assert search.selected is fused_compiled.plan

    fallback_backend = make_backend(dump_dir=tmp_path / "fallback")
    fallback = torch.compile(
        ExternalIntermediate().eval(),
        backend=fallback_backend,
        fullgraph=True,
        dynamic=False,
    )
    expected_intermediate = torch.relu(source)
    with torch.no_grad():
        actual_intermediate, actual_result = fallback(source)
    torch.testing.assert_close(actual_intermediate, expected_intermediate)
    torch.testing.assert_close(actual_result, (expected_intermediate + 1) * 2)

    fallback_compiled = fallback_backend.megabake_last_compiled
    assert [selection.region_id for selection in fallback_compiled.plan.selection.selections] == [
        "region:relu",
        "region:add",
        "region:mul",
    ]
    assert any(
        rejection.region_id == "region:mul:fused"
        for rejection in fallback_compiled.plan.selection.rejections
    )
