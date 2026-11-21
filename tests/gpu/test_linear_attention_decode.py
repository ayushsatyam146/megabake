"""Linear-attention decode on the GPU: Gated DeltaNet (Qwen3.5), Kimi Delta
Attention and Mamba2 compile into one megakernel whose linear layers run the
declared ``megabake::linear_attention`` body, and match eager decode."""

import copy
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))


def _models():
    from test_linear_attention import _kimi, _mamba2, _qwen35

    # Production head dims (128 key/value dims; Mamba2 state 128, head 64),
    # and enough layers for the table launch every real model takes: a
    # smaller launch's GEMMs leave other bodies only 2 KiB of shared memory.
    # Qwen3.5 runs linear layers only: its full-attention layers fault at these
    # toy widths with or without linear attention (the 0.8B checkpoint runs;
    # tools/greedy_check.py covers it).
    return {
        "qwen3_5": lambda: _qwen35(head_dim=128, hidden=128, value_heads=4, layers=("linear_attention",) * 8),
        "kimi_linear": lambda: _kimi(head_dim=128, layers=8),
        "mamba2": lambda: _mamba2(state_size=128, head_dim=64, hidden=128, layers=24),
    }


@pytest.mark.parametrize("name", ["qwen3_5", "kimi_linear", "mamba2"])
def test_linear_attention_decode_matches_eager(name, tmp_path):
    from megabake import make_backend
    from megabake.fx_handler import DecodeStep

    torch._dynamo.reset()
    torch.manual_seed(11)
    model = _models()[name]().to("cuda", torch.bfloat16).eval()
    exact = DecodeStep(copy.deepcopy(model).float(), capacity=8)
    step = DecodeStep(model, capacity=8)
    shapes = step.state_shapes()
    state = [torch.zeros(s, dtype=d, device="cuda") for layer in shapes for s, d in layer]
    reference_state = [tensor.clone() for tensor in state]
    # .float() of an FP32 state is the same tensor: clone, or the FP32 step
    # would update the compiled step's state too.
    exact_state = [tensor.float().clone() for tensor in state]
    backend = make_backend(mode="decode", compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase")
    optimized = torch.compile(step, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for position, token in enumerate((5, 17, 3, 42)):
            token = torch.tensor([[token]], device="cuda")
            positions = torch.tensor([[position]], device="cuda")
            actual = optimized(token, positions, *state)
            expected = step(token, positions, *reference_state)
            exact(token, positions, *exact_state)
            error = (actual.float() - expected.float()).norm() / expected.float().norm()
            assert error.item() < 0.02, f"step {position}: relative logit error {error.item():.4f}"

    def relative(value, target):
        return ((value.float() - target).norm() / target.norm().clamp_min(1e-12)).item()

    # Every state is as close to the FP32 model's as eager BF16's is.
    for actual_state, eager_state, fp32_state in zip(state, reference_state, exact_state, strict=True):
        if actual_state.dtype == torch.int32:
            assert torch.equal(actual_state, eager_state)
            continue
        assert relative(actual_state, fp32_state) <= 1.25 * relative(eager_state, fp32_state) + 1e-3

    compiled = backend.megabake_last_compiled
    bodies = [phase.body.name for phase in compiled.plan.phases]
    layers = sum(1 for kind in step.layer_types if kind != "full_attention")
    assert bodies.count("linear_attention_decode_heads") == layers
