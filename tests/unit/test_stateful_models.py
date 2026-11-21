from types import SimpleNamespace

import pytest
import torch

from benchmarks.stateful_models import StatefulCausalLM, _rotary_positions


def test_qwen35_text_rotary_positions_expand_all_three_axes():
    position_ids = torch.tensor([[3, 4], [8, 9]])
    actual = _rotary_positions(SimpleNamespace(model_type="qwen3_5_text"), position_ids)
    assert actual.shape == (3, 2, 2)
    assert torch.equal(actual, position_ids.unsqueeze(0).expand(3, -1, -1))


def test_other_text_rotary_positions_keep_original_shape():
    position_ids = torch.tensor([[3, 4]])
    assert _rotary_positions(SimpleNamespace(model_type="llama"), position_ids) is position_ids


def test_qwen35_hybrid_decode_matches_native_hugging_face():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )
    torch.manual_seed(7)
    model = transformers.Qwen3_5ForCausalLM(config).float().eval()
    tokens = (1, 2, 3)
    expected = []
    cache = None
    with torch.no_grad():
        for token in tokens:
            result = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            expected.append(result.logits)
        step = StatefulCausalLM(model).eval()
        state = [
            torch.zeros(shape, dtype=torch.int32 if dtype == torch.int32 else torch.float32)
            for layer in step.state_shapes(8)
            for shape, dtype in layer
        ]
        for position, token in enumerate(tokens):
            actual = step(torch.tensor([[token]]), torch.tensor([[position]]), *state)
            torch.testing.assert_close(actual, expected[position], rtol=1e-4, atol=1e-5)
