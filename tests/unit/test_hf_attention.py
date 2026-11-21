import pytest
import torch

from megabake import canonicalize_semantics, form_regions, import_fx, make_backend
from megabake.fx_handler import register_transformers_attention
from megabake.ops import Attention

transformers = pytest.importorskip("transformers")


def _tiny_models():
    from transformers import (
        Gemma3ForCausalLM,
        Gemma3TextConfig,
        LlamaConfig,
        LlamaForCausalLM,
    )

    llama = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    # One KV head, sliding and full layers, and a query_pre_attn_scalar scale.
    gemma = Gemma3TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        query_pre_attn_scalar=24,
        sliding_window=4,
        max_position_embeddings=64,
        layer_types=["sliding_attention", "full_attention"],
    )
    return (("llama", LlamaForCausalLM, llama), ("gemma3", Gemma3ForCausalLM, gemma))


@pytest.mark.parametrize("name,model_class,config", _tiny_models())
def test_megabake_attention_matches_eager_bit_for_bit(name, model_class, config):
    implementation = register_transformers_attention()
    torch.manual_seed(7)
    model = model_class(config).to(torch.bfloat16).eval()
    ids = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        model.set_attn_implementation("eager")
        eager = model(ids, use_cache=False).logits
        model.set_attn_implementation(implementation)
        declared = model(ids, use_cache=False).logits
    assert torch.equal(declared, eager)


@pytest.mark.parametrize("name,model_class,config", _tiny_models())
def test_declared_attention_forms_one_region_per_layer(name, model_class, config):
    implementation = register_transformers_attention()
    torch.manual_seed(7)
    model = model_class(config).to(torch.bfloat16).eval()
    model.set_attn_implementation(implementation)
    model.config.use_cache = False
    ids = torch.randint(0, config.vocab_size, (1, 6))
    backend = make_backend(compile_kernel=False, cache=False)
    with torch.no_grad():
        torch.compile(model, backend=backend, fullgraph=True, dynamic=False)(ids)
    prepared = backend.megabake_last_prepared
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    declared = [
        candidate
        for region in graph.regions
        for candidate in region.fusion_candidates
        if isinstance(candidate.semantics, Attention)
    ]
    assert len(declared) == config.num_hidden_layers
    grouping = declared[0].semantics.grouping
    assert grouping.query_heads == config.num_attention_heads
    assert grouping.kv_heads == config.num_key_value_heads
    assert all(candidate.source_op_ids[0].startswith("op:attention") for candidate in declared)


@pytest.mark.parametrize("name,model_class,config", _tiny_models())
def test_decode_step_matches_hugging_face_decode(name, model_class, config):
    from megabake.fx_handler import DecodeStep

    config.sliding_window = 16  # the cache capacity below fits inside the window
    torch.manual_seed(7)
    model = model_class(config).float().eval()
    tokens = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        model.set_attn_implementation("eager")
        expected = []
        cache = None
        for position in range(tokens.shape[1]):
            out = model(tokens[:, position : position + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            expected.append(out.logits)
        step = DecodeStep(model, capacity=8)
        state = [
            torch.zeros(s, dtype=d if d == torch.int32 else torch.float32)
            for layer in step.state_shapes()
            for s, d in layer
        ]
        for position in range(tokens.shape[1]):
            actual = step(tokens[:, position : position + 1], torch.tensor([[position]]), *state)
            torch.testing.assert_close(actual, expected[position], rtol=1e-4, atol=1e-4)
    assert all(int(length.item()) == tokens.shape[1] for length in state[2::3])


@pytest.mark.parametrize("name,model_class,config", _tiny_models())
def test_decode_step_captures_one_kv_attention_region_per_layer(name, model_class, config):
    from megabake.fx_handler import DecodeStep

    config.sliding_window = 16
    torch.manual_seed(7)
    step = DecodeStep(model_class(config).to(torch.bfloat16).eval(), capacity=8)
    state = [torch.zeros(s, dtype=d) for layer in step.state_shapes() for s, d in layer]
    backend = make_backend(mode="decode", compile_kernel=False, cache=False)
    with torch.no_grad():
        torch.compile(step, backend=backend, fullgraph=True, dynamic=False)(
            torch.tensor([[3]]), torch.tensor([[0]]), *state
        )
    prepared = backend.megabake_last_prepared
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    stateful = [
        candidate
        for region in graph.regions
        for candidate in region.fusion_candidates
        if candidate.id.endswith(":kv_attention")
    ]
    assert len(stateful) == config.num_hidden_layers
