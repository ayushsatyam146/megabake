"""``megabake::linear_attention``: one declared op for Gated DeltaNet, Kimi
Delta Attention and Mamba2 decode, its regions, and its body."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from megabake import canonicalize_semantics, form_regions, import_fx, make_backend
from megabake.ops import Attention

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))


def _qwen35(head_dim=16, value_heads=2, hidden=32, layers=("linear_attention", "full_attention")):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=hidden, intermediate_size=64, num_hidden_layers=len(layers),
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=value_heads,
        linear_key_head_dim=head_dim, linear_value_head_dim=head_dim, layer_types=list(layers),
    )
    model = Qwen3_5ForCausalLM(config)
    with torch.no_grad():
        for layer in model.model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.A_log.uniform_(-1.0, 1.0)
                layer.linear_attn.dt_bias.uniform_(-1.0, 1.0)
                layer.linear_attn.norm.weight.uniform_(0.5, 1.5)
    return model


def _kimi(head_dim=16, layers=2):
    from transformers import KimiLinearConfig, KimiLinearForCausalLM

    config = KimiLinearConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=64,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=2,
        linear_attn_config={"head_dim": head_dim, "num_heads": 2, "short_conv_kernel_size": 4},
        layer_types=["linear_attention"] * layers, mlp_layer_types=["dense"] * layers,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = KimiLinearForCausalLM(config)
    with torch.no_grad():
        for layer in model.model.layers:
            gate = layer.self_attn.forget_gate
            gate.A_log.uniform_(-1.0, 1.0)
            gate.dt_bias.uniform_(-1.0, 1.0)
    return model


def _mamba2(state_size=16, head_dim=16, hidden=32, layers=2):
    from transformers import Mamba2Config, Mamba2ForCausalLM

    config = Mamba2Config(
        vocab_size=64, hidden_size=hidden, num_hidden_layers=layers, state_size=state_size,
        expand=2, head_dim=head_dim, num_heads=2 * hidden // head_dim, n_groups=2, conv_kernel=4,
    )
    model = Mamba2ForCausalLM(config)
    with torch.no_grad():
        for layer in model.backbone.layers:
            layer.mixer.A_log.uniform_(-1.0, 1.0)
            layer.mixer.dt_bias.uniform_(-1.0, 1.0)
            layer.mixer.D.uniform_(0.5, 1.5)
    return model


MODELS = {"qwen3_5": _qwen35, "kimi_linear": _kimi, "mamba2": _mamba2}


def _hugging_face_logits(model, tokens):
    """Logits of every position from one cache-free forward (HF's chunked
    linear attention), which a chain of decode steps must reproduce."""
    return model(tokens, use_cache=False).logits


@pytest.mark.parametrize("name", sorted(MODELS))
def test_decode_steps_match_hugging_face_prefill(name):
    from megabake.fx_handler import DecodeStep

    torch.manual_seed(3)
    model = MODELS[name]().float().eval()
    tokens = torch.randint(0, model.config.vocab_size, (1, 6))
    with torch.no_grad():
        model.set_attn_implementation("eager")
        expected = _hugging_face_logits(model, tokens)
        step = DecodeStep(model, capacity=8)
        state = [
            torch.zeros(s, dtype=d)
            for layer in step.state_shapes(dtype=torch.float32)
            for s, d in layer
        ]
        for position in range(tokens.shape[1]):
            actual = step(tokens[:, position : position + 1], torch.tensor([[position]]), *state)
            torch.testing.assert_close(actual[:, -1], expected[:, position], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("value_heads", [2, 4])
def test_declared_gated_delta_matches_the_decomposed_step_bit_for_bit(value_heads):
    import stateful_models

    torch.manual_seed(5)
    wrapper = stateful_models.StatefulCausalLM(_qwen35(value_heads=value_heads).to(torch.bfloat16).eval())
    runs = {}
    for declared in (False, True):
        stateful_models.DECLARED_LINEAR_ATTENTION = declared
        state = [torch.zeros(s, dtype=d) for layer in wrapper.state_shapes(8) for s, d in layer]
        with torch.no_grad():
            logits = torch.stack([
                wrapper(torch.tensor([[token]]), torch.tensor([[position]]), *state)
                for position, token in enumerate((3, 17, 42, 7, 9))
            ])
        runs[declared] = (logits, state)
    stateful_models.DECLARED_LINEAR_ATTENTION = True
    assert torch.equal(runs[False][0], runs[True][0])
    tokens = 5
    for decomposed, declared in zip(runs[False][1], runs[True][1], strict=True):
        if declared.dim() == 3 and declared.shape[-1] == decomposed.shape[-1] + 1:
            # The declared window is a ring (token t in slot t % kernel); the
            # decomposed one holds the last kernel - 1 tokens, oldest first.
            kernel = declared.shape[-1]
            declared = declared[:, :, [(tokens - (kernel - 1) + j) % kernel for j in range(kernel - 1)]]
        assert torch.equal(decomposed, declared)


def _linear_candidates(model):
    from megabake.fx_handler import DecodeStep

    torch._dynamo.reset()
    step = DecodeStep(model.to(torch.bfloat16).eval(), capacity=8)
    state = [torch.zeros(s, dtype=d) for layer in step.state_shapes() for s, d in layer]
    backend = make_backend(mode="decode", compile_kernel=False, cache=False)
    with torch.no_grad():
        torch.compile(step, backend=backend, fullgraph=True, dynamic=False)(
            torch.tensor([[3]]), torch.tensor([[0]]), *state
        )
    prepared = backend.megabake_last_prepared
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    found = [
        candidate
        for region in graph.regions
        for candidate in region.fusion_candidates
        if isinstance(candidate.semantics, Attention) and candidate.semantics.algorithm == "linear"
    ]
    return graph, found


@pytest.mark.parametrize(
    "name,expected",
    [
        # (update, decay_per_key, state_layout, skip, norm, output gate, key heads, heads)
        ("qwen3_5", ("delta", False, "kv", False, True, "silu", 2, 2)),
        ("kimi_linear", ("delta", True, "kv", False, True, "sigmoid", 2, 2)),
        ("mamba2", ("additive", False, "vk", True, False, "silu", 2, 4)),
    ],
)
def test_each_linear_layer_is_one_declared_attention_region(name, expected):
    torch.manual_seed(3)
    model = MODELS[name]()
    graph, found = _linear_candidates(model)
    layers = sum(kind != "full_attention" for kind in (getattr(model.config, "layer_types", None) or ["mamba"] * 2))
    assert len(found) == layers
    semantics = found[0].semantics
    recurrence = semantics.recurrent
    assert (
        recurrence.update, recurrence.decay_per_key, recurrence.state_layout, recurrence.skip,
        recurrence.norm, recurrence.output_gate, semantics.grouping.kv_heads, semantics.grouping.query_heads,
    ) == expected
    assert found[0].tensor_program.family == "attention"
    (operation,) = (graph.ops[op_id] for op_id in found[0].source_op_ids)
    # Only the state and the ring convolution window are written; the weights
    # stay read-only inputs.
    conv_state = next(load.value_id for load in found[0].tensor_program.loads if load.id == "attention:conv_state")
    assert recurrence.conv_ring
    assert set(operation.effects["mutated_operands"]) == {recurrence.state_value_id, conv_state}


@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("threads", [128, 256])
def test_attention_provider_emits_a_compilable_linear_attention_body(name, threads, tmp_path):
    """The attention provider serves linear-attention regions, and each
    recurrence's body compiles for sm90a at production head dims (no GPU)."""
    cute = pytest.importorskip("cutlass.cute")
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    from megabake.bodies.cuda.sm90.attention import AttentionBodyProvider
    from megabake.bodies.cuda.sm90.linear_attention import BODY_NAME
    from megabake.bodies.interface import SelectionPolicy, TargetFacts
    from megabake.graph import Region

    torch.manual_seed(3)
    model = {"qwen3_5": lambda: _qwen35(head_dim=128, hidden=64, layers=("linear_attention",)),
             "kimi_linear": lambda: _kimi(head_dim=128),
             "mamba2": lambda: _mamba2(state_size=128, head_dim=64, hidden=128)}[name]()
    _, found = _linear_candidates(model)
    candidate = found[0]
    region = Region(
        id=candidate.id, source_op_ids=candidate.source_op_ids,
        external_inputs=candidate.external_inputs, live_outputs=candidate.live_outputs,
        tensor_programs=(candidate.tensor_program,), semantics=candidate.semantics,
    )
    provider = AttentionBodyProvider()
    target = TargetFacts("sm90a", 32, 1024, 232448, 255, 132, True, True)
    proposed = provider.get_candidates(region, target, SelectionPolicy(mode="decode"))
    assert proposed.rejections == () and [c.name for c in proposed.configurations] == [BODY_NAME]
    program = candidate.tensor_program
    config = provider.widen(program, proposed.configurations[0], threads)
    body = provider.emit(program, config)
    assert body.tile_count == candidate.semantics.grouping.query_heads * config.tiles.cta_shape[0]
    path = tmp_path / "body.py"
    path.write_text(body.source)
    spec = importlib.util.spec_from_file_location(f"linear_body_{name}_{threads}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dtypes = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32, "torch.int64": torch.int64, "torch.int32": torch.int32}
    tensors = [torch.zeros(load.source_shape, dtype=dtypes[load.dtype]) for load in program.loads]
    tensors.append(torch.zeros(program.store.shape, dtype=torch.bfloat16))

    @cute.jit
    def launch(work: cute.Tensor, bindings: tuple, scratch: cute.Tensor):
        module.run(work, bindings, scratch, cutlass.Int32(0)).launch(grid=[2, 1, 1], block=[threads, 1, 1])

    cute.compile(
        launch,
        from_dlpack(torch.arange(body.tile_count, dtype=torch.int64)),
        tuple(from_dlpack(tensor) for tensor in tensors),
        from_dlpack(torch.zeros(config.scratch.size_bytes // 4)),
    )
