import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from megabake import make_backend
from megabake.fx_handler.export import capture_exported_program
from megabake.fx_handler.transformers import capture_transformer
from megabake.graph import dump_compute_graph, import_fx


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.register_buffer("scale", torch.tensor([2.0]))

    def forward(self, value):
        return self.linear(value) * self.scale


def test_exported_fx_rebinds_inputs_parameters_and_uses_post_grad_cache():
    torch.manual_seed(11)
    model = Model().eval()
    first = torch.randn(3, 4)
    second = torch.randn(2, 4)
    exported = torch.export.export(model, (first,))
    backend = make_backend()
    capture = capture_exported_program(exported, args=(first,), model=model, backend=backend)
    assert capture.context.phase == "post_grad"
    assert capture.context.cache_state == "miss"
    assert capture.context.parameter_names == {"linear.weight", "linear.bias"}
    assert isinstance(capture.graph, torch.fx.GraphModule)
    assert capture.graph is capture.context.graph

    with torch.no_grad():
        assert torch.allclose(capture.run(first), model(first))
        assert torch.allclose(capture.run(second), model(second))
        model.linear.weight.add_(0.5)
        assert torch.allclose(capture.run(second), model(second))

    repeat = capture_exported_program(exported, args=(first,), model=model, backend=backend)
    assert repeat.context.phase == "post_grad"
    assert repeat.context.cache_state == "hit"
    assert backend.megabake_cache.hits == 1
    assert backend.megabake_cache.misses == 1
    with torch.no_grad():
        assert torch.allclose(repeat.run(first), model(first))


def test_capture_transformer_extracts_fx_graph_without_torch_compile():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    backend = make_backend(cache=False)
    capture = capture_transformer(model, input_ids, device="cpu", backend=backend)
    assert capture.context.phase == "post_grad"
    assert capture.context.cache_state == "miss"
    assert any(name == "input_ids" for name in capture.context.input_order)
    with torch.no_grad():
        expected = model(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=False,
        )
        torch.testing.assert_close(capture.run(input_ids), expected)


def test_capture_transformer_imports_and_dumps_compute_graph(tmp_path):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    backend = make_backend(cache=False)
    capture = capture_transformer(model, input_ids, device="cpu", backend=backend)
    graph = import_fx(capture.graph, capture.context)
    path = dump_compute_graph(graph, tmp_path / "compute_graph.txt")

    text = path.read_text(encoding="utf-8")
    assert "## Values" in text
    assert "## Operations" in text
    assert "## Output Reconstruction" in text
    assert f"values={len(graph.values)}" in text
    assert f"operations={len(graph.ops)}" in text
