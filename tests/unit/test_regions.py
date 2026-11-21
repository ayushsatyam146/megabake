from __future__ import annotations

from types import SimpleNamespace

import torch

from megabake import (
    canonicalize_semantics,
    dump_region_graph,
    form_regions,
    prepare_fx_graph,
)
from megabake.graph import import_fx
from megabake.ir import Accumulator, Binary, Cast, Load, TensorStore
from megabake.ops import (
    HF_EAGER_NUMERICS,
    Attention,
    CacheEffects,
    GemmEpilogue,
    HeadGrouping,
    Mask,
    RMSNorm,
    Rope,
    ScalarChain,
    ViewPlusGemm,
)


def _context(graph: torch.fx.GraphModule, *values: torch.Tensor) -> SimpleNamespace:
    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    return SimpleNamespace(
        input_order=[node.name for node in placeholders],
        input_specs=[
            SimpleNamespace(
                name=node.name,
                kind="runtime",
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                stride=tuple(value.stride()),
            )
            for node, value in zip(placeholders, values)
        ],
    )


def _import(model: torch.nn.Module, *values: torch.Tensor):
    exported = torch.export.export(model, values)
    prepared = prepare_fx_graph(
        exported.graph_module,
        values,
        output_tree_spec=exported.call_spec.out_spec,
    )
    return canonicalize_semantics(import_fx(prepared.graph, prepared.context))


def test_form_regions_preserves_intermediate_graph_output_and_operation_coverage():
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            projected = torch.nn.functional.linear(x, weight, bias)
            activated = torch.relu(projected)
            return activated.sum(dim=-1), activated

    torch.manual_seed(41)
    x = torch.randn(2, 3)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)
    graph = form_regions(_import(Model(), x, weight, bias))

    assert len(graph.regions) == len(graph.ops)
    assert [region.source_op_ids for region in graph.regions] == [
        (operation_id,) for operation_id in graph.ops
    ]
    activated = graph.values["value:relu"]
    activated_region = next(
        region for region in graph.regions if activated.producer in region.source_op_ids
    )
    assert activated_region.live_outputs == (activated.id,)
    assert graph.outputs == (
        graph.values["value:sum_1"].id,
        activated.id,
    )


def test_form_regions_builds_executable_vector_chain_candidate_and_keeps_baselines():
    class Model(torch.nn.Module):
        def forward(self, value):
            return (torch.relu(value) + 1) * 2

    torch.manual_seed(59)
    value = torch.randn(2, 3)
    graph = form_regions(_import(Model(), value))

    assert len(graph.regions) == len(graph.ops)
    candidate = graph.regions[-1].fusion_candidates[0]
    assert candidate.id == "region:mul:fused"
    assert candidate.source_op_ids == ("op:relu", "op:add", "op:mul")
    assert candidate.external_inputs == ("value:arg0_1",)
    assert candidate.live_outputs == ("value:mul",)
    assert candidate.executable
    assert candidate.cost_bytes == 2 * 3 * 4 * 4
    assert candidate.tensor_program.loads[0].value_id == "value:arg0_1"
    assert candidate.semantics == ScalarChain(saved_bytes=2 * 3 * 4 * 4)


def test_form_regions_builds_executable_scalar_dag_candidate():
    class Model(torch.nn.Module):
        def forward(self, value):
            activated = torch.relu(value)
            return (activated + 1) * (activated * 2)

    torch.manual_seed(59)
    value = torch.randn(2, 3)
    graph = form_regions(_import(Model(), value))

    candidate = graph.regions[-1].fusion_candidates[0]
    assert candidate.id == "region:mul_1:fused"
    assert set(candidate.source_op_ids) == {
        "op:relu",
        "op:add",
        "op:mul_1",
        "op:mul",
    }
    assert candidate.executable
    assert candidate.tensor_program is not None
    assert candidate.tensor_program.loads[0].value_id == "value:arg0_1"


def test_form_regions_rejects_vector_chain_when_intermediate_is_external():
    class Model(torch.nn.Module):
        def forward(self, value):
            activated = torch.relu(value)
            return activated, activated + 1

    torch.manual_seed(61)
    value = torch.randn(2, 3)
    graph = form_regions(_import(Model(), value))

    # The externally used value stays outside the chain, so nothing fuses.
    assert not any(
        candidate.executable
        for region in graph.regions
        for candidate in region.fusion_candidates
    )
    assert len(graph.regions) == len(graph.ops)


def _epilogue_candidates(graph):
    return [
        candidate
        for region in graph.regions
        for candidate in region.fusion_candidates
        if isinstance(candidate.semantics, GemmEpilogue)
    ]


def test_form_regions_pairs_gate_and_up_gemms_with_their_activation():
    class Model(torch.nn.Module):
        def forward(self, source, gate_weight, up_weight):
            gate = torch.nn.functional.linear(source, gate_weight)
            up = torch.nn.functional.linear(source, up_weight)
            return torch.nn.functional.silu(gate) * up

    torch.manual_seed(67)
    source = torch.randn(4, 8, dtype=torch.bfloat16)
    gate_weight = torch.randn(6, 8, dtype=torch.bfloat16)
    up_weight = torch.randn(6, 8, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, gate_weight, up_weight))

    candidate = next(
        candidate
        for candidate in _epilogue_candidates(graph)
        if candidate.semantics.paired
    )
    assert candidate.source_op_ids == (
        "op:mm",
        "op:mm_1",
        "op:convert_element_type_4",
        "op:neg",
        "op:exp",
        "op:add",
        "op:div",
        "op:convert_element_type_5",
        "op:mul",
    )
    assert candidate.executable
    assert candidate.live_outputs == ("value:mul",)
    program = candidate.tensor_program
    assert program.family == "contraction"
    assert [load.value_id for load in program.loads] == [
        "value:arg0_1",
        "value:permute",
        "value:permute_1",
    ]
    assert program.store == TensorStore("value:mul", (4, 6), "torch.bfloat16", (6, 1), True)
    # Each product is rounded to BF16 before the activation, as in eager.
    assert program.post.rhs == Cast(
        Accumulator("torch.float32", 1), "torch.bfloat16", "round_nearest_even"
    )


def test_form_regions_does_not_pair_gemms_whose_results_escape():
    class Model(torch.nn.Module):
        def forward(self, source, gate_weight, up_weight):
            gate = torch.nn.functional.linear(source, gate_weight)
            up = torch.nn.functional.linear(source, up_weight)
            return torch.nn.functional.silu(gate) * up, gate

    torch.manual_seed(67)
    source = torch.randn(4, 8, dtype=torch.bfloat16)
    gate_weight = torch.randn(6, 8, dtype=torch.bfloat16)
    up_weight = torch.randn(6, 8, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, gate_weight, up_weight))

    assert not any(
        candidate.semantics.paired for candidate in _epilogue_candidates(graph)
    )


def test_form_regions_fuses_residual_add_into_gemm_epilogue():
    class Model(torch.nn.Module):
        def forward(self, source, weight, residual):
            return residual + torch.nn.functional.linear(source, weight).view(1, 4, 16)

    torch.manual_seed(71)
    source = torch.randn(4, 8, dtype=torch.bfloat16)
    weight = torch.randn(16, 8, dtype=torch.bfloat16)
    residual = torch.randn(1, 4, 16, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, weight, residual))

    (candidate,) = _epilogue_candidates(graph)
    assert candidate.source_op_ids == ("op:mm", "op:add")
    program = candidate.tensor_program
    residual_load = program.loads[2]
    # The [1, 4, 16] residual is read at GEMM output (row, column).
    assert residual_load.value_id == "value:arg2_1"
    assert residual_load.axes == (0, "o0", "o1")
    assert program.store == TensorStore("value:add", (4, 16), "torch.bfloat16", (16, 1), True)
    assert program.post == Binary(
        "add",
        Load("load:2", "torch.bfloat16"),
        Cast(Accumulator("torch.float32"), "torch.bfloat16", "round_nearest_even"),
        "torch.bfloat16",
    )


def test_form_regions_matches_view_plus_gemm():
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.nn.functional.linear(x, weight, bias)

    torch.manual_seed(43)
    x = torch.randn(2, 3)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)
    graph = form_regions(_import(Model(), x, weight, bias))

    gemm_region = next(
        region for region in graph.regions if region.id == "region:addmm"
    )
    assert gemm_region.semantics == ViewPlusGemm(view="op:permute", source="value:arg1_1")


def test_dump_region_graph_reports_regions_dependencies_and_outputs(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            projected = torch.nn.functional.linear(x, weight, bias)
            activated = torch.relu(projected)
            return activated.sum(dim=-1), activated

    torch.manual_seed(53)
    x = torch.randn(2, 3)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)
    graph = form_regions(_import(Model(), x, weight, bias))
    path = dump_region_graph(graph, tmp_path / "nested" / "region_graph.txt")

    text = path.read_text(encoding="utf-8")
    assert "# RegionGraph" in text
    assert "values=7 operations=4 regions=4 inputs=3 outputs=2" in text
    assert "region:addmm source_ops=['op:addmm']" in text
    assert "external_inputs=['value:arg2_1', 'value:arg0_1', 'value:permute']" in text
    assert "region:permute -> region:addmm value=value:permute" in text
    assert "region:addmm -> region:relu value=value:addmm" in text
    assert "region:relu -> region:sum_1 value=value:relu" in text
    assert "region:sum_1 -> output value=value:sum_1" in text
    assert "region:relu -> output value=value:relu" in text


def test_form_regions_keeps_unfamiliar_operations_representable():
    def unknown_operator(value):
        return value * 2

    class Model(torch.nn.Module):
        def forward(self, x):
            return unknown_operator(x) + 1

    torch.manual_seed(47)
    x = torch.randn(2, 3)
    graph_module = torch.fx.symbolic_trace(Model())
    unknown_node = next(
        node for node in graph_module.graph.nodes if node.op == "call_function"
    )
    unknown_node.meta["val"] = x
    unknown_node.args = ()
    unknown_operation_id = f"op:{unknown_node.name}"
    graph = form_regions(
        canonicalize_semantics(import_fx(graph_module, _context(graph_module, x)))
    )

    unknown_region = graph.regions[0]
    assert unknown_region.source_op_ids == (unknown_operation_id,)
    assert unknown_region.semantics is None
    assert graph.ops[unknown_operation_id].tensor_program is None
    assert len(graph.regions) == len(graph.ops)


def test_form_regions_matches_exact_rms_norm_with_residual_and_casts():
    class Model(torch.nn.Module):
        def forward(self, source, weight, residual):
            widened = source.float()
            variance = torch.pow(widened, 2).mean(dim=-1, keepdim=True)
            scale = torch.rsqrt(variance + 1e-5)
            normalized = scale * widened
            affine = normalized * weight
            return affine.to(torch.bfloat16) + residual

    torch.manual_seed(24)
    source = torch.randn(3, 7, dtype=torch.bfloat16)
    weight = torch.randn(7, dtype=torch.bfloat16)
    residual = torch.randn(3, 7, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, weight, residual))

    tail = next(region for region in graph.regions if region.id == "region:add_1")
    candidate = next(
        candidate
        for candidate in tail.fusion_candidates
        if isinstance(candidate.semantics, RMSNorm)
    )
    assert candidate.semantics == RMSNorm(
        width=7, epsilon=1e-5, weight="value:arg1_1", residual="value:arg2_1"
    )
    program = candidate.tensor_program
    assert program is not None
    assert program.source_target == "rms_norm"
    assert program.reduction_axes == ("r0",)
    assert len(program.loads) == 5
    assert {load.value_id for load in program.loads} == {
        "value:arg0_1",
        "value:arg1_1",
        "value:arg2_1",
    }


def test_form_regions_matches_exact_rope_and_keeps_unfamiliar_variants_generic():
    class Model(torch.nn.Module):
        def forward(self, source, cosine, sine):
            half = source.shape[-1] // 2
            first = (
                source[..., :half] * cosine[..., :half]
                - source[..., half:] * sine[..., :half]
            )
            second = (
                source[..., half:] * sine[..., half:]
                + source[..., :half] * cosine[..., half:]
            )
            return torch.cat((first, second), dim=-1)

    torch.manual_seed(24)
    source = torch.randn(3, 6, dtype=torch.bfloat16)
    cosine = torch.randn(3, 6, dtype=torch.bfloat16)
    sine = torch.randn(3, 6, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, cosine, sine))

    tail = next(region for region in graph.regions if region.id == "region:cat")
    candidate = next(
        candidate
        for candidate in tail.fusion_candidates
        if isinstance(candidate.semantics, Rope)
    )
    assert candidate.semantics == Rope(width=6, rotary_width=6, form="split_halves")
    program = candidate.tensor_program
    assert program is not None
    assert program.source_target == "rope"
    assert len(program.loads) == 5
    assert {load.value_id for load in program.loads} == {
        "value:arg0_1",
        "value:arg1_1",
        "value:arg2_1",
    }

    class CastModel(Model):
        def forward(self, source, cosine, sine):
            return super().forward(source.float(), cosine.float(), sine.float())

    cast_graph = form_regions(_import(CastModel(), source, cosine, sine))
    assert not any(
        isinstance(candidate.semantics, Rope)
        for region in cast_graph.regions
        for candidate in region.fusion_candidates
    )

    half_graph = form_regions(
        _import(
            Model(),
            source.float().to(torch.float16),
            cosine.float().to(torch.float16),
            sine.float().to(torch.float16),
        )
    )
    assert not any(
        isinstance(candidate.semantics, Rope)
        for region in half_graph.regions
        for candidate in region.fusion_candidates
    )


def _grouped_attention(source, key, value, mask):
    query = source.permute(0, 2, 1, 3)
    grouped_key = key.permute(0, 2, 1, 3).repeat_interleave(
        source.shape[2] // key.shape[2], dim=1
    )
    grouped_value = value.permute(0, 2, 1, 3).repeat_interleave(
        source.shape[2] // value.shape[2], dim=1
    )
    scores = (
        query @ grouped_key.transpose(-2, -1)
    ) * source.shape[-1] ** -0.5 + mask
    probabilities = torch.softmax(scores.float(), dim=-1).to(source.dtype)
    return probabilities @ grouped_value, scores


def test_task25_matches_grouped_attention_access_mask_and_outputs():
    class Model(torch.nn.Module):
        def forward(self, source, key, value, mask):
            return _grouped_attention(source, key, value, mask)

    torch.manual_seed(25)
    source = torch.randn(2, 4, 4, 8, dtype=torch.bfloat16)
    key = torch.randn(2, 4, 2, 8, dtype=torch.bfloat16)
    value = torch.randn(2, 4, 2, 8, dtype=torch.bfloat16)
    mask = torch.randn(2, 1, 4, 4)
    graph = form_regions(_import(Model(), source, key, value, mask))

    candidate = next(
        candidate
        for region in graph.regions
        for candidate in region.fusion_candidates
        if isinstance(candidate.semantics, Attention)
    )
    assert candidate.executable
    assert candidate.rejections == ()
    assert candidate.external_inputs == (
        "value:arg0_1",
        "value:arg1_1",
        "value:arg2_1",
        "value:arg3_1",
    )
    attention = candidate.semantics
    assert attention.algorithm == "stable_softmax"
    assert attention.grouping == HeadGrouping(batch=2, query_heads=4, kv_heads=2, head_dim=8)
    assert attention.grouping.group_size == 2
    assert attention.q.value_id == "value:arg0_1"
    assert attention.k.value_id == "value:arg1_1"
    assert attention.v.value_id == "value:arg2_1"
    assert attention.mask == Mask("additive_tensor")
    assert attention.numerics == HF_EAGER_NUMERICS
    assert "value:add" in attention.outputs
    assert attention.cache == CacheEffects(
        "read_only", reads=("value:arg1_1", "value:arg2_1")
    )


def test_task25_matches_typed_causal_mask_with_offset():
    class Model(torch.nn.Module):
        def forward(self, source, key, value):
            rows = torch.arange(source.shape[1]).view(1, 1, -1, 1)
            columns = torch.arange(key.shape[1]).view(1, 1, 1, -1)
            causal = torch.where(columns <= rows + 2, 0.0, -1e4)
            return _grouped_attention(source, key, value, causal)[0]

    torch.manual_seed(25)
    source = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
    key = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
    value = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, key, value))

    candidate = next(
        candidate
        for region in graph.regions
        for candidate in region.fusion_candidates
        if isinstance(candidate.semantics, Attention)
    )
    mask = candidate.semantics.mask
    assert mask.kind == "typed_where"
    assert mask.causal_offset == 2
    assert mask.window_offsets is None


def test_task25_matches_window_mask_bounds():
    class Model(torch.nn.Module):
        def forward(self, source, key, value):
            rows = torch.arange(source.shape[1]).view(1, 1, -1, 1)
            columns = torch.arange(key.shape[1]).view(1, 1, 1, -1)
            inside = (columns <= rows + 2) & (columns >= rows - 1)
            window = torch.where(inside, 0.0, -1e4)
            return _grouped_attention(source, key, value, window)[0]

    torch.manual_seed(25)
    source = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
    key = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
    value = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, key, value))

    candidate = next(
        candidate
        for region in graph.regions
        for candidate in region.fusion_candidates
        if isinstance(candidate.semantics, Attention)
    )
    mask = candidate.semantics.mask
    assert mask.kind == "typed_where"
    assert mask.causal_offset is None
    assert mask.window_offsets == (-1, 2)


def test_task25_preserves_functional_kv_cache_append_effects():
    class Model(torch.nn.Module):
        def forward(self, source, past_key, key, past_value, value, mask):
            full_key = torch.cat((past_key, key), dim=1)
            full_value = torch.cat((past_value, value), dim=1)
            output, scores = _grouped_attention(
                source, full_key, full_value, mask
            )
            return output, scores, full_key, full_value

    torch.manual_seed(25)
    source = torch.randn(1, 2, 2, 8, dtype=torch.bfloat16)
    past_key = torch.randn(1, 1, 2, 8, dtype=torch.bfloat16)
    key = torch.randn(1, 2, 2, 8, dtype=torch.bfloat16)
    past_value = torch.randn(1, 1, 2, 8, dtype=torch.bfloat16)
    value = torch.randn(1, 2, 2, 8, dtype=torch.bfloat16)
    mask = torch.randn(1, 1, 2, 3)
    graph = form_regions(
        _import(Model(), source, past_key, key, past_value, value, mask)
    )

    candidate = next(
        candidate
        for region in graph.regions
        for candidate in region.fusion_candidates
        if isinstance(candidate.semantics, Attention)
    )
    attention = candidate.semantics
    assert attention.cache == CacheEffects(
        "functional_append",
        reads=("value:arg1_1", "value:arg3_1"),
        appends=("value:arg2_1", "value:arg4_1"),
        updates=(attention.k.value_id, attention.v.value_id),
    )


def test_task25_soft_cap_and_row_dependent_hooks_miss_but_preserve_source():
    class Model(torch.nn.Module):
        def forward(self, source, key, value):
            scores = (source @ key.transpose(-2, -1)) * source.shape[-1] ** -0.5
            capped = 30.0 * torch.tanh(scores / 30.0)
            probabilities = torch.softmax(capped.float(), dim=-1).to(source.dtype)
            return probabilities @ value

    torch.manual_seed(25)
    source = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    key = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    value = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, key, value))

    assert not any(
        isinstance(candidate.semantics, Attention)
        for region in graph.regions
        for candidate in region.fusion_candidates
    )
    assert len(graph.regions) == len(graph.ops)
    assert all(
        len({operation_id for operation_id in region.source_op_ids}) == 1
        for region in graph.regions
    )


def test_task25_rejects_row_dependent_score_hook_and_preserves_source():
    class Model(torch.nn.Module):
        def forward(self, source, key, value):
            scores = (source @ key.transpose(-2, -1)) * source.shape[-1] ** -0.5
            centered = scores - scores.mean(dim=-1, keepdim=True)
            probabilities = torch.softmax(centered.float(), dim=-1).to(source.dtype)
            return probabilities @ value

    torch.manual_seed(25)
    source = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    key = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    value = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, key, value))

    assert not any(
        isinstance(candidate.semantics, Attention)
        for region in graph.regions
        for candidate in region.fusion_candidates
    )
    assert len(graph.regions) == len(graph.ops)


def test_form_regions_rope_reads_each_half_at_its_source_slices():
    class Model(torch.nn.Module):
        def forward(self, source, cosine, sine):
            half = source.shape[-1] // 2
            first = (
                source[..., :half] * cosine[..., :half]
                - source[..., half:] * sine[..., :half]
            )
            second = (
                source[..., half:] * sine[..., :half]
                + source[..., :half] * cosine[..., :half]
            )
            return torch.cat((first, second), dim=-1)

    source = torch.randn(3, 6, dtype=torch.bfloat16)
    cosine = torch.randn(3, 6, dtype=torch.bfloat16)
    sine = torch.randn(3, 6, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), source, cosine, sine))

    program = next(
        candidate.tensor_program
        for region in graph.regions
        for candidate in region.fusion_candidates
        if isinstance(candidate.semantics, Rope)
    )

    def offset(load):
        axis = load.axes[-1]
        return 0 if isinstance(axis, str) else axis.offset

    trig_offsets = {
        (load.value_id, offset(load))
        for load in program.loads
        if load.value_id != "value:arg0_1"
    }
    # The second half reuses the first-half trig values, so it reads them
    # half a width earlier than its own output coordinate.
    assert trig_offsets == {
        ("value:arg1_1", 0),
        ("value:arg1_1", -3),
        ("value:arg2_1", 0),
        ("value:arg2_1", -3),
    }


def test_task25_feature_axis_cat_is_not_a_cache_append():
    class Model(torch.nn.Module):
        def forward(self, source, key, value, mask):
            half = key.shape[-1] // 2
            rotated = torch.cat((-key[..., half:], key[..., :half]), dim=-1)
            output, _ = _grouped_attention(source, rotated, value, mask)
            return output

    torch.manual_seed(25)
    source = torch.randn(1, 2, 2, 8, dtype=torch.bfloat16)
    key = torch.randn(1, 2, 2, 8, dtype=torch.bfloat16)
    value = torch.randn(1, 2, 2, 8, dtype=torch.bfloat16)
    mask = torch.randn(1, 1, 2, 2)
    graph = form_regions(_import(Model(), source, key, value, mask))

    candidate = next(
        candidate
        for region in graph.regions
        for candidate in region.fusion_candidates
        if isinstance(candidate.semantics, Attention)
    )
    assert candidate.semantics.cache.policy == "read_only"
    assert candidate.semantics.cache.appends == ()


def test_attention_matches_a_single_kv_head():
    # Gemma 3 270M: one KV head, so the size-1 head axis of K and V carries an
    # arbitrary stride and the strides alone cannot tell heads from tokens.
    class Model(torch.nn.Module):
        def forward(self, query, key, value, mask):
            query = query.transpose(1, 2)  # [b, s, h, d] -> [b, h, s, d]
            value = value.transpose(1, 2)  # key arrives as [b, h, s, d]
            heads = query.shape[1]
            key = key.expand(-1, heads, -1, -1)
            value = value.expand(-1, heads, -1, -1)
            scores = query @ key.transpose(-2, -1) * query.shape[-1] ** -0.5 + mask
            probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            return probabilities @ value

    torch.manual_seed(3)
    query = torch.randn(1, 4, 4, 8, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 4, 8, dtype=torch.bfloat16)
    value = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
    mask = torch.randn(1, 1, 4, 4, dtype=torch.bfloat16)
    graph = form_regions(_import(Model(), query, key, value, mask))

    candidate = next(
        candidate
        for region in graph.regions
        for candidate in region.fusion_candidates
        if isinstance(candidate.semantics, Attention)
    )
    assert candidate.semantics.grouping == HeadGrouping(
        batch=1, query_heads=4, kv_heads=1, head_dim=8
    )
