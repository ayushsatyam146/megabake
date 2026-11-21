from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megabake.bodies import (
    CandidateSet,
    InvalidBodyConfig,
    SelectionPolicy,
    TargetFacts,
    UnsupportedContractionBody,
    emit_contraction_body,
    emit_map_body,
    emit_reduction_body,
    validate_body_config,
)
from megabake.bodies.catalog import get_candidates
from megabake.bodies.cuda.sm90.gemm import gemm_candidates
from megabake.frontend import prepare_fx_graph
from megabake.graph import Region, import_fx
from megabake.ir import Accumulator, Binary, Cast, Constant, Unary
from megabake.lowering import canonicalize_semantics
from megabake.ops import ViewPlusGemm
from megabake.regions import form_regions


def _target(**changes):
    values = {
        "architecture": "sm90a",
        "threads_per_warp": 32,
        "max_threads_per_cta": 1024,
        "max_shared_memory_per_cta": 232_448,
        "max_registers_per_thread": 255,
        "multiprocessor_count": 60,
        "supports_tma": True,
        "supports_wgmma": True,
    }
    values.update(changes)
    return TargetFacts(**values)


def _policy(**changes):
    values = {
        "allow_generic": True,
        "allow_tuned": True,
        "numerical_policy": "source_semantics",
        "require_measured": False,
    }
    values.update(changes)
    return SelectionPolicy(**values)


def _regions(model, *inputs):
    exported = torch.export.export(model, inputs)
    prepared = prepare_fx_graph(exported.graph_module, inputs)
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    return graph.regions


def test_catalog_returns_generic_map_contract():
    class Model(torch.nn.Module):
        def forward(self, value):
            return torch.relu(value)

    region = _regions(Model(), torch.randn(3, 5))[0]
    result = get_candidates(region, _target(), _policy())

    assert isinstance(result, CandidateSet)
    assert len(result.configurations) == 1
    config = result.configurations[0]
    assert config.provider == "generic"
    assert config.semantic.operation_family == "elementwise"
    assert config.entrypoint.name == "run"
    assert config.entrypoint.arguments == (
        "work_iterator",
        "tensor_bindings",
        "scratch",
        "epilogue",
    )
    assert not config.entrypoint.whole_kernel_launch
    assert config.participants.participating_threads == 128
    assert config.completion.outputs_ready.startswith("assigned output elements")
    assert config.completion.scratch_reusable == "scratch is reusable when run returns"


def test_emit_map_body_generates_copy_source():
    class Model(torch.nn.Module):
        def forward(self, value):
            return value.clone()

    region = _regions(Model(), torch.randn(5, 70))[0]
    config = get_candidates(region, _target(), _policy()).configurations[0]
    body = emit_map_body(region.tensor_programs[0], config)

    assert body.function_name == "run"
    # 350 elements, four per thread: one tile of 128 x 4.
    assert body.tile_count == 1
    assert body.tile_elements == 4 * config.participants.participating_threads
    assert body.work_iterator_dtype == "torch.int64"
    assert "tensor_bindings[0][o0_0, o1_0]" in body.source
    assert "output[o0_3, o1_3] = value_3" in body.source


def test_emit_map_body_generates_zero_rank_scalar_source():
    class Model(torch.nn.Module):
        def forward(self, value):
            return torch.scalar_tensor(4.0)

    region = _regions(Model(), torch.randn(1))[0]
    config = get_candidates(region, _target(), _policy()).configurations[0]
    body = emit_map_body(region.tensor_programs[0], config)

    assert body.tile_count == 1
    assert "output[0] = value" in body.source


def test_emit_reduction_body_generates_sum_mean_and_norm_sources():
    class Sum(torch.nn.Module):
        def forward(self, value):
            return value.sum(dim=-1)

    class Mean(torch.nn.Module):
        def forward(self, value):
            return value.mean(dim=-1)

    sum_region = _regions(Sum(), torch.randn(5, 1003))[0]
    mean_region = _regions(Mean(), torch.randn(5, 1003))[0]
    sum_config = get_candidates(sum_region, _target(), _policy()).configurations[0]
    mean_config = get_candidates(mean_region, _target(), _policy()).configurations[0]
    sum_body = emit_reduction_body(
        sum_region.tensor_programs[0],
        sum_config,
    )
    mean_body = emit_reduction_body(
        mean_region.tensor_programs[0],
        mean_config,
    )

    assert sum_config.scratch.size_bytes == 16
    assert sum_config.tiles.work_partition == "one output row per CTA"
    assert sum_config.evidence.revision.startswith(
        "mirage@f9eb70c254acefc9f3667b2a973d0dcf25471fce"
    )
    assert sum_body.tile_count == mean_body.tile_count == 5
    assert sum_body.reduction_tile == 128
    assert sum_body.scratch_elements == 4
    assert sum_body.scratch_encoding == (
        "scratch[block, warp] holds one accumulator partial per warp"
    )
    assert sum_body.phases == (
        "thread_partial",
        "warp_partial",
        "cta_merge",
        "epilogue",
    )
    for body in (sum_body, mean_body):
        assert "cutlass.range(thread, 1003, 128, unroll_full=True)" in body.source
        assert "cute.arch.warp_reduction_sum(accumulator)" in body.source
        assert "cute.arch.sync_threads()" in body.source
        assert "output[o0] = value" in body.source
    assert "cutlass.Int64(1003)).to(cutlass.Float32)" in mean_body.source


def test_emit_reduction_body_generates_softmax_maximum():
    class Amax(torch.nn.Module):
        def forward(self, value):
            return value.amax(dim=-1, keepdim=True)

    region = _regions(Amax(), torch.randn(5, 1003))[0]
    config = get_candidates(region, _target(), _policy()).configurations[0]
    body = emit_reduction_body(region.tensor_programs[0], config)

    # NaN-propagating maximum, inline: codegen keeps only a body's run().
    assert "lambda left, right: cute.arch.fmax(left, right, nan=True)" in body.source
    assert "accumulator = cute.arch.fmax(accumulator, scratch[block, warp_index], nan=True)" in body.source
    assert "exact_maximum" not in body.source


def test_emit_reduction_body_preserves_norm_epsilon_and_cast():
    class Sum(torch.nn.Module):
        def forward(self, value):
            return value.sum(dim=-1)

    region = _regions(Sum(), torch.randn(4, 129))[0]
    program = region.tensor_programs[0]
    config = get_candidates(region, _target(), _policy()).configurations[0]
    post = Cast(
        Unary(
            "rsqrt",
            Binary(
                "add",
                Binary(
                    "div",
                    Cast(
                        Accumulator(program.accumulator_dtype),
                        program.accumulator_dtype,
                        "none",
                    ),
                    Constant(129, program.accumulator_dtype),
                    program.accumulator_dtype,
                ),
                Constant(1e-6, program.accumulator_dtype),
                program.accumulator_dtype,
            ),
            program.accumulator_dtype,
        ),
        "torch.bfloat16",
        "none",
    )
    program = replace(program, post=post, store=replace(program.store, dtype="torch.bfloat16"))
    output_layout = replace(config.layouts.outputs[program.store.value_id], dtype="torch.bfloat16")
    config = replace(
        config,
        layouts=replace(config.layouts, outputs={program.store.value_id: output_layout}),
    )

    body = emit_reduction_body(program, config)

    assert "cutlass.Float32(1e-06)" in body.source
    assert ".to(cutlass.BFloat16)" in body.source


def test_emit_contraction_body_generates_gemm_source():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(5, 70, dtype=torch.bfloat16)
    weight = torch.randn(33, 70, dtype=torch.bfloat16)
    region = next(
        region
        for region in _regions(Model(), source, weight)
        if isinstance(region.semantics, ViewPlusGemm)
    )
    config = get_candidates(region, _target(), _policy()).configurations[0]

    body = emit_contraction_body(region.tensor_programs[0], config)

    assert config.provider == "generic"
    assert config.tiles.cta_shape == (128, 1)
    assert config.tiles.work_partition == "one output element per thread"
    assert config.tiles.reduction_tile == (32,)
    assert config.evidence.revision.startswith("mirage@f9eb70c254acefc9f3667b2a973d0dcf25471fce")
    assert body.function_name == "run"
    assert body.threads_per_cta == 128
    assert body.output_tile_elements == 128
    assert body.reduction_chunk == 32
    assert body.tile_count == 2
    assert body.phases == ("output_tile", "bounded_reduction", "epilogue")
    assert "if linear_index < output_count" in body.source
    assert "cutlass.range(0, 70, 32)" in body.source
    assert "if reduction_index < 70" in body.source
    assert "accumulator = cutlass.Float32(cutlass.Float32(0))" in body.source
    assert "tensor_bindings[0][o0, r0]" in body.source
    assert "tensor_bindings[1][r0, o1]" in body.source
    assert "stored_value = cutlass.BFloat16((accumulator).to(cutlass.BFloat16))" in body.source
    assert "output[o0, o1] = stored_value" in body.source


def test_emit_map_body_decomposes_coordinates_by_output_strides():
    class Model(torch.nn.Module):
        def forward(self, value):
            return value + 1

    row_major = torch.randn(5, 70)
    column_major = row_major.t()
    programs = []
    for value in (row_major, column_major):
        region = _regions(Model(), value)[0]
        config = get_candidates(region, _target(), _policy()).configurations[0]
        programs.append((region.tensor_programs[0], config))

    row_body = emit_map_body(*programs[0])
    column_body = emit_map_body(*programs[1])

    assert "o1_0 = remainder_0 % 70" in row_body.source
    assert "remainder_0 = remainder_0 // 70" in row_body.source
    assert "o0_0 = remainder_0 % 5" in row_body.source
    assert "o0_0 = remainder_0 % 70" in column_body.source
    assert "remainder_0 = remainder_0 // 70" in column_body.source
    assert "o1_0 = remainder_0 % 5" in column_body.source


def test_emit_contraction_body_preserves_bias_partial_dimensions_and_cast():
    class Model(torch.nn.Module):
        def forward(self, source, weight, bias):
            return torch.nn.functional.linear(source, weight, bias)

    source = torch.randn(7, 67, dtype=torch.bfloat16)
    weight = torch.randn(29, 67, dtype=torch.bfloat16)
    bias = torch.randn(29, dtype=torch.bfloat16)
    region = next(
        region
        for region in _regions(Model(), source, weight, bias)
        if region.tensor_programs and region.tensor_programs[0].family == "contraction"
    )
    program = region.tensor_programs[0]
    config = get_candidates(region, _target(), _policy()).configurations[0]
    program = replace(
        program,
        post=Cast(program.post.operand, "torch.float32", "none"),
        store=replace(program.store, dtype="torch.float32"),
    )
    output_layout = replace(config.layouts.outputs[program.store.value_id], dtype="torch.float32")
    config = replace(
        config,
        layouts=replace(config.layouts, outputs={program.store.value_id: output_layout}),
    )

    body = emit_contraction_body(program, config)

    assert body.tile_count == 2
    assert "cutlass.range(0, 67, 32)" in body.source
    assert "if reduction_index < 67" in body.source
    assert "tensor_bindings[2][o1]" in body.source
    assert ".to(cutlass.Float32)" in body.source


def test_emit_contraction_body_rejects_incompatible_schedule():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(5, 70, dtype=torch.bfloat16)
    weight = torch.randn(33, 70, dtype=torch.bfloat16)
    region = next(
        region
        for region in _regions(Model(), source, weight)
        if isinstance(region.semantics, ViewPlusGemm)
    )
    config = get_candidates(region, _target(), _policy()).configurations[0]

    bad_tiles = replace(config.tiles, cta_shape=(32, 32))
    with pytest.raises(UnsupportedContractionBody, match="flat"):
        emit_contraction_body(region.tensor_programs[0], replace(config, tiles=bad_tiles))

    bad_reduction = replace(config.tiles, reduction_tile=())
    with pytest.raises(UnsupportedContractionBody, match="reduction chunk"):
        emit_contraction_body(region.tensor_programs[0], replace(config, tiles=bad_reduction))


def test_catalog_describes_g0_range_and_rejects_incompatible_target():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(4, 576, dtype=torch.bfloat16)
    weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    region = next(
        region
        for region in _regions(Model(), source, weight)
        if isinstance(region.semantics, ViewPlusGemm)
    )
    result = get_candidates(region, _target(), _policy())

    assert [config.provider for config in result.configurations] == [
        "generic",
        "g0_range",
        "g0_range",
    ]
    tuned = result.configurations[1]
    assert tuned.name == "g0_range_64x64"
    assert tuned.pipeline.stage_count == 13
    larger_tile = result.configurations[2]
    assert larger_tile.name == "g0_range_64x128"
    assert larger_tile.pipeline.stage_count == 8
    assert tuned.participants.registers_per_thread == 232
    assert tuned.scratch.size_bytes == 230_400
    assert larger_tile.scratch.size_bytes == 214_016
    assert tuned.evidence.revision == "0b55a2f691d69981583568fd9eb69687b1f0de8a"
    assert "WGMMA accumulators" in tuned.completion.register_fragments

    rejected = get_candidates(region, _target(max_shared_memory_per_cta=230_399), _policy())
    assert [config.provider for config in rejected.configurations] == [
        "generic",
        "g0_range",
    ]
    g0_rejections = [r for r in rejected.rejections if r.provider == "g0_range"]
    assert any("shared memory" in r.reason for r in g0_rejections)


def test_gemm_candidates_reject_incompatible_operand_layout():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(4, 576, dtype=torch.bfloat16)
    weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    region = next(
        region
        for region in _regions(Model(), source, weight)
        if isinstance(region.semantics, ViewPlusGemm)
    )
    program = region.tensor_programs[0]

    assert len(gemm_candidates(region, _target()).configurations) == 2

    left = replace(program.loads[0], strides=(1, 576))
    incompatible = replace(program, loads=(left, program.loads[1]))
    bad_region = replace(region, tensor_programs=(incompatible,))
    result = gemm_candidates(bad_region, _target())

    assert result.configurations == ()
    assert result.rejections[0].provider == "g0_range"
    assert "K-major" in result.rejections[0].reason


def test_region_without_tensor_program_is_rejected():
    region = Region(
        id="region:unknown",
        source_op_ids=("op:unknown",),
        external_inputs=(),
        live_outputs=(),
        tensor_programs=(),
    )
    result = get_candidates(region, _target(), _policy())

    assert result.configurations == ()
    assert result.rejections[0].reason == "region has no tensor program"


def test_invalid_participation_and_layout_are_rejected():
    class Model(torch.nn.Module):
        def forward(self, value):
            return torch.relu(value)

    region = _regions(Model(), torch.randn(3, 5))[0]
    config = get_candidates(region, _target(), _policy()).configurations[0]

    too_many_threads = replace(
        config.participants, participating_threads=2048, waiting_threads=2048
    )
    with pytest.raises(InvalidBodyConfig, match="threads"):
        validate_body_config(replace(config, participants=too_many_threads), _target())

    first_input = next(iter(config.layouts.inputs))
    bad_layout = replace(config.layouts.inputs[first_input], strides=(1,))
    bad_inputs = {**config.layouts.inputs, first_input: bad_layout}
    with pytest.raises(InvalidBodyConfig, match="mismatched axes and strides"):
        validate_body_config(replace(config, layouts=replace(config.layouts, inputs=bad_inputs)))


def test_emit_map_body_binds_each_load_once():
    class Model(torch.nn.Module):
        def forward(self, left, right):
            return left * right + left

    region = _regions(Model(), torch.randn(4, 8), torch.randn(4, 8))[0]
    config = get_candidates(region, _target(), _policy()).configurations[0]
    body = emit_map_body(region.tensor_programs[0], config)

    # Once per element a thread covers, all before the first store.
    first_store = body.source.index("output[")
    for index, _ in enumerate(region.tensor_programs[0].loads):
        for element in range(4):
            assert body.source.count(f"load_{index}_{element} = ") == 1
            assert body.source.index(f"load_{index}_{element} = ") < first_store
    assert "value_0 = " in body.source and "tensor_bindings[0][o0_0, o1_0]" in body.source


def test_swap_ab_serves_small_m_launches_with_transposed_tiles():
    from megabake.bodies.cuda.sm90.gemm import is_swap_ab, kernel_tile, wants_swap_ab

    assert wants_swap_ab([(4, 2048), (4, 8192), (4, 49152)])
    assert wants_swap_ab([(16, 576)])
    assert wants_swap_ab([(64, 2048)])
    assert not wants_swap_ab([(4, 2048), (256, 2048)])
    assert not wants_swap_ab([])
    body = SimpleNamespace(
        provider="g0_range",
        name="g0_range_swapab_16x64",
        tiles=SimpleNamespace(cta_shape=(16, 64)),
    )
    assert is_swap_ab(body)
    assert kernel_tile(body) == (64, 16)
    body.name = "g0_range_64x64"
    body.tiles.cta_shape = (64, 64)
    assert not is_swap_ab(body)
    assert kernel_tile(body) == (64, 64)


def test_catalog_offers_a_target_the_generic_and_its_generation_providers():
    from megabake.bodies import catalog
    from megabake.bodies.cuda.sm90.gemm import HopperLaunchOptions

    def names(target):
        return [provider.provider_name for provider in catalog.providers(target)]

    assert names(_target()) == ["generic", "g0_range", "gemv", "attention"]
    assert names(_target(architecture="sm100")) == ["generic", "blackwell_mma", "sm100_attention"]
    assert names(_target(architecture="sm80")) == ["generic"]
    assert catalog.provider("attention").provider_name == "attention"

    class Model(torch.nn.Module):
        def forward(self, value, weight):
            return value @ weight

    region = next(
        region
        for region in _regions(
            Model(),
            torch.randn(4, 64, dtype=torch.bfloat16),
            torch.randn(64, 64, dtype=torch.bfloat16),
        )
        if region.tensor_programs[0].family == "contraction"
    )
    graph = SimpleNamespace(
        ops={"op:mm": SimpleNamespace(tensor_program=region.tensor_programs[0])}
    )
    assert catalog.launch_options(graph, _target(), table_launch=True) == HopperLaunchOptions(
        swap_ab=True, swap_ab_rows=16
    )
    assert catalog.launch_options(graph, _target(), table_launch=False) == HopperLaunchOptions()
    assert catalog.launch_options(graph, _target(architecture="sm100"), table_launch=True) is None
