"""Blackwell GEMM bodies, including the SM100 short-row path."""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass, replace

from ....graph import Region
from ....ir import Accumulator, Binary, Cast, Constant, Load, TensorProgram
from ....ops import GemmEpilogue
from ...interface import (
    BodyConfig,
    CandidateSet,
    ParticipantContract,
    ProviderRejection,
    ResourceEvidence,
    SelectionPolicy,
    TargetFacts,
    TileContract,
    validate_body_config,
    widened_participants,
)
from ..generic import GeneratedContractionBody, generic_config
from .config import persistent_gemm_settings

BLACKWELL_ARCHITECTURES = frozenset({"sm100", "sm103", "sm120", "sm121"})


@dataclass(frozen=True)
class BlackwellLaunchOptions:
    tma_gemm: bool = True


def paired_contraction(program):
    def reads_second(expression):
        if isinstance(expression, Accumulator):
            return expression.index == 1
        return any(
            reads_second(getattr(expression, name))
            for name in ("operand", "lhs", "rhs", "condition", "true_value", "false_value")
            if getattr(expression, name, None) is not None
        )

    return program.post is not None and reads_second(program.post)


def epilogue_loads(program):
    return program.loads[3:] if paired_contraction(program) else program.loads[2:]


def epilogue_source(program, symbol):
    from ..generic import _cast_expression, _emit_expression

    loads = epilogue_loads(program)
    value = _cast_expression(
        _emit_expression(
            program.post,
            {load.id: index for index, load in enumerate(loads)},
            {"o0": "o0", "o1": "o1"},
            tuple(loads),
        ),
        program.store.dtype,
    )
    return (
        "@cute.jit\n"
        f"def {symbol}(accumulator, accumulator_1, o0, o1, tensor_bindings):\n"
        f"    return {value.code}\n"
    )


def split_k(body):
    match = re.search(r"_k(\d+)(?:_epilogue)?$", body.name)
    return int(match.group(1)) if match else 1


def cute_gemm_tensors(a, b, c, splits=1):
    from cutlass.cute.runtime import from_dlpack

    if splits != 1:
        raise ValueError("SM100 TMA descriptors require full operands")
    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if b.ndim > 2:
        b = b.reshape(-1, b.shape[-1])
    if any(tensor.data_ptr() % 16 for tensor in (a, b, c)):
        raise ValueError("GEMM tensor base addresses must be 16-byte aligned")

    def view(tensor):
        leading_dim = next(index for index, stride in enumerate(tensor.stride()) if stride == 1)
        return from_dlpack(tensor.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
            leading_dim=leading_dim
        )

    return view(a), view(b), view(c)


def _sm100_tma_candidate(program, base, target, policy, paired, fused=False):
    """Return the short-row body when its contracts hold."""
    mode_enabled = policy.mode == "prefill" or (
        policy.mode == "decode" and os.environ.get("MEGABAKE_SM100_TCGEN_DECODE", "1") != "0"
    )
    if (
        paired
        or target.architecture != "sm100"
        or not target.supports_tma
        or os.environ.get("MEGABAKE_SM100_TCGEN", "1") != "1"
        or (
            isinstance(policy.target_options, BlackwellLaunchOptions)
            and not policy.target_options.tma_gemm
        )
        or not policy.table_launch
        or policy.event_schedule
        or not mode_enabled
        or program.store.dtype != "torch.bfloat16"
    ):
        return None
    rows, columns = program.store.shape
    reduction = next(d.size for d in program.iteration_domains if d.axis == "r0")
    left, right = program.loads[:2]
    left_strides = dict(zip(left.axes, left.strides, strict=True))
    right_strides = dict(zip(right.axes, right.strides, strict=True))
    tile_setting = os.environ.get("MEGABAKE_SM100_TCGEN_TILE", "auto")
    feature_tile = (
        64 if tile_setting == "64" or (tile_setting == "auto" and columns % 128 != 0) else 128
    )
    from .tuning import choice_for_shape

    measured = choice_for_shape(target, rows, columns, reduction)
    if measured is not None:
        feature_tile = measured[0]
    token_setting = os.environ.get("MEGABAKE_SM100_TOKEN_TILE", "8")
    token_tile = int(token_setting) if token_setting in {"8", "16", "32"} else 8
    row_setting = os.environ.get("MEGABAKE_SM100_TCGEN_MAX_ROWS", "32")
    max_rows = int(row_setting) if row_setting in {"32", "64", "128"} else 32
    if (
        not 1 <= rows <= max_rows
        or columns < 128
        or columns % feature_tile != 0
        or reduction % 64 != 0
        or left_strides != {"o0": reduction, "r0": 1}
        or right_strides != {"r0": 1, "o1": reduction}
        or program.store.strides != (columns, 1)
    ):
        return None
    split_setting = os.environ.get("MEGABAKE_SM100_SPLIT_K", "8")
    split_limit = int(split_setting) if split_setting in {"2", "4", "8"} else 1
    output_tiles = ((rows + token_tile - 1) // token_tile) * (columns // feature_tile)
    # Avoid creating a second wave of split tiles.
    splits = 1
    for count in (2, 4, 8):
        if (
            count <= split_limit
            and reduction % (64 * count) == 0
            and output_tiles * count <= target.multiprocessor_count
        ):
            splits = count
    if measured is not None:
        splits = measured[1]
    settings = persistent_gemm_settings()
    persistent, persistent_width, acc_stages, cutlass_fork = (
        settings.enabled,
        settings.width,
        settings.stages,
        settings.cutlass_fork,
    )
    from ..generic import _layouts

    return replace(
        base,
        provider="sm100_tma",
        selection_priority=-2,
        name=(
            f"sm100_tma_swapab_{token_tile}x{feature_tile}"
            + (f"_persistent{acc_stages}" if persistent else "")
            + ("_w128" if persistent and persistent_width == 128 else "")
            + ("_fork" if cutlass_fork else "")
            + (f"_k{splits}" if splits > 1 else "")
            + ("_epilogue" if fused else "")
        ),
        layouts=_layouts(program, alignment_bytes=16),
        tiles=TileContract(
            cta_shape=(token_tile, feature_tile),
            cluster_shape=(1, 1),
            work_partition=f"persistent range of transposed {feature_tile}x{token_tile} output tiles",
            mma_family="tcgen05",
            reduction_tile=(64,),
        ),
        participants=ParticipantContract(
            participating_threads=persistent_width if persistent else 128,
            waiting_threads=persistent_width if persistent else 128,
            warp_roles=(
                ("epilogue",) * 4
                + (("mma", "tma") if cutlass_fork else ("tma", "mma"))
                + ("idle", "idle")
                if persistent and persistent_width == 256
                else ("tma_epilogue", "mma_epilogue", "epilogue", "epilogue")
                if persistent
                else ("tma_mma", "epilogue", "epilogue", "epilogue")
            ),
            registers_per_thread=128,
        ),
        pipeline=replace(
            base.pipeline,
            stage_count=4,
            barriers=("tma_umma", "umma_epilogue"),
            async_operations=("tma_load", "tcgen05_mma"),
            descriptors={"A": "TMA weight", "B": "TMA activation"},
            drain_rule="wait for the TMEM result and free the allocation before return",
        ),
        scratch=replace(
            base.scratch, size_bytes=73_728 if token_tile == 8 else 86_016, alignment_bytes=128
        ),
        completion=replace(
            base.completion,
            register_fragments="TMEM accumulators are copied to registers and stored",
        ),
        evidence=ResourceEvidence(
            measured=False,
            source="experimental SM100 persistent pipeline"
            if persistent
            else "experimental SM100 TMA/tcgen05 pipeline",
            revision="cutlass-4.8.0-persistent" if persistent else "cutlass-4.8.0-tutorial",
            resources={
                "threads": persistent_width if persistent else 128,
                "shared_bytes": 73_728 if token_tile == 8 else 86_016,
            },
            measurements={},
        ),
    )


def _epilogue_rejection(program, policy):
    from ....ir import AxisAffine
    from ..generic import UnsupportedMapBody

    if not policy.table_launch:
        return "fused epilogues require a table launch"
    if paired_contraction(program):
        first, second = program.loads[1:3]
        if (first.axes, first.strides, first.dtype, first.source_shape) != (
            second.axes,
            second.strides,
            second.dtype,
            second.source_shape,
        ):
            return "paired operands must share one layout"
    for load in epilogue_loads(program):
        for axis in load.axes:
            references = (
                (axis,)
                if isinstance(axis, str)
                else tuple(term.axis for term in axis.terms)
                if isinstance(axis, AxisAffine)
                else ()
                if isinstance(axis, int)
                else None
            )
            if references is None or not set(references) <= {"o0", "o1"}:
                return "epilogue operands must be indexed by the output coordinates"
    try:
        epilogue_source(program, "_epilogue")
    except UnsupportedMapBody as error:
        return f"epilogue expression is not supported: {error}"
    return None


class BlackwellGemmProvider:
    provider_name = "blackwell_mma"

    def emit(
        self,
        program: TensorProgram,
        config: BodyConfig,
        *,
        defer_length_publish: bool = False,
        workers: int | None = None,
    ) -> GeneratedContractionBody:
        generated = emit_blackwell_gemm(program, config)
        if config.participants.participating_threads == 256:
            marker = "    workers, _, _ = cute.arch.grid_dim()\n"
            if generated.source.count(marker) != 1:
                raise ValueError("unexpected warp-MMA source while widening the block")
            head, body = generated.source.split(marker, 1)
            generated = replace(
                generated,
                source=head + marker + "    if thread < 128:\n" + textwrap.indent(body, "    "),
                threads_per_cta=256,
            )
        return generated

    def widen(self, program: TensorProgram, config: BodyConfig, threads: int) -> BodyConfig:
        if threads == config.participants.participating_threads:
            return config
        if threads != 256 or config.participants.participating_threads != 128:
            raise ValueError("the warp-MMA GEMM body requires 128 active threads")
        participants = widened_participants(config, threads)
        return replace(
            config,
            participants=replace(
                participants,
                warp_roles=config.participants.warp_roles + ("idle",) * 4,
            ),
        )

    def get_candidates(
        self, region: Region, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        def reject(reason):
            return CandidateSet((), (ProviderRejection(self.provider_name, reason),))

        if target.architecture not in BLACKWELL_ARCHITECTURES:
            return reject("provider requires a Blackwell target")
        if not policy.allow_tuned:
            return reject("tuned providers disabled")
        if len(region.tensor_programs) != 1:
            return reject("provider requires a single tensor program")
        program = region.tensor_programs[0]
        paired = isinstance(region.semantics, GemmEpilogue) and region.semantics.paired
        if paired and os.environ.get("MEGABAKE_SM100_PAIRED_MATVEC") != "1":
            return reject("experimental paired matvec is disabled")
        if (
            program.family != "contraction"
            or (len(program.loads) != 3 if paired else len(program.loads) < 2)
            or len(program.store.shape) != 2
            or program.reduction_axes != ("r0",)
        ):
            return reject("provider supports plain or paired 2-D GEMM only")
        if any(domain.size < 1 for domain in program.iteration_domains):
            return reject("provider requires nonempty GEMM dimensions")
        dtype = program.store.dtype
        if dtype not in {"torch.float16", "torch.bfloat16"} or any(
            load.dtype != dtype for load in program.loads[: 3 if paired else 2]
        ):
            return reject("provider requires matching FP16 or BF16 operands and output")
        left, right = program.loads[:2]
        if left.axes != ("o0", "r0") or right.axes not in {("r0", "o1"), ("o1", "r0")}:
            return reject("provider requires plain GEMM indexing")
        if any(stride < 0 for load in program.loads for stride in load.strides):
            return reject("negative operand strides are unsupported")
        # Reject scaled contractions and fused epilogues rather than silently
        # substituting plain GEMM for a different tensor program.
        acc = Accumulator("torch.float32")
        product = Binary(
            "mul",
            Cast(Load(left.id, dtype), "torch.float32", "round_nearest_even"),
            Cast(Load(right.id, dtype), "torch.float32", "round_nearest_even"),
            "torch.float32",
        )
        if (
            program.accumulator_dtype != "torch.float32"
            or program.init != Constant(0.0, "torch.float32")
            or program.combine != Binary("add", acc, product, "torch.float32")
        ):
            return reject("provider requires an unscaled FP32 accumulation and result cast")
        fused = not paired and (
            len(program.loads) != 2 or program.post != Cast(acc, dtype, "round_nearest_even")
        )
        if fused:
            epilogue_default = "1" if target.architecture == "sm100" else "0"
            if os.environ.get("MEGABAKE_SM100_EPILOGUE", epilogue_default) != "1":
                return reject("experimental SM100 fused epilogue is disabled")
            reason = _epilogue_rejection(program, policy)
            if reason is not None:
                return reject(reason)
        if paired:
            from ..generic import UnsupportedMapBody, _emit_expression

            second = program.loads[2]
            if (second.axes, second.strides, second.dtype, second.source_shape) != (
                right.axes,
                right.strides,
                right.dtype,
                right.source_shape,
            ):
                return reject("paired weights must have the same shape and layout")
            if program.post is None:
                return reject("paired GEMM requires an epilogue")
            try:
                expression = _emit_expression(program.post, {}, {"o0": "0", "o1": "0"}, ())
            except UnsupportedMapBody:
                return reject("paired GEMM epilogue is unsupported")
            accumulator_indices = {
                int(match.group(1) or 0)
                for match in re.finditer(r"\baccumulator(?:_(\d+))?\b", expression.code)
            }
            if accumulator_indices != {0, 1}:
                return reject("paired epilogue must consume its two accumulators")
            # The epilogue recognizer only accepts contiguous outputs, so one
            # row is enough for the body to store through a flat view.
            if program.store.shape[0] != 1:
                return reject("paired matvec output must be a single contiguous row")
        base = generic_config(region, program)
        mma = replace(
            base,
            provider=self.provider_name,
            name="blackwell_mma_16x32x16",
            tiles=TileContract(
                cta_shape=(16, 32),
                cluster_shape=(1, 1),
                work_partition="persistent range of output tiles",
                mma_family="mma.sync",
                reduction_tile=(16,),
            ),
            participants=ParticipantContract(
                participating_threads=128,
                waiting_threads=128,
                warp_roles=("mma",) * 4,
                registers_per_thread=64,
            ),
            completion=replace(
                base.completion,
                register_fragments="warp MMA FP32 accumulators are cast and stored before return",
            ),
            evidence=ResourceEvidence(
                measured=False,
                source="megabake Blackwell warp MMA baseline",
                revision="blackwell-mma-v1",
                resources={"threads": 128, "shared_bytes": 0},
                measurements={},
            ),
        )
        tma = _sm100_tma_candidate(program, base, target, policy, paired, fused)
        if fused:
            return (
                CandidateSet((tma,), ())
                if tma is not None
                else reject("SM100 fused epilogues require the compatible TMA body")
            )
        # Warp reduction changes BF16 accumulation order and can change greedy
        # tokens. Keep it an explicit experiment until wider model validation.
        if not paired and os.environ.get("MEGABAKE_SM100_MATVEC") != "1":
            return CandidateSet((tma, mma) if tma is not None else (mma,), ())
        contiguous_k = right.strides[right.axes.index("r0")] == 1
        rows = program.store.shape[0]
        if (
            target.architecture == "sm100"
            and 1 <= rows <= 4
            and (not paired or rows == 1)
            and program.store.shape[1] >= 128
            and contiguous_k
        ):
            name = (
                "blackwell_matvec_pair_1x16"
                if paired
                else "blackwell_matvec_1x16"
                if rows == 1
                else f"blackwell_smallrows_{rows}x16"
            )
            matvec = replace(
                base,
                provider=self.provider_name,
                selection_priority=-1,
                name=name,
                tiles=TileContract(
                    cta_shape=(rows, 16),
                    cluster_shape=(1, 1),
                    work_partition="four warps, each reducing four output columns over K",
                    mma_family=None,
                    reduction_tile=(32,),
                ),
                participants=ParticipantContract(
                    participating_threads=128,
                    waiting_threads=128,
                    warp_roles=("matvec",) * 4,
                    registers_per_thread=64,
                ),
                completion=replace(
                    base.completion,
                    register_fragments="warp-reduced FP32 accumulators are cast and stored before return",
                ),
                evidence=ResourceEvidence(
                    measured=False,
                    source="SM100 warp-parallel small-row contraction",
                    revision="sm100-matvec-v1",
                    resources={"threads": 128, "shared_bytes": 0},
                    measurements={},
                ),
            )
            if tma is not None:
                return CandidateSet((tma, matvec, mma), ())
            return CandidateSet((matvec,) if paired else (matvec, mma), ())
        if tma is not None:
            return CandidateSet((tma, mma), ())
        if paired:
            return reject("paired GEMM is supported only by SM100 single-row matvec")
        return CandidateSet((mma,), ())


def emit_blackwell_gemm(program: TensorProgram, config: BodyConfig) -> GeneratedContractionBody:
    """Emit a synchronous 16x32 CTA tile with predicated M/N/K tails."""
    validate_body_config(config)
    if config.provider == "blackwell_mma" and config.name.startswith("blackwell_smallrows_"):
        return _emit_blackwell_smallrows(program, config)
    if config.provider == "blackwell_mma" and config.name in {
        "blackwell_matvec_1x16",
        "blackwell_matvec_pair_1x16",
    }:
        return _emit_blackwell_matvec(program, config)
    if config.provider != "blackwell_mma" or config.tiles.cta_shape != (16, 32):
        raise ValueError("Blackwell GEMM requires the 16x32 warp MMA body contract")
    rows, columns = program.store.shape
    reduction = next(d.size for d in program.iteration_domains if d.axis == "r0")
    dtype = "cutlass.BFloat16" if program.store.dtype == "torch.bfloat16" else "cutlass.Float16"
    right_index = "(k, n)" if program.loads[1].axes == ("r0", "o1") else "(n, k)"
    ntiles = (columns + 31) // 32
    source = f"""import cutlass
import cutlass.cute as cute

@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op({dtype}, cutlass.Float32, (16, 8, 16)), (1, 4, 1))
    thr_mma = mma.get_slice(thread)
    ca = thr_mma.partition_A(cute.make_identity_tensor((16, 16)))
    cb = thr_mma.partition_B(cute.make_identity_tensor((32, 16)))
    cc = thr_mma.partition_C(cute.make_identity_tensor((16, 32)))
    fa = cute.make_rmem_tensor(ca.shape, {dtype})
    fb = cute.make_rmem_tensor(cb.shape, {dtype})
    fc = cute.make_rmem_tensor(cc.shape, cutlass.Float32)
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        tile_id = cutlass.Int32(work_iterator[tile])
        row_base = (tile_id // {ntiles}) * 16
        col_base = (tile_id % {ntiles}) * 32
        fc.fill(0.0)
        for kbase in cutlass.range(0, {reduction}, 16):
            for i in cutlass.range_constexpr(cute.size(fa)):
                m = row_base + ca[i][0]
                k = kbase + ca[i][1]
                fa[i] = {dtype}(0.0)
                if m < {rows} and k < {reduction}:
                    fa[i] = tensor_bindings[0][m, k]
            for i in cutlass.range_constexpr(cute.size(fb)):
                n = col_base + cb[i][0]
                k = kbase + cb[i][1]
                fb[i] = {dtype}(0.0)
                if n < {columns} and k < {reduction}:
                    fb[i] = tensor_bindings[1][{right_index}]
            cute.gemm(mma, fc, fa, fb, fc)
        for i in cutlass.range_constexpr(cute.size(fc)):
            m = row_base + cc[i][0]
            n = col_base + cc[i][1]
            if m < {rows} and n < {columns}:
                tensor_bindings[2][m, n] = {dtype}(fc[i])
"""
    return GeneratedContractionBody(
        source=source,
        threads_per_cta=128,
        output_tile_elements=512,
        reduction_chunk=16,
        tile_count=((rows + 15) // 16) * ntiles,
        work_step=1,
        work_iterator_encoding="one row-major 16x32 output tile ordinal per CTA",
        phases=("output_tile", "warp_mma", "epilogue"),
    )


def _emit_blackwell_matvec(program: TensorProgram, config: BodyConfig) -> GeneratedContractionBody:
    """One warp reduces K for four columns, with FP32 accumulation."""
    rows, columns = program.store.shape
    if rows != 1 or config.tiles.cta_shape != (1, 16):
        raise ValueError("Blackwell matvec requires one row and a 1x16 tile")
    reduction = next(d.size for d in program.iteration_domains if d.axis == "r0")
    dtype = "cutlass.BFloat16" if program.store.dtype == "torch.bfloat16" else "cutlass.Float16"
    paired = config.name == "blackwell_matvec_pair_1x16"
    transposed = program.loads[1].axes == ("r0", "o1")
    bindings = 4 if paired else 3
    output_view = (
        f"    output_row = cute.make_tensor(tensor_bindings[{bindings - 1}].iterator, "
        f"cute.make_layout({columns}))\n"
        if paired
        else ""
    )
    cols = "\n".join(f"        col{i} = tile_id * 16 + warp * 4 + {i}" for i in range(4))
    init = "\n".join(
        f"        acc{i} = cutlass.Float32(0.0)"
        + (f"\n        other{i} = cutlass.Float32(0.0)" if paired else "")
        for i in range(4)
    )
    accum = "\n".join(
        f"            if kk < {reduction} and col{i} < {columns}:\n"
        f"                acc{i} = acc{i} + x * tensor_bindings[1]["
        f"{'kk, col' + str(i) if transposed else 'col' + str(i) + ', kk'}].to(cutlass.Float32)"
        + (
            f"\n                other{i} = other{i} + x * tensor_bindings[2]["
            f"{'kk, col' + str(i) if transposed else 'col' + str(i) + ', kk'}].to(cutlass.Float32)"
            if paired
            else ""
        )
        for i in range(4)
    )
    stores = []
    for i in range(4):
        stores.append(f"        sum{i} = cute.arch.warp_reduction_sum(acc{i})")
        if paired:
            from ..generic import _cast_expression, _emit_expression

            stores.append(f"        other_sum{i} = cute.arch.warp_reduction_sum(other{i})")
            result = (
                _cast_expression(
                    _emit_expression(program.post, {}, {"o0": "0", "o1": f"col{i}"}, ()),
                    program.store.dtype,
                )
                .code.replace("accumulator_1", f"other_sum{i}")
                .replace("accumulator", f"sum{i}")
            )
        else:
            result = f"{dtype}(sum{i})"
        stores.append(f"        if lane == 0 and col{i} < {columns}:")
        target = "output_row" if paired else f"tensor_bindings[{bindings - 1}]"
        coordinates = f"col{i}" if paired else f"0, col{i}"
        stores.append(f"            {target}[{coordinates}] = {result}")
    stores = "\n".join(stores)
    source = f"""import cutlass
import cutlass.cute as cute

@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[{", ".join(["cute.Tensor"] * bindings)}],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    lane = thread % 32
    warp = thread // 32
{output_view}    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        tile_id = cutlass.Int32(work_iterator[tile])
{cols}
{init}
        for kk_base in cutlass.range(0, {reduction}, 32):
            kk = kk_base + lane
            x = cutlass.Float32(0.0)
            if kk < {reduction}:
                x = tensor_bindings[0][0, kk].to(cutlass.Float32)
{accum}
{stores}
"""
    return GeneratedContractionBody(
        source=source,
        threads_per_cta=128,
        output_tile_elements=16,
        reduction_chunk=32,
        tile_count=(columns + 15) // 16,
        work_iterator_encoding="one row-major 1x16 output tile ordinal per CTA",
        phases=("output_tile", "warp_reduce", "epilogue"),
    )


def _emit_blackwell_smallrows(
    program: TensorProgram, config: BodyConfig
) -> GeneratedContractionBody:
    """Reuse each weight across two to four rows while warps reduce K."""
    rows, columns = program.store.shape
    if rows not in {2, 3, 4} or config.tiles.cta_shape != (rows, 16):
        raise ValueError("Blackwell small-row body requires two to four rows")
    reduction = next(d.size for d in program.iteration_domains if d.axis == "r0")
    dtype = "cutlass.BFloat16" if program.store.dtype == "torch.bfloat16" else "cutlass.Float16"
    transposed = program.loads[1].axes == ("r0", "o1")
    cols = "\n".join(f"        col{i} = tile_id * 16 + warp * 4 + {i}" for i in range(4))
    init = "\n".join(
        f"        acc{row}_{col} = cutlass.Float32(0.0)" for row in range(rows) for col in range(4)
    )
    inputs = "\n".join(
        f"                x{row} = tensor_bindings[0][{row}, kk].to(cutlass.Float32)"
        for row in range(rows)
    )
    weights = "\n".join(
        f"                w{col} = cutlass.Float32(0.0)\n"
        f"                if col{col} < {columns}:\n"
        f"                    w{col} = tensor_bindings[1]["
        f"{'kk, col' + str(col) if transposed else 'col' + str(col) + ', kk'}].to(cutlass.Float32)"
        for col in range(4)
    )
    products = "\n".join(
        f"                acc{row}_{col} = acc{row}_{col} + x{row} * w{col}"
        for row in range(rows)
        for col in range(4)
    )
    stores = "\n".join(
        f"        sum{row}_{col} = cute.arch.warp_reduction_sum(acc{row}_{col})\n"
        f"        if lane == 0 and col{col} < {columns}:\n"
        f"            tensor_bindings[2][{row}, col{col}] = {dtype}(sum{row}_{col})"
        for row in range(rows)
        for col in range(4)
    )
    source = f"""import cutlass
import cutlass.cute as cute

@cute.kernel
def run(
    work_iterator: cute.Tensor,
    tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor],
    scratch,
    epilogue,
):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    workers, _, _ = cute.arch.grid_dim()
    lane = thread % 32
    warp = thread // 32
    for tile in cutlass.range(block, cute.size(work_iterator), workers):
        tile_id = cutlass.Int32(work_iterator[tile])
{cols}
{init}
        for kk_base in cutlass.range(0, {reduction}, 32):
            kk = kk_base + lane
            if kk < {reduction}:
{inputs}
{weights}
{products}
{stores}
"""
    return GeneratedContractionBody(
        source=source,
        threads_per_cta=128,
        output_tile_elements=rows * 16,
        reduction_chunk=32,
        tile_count=(columns + 15) // 16,
        work_iterator_encoding=f"one row-major {rows}x16 output tile ordinal per CTA",
        phases=("output_tile", "warp_reduce", "epilogue"),
    )
