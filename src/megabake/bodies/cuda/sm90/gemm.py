"""MegaBake's G0 GEMM body: provider, table-launch descriptors, fused epilogues.

The kernel itself is ``cutlass_gemm.HopperWgmmaGemmPersistentKernel``, MegaBake's
fork of NVIDIA CUTLASS's persistent Hopper GEMM (the pinned upstream file is in
``cutlass_upstream/``; ``tools/vendor_cutlass.py`` shows the changes); this
module only drives it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
import os
import re
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.tensor_utils import TensorMapManager, TensorMapUpdateMode

from .cutlass_gemm import HopperWgmmaGemmPersistentKernel

from ....graph import ComputeGraph, Region
from ....ir import TensorProgram
from ....ops import ViewPlusGemm
from ...interface import (
    BodyConfig,
    CandidateSet,
    CompletionContract,
    InvalidBodyConfig,
    LayoutContract,
    ParticipantContract,
    PipelineContract,
    ProviderRejection,
    ResourceEvidence,
    ScratchContract,
    SelectionPolicy,
    SemanticContract,
    TargetFacts,
    TensorLayout,
    TileContract,
    access_effects,
    validate_body_config,
)


# Pipeline barriers (1 KiB), eight 64x64x64 BF16 A/B stages, and four 64x32
# BF16 epilogue stages; HopperWgmmaGemmPersistentKernel derives eight stages
# when this is its shared-memory capacity.
TABLE_GEMM_SHARED_BYTES = 1024 + 8 * 2 * 64 * 64 * 2 + 4 * 64 * 32 * 2

_TABLE_TENSORMAPS = TensorMapManager(TensorMapUpdateMode.GMEM, 128)

# Table GEMM descriptors: A, B, C, and the second A of a paired GEMM.
TABLE_DESCRIPTORS_PER_GEMM = 4


@cute.jit
def table_descriptor(
    descriptors: cute.Tensor,
    gemm: cutlass.Int32,
    operand: cutlass.Constexpr,
    space: cutlass.Constexpr = cute.AddressSpace.generic,
):
    """Pointer to the A (0), B (1), C (2), or second A (3) TMA descriptor of
    one table GEMM.

    ``descriptors`` is an Int64 buffer holding TABLE_DESCRIPTORS_PER_GEMM
    128-byte descriptors per GEMM phase. TMA copies take generic pointers;
    descriptor writes take global ones."""
    return _TABLE_TENSORMAPS.get_tensormap_ptr(
        descriptors.iterator + (gemm * TABLE_DESCRIPTORS_PER_GEMM + operand) * 16, space
    )


@cute.jit
def table_descriptors(descriptors: cute.Tensor, gemm: cutlass.Int32):
    return (
        table_descriptor(descriptors, gemm, 0),
        table_descriptor(descriptors, gemm, 1),
        table_descriptor(descriptors, gemm, 2),
        table_descriptor(descriptors, gemm, 3),
    )


@cute.jit
def _bf16_tensor(address: cutlass.Int64, shape: cutlass.Constexpr, stride: cutlass.Constexpr):
    return cute.make_tensor(
        cute.make_ptr(cutlass.BFloat16, address, cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout(shape, stride=stride),
    )


@cute.jit
def build_table_descriptors(
    prepared,
    descriptors: cute.Tensor,
    gemm: cutlass.Int32,
    a_address: cutlass.Int64,
    b_address: cutlass.Int64,
    c_address: cutlass.Int64,
    mnk: cutlass.Constexpr,
    c_m_major: cutlass.Constexpr = False,
    splits: cutlass.Constexpr = 1,
    a2_address=None,
):
    """Write one GEMM phase's descriptors from its variant's templates.

    Called by one full warp: the descriptor update is warp-collective, so a
    single thread deadlocks. The caller publishes the writes with
    ``fence_tma_desc_release`` before the grid barrier, and the GEMM body
    acquires them before its first TMA copy. Split-K variants see A and B as
    ``(rows, k / splits, splits)``: the L mode selects a split's K range.
    Paired variants also take the second A, which shares A's template."""
    m, n, k = mnk
    tensor_bindings, epilogue, _ = prepared
    atoms = (tensor_bindings[0], tensor_bindings[2], epilogue[0], tensor_bindings[0])
    chunk = k // splits
    tensors = (
        _bf16_tensor(a_address, (m, chunk, splits), (k, 1, chunk if splits > 1 else m * k)),
        _bf16_tensor(b_address, (n, chunk, splits), (k, 1, chunk if splits > 1 else n * k)),
        # swapAB writes C^T, so its C is M-major.
        _bf16_tensor(c_address, (m, n, 1), (1, m, m * n) if c_m_major else (n, 1, m * n)),
    )
    operands = 3
    if cutlass.const_expr(a2_address is not None):
        tensors = (
            *tensors,
            _bf16_tensor(a2_address, (m, chunk, splits), (k, 1, chunk if splits > 1 else m * k)),
        )
        operands = 4
    for operand in cutlass.range_constexpr(operands):
        pointer = table_descriptor(descriptors, gemm, operand, cute.AddressSpace.gmem)
        with cute.arch.elect_one():
            cute.nvgpu.cpasync.copy_tensormap(atoms[operand], pointer)
        cute.arch.sync_warp()
        cute.nvgpu.cpasync.update_tma_descriptor(atoms[operand], tensors[operand], pointer)
        cute.arch.sync_warp()


def union_shared_storage(kernels):
    """One shared-storage struct that every prepared variant fits: each
    variant reads its own layouts from the same fields, so every field takes
    the largest size any variant needs."""
    stages, a_elements, b_elements, c_elements = (
        max(sizes) for sizes in zip(*(kernel.storage_sizes for kernel in kernels))
    )
    first = kernels[0]
    align = first.buffer_align_bytes

    @cute.struct
    class SharedStorage:
        mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, stages * 2]
        split_arrival: cute.struct.MemRange[cutlass.Int32, 4]
        sA: cute.struct.Align[cute.struct.MemRange[first.a_dtype, a_elements], align]
        sB: cute.struct.Align[cute.struct.MemRange[first.b_dtype, b_elements], align]
        sC: cute.struct.Align[cute.struct.MemRange[first.c_dtype, c_elements], align]

    return SharedStorage


def table_gemm_shared_bytes(tiles: Sequence[tuple[int, int]], capacity: int) -> int:
    """Shared bytes of ``union_shared_storage`` for BF16 variants with these
    CUTLASS tiles, each staged within ``capacity`` (see ``_compute_stages``)."""
    stages = a_bytes = b_bytes = c_bytes = 0
    for tile_m, tile_n in tiles:
        ab_bytes = (tile_m + tile_n) * 64 * 2
        epilogue_bytes = 4 * min(64, tile_m) * min(32, tile_n) * 2
        stage_count = (capacity - 1024 - epilogue_bytes) // ab_bytes
        stages = max(stages, stage_count)
        a_bytes = max(a_bytes, stage_count * tile_m * 64 * 2)
        b_bytes = max(b_bytes, stage_count * tile_n * 64 * 2)
        c_bytes = max(c_bytes, epilogue_bytes)
    barrier_bytes = stages * 2 * 8 + 4 * 4
    return -(-barrier_bytes // 1024) * 1024 + a_bytes + b_bytes + c_bytes


def plain_contraction(program) -> bool:
    """Whether ``program``'s result is its accumulator rounded once."""
    from ....ir import Accumulator, Cast

    return (
        len(program.loads) == 2
        and isinstance(program.post, Cast)
        and isinstance(program.post.operand, Accumulator)
    )


def paired_contraction(program) -> bool:
    """Whether ``program`` multiplies one left operand by two right operands
    (loads 1 and 2), its post reading accumulator 1 for the second."""
    from ....ir import Accumulator

    def reads_second(expression) -> bool:
        if isinstance(expression, Accumulator):
            return expression.index == 1
        return any(
            reads_second(getattr(expression, name))
            for name in ("operand", "lhs", "rhs", "condition", "true_value", "false_value")
            if getattr(expression, name, None) is not None
        )

    return program.post is not None and reads_second(program.post)


def epilogue_loads(program) -> tuple:
    """The loads a fused epilogue reads beyond the GEMM operands."""
    return program.loads[3:] if paired_contraction(program) else program.loads[2:]


def epilogue_source(program, symbol: str) -> str:
    """A CuTe function ``symbol(accumulator, accumulator_1, o0, o1,
    tensor_bindings)`` evaluating ``program.post`` at output (o0, o1), with
    ``tensor_bindings`` holding the tensors of ``epilogue_loads(program)``."""
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


def _epilogue_rejection(program, policy: SelectionPolicy) -> str | None:
    """Why ``program``'s fused epilogue does not fit the GEMM, or None."""
    from ....ir import AxisAffine
    from ..generic import UnsupportedMapBody

    if not (policy.table_launch and launch_options_of(policy).swap_ab):
        return "fused epilogues run in swapAB table GEMMs, which store from registers"
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


class G0RangeBodyProvider:
    provider_name = "g0_range"

    def emit(
        self,
        program: TensorProgram,
        config: BodyConfig,
        *,
        defer_length_publish: bool = False,
        workers: int | None = None,
    ):
        # The persistent TMA/WGMMA kernel is instantiated once per tile shape
        # in the kernel module, with its tensor maps and shared storage; codegen
        # emits that, not a nested per-phase body.
        raise NotImplementedError("g0_range GEMMs are emitted with the kernel module")

    def widen(self, program: TensorProgram, config: BodyConfig, threads: int) -> BodyConfig:
        # The GEMM runs its own warp-specialized CTA, whatever the launch's.
        return config

    def get_candidates(
        self, region: Region, target: TargetFacts, policy: SelectionPolicy
    ) -> CandidateSet:
        def reject(reason: str) -> CandidateSet:
            return CandidateSet((), (ProviderRejection(self.provider_name, reason),))

        if not policy.allow_tuned:
            return reject("tuned providers disabled")
        options = launch_options_of(policy)
        if not options.tma_gemm:
            return reject("TMA GEMM bodies are disabled by policy")
        if not region.tensor_programs or len(region.tensor_programs) != 1:
            return reject("region has no single tensor program")

        program = region.tensor_programs[0]
        if program.family != "contraction":
            return reject("provider supports contraction only")
        paired = paired_contraction(program)
        fused = paired or not (
            plain_contraction(program)
            or len(program.loads) == 3 and _bias_epilogue_rejection(program) is None
        )
        if fused:
            reason = _epilogue_rejection(program, policy)
            if reason is not None:
                return reject(reason)
        elif len(program.loads) == 3 and not policy.table_launch:
            return reject("bias epilogues need a table launch")
        if len(program.store.shape) != 2:
            return reject("provider supports two-dimensional GEMM only")
        operands = program.loads[: 3 if paired else 2 if fused else len(program.loads)]
        if (
            any(load.dtype != "torch.bfloat16" for load in operands)
            or program.store.dtype != "torch.bfloat16"
        ):
            return reject("provider supports BF16 only")

        rows, columns = program.store.shape
        reduction_size = next(
            domain.size
            for domain in program.iteration_domains
            if domain.axis in program.reduction_axes
        )
        if rows < 1 or columns < 64 or reduction_size < 64:
            return reject("M must be positive and N/K must be nonempty multiples of 64")
        if columns % 64 or reduction_size % 64:
            return reject("N and K must be multiples of 64")

        left, right = program.loads[:2]
        left_strides = dict(zip(left.axes, left.strides, strict=True))
        right_strides = dict(zip(right.axes, right.strides, strict=True))
        if (
            left.axes != ("o0", "r0")
            or left_strides != {"o0": reduction_size, "r0": 1}
            or "r0" not in right_strides
            or "o1" not in right_strides
            or right_strides["r0"] != 1
            or right_strides["o1"] != reduction_size
            or program.store.strides != (columns, 1)
        ):
            return reject("provider requires K-major BF16 operands and row-major output")
        if target.architecture != "sm90a":
            return reject("provider targets sm90a only")
        if not target.supports_tma or not target.supports_wgmma:
            return reject("target lacks TMA or WGMMA")

        if policy.table_launch:
            # Every table-launched GEMM shares one staged shared-memory budget,
            # sized to leave room for the other bodies' scratch.
            if options.swap_ab:
                token_tile = options.swap_ab_rows
                if rows > token_tile:
                    return reject(f"swapAB GEMMs of this launch take at most {token_tile} rows")
                tiles = -(-columns // 64)
                splits = (
                    split_k_count(tiles, reduction_size // 64, target.multiprocessor_count)
                    if options.split_k and not fused
                    else 1
                )
                if options.splits and not fused:
                    splits = dict(options.splits).get((tiles, reduction_size // 64), 1)
                forced = _forced_splits(tiles, reduction_size // 64)
                if forced is not None:
                    splits = forced
                return CandidateSet(
                    (
                        _g0_config(
                            region, program, (token_tile, 64), 8,
                            TABLE_GEMM_SHARED_BYTES, swap_ab=True, splits=splits,
                            epilogue=fused, paired=paired,
                        ),
                    ),
                    (),
                )
            return CandidateSet(
                (_g0_config(region, program, (64, 64), 8, TABLE_GEMM_SHARED_BYTES),),
                (),
            )
        return CandidateSet(
            (
                _g0_config(region, program, (64, 64), 13, 230_400),
                _g0_config(region, program, (64, 128), 8, 214_016),
            ),
            (),
        )


# Mirage's linear_swapAB_hopper: with 1-16 token rows, a 64-row activation
# tile is mostly padding, so each pipeline stage carries little weight data.
# Swapping the operands puts 64 weight rows on WGMMA's M side and the tokens
# on N=16. On 60 SMs this doubles decode GEMM bandwidth (0.84-1.04 to
# 1.78-2.07 TB/s), matching cuBLAS.
SWAP_AB_MAX_ROWS = 64
# WGMMA N tiles for the token rows of swapAB launches; a launch takes the
# smallest that holds its largest GEMM.
SWAP_AB_ROW_TILES = (16, 32, 64)


# Split-K (Mirage splits small-N linears the same way): a decode projection
# with fewer 64-row weight tiles than SMs leaves most SMs idle, and each busy
# SM cannot pull its share of HBM bandwidth alone. Splitting K multiplies the
# work tiles; the last split of each tile sums the FP32 partials in order.
SPLIT_K_MAX = 16
# Each split keeps at least this many 64-wide K tiles in its pipeline.
SPLIT_K_MIN_K_TILES = 2


def split_k_count(tiles: int, k_tiles: int, workers: int) -> int:
    """The largest split of ``k_tiles`` that keeps ``tiles * splits`` within
    ``workers`` and every split at least ``SPLIT_K_MIN_K_TILES`` deep."""
    best = 1
    for splits in range(2, SPLIT_K_MAX + 1):
        if (
            k_tiles % splits == 0
            and k_tiles // splits >= SPLIT_K_MIN_K_TILES
            and tiles * splits <= workers
        ):
            best = splits
    return best


def _forced_splits(tiles: int, k_tiles: int) -> int | None:
    """Experiment knob: ``MEGABAKE_SPLIT_K_FORCE="tiles,k_tiles:splits;..."``
    forces the split of GEMMs with that many output tiles and K tiles."""
    spec = os.environ.get("MEGABAKE_SPLIT_K_FORCE")
    if not spec:
        return None
    for entry in spec.split(";"):
        shape, splits = entry.split(":")
        if tuple(int(v) for v in shape.split(",")) == (tiles, k_tiles):
            splits = int(splits)
            if k_tiles % splits or k_tiles // splits < SPLIT_K_MIN_K_TILES:
                raise ValueError(f"cannot split {k_tiles} K tiles {splits} ways")
            return splits
    return None


def has_bias(body: BodyConfig) -> bool:
    """Whether a G0 GEMM body adds a bias in its epilogue (addmm)."""
    return body.name.endswith("_bias")


def has_fused_epilogue(body: BodyConfig) -> bool:
    """Whether a G0 GEMM body evaluates its program's post expression as a
    generated epilogue (see ``epilogue_source``)."""
    return body.name.endswith("_epilogue")


def is_paired(body: BodyConfig) -> bool:
    """Whether a G0 GEMM body computes a paired contraction."""
    return "_paired" in body.name


def _bias_epilogue_rejection(program) -> str | None:
    """Why ``program``'s bias does not fit the GEMM epilogue, or None.

    The epilogue adds a contiguous BF16 bias over the output columns to the
    FP32 accumulators and rounds once, exactly ``post`` with alpha = beta = 1.
    """
    from ....ir import Accumulator, Binary, Cast, Load

    bias = program.loads[2]
    columns = program.store.shape[1]
    axes = dict(zip(bias.axes, bias.strides, strict=True))
    if bias.dtype != "torch.bfloat16":
        return "bias must be BF16"
    if axes.get("o1") != 1 or any(
        not isinstance(axis, int) for axis in bias.axes if axis != "o1"
    ):
        return "bias must be contiguous over the output columns"
    if bias.source_shape is not None and prod(bias.source_shape) != columns:
        return "bias must hold one value per output column"
    product = program.combine.rhs if isinstance(program.combine, Binary) else None
    if not (
        isinstance(product, Binary)
        and product.operator == "mul"
        and all(isinstance(operand, Cast) for operand in (product.lhs, product.rhs))
    ):
        return "only alpha = 1 is supported"
    post = program.post
    if not (
        isinstance(post, Cast)
        and isinstance(post.operand, Binary)
        and post.operand.operator == "add"
        and isinstance(post.operand.lhs, Accumulator)
        and isinstance(post.operand.rhs, Cast)
        and isinstance(post.operand.rhs.operand, Load)
        and post.operand.rhs.operand.load_id == bias.id
    ):
        return "only beta = 1 is supported"
    return None


def split_k(body: BodyConfig) -> int:
    """K splits of a G0 GEMM body (1 without split-K)."""
    match = re.search(r"_k(\d+)(?=_|$)", body.name)
    return int(match.group(1)) if match else 1


def is_swap_ab(body: BodyConfig) -> bool:
    return body.provider == G0RangeBodyProvider.provider_name and body.name.startswith(
        "g0_range_swapab_"
    )


def kernel_tile(body: BodyConfig) -> tuple[int, int]:
    """The CUTLASS CTA tile. ``body.tiles.cta_shape`` is in output (rows,
    columns) coordinates, which swapAB transposes."""
    rows, columns = body.tiles.cta_shape[:2]
    return (columns, rows) if is_swap_ab(body) else (rows, columns)


def swap_ab_rows(shapes: Sequence[tuple[int, int]]) -> int:
    """The swapAB token tile for a launch's ``(rows, columns)`` GEMMs."""
    largest = max((rows for rows, _ in shapes), default=0)
    return next(tile for tile in SWAP_AB_ROW_TILES if largest <= tile)


def wants_swap_ab(shapes: Sequence[tuple[int, int]]) -> bool:
    """Whether every ``(rows, columns)`` GEMM of a launch is small-M."""
    return bool(shapes) and all(rows <= SWAP_AB_MAX_ROWS for rows, _ in shapes)


@dataclass(frozen=True)
class HopperLaunchOptions:
    """Launch-wide GEMM choices (``SelectionPolicy.target_options`` on sm90a)."""

    tma_gemm: bool = True
    # Table-launched GEMMs with at most ``swap_ab_rows`` rows put the weight
    # on WGMMA's M side (Mirage's swapAB). One orientation serves the whole
    # launch because its GEMMs share one shared-memory layout.
    swap_ab: bool = False
    # The token tile (WGMMA N) of swapAB launches: 16, 32, or 64 rows.
    swap_ab_rows: int = 16
    # Split the K range of swapAB GEMMs that have fewer weight tiles than
    # SMs (see split_k_count).
    split_k: bool = False
    # The planner's choice with a device tree (``plan_splits``): K splits of
    # the GEMMs with (output tiles, K tiles), 1 for shapes not listed.
    splits: tuple[tuple[tuple[int, int], int], ...] = ()


def launch_options(
    graph: ComputeGraph, table_launch: bool, device: Any = None
) -> HopperLaunchOptions:
    """SwapAB when every GEMM of a table launch is small-M. With a device
    tree the planner chooses each GEMM shape's K split from the cost model
    (``plan_splits``); without one, split-K is on request
    (``MEGABAKE_SPLIT_K=1``)."""
    programs = tuple(
        operation.tensor_program
        for operation in graph.ops.values()
        if isinstance(operation.tensor_program, TensorProgram)
        and operation.tensor_program.family == "contraction"
        and len(operation.tensor_program.store.shape) == 2
    )
    shapes = tuple(tuple(program.store.shape) for program in programs)
    swap_ab = table_launch and wants_swap_ab(shapes)
    rows = swap_ab_rows(shapes) if swap_ab else 16
    forced = os.environ.get("MEGABAKE_SPLIT_K") == "1"
    return HopperLaunchOptions(
        swap_ab=swap_ab,
        swap_ab_rows=rows,
        split_k=swap_ab and forced,
        # Only on a calibrated tree: a spec tree's rates are assumptions.
        splits=plan_splits(programs, rows, device)
        if swap_ab and device is not None and device.measured and not forced
        else (),
    )


def plan_splits(
    programs: Sequence[TensorProgram], token_rows: int, tree: Any
) -> tuple[tuple[tuple[int, int], int], ...]:
    """The K split of each swapAB GEMM shape class of a launch, from the cost
    model: ``((output tiles, K tiles), splits)`` for the shapes that split.

    Splitting is a launch-wide trade. A split GEMM puts more leaves on the
    root edge, which pays where the edge is not yet saturated; but one split
    GEMM takes the producer away from every GEMM of the kernel (its pages
    hold whole K ranges), so the unsplit kernel keeps its staged starts. Both
    kernels are joined class by class and the split one must win by the
    model's margin."""
    from ....costmodel import Interval, gemm_work, interval_time
    from ....einsum import einsum_view
    from ....runtime import producer_warp_enabled
    from ....search import Candidate, ShapeClass, best_of, join

    workers = tree.workers
    counts: dict[tuple[int, int, int], int] = {}
    for program in programs:
        view = einsum_view(program)
        reduction = next(view.ranks[axis] for axis in program.reduction_axes)
        columns = program.store.shape[1]
        if columns % 64 or reduction % 64:
            continue
        key = (-(-columns // 64), reduction // 64, program.store.shape[0])
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ()
    staged = token_rows == 16 and producer_warp_enabled()
    if staged:
        # Keep the producer. Its gain is run-ahead: the next interval's weights
        # load during norms, attention and barriers. The per-interval model
        # below has no term for that (fitted on a full H200, staged_start_s
        # 1.44 us against start_s 1.42 us), so it prefers split-K. Measured on
        # Llama-3.2-1B decode, 132 SMs (2026-11-20): split q/k/v with no
        # producer 1,327-1,330 us; unsplit with the producer 1,176 us, and
        # 1,159 us with counted GEMMs, which need the producer.
        return ()

    def seconds(tiles: int, k_tiles: int, token: int, splits: int, staged: bool) -> float:
        tile_bytes = 64 * k_tiles * 64 * 2 + (token * k_tiles * 64 + token * 64) * 2
        work = gemm_work(
            "gemm", tree, tiles=tiles, tile_bytes=tile_bytes,
            tile_flops=2.0 * token * 64 * k_tiles * 64, splits=splits, staged=staged,
        )
        return interval_time(Interval((work,), workers, "none"), tree).critical_s

    def classes(split: bool) -> list[ShapeClass]:
        result = []
        for (tiles, k_tiles, token), count in counts.items():
            options = [1]
            if split:
                options += [
                    splits
                    for splits in range(2, SPLIT_K_MAX + 1)
                    if k_tiles % splits == 0
                    and k_tiles // splits >= SPLIT_K_MIN_K_TILES
                    and tiles * splits <= workers
                ]
            result.append(
                ShapeClass(
                    (tiles, k_tiles, token),
                    count,
                    tuple(
                        Candidate(
                            splits,
                            seconds(tiles, k_tiles, token, splits, staged and not split),
                            # Split partials: FP32 accumulators in phase scratch.
                            {"phase_scratch_bytes": 0.0 if splits == 1 else token * 64 * 4.0 * splits},
                        )
                        for splits in options
                    ),
                )
            )
        return result

    alternatives = {"unsplit": join(classes(False)), "split": join(classes(True))}
    if best_of(alternatives) != "split":
        return ()
    return tuple(
        sorted(
            ((tiles, k_tiles), splits)
            for (tiles, k_tiles, _), splits in alternatives["split"].choices.items()
            if splits > 1
        )
    )


def launch_options_of(policy: SelectionPolicy) -> HopperLaunchOptions:
    options = policy.target_options
    return options if isinstance(options, HopperLaunchOptions) else HopperLaunchOptions()


def gemm_candidates(
    region: Region,
    target: TargetFacts,
    policy: SelectionPolicy | None = None,
) -> CandidateSet:
    result = G0RangeBodyProvider().get_candidates(
        region, target, policy or SelectionPolicy()
    )
    configurations = []
    rejections = list(result.rejections)
    for config in result.configurations:
        try:
            validate_body_config(config, target, policy or SelectionPolicy())
        except InvalidBodyConfig as error:
            rejections.append(ProviderRejection(config.provider, str(error)))
        else:
            configurations.append(config)
    return CandidateSet(tuple(configurations), tuple(rejections))


def g0_range_body(config: BodyConfig) -> HopperWgmmaGemmPersistentKernel:
    return HopperWgmmaGemmPersistentKernel(
        cutlass.Float32, config.tiles.cta_shape, (1, 1), 1, True
    )


def cute_gemm_tensors(a, b, c, splits: int = 1):
    """CuTe ``(A, B, C)`` with an L mode. Split-K views A and B as
    ``(rows, k / splits, splits)``, so L selects a split's K range."""
    from cutlass.cute.runtime import from_dlpack

    # Activations can arrive as [batch, sequence, hidden]; swapAB binds them
    # as B rather than A.
    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if b.ndim > 2:
        b = b.reshape(-1, b.shape[-1])
    if any(tensor.data_ptr() % 16 for tensor in (a, b, c)):
        raise ValueError("GEMM tensor base addresses must be 16-byte aligned")

    def view(tensor):
        leading_dim = next(
            index for index, stride in enumerate(tensor.stride()) if stride == 1
        )
        return from_dlpack(tensor.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
            leading_dim=leading_dim
        )

    def split_view(tensor):
        rows, reduction = tensor.shape
        chunk = reduction // splits
        split = tensor.as_strided((rows, chunk, splits), (tensor.stride(0), 1, chunk))
        return from_dlpack(split, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    if splits > 1:
        return split_view(a), split_view(b), view(c)
    return view(a), view(b), view(c)


class CompiledG0RangeGemm:
    def __init__(self, compiled):
        self.compiled = compiled

    def __call__(self, a, b, c, stream):
        return self.compiled(*cute_gemm_tensors(a, b, c), stream)


def g0_range_resources(compiled, threads, shared_bytes):
    compiled = getattr(compiled, "compiled", compiled)
    resources = []
    for symbol in compiled.kernel_info:
        status, kernel = cuda.cuLibraryGetKernel(
            cuda.CUlibrary(int(compiled.library)), symbol.encode()
        )
        if int(status) != 0:
            raise RuntimeError(f"failed to open compiled kernel {symbol}: {status}")
        values = {"name": symbol}
        for label, attribute in (
            ("registers_per_thread", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS),
            ("static_shared_bytes", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
            ("local_bytes_per_thread", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
        ):
            status, value = cuda.cuKernelGetAttribute(attribute, kernel, 0)
            if int(status) != 0:
                raise RuntimeError(f"failed to query {label}: {status}")
            values[label] = value
        status, function = cuda.cuKernelGetFunction(kernel)
        if int(status) != 0:
            raise RuntimeError(f"failed to get kernel function: {status}")
        status, = cuda.cuFuncSetAttribute(
            function,
            cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_bytes,
        )
        if int(status) != 0:
            raise RuntimeError(f"failed to admit {shared_bytes} shared bytes: {status}")
        status, active_blocks = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            function, threads, shared_bytes
        )
        if int(status) != 0:
            raise RuntimeError(f"failed to query occupancy: {status}")
        values.update(
            {"shared_storage_bytes": shared_bytes, "active_blocks_per_sm": active_blocks}
        )
        resources.append(values)
    if len(resources) != 1:
        raise RuntimeError(f"expected one GEMM kernel, found {len(resources)}")
    return resources[0]


def compile_g0_range_gemm(body, a, b, c, max_active_clusters, stream):
    return CompiledG0RangeGemm(
        cute.compile(body, *cute_gemm_tensors(a, b, c), max_active_clusters, stream)
    )


def _g0_config(
    region: Region,
    program,
    tile: tuple[int, int],
    stages: int,
    shared_bytes: int,
    *,
    swap_ab: bool = False,
    splits: int = 1,
    epilogue: bool = False,
    paired: bool = False,
) -> BodyConfig:
    return BodyConfig(
        provider=G0RangeBodyProvider.provider_name,
        name=(
            f"g0_range_{'swapab_' if swap_ab else ''}{tile[0]}x{tile[1]}"
            + (f"_k{splits}" if splits > 1 else "")
            + ("_paired" if paired else "")
            + ("_epilogue" if epilogue else "_bias" if len(program.loads) == 3 else "")
        ),
        semantic=SemanticContract(
            operation_family="view_plus_gemm"
            if isinstance(region.semantics, ViewPlusGemm)
            else "contraction",
            source_op_ids=region.source_op_ids,
            numerical_policy="source_semantics",
            expression_hooks=(),
            access_effects=access_effects(program),
        ),
        layouts=_layouts(program, alignment_bytes=16),
        tiles=TileContract(
            cta_shape=tile,
            cluster_shape=(1, 1),
            work_partition=(
                f"persistent range of output tiles, K split {splits} ways"
                if splits > 1
                else "persistent range of output tiles"
            ),
            mma_family="wgmma",
            reduction_tile=(64,),
        ),
        participants=ParticipantContract(
            participating_threads=256,
            waiting_threads=256,
            warp_roles=(
                "tma_load",
                "tma_load",
                "tma_load",
                "tma_load",
                "wgmma_epilogue",
                "wgmma_epilogue",
                "wgmma_epilogue",
                "wgmma_epilogue",
            ),
            registers_per_thread=232,
        ),
        pipeline=PipelineContract(
            stage_count=stages,
            barriers=("producer_tail", "wgmma_wait_group_0", "tma_store_producer_tail"),
            async_operations=("tma_load", "wgmma", "tma_store"),
            descriptors={
                "A": "TMA input descriptor",
                "B": "TMA input descriptor",
                "C": "TMA output descriptor",
            },
            drain_rule=(
                "run returns only after input producer tail, WGMMA wait, "
                "and TMA store producer tail"
            ),
        ),
        scratch=ScratchContract(
            owner="cta",
            size_bytes=shared_bytes,
            alignment_bytes=128,
        ),
        completion=CompletionContract(
            outputs_ready=(
                "all assigned range outputs are globally stored when run returns; "
                "another CTA may read them after the enclosing kernel grid-synchronizes"
            ),
            scratch_reusable="scratch is reusable when run returns",
            register_fragments=(
                "WGMMA accumulators feed the body epilogue without conversion"
            ),
        ),
        evidence=ResourceEvidence(
            measured=True,
            source=(
                "src/megabake/bodies/cuda/sm90/gemm.py from NVIDIA CUTLASS; "
                "experiments/g0_2026_10_09/resident_range.json"
            ),
            revision="0b55a2f691d69981583568fd9eb69687b1f0de8a",
            resources={
                "threads": 256,
                "registers_per_thread": 232,
                "shared_storage_bytes": shared_bytes,
                "active_blocks_per_sm": 1,
            },
            measurements={
                "selected_tile": tile,
                "pipeline_stages": stages,
                "max_admitted_workers": 60,
                "resident_range_gpu_us_by_m": {4: 4.057, 256: 4.339}
                if tile == (64, 64)
                else {256: 5.602, 1024: 9.458},
                "standalone_persistent_gpu_us_by_m": {4: 3.086, 256: 4.019}
                if tile == (64, 64)
                else {256: 4.019},
            },
        ),
    )


def _layouts(program, alignment_bytes: int) -> LayoutContract:
    domain_sizes = {domain.axis: domain.size for domain in program.iteration_domains}
    inputs = {
        load.value_id: TensorLayout(
            dtype=load.dtype,
            axes=tuple(load.axes),
            strides=tuple(load.strides),
            alignment_bytes=alignment_bytes,
            # Epilogue operands may index by affine maps of the output axes;
            # their shape is the source tensor's.
            shape=tuple(
                1 if isinstance(axis, int) else domain_sizes[axis] for axis in load.axes
            )
            if all(isinstance(axis, (int, str)) for axis in load.axes)
            else load.source_shape,
        )
        for load in program.loads
    }
    output_axes = tuple(
        domain.axis
        for domain in program.iteration_domains
        if domain.axis not in program.reduction_axes
    )
    outputs = {
        program.store.value_id: TensorLayout(
            dtype=program.store.dtype,
            axes=output_axes,
            strides=tuple(program.store.strides),
            alignment_bytes=alignment_bytes,
            shape=tuple(program.store.shape),
        )
    }
    return LayoutContract(inputs, outputs)
