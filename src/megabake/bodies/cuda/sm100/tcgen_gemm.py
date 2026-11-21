#################################################################################################
#
# Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""Experimental persistent SM100 TMA/tcgen05 GEMM for short inputs.

Adapted from NVIDIA CUTLASS 4.8.0's Blackwell tutorial GEMM. The license and
copyright notice above apply to this file.
"""

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from .descriptors import table_descriptor

io_dtype = cutlass.BFloat16
acc_dtype = cutlass.Float32
threads_per_cta = 128

ab_stages = 4
acc_stage = 1


@cute.jit
def _finish_value(accumulator, output_coord, epilogues, epilogue_state):
    value = accumulator.to(io_dtype)
    if cutlass.const_expr(len(epilogues) > 0):
        kind, tensors = epilogue_state()
        for index in cutlass.range_constexpr(len(epilogues)):
            if kind == index + 1:
                value = epilogues[index](
                    accumulator,
                    cutlass.Float32(0.0),
                    output_coord[1],
                    output_coord[0],
                    tensors[index],
                )
    return value


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32
    split_arrival: cutlass.Int32
    sA: cute.struct.Align[cute.struct.MemRange[io_dtype, 128 * 64 * ab_stages], 128]
    sB: cute.struct.Align[cute.struct.MemRange[io_dtype, 8 * 64 * ab_stages], 128]


@cute.struct
class SharedStorageWide:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32
    split_arrival: cutlass.Int32
    sA: cute.struct.Align[cute.struct.MemRange[io_dtype, 128 * 64 * ab_stages], 128]
    sB: cute.struct.Align[cute.struct.MemRange[io_dtype, 32 * 64 * ab_stages], 128]


@cute.jit
def _run_tcgen_tile(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    tile_index: cutlass.Int32,
    storage,
    desc_a,
    desc_b,
    c_address: cutlass.Int64,
    feature_tile: cutlass.Constexpr,
    token_tile: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    split_state,
    epilogues,
    epilogue_state,
    warp_specialized: cutlass.Constexpr = False,
):
    mma_tiler_mnk = (feature_tile, token_tile, 64)
    mC_mnl = cute.make_tensor(
        cute.make_ptr(io_dtype, c_address, cute.AddressSpace.gmem, assumed_align=16), mC_mnl.layout
    )
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    feature_tiles = cute.ceil_div(cute.size(mC_mnl, mode=[0]), feature_tile)
    output_tile = tile_index // splits
    split_index = tile_index % splits
    bidx, bidy = output_tile % feature_tiles, output_tile // feature_tiles
    mma_coord_mnk = (bidx, bidy, None)

    sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.fence_tma_desc_acquire(desc_a)
        cpasync.fence_tma_desc_acquire(desc_b)
        # Earlier megakernel phases write the activation through the generic
        # proxy; make those writes visible to the async TMA proxy.
        cute.arch.fence_proxy("async.global")

    num_tma_copy_bytes = cute.size_in_bytes(
        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            threads_per_cta,
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    # Wait for warp 0's TMEM allocation before retrieving its pointer.
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 1 if cutlass.const_expr(feature_tile == 64) else 4
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    if cutlass.const_expr(feature_tile == 64):
        tmem_atom = cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition.x1), cutlass.Float32)
    else:
        tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x2), cutlass.Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)
    identity = cute.make_identity_tensor(mC_mnl.shape)
    gCoord = cute.local_tile(identity, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    tCgCoord = thr_mma.partition_C(gCoord)
    gCoord_epi = cute.zipped_divide(tCgCoord, epi_tiler)
    tDgCoord = tmem_thr_copy.partition_D(gCoord_epi)

    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    num_k_tiles = cute.size(gA, mode=[2]) // splits
    k_start = split_index * num_k_tiles
    if cutlass.const_expr(warp_specialized):
        # Producer/consumer warp roles follow the CUTLASS persistent GEMM. All
        # four warps join the epilogue once the MMA warp completes this tile.
        if warp_idx == 0:
            for _ in cutlass.range(0, num_k_tiles, 1, unroll=1):
                ab_empty = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, k_start + ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    tma_desc_ptr=desc_a,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, k_start + ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    tma_desc_ptr=desc_b,
                )
        if warp_idx == 1:
            acc_empty = acc_producer.acquire_and_advance()
            for _ in cutlass.range(0, num_k_tiles, 1, unroll=1):
                ab_full = ab_consumer.wait_and_advance()
                num_k_blocks = cute.size(tCrA, mode=[2])
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, ab_full.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_full.release()
            acc_empty.commit()
    else:
        if warp_idx == 0:
            acc_empty = acc_producer.acquire_and_advance()
            for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
                ab_empty = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, k_start + ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    tma_desc_ptr=desc_a,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, k_start + ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    tma_desc_ptr=desc_b,
                )

                ab_full = ab_consumer.wait_and_advance()
                num_k_blocks = cute.size(tCrA, mode=[2])
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, ab_full.index)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[k_block_coord],
                        tCrB[k_block_coord],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                ab_full.release()

            acc_empty.commit()

    # The specialized load warp can finish before the MMA warp. Keep its
    # allocation permit until the accumulator is complete in that variant.
    if cutlass.const_expr(not warp_specialized):
        tmem.relinquish_alloc_permit()

    acc_full = acc_consumer.wait_and_advance()
    if cutlass.const_expr(warp_specialized):
        tmem.relinquish_alloc_permit()

    if cutlass.const_expr(splits == 1):
        for i in cutlass.range(cute.size(tDtC, mode=[2])):
            cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
            dst = tDgC[None, None, i]
            coord = tDgCoord[None, None, i]
            for value_idx in cutlass.range_constexpr(cute.size(tCrC)):
                value_coord = cute.idx2crd(value_idx, tCrC.shape)
                output_coord = coord[value_coord]
                if output_coord[0] < cute.size(mC_mnl, mode=[0]) and output_coord[1] < cute.size(
                    mC_mnl, mode=[1]
                ):
                    dst[value_coord] = _finish_value(
                        tCrAcc[value_coord], output_coord, epilogues, epilogue_state
                    )
    else:
        # Each split publishes FP32 partials. The final arriving CTA sums
        # them in split order, then writes the BF16 output exactly once.
        partials, counters, partial_base, counter_base = split_state
        tile_elements = feature_tile * token_tile
        own = partial_base + (output_tile * splits + split_index) * tile_elements
        for i in cutlass.range(cute.size(tDtC, mode=[2])):
            cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
            coord = tDgCoord[None, None, i]
            for value_idx in cutlass.range_constexpr(cute.size(tCrAcc)):
                value_coord = cute.idx2crd(value_idx, tCrAcc.shape)
                output_coord = coord[value_coord]
                if output_coord[0] < cute.size(mC_mnl, mode=[0]) and output_coord[1] < cute.size(
                    mC_mnl, mode=[1]
                ):
                    offset = (output_coord[0] % feature_tile) * token_tile + output_coord[
                        1
                    ] % token_tile
                    cute.arch.store(partials.iterator + own + offset, tCrAcc[value_coord], cop="cg")
        cute.arch.fence_acq_rel_gpu()
        cute.arch.sync_threads()
        if tidx == 0:
            arrived = cute.arch.atomic_add(
                counters.iterator + counter_base + output_tile,
                cutlass.Int32(1),
                sem="acq_rel",
                scope="gpu",
            )
            cute.arch.store(storage.split_arrival.ptr, arrived)
        cute.arch.sync_threads()
        last = cute.arch.load(storage.split_arrival.ptr, cutlass.Int32) == splits - 1
        if last:
            cute.arch.fence_acq_rel_gpu()
            first = partial_base + output_tile * splits * tile_elements
            for i in cutlass.range(cute.size(tDtC, mode=[2])):
                dst = tDgC[None, None, i]
                coord = tDgCoord[None, None, i]
                for value_idx in cutlass.range_constexpr(cute.size(tCrC)):
                    value_coord = cute.idx2crd(value_idx, tCrC.shape)
                    output_coord = coord[value_coord]
                    if output_coord[0] < cute.size(mC_mnl, mode=[0]) and output_coord[
                        1
                    ] < cute.size(mC_mnl, mode=[1]):
                        offset = (output_coord[0] % feature_tile) * token_tile + output_coord[
                            1
                        ] % token_tile
                        total = cutlass.Float32(0.0)
                        for split in cutlass.range_constexpr(splits):
                            total += cute.arch.load(
                                partials.iterator + first + split * tile_elements + offset,
                                cutlass.Float32,
                                cop="cg",
                            )
                        dst[value_coord] = _finish_value(
                            total, output_coord, epilogues, epilogue_state
                        )
            cute.arch.sync_threads()
            if tidx == 0:
                cute.arch.atomic_exch(
                    counters.iterator + counter_base + output_tile,
                    cutlass.Int32(0),
                    sem="relaxed",
                    scope="gpu",
                )
    acc_full.release()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def build_sm100_table_descriptors(
    prepared,
    descriptors: cute.Tensor,
    gemm: cutlass.Int32,
    a_address: cutlass.Int64,
    b_address: cutlass.Int64,
    c_address: cutlass.Int64,
    mnk: cutlass.Constexpr,
    c_m_major: cutlass.Constexpr = False,
    splits: cutlass.Constexpr = 1,
):
    """Create per-phase TMA maps for the weight and activation operands."""
    m, n, k = mnk
    tensor_bindings, _, _ = prepared
    atoms = (tensor_bindings[0], tensor_bindings[2])
    a = cute.make_tensor(
        cute.make_ptr(io_dtype, a_address, cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout((m, k), stride=(k, 1)),
    )
    b = cute.make_tensor(
        cute.make_ptr(io_dtype, b_address, cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout((n, k), stride=(k, 1)),
    )
    tensors = (a, b)
    for operand in cutlass.range_constexpr(2):
        pointer = table_descriptor(descriptors, gemm, operand, cute.AddressSpace.gmem)
        with cute.arch.elect_one():
            cpasync.copy_tensormap(atoms[operand], pointer)
        cute.arch.sync_warp()
        cpasync.update_tma_descriptor(atoms[operand], tensors[operand], pointer)
        cute.arch.sync_warp()


class Sm100TmaGemm:
    """SM100 BF16 TMA/tcgen05 body for short rows in a table launch."""

    shared_storage = SharedStorage

    def __init__(
        self,
        feature_tile: int = 128,
        splits: int = 1,
        token_tile: int = 8,
        epilogues=(),
        warp_specialized: bool = False,
    ):
        if feature_tile not in (64, 128):
            raise ValueError("SM100 tcgen feature tile must be 64 or 128")
        self.feature_tile = feature_tile
        self.splits = splits
        if token_tile not in (8, 16, 32):
            raise ValueError("SM100 tcgen token tile must be 8, 16 or 32")
        self.token_tile = token_tile
        self.shared_storage = SharedStorage if token_tile == 8 else SharedStorageWide
        self.epilogues = epilogues
        self.warp_specialized = warp_specialized

    @cute.jit
    def prepare(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, max_active_clusters):
        a2 = a[None, None, 0]
        b2 = b[None, None, 0]
        c2 = c[None, None, 0]
        mma_inst_shape_mnk = (self.feature_tile, self.token_tile, 16)
        mma_tiler_mnk = (self.feature_tile, self.token_tile, 64)
        op = tcgen05.MmaF16BF16Op(
            io_dtype,
            acc_dtype,
            mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a2.element_type, ab_stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b2.element_type, ab_stages
        )
        a_stage = cute.select(a_layout, mode=[0, 1, 2])
        b_stage = cute.select(b_layout, mode=[0, 1, 2])
        copy_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_atom, a_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            copy_op, a2, a_stage, mma_tiler_mnk, tiled_mma
        )
        b_atom, b_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            copy_op, b2, b_stage, mma_tiler_mnk, tiled_mma
        )
        return (
            (a_atom, a_tensor, b_atom, b_tensor, tiled_mma, None, a_layout, b_layout),
            (None, c2, None),
            None,
        )

    @cute.jit
    def run_table(
        self,
        prepared,
        storage,
        descriptors,
        rotated_block,
        split_state=None,
        c_address=None,
        epilogue_state=None,
    ):
        operands, epilogue, _ = prepared
        a_atom, a_tensor, b_atom, b_tensor, tiled_mma, _, a_layout, b_layout = operands
        _, c_tensor, _ = epilogue
        desc_a, desc_b, _, _ = descriptors
        _, _, workers = cute.arch.grid_dim()
        tiles = cute.ceil_div(cute.size(c_tensor, mode=[0]), self.feature_tile) * cute.ceil_div(
            cute.size(c_tensor, mode=[1]), self.token_tile
        )
        for tile_index in cutlass.range(rotated_block, tiles * self.splits, workers):
            _run_tcgen_tile(
                tiled_mma,
                a_atom,
                a_tensor,
                b_atom,
                b_tensor,
                c_tensor,
                a_layout,
                b_layout,
                tile_index,
                storage,
                desc_a,
                desc_b,
                c_address,
                self.feature_tile,
                self.token_tile,
                self.splits,
                split_state,
                self.epilogues,
                epilogue_state,
                self.warp_specialized,
            )
