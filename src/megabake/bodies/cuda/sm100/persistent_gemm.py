# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
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

"""Experimental cooperative SM100 persistent GEMM.

The warp-specialized producer/consumer structure is adapted from NVIDIA
CUTLASS 4.8.0's Blackwell dense_gemm_persistent.py.  MegaBake supplies its
own phase-table scheduler, dynamic tensor maps, output predicates and fused
epilogues.  Four warps perform the epilogue, one issues TMA, one issues MMA,
and two are idle to fit MegaBake's 256-thread cooperative block contract.
"""

import cutlass
from cutlass import cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from .tcgen_gemm import Sm100TmaGemm, _finish_value, ab_stages, acc_dtype, io_dtype


@cute.struct
class PersistentStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
    tmem_holding_buf: cutlass.Int32
    split_arrival: cutlass.Int32
    sA: cute.struct.Align[cute.struct.MemRange[io_dtype, 128 * 64 * ab_stages], 128]
    sB: cute.struct.Align[cute.struct.MemRange[io_dtype, 8 * 64 * ab_stages], 128]


@cute.struct
class PersistentStorageWide:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
    tmem_holding_buf: cutlass.Int32
    split_arrival: cutlass.Int32
    sA: cute.struct.Align[cute.struct.MemRange[io_dtype, 128 * 64 * ab_stages], 128]
    sB: cute.struct.Align[cute.struct.MemRange[io_dtype, 32 * 64 * ab_stages], 128]


@cute.jit
def _persistent_run(
    tiled_mma,
    tma_atom_a,
    mA_mkl,
    tma_atom_b,
    mB_nkl,
    mC_mnl,
    a_smem_layout,
    b_smem_layout,
    storage,
    desc_a,
    desc_b,
    c_address,
    feature_tile: cutlass.Constexpr,
    token_tile: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    acc_stages: cutlass.Constexpr,
    split_state,
    start_tile,
    workers,
    tiles,
    epilogues,
    epilogue_state,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    mma_tiler_mnk = (feature_tile, token_tile, 64)
    feature_tiles = cute.ceil_div(cute.size(mC_mnl, mode=[0]), feature_tile)
    mC_mnl = cute.make_tensor(
        cute.make_ptr(io_dtype, c_address, cute.AddressSpace.gmem, assumed_align=16), mC_mnl.layout
    )
    sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
    num_bytes = cute.size_in_bytes(
        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    # UMMA releases an A/B stage with one signal per warp.  As in the
    # persistent CUTLASS example, multicast signaling converts the warp
    # consumer group to that single arrival even for a one-CTA cluster.
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
        tx_count=num_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        enable_multicast_signaling=True,
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()
    allocation_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=160)
    epilogue_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=128)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr, barrier_for_retrieve=allocation_barrier
    )
    thr_mma = tiled_mma.get_slice(0)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    if cutlass.const_expr(acc_stages == 2):
        fake_acc = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
    else:
        fake_acc = tiled_mma.make_fragment_C(acc_shape)

    if warp == 4:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.fence_tma_desc_acquire(desc_a)
        cpasync.fence_tma_desc_acquire(desc_b)
        cute.arch.fence_proxy("async.global")

        for tile in cutlass.range(start_tile, tiles, workers):
            output_tile = tile // splits
            split_index = tile % splits
            coord = (output_tile % feature_tiles, output_tile // feature_tiles, None)
            gA = cute.local_tile(mA_mkl, mma_tiler_mnk, coord, proj=(1, None, 1))
            gB = cute.local_tile(mB_nkl, mma_tiler_mnk, coord, proj=(None, 1, 1))
            tCgA = thr_mma.partition_A(gA)
            tCgB = thr_mma.partition_B(gB)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )
            k_tiles = cute.size(gA, mode=[2]) // splits
            for k_tile in cutlass.range(k_tiles):
                empty = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, split_index * k_tiles + k_tile)],
                    tAsA[(None, empty.index)],
                    tma_bar_ptr=empty.barrier,
                    tma_desc_ptr=desc_a,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, split_index * k_tiles + k_tile)],
                    tBsB[(None, empty.index)],
                    tma_bar_ptr=empty.barrier,
                    tma_desc_ptr=desc_b,
                )

    if warp == 5:
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        acc_base = cute.make_tensor(tmem_ptr, fake_acc.layout)
        k_tiles = cute.ceil_div(cute.size(mA_mkl, mode=[1]), 64) // splits
        for _tile in cutlass.range(start_tile, tiles, workers):
            empty_acc = acc_producer.acquire_and_advance()
            if cutlass.const_expr(acc_stages == 2):
                acc = acc_base[(None, None, None, empty_acc.index)]
            else:
                acc = acc_base
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for _k in cutlass.range(k_tiles):
                full = ab_consumer.wait_and_advance()
                for k_block in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    k_coord = (None, None, k_block, full.index)
                    cute.gemm(tiled_mma, acc, tCrA[k_coord], tCrB[k_coord], acc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                full.release()
            empty_acc.commit()

    if warp < 4:
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        acc_base = cute.make_tensor(tmem_ptr, fake_acc.layout)
        if cutlass.const_expr(acc_stages == 2):
            acc = acc_base[(None, None, None, 0)]
        else:
            acc = acc_base
        subtile_cnt = 1 if cutlass.const_expr(feature_tile == 64) else 4
        epi_tiler = ((cute.size(acc, mode=[0, 0]), cute.size(acc, mode=[0, 1]) // subtile_cnt),)
        acc_epi = cute.zipped_divide(acc, epi_tiler)
        if cutlass.const_expr(feature_tile == 64):
            atom = cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition.x1), cutlass.Float32)
        else:
            atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x2), cutlass.Float32)
        copy = tcgen05.make_tmem_copy(atom, acc_epi[None, 0])
        thread_copy = copy.get_slice(tidx)
        identity = cute.make_identity_tensor(mC_mnl.shape)

        for tile in cutlass.range(start_tile, tiles, workers):
            full_acc = acc_consumer.wait_and_advance()
            if cutlass.const_expr(acc_stages == 2):
                acc = acc_base[(None, None, None, full_acc.index)]
                acc_epi = cute.zipped_divide(acc, epi_tiler)
            src = thread_copy.partition_S(acc_epi)
            output_tile = tile // splits
            split_index = tile % splits
            coord = (output_tile % feature_tiles, output_tile // feature_tiles, None)
            gC = cute.local_tile(mC_mnl, mma_tiler_mnk, coord, proj=(1, 1, None))
            gCoord = cute.local_tile(identity, mma_tiler_mnk, coord, proj=(1, 1, None))
            tCgC = thr_mma.partition_C(gC)
            tCgCoord = thr_mma.partition_C(gCoord)
            dst = thread_copy.partition_D(cute.zipped_divide(tCgC, epi_tiler))
            coords = thread_copy.partition_D(cute.zipped_divide(tCgCoord, epi_tiler))
            values = cute.make_rmem_tensor(dst[None, None, 0].shape, acc_dtype)
            if cutlass.const_expr(splits == 1):
                for i in cutlass.range(cute.size(src, mode=[2])):
                    cute.copy(copy, src[None, None, i], values)
                    out = dst[None, None, i]
                    out_coords = coords[None, None, i]
                    for index in cutlass.range_constexpr(cute.size(values)):
                        val_coord = cute.idx2crd(index, values.shape)
                        output_coord = out_coords[val_coord]
                        if output_coord[0] < cute.size(mC_mnl, mode=[0]) and output_coord[
                            1
                        ] < cute.size(mC_mnl, mode=[1]):
                            out[val_coord] = _finish_value(
                                values[val_coord], output_coord, epilogues, epilogue_state
                            )
            else:
                partials, counters, partial_base, counter_base = split_state
                tile_elements = feature_tile * token_tile
                own = partial_base + (output_tile * splits + split_index) * tile_elements
                for i in cutlass.range(cute.size(src, mode=[2])):
                    cute.copy(copy, src[None, None, i], values)
                    out_coords = coords[None, None, i]
                    for index in cutlass.range_constexpr(cute.size(values)):
                        val_coord = cute.idx2crd(index, values.shape)
                        output_coord = out_coords[val_coord]
                        if output_coord[0] < cute.size(mC_mnl, mode=[0]) and output_coord[
                            1
                        ] < cute.size(mC_mnl, mode=[1]):
                            offset = (output_coord[0] % feature_tile) * token_tile + output_coord[
                                1
                            ] % token_tile
                            cute.arch.store(
                                partials.iterator + own + offset, values[val_coord], cop="cg"
                            )
                cute.arch.fence_acq_rel_gpu()
                epilogue_barrier.arrive_and_wait()
                if tidx == 0:
                    arrived = cute.arch.atomic_add(
                        counters.iterator + counter_base + output_tile,
                        cutlass.Int32(1),
                        sem="acq_rel",
                        scope="gpu",
                    )
                    cute.arch.store(storage.split_arrival.ptr, arrived)
                epilogue_barrier.arrive_and_wait()
                last = cute.arch.load(storage.split_arrival.ptr, cutlass.Int32) == splits - 1
                if last:
                    cute.arch.fence_acq_rel_gpu()
                    first = partial_base + output_tile * splits * tile_elements
                    for i in cutlass.range(cute.size(src, mode=[2])):
                        out = dst[None, None, i]
                        out_coords = coords[None, None, i]
                        for index in cutlass.range_constexpr(cute.size(values)):
                            val_coord = cute.idx2crd(index, values.shape)
                            output_coord = out_coords[val_coord]
                            if output_coord[0] < cute.size(mC_mnl, mode=[0]) and output_coord[
                                1
                            ] < cute.size(mC_mnl, mode=[1]):
                                offset = (
                                    output_coord[0] % feature_tile
                                ) * token_tile + output_coord[1] % token_tile
                                total = cutlass.Float32(0.0)
                                for split in cutlass.range_constexpr(splits):
                                    total += cute.arch.load(
                                        partials.iterator + first + split * tile_elements + offset,
                                        cutlass.Float32,
                                        cop="cg",
                                    )
                                out[val_coord] = _finish_value(
                                    total, output_coord, epilogues, epilogue_state
                                )
                    epilogue_barrier.arrive_and_wait()
                    if tidx == 0:
                        cute.arch.atomic_exch(
                            counters.iterator + counter_base + output_tile,
                            cutlass.Int32(0),
                            sem="relaxed",
                            scope="gpu",
                        )
            full_acc.release()
        epilogue_barrier.arrive_and_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)
    cute.arch.sync_threads()


class Sm100PersistentGemm(Sm100TmaGemm):
    """Experimental warp-specialized SM100 table body."""

    def __init__(self, feature_tile=128, splits=1, token_tile=8, epilogues=(), acc_stages=1):
        super().__init__(feature_tile, splits, token_tile, epilogues)
        if acc_stages not in (1, 2):
            raise ValueError("persistent SM100 GEMM supports one or two accumulator stages")
        self.acc_stages = acc_stages
        if acc_stages == 2:
            self.shared_storage = PersistentStorage if token_tile == 8 else PersistentStorageWide

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
        if rotated_block < tiles * self.splits:
            _persistent_run(
                tiled_mma,
                a_atom,
                a_tensor,
                b_atom,
                b_tensor,
                c_tensor,
                a_layout,
                b_layout,
                storage,
                desc_a,
                desc_b,
                c_address,
                self.feature_tile,
                self.token_tile,
                self.splits,
                self.acc_stages,
                split_state,
                rotated_block,
                workers,
                tiles * self.splits,
                self.epilogues,
                epilogue_state,
            )
