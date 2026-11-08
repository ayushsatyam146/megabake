# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Vendored and explicitly adapted from NVIDIA CUTLASS revision
# 0b55a2f691d69981583568fd9eb69687b1f0de8a:
# examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py
# (SHA-256 65088a1fd7655ed46ad54fecd08e1461a334fb38f1b2d1da7cf252e465afc130).
#
# The unmodified upstream file is kept in cutlass_upstream/; review what
# MegaBake changed with `tools/vendor_cutlass.py` (a diff against it).

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

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

from collections.abc import Sequence
import contextlib
from typing import Optional, Tuple, Type
import math
from math import prod
import re
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.pipeline.helpers import MbarrierArray
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.tensor_utils import TensorMapManager, TensorMapUpdateMode
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

"""
A high-performance batched dense GEMM (C = A * B) example for the NVIDIA Hopper architecture
using CuTe DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with MMA between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and MMA

This GEMM works as follows:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp:
   - Perform matrix multiply-accumulate (MMA) operations using WGMMA instruction.
   - Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Hopper WGMMA instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

To run this example:

.. code-block:: bash

    python examples/hopper/dense_gemm_persistent.py                        \
      --mnkl 8192,8192,8192,1 --tile_shape_mn 128,256                      \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --c_major n

The above example command compute batched gemm with M=8192, N=8192, K=8192,
batch_count=1. The Hopper WGMMA tile shape is 128x256x64 and the cluster shape
is (1,1). The input, mma accumulator and output data type are set as fp16, fp32
and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/hopper/dense_gemm.py                               \
      --mnkl 8192,8192,8192,1 --tile_shape_mn 128,256                      \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --c_major n

Constraints are same as dense_gemm.py:
* Supported input data types: fp16, fp8 (e4m3fn, e5m2), int8, uint8
* For fp16 types, A and B must have the same data type
* For fp8 types, A and B can have different types (e4m3fn or e5m2)
* For 8-bit integer types, A and B can have different types (int8 or uint8)
* 8-bit types (e4m3fn, e5m2, int8, uint8) only support k-major layout
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* CTA tile shape K must be 64
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
"""


# MegaBake: the slot of the GEMM shared storage's split_arrival that holds
# the streamed GEMM's tag; the next slot holds its streamed K tiles.
STREAM_TAG_SLOT = 2
# Split-K: the count of published non-owner partials (see _reduce_splits).
SPLIT_PUBLISHED_SLOT = 1

# Diagnostics, off unless MEGABAKE_GEMM_STAMPS=1 (tools/gemm_stages.py): every
# variant writes globaltimer stamps per (CTA, work item) into one global Int32
# buffer; the last phase to run each slot leaves its stamps there.
GEMM_STAMP_CTAS = 160
GEMM_STAMP_ITEMS = 24
GEMM_STAMP_FIELDS = 12
GEMM_STAMP_VARIANTS = 12
_STAMP_BUFFER = None


def gemm_stamp_address() -> int:
    """The stamp buffer's address (0 without MEGABAKE_GEMM_STAMPS=1)."""
    global _STAMP_BUFFER
    import os

    if os.environ.get("MEGABAKE_GEMM_STAMPS") != "1":
        return 0
    if _STAMP_BUFFER is None:
        import torch

        _STAMP_BUFFER = torch.zeros(
            GEMM_STAMP_VARIANTS * GEMM_STAMP_CTAS * GEMM_STAMP_ITEMS * GEMM_STAMP_FIELDS,
            dtype=torch.int32,
            device="cuda",
        )
    return _STAMP_BUFFER.data_ptr()


def gemm_stamps():
    """(variant tags, CTAs, items, fields) stamps; fields: 0 producer item
    start, 1 producer last issue, 2 consumer item start, 3 first stage ready,
    4 mainloop done, 5 reduction done, 6 store done, 7 variant tag, 8 body
    entry (item 0), 9 after the pipeline-init CTA sync (item 0), 10 producer
    issued the item's first TMA load, 11 consumer leaves the body (item 0)."""
    return _STAMP_BUFFER.view(
        GEMM_STAMP_VARIANTS, GEMM_STAMP_CTAS, GEMM_STAMP_ITEMS, GEMM_STAMP_FIELDS
    )


@contextlib.contextmanager
def _without_mbarrier_init():
    """MegaBake: ``PipelineTmaAsync.create`` initializes its mbarrier arrays
    in their constructors; while this is active, it emits no init."""
    initialize = MbarrierArray.mbarrier_init
    MbarrierArray.mbarrier_init = lambda self, **_: None
    try:
        yield
    finally:
        MbarrierArray.mbarrier_init = initialize


@dsl_user_op
def store_bf16_if(address, value, predicate, *, loc=None, ip=None):
    """MegaBake: store the BF16 ``value`` at global ``address`` when
    ``predicate`` holds, as one predicated instruction rather than a branch.
    Divergent branches around the direct-store epilogue can make ptxas
    serialize every wgmma of the kernel (C7519)."""
    bits = llvm.bitcast(ir.IntegerType.get_signless(16), value.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    llvm.inline_asm(
        res=None,
        operands_=[
            cutlass.Int64(address).ir_value(loc=loc, ip=ip),
            bits,
            cutlass.Int32(predicate).ir_value(loc=loc, ip=ip),
        ],
        asm_string="{ .reg .pred p; setp.ne.b32 p, $2, 0; @p st.global.b16 [$0], $1; }",
        constraints="l,h,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# Helpers to parse args
class HopperWgmmaGemmPersistentKernel:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: Supported A/B data types:
        - Float16
          A and B must have the same data type
        - Float8E4M3FN/Float8E5M2
          A and B can have different types (Float8E4M3FN/Float8E5M2)
          only support k-major layout
        - Int8/Uint8
          A and B can have different types (Int8/Uint8)
          only support k-major layout

    :note: Supported accumulation types:
        - Float32/Float16 (for all floating point inputs)
        - Int32 (for Int8/Uint8 inputs)

    :note: Constraints:
        - CTA tile M must be 64/128
        - CTA tile N must be 64/128/256
        - CTA tile K must be 64
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = HopperWgmmaGemmPersistentKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     tile_shape_mn=(128, 256),
        ...     cluster_shape_mn=(1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        smem_capacity: int | None = None,
        reallocate_registers: bool = True,
        splits: int = 1,
        bias_features: int = 0,
        bias_along_m: bool = False,
        direct_store: bool = False,
        epilogues: tuple = (),
        paired: bool = False,
        stream_stages: int = 0,
        stamp_address: int = 0,
        stamp_tag: int = 0,
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mn: Shape of the CTA tile (M,N)
        :type tile_shape_mn: Tuple[int, int]
        :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
        :type cluster_shape_mn: Tuple[int, int]
        """

        self.acc_dtype = acc_dtype

        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.mma_inst_shape_mn = None
        # K dimension is deferred in _setup_attributes
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        # For large tile size, using two warp groups is preferred because using only one warp
        # group may result in register spill
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = 1
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * 32
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.epi_store_warp_id = (
            self.num_dma_warp_groups * self.num_warps_per_warp_group
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        # MegaBake: a mixed megakernel shares shared memory with other bodies,
        # so it may cap the pipeline's budget instead of filling the SM.
        self.smem_capacity = smem_capacity or utils.get_smem_capacity_in_bytes("sm_90")
        # MegaBake: warp-specialized register reallocation is only valid when
        # the GEMM owns the kernel; mixed megakernels run other bodies on the
        # same warps afterwards and keep the kernel-wide register count.
        self.reallocate_registers = reallocate_registers
        # MegaBake: split-K. Work tiles are (m, n, split); the L mode of A and
        # B selects the split's K range. Each split stores its FP32
        # accumulators, and the last split of a tile to finish sums all of
        # them in split order and runs the epilogue, so results do not depend
        # on which CTA finishes last.
        self.splits = splits
        # MegaBake: a BF16 bias over the output features (addmm), added to
        # the FP32 accumulators before the single rounding to BF16. swapAB
        # GEMMs carry the features on CUTLASS M, the others on N.
        self.bias_features = bias_features
        self.bias_along_m = bias_along_m
        # MegaBake: small-tile table GEMMs (swapAB decode) store each
        # thread's results straight from registers to global memory, at the
        # phase's own C address. That skips the shared-memory staging, the TMA
        # store and the wait for its completion before the CTA can arrive at
        # the next barrier; the barrier then orders the plain stores.
        # With split-K, the tile's owner (its last split) stores it directly.
        self.direct_store = direct_store
        # MegaBake: fused epilogues, each epilogue(accumulator, accumulator_1,
        # o0, o1, tensors) -> C value, evaluated per output element of a
        # direct-store GEMM; o0 and o1 are the output row and column of the
        # source (un-swapped) GEMM, and tensors are its extra operands. Each
        # phase picks one at run time (kind 1..n; kind 0 stores the plain
        # result), so phases of one shape share one kernel body.
        if (epilogues or paired) and not direct_store:
            raise ValueError("fused epilogues need a direct-store GEMM")
        if paired and not epilogues:
            raise ValueError("paired GEMMs need an epilogue to combine their products")
        self.epilogues = tuple(epilogues)
        # MegaBake: a paired GEMM computes two products of the same B with
        # two A operands of one shape (gate and up projections). Its 128-row
        # tile holds 64 rows of each A: TMA loads them into the two halves of
        # the A stage from two descriptors, so each thread's accumulators
        # pair row r of the first product with row r of the second.
        if paired and tile_shape_mn[0] != 128:
            raise ValueError("paired GEMMs use a 128-row tile")
        self.paired = paired
        # MegaBake: weight streaming. A table launch may load A (the weight of
        # a swapAB GEMM) for up to this many leading K tiles of a CTA's first
        # tile into the pipeline stages before the phase starts, from the
        # previous grid barrier (preload_table). The phase then initializes
        # no mbarriers and loads only B for those K tiles. A shared-memory tag
        # names the GEMM whose stages a CTA loaded.
        self.stream_stages = stream_stages
        self.stamp_address = stamp_address
        self.stamp_tag = stamp_tag

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )
        # MegaBake: split-K partials are released by an idle DMA warp, so the
        # GPU-scope release fence never stalls the WGMMA warpgroup (see
        # _reduce_splits).
        self.release_warp_id = self.load_warp_id + 1
        # An owner's MMA threads wait here until the release warp has seen
        # every other split arrive. Both sides block, so generations never mix.
        self.split_ready_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.num_mma_threads + 32
        )

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        """

        # check the cta tile shape
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        # N=16/32 serve swapAB GEMMs, whose N is the 1-32 token rows.
        if self.tile_shape_mnk[1] not in [16, 32, 64, 128, 256]:
            raise ValueError("CTA tile shape N must be 16/32/64/128/256")

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )
        # TMA loads A in tiles of a_tma_tile into a_tma_smem_layout_staged,
        # and the tile scheduler walks schedule_tile_mnk tiles of C.
        self.a_tma_tile = (self.tile_shape_mnk[0], self.tile_shape_mnk[2])
        self.a_tma_smem_layout_staged = self.a_smem_layout_staged
        self.schedule_tile_mnk = self.tile_shape_mnk
        if self.paired:
            self.a_tma_tile = (64, self.tile_shape_mnk[2])
            self.a_tma_smem_layout_staged = self._paired_a_smem_layout_staged()
            # Each C tile holds 64 outputs: row r of both products.
            self.schedule_tile_mnk = (64, *self.tile_shape_mnk[1:])

    def _paired_a_smem_layout_staged(self):
        """The paired A stages as TMA writes them: stage s of the 128-row
        layout is half-stages 2s (first A) and 2s + 1 (second A) of a 64-row
        layout over the same memory, because the swizzle atoms stack along M
        and rows 64-127 of a stage start where the next 64-row stage would."""
        return self._make_smem_layouts(
            (64, *self.tile_shape_mnk[1:]),
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage * 2,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )[0]

    @cute.jit
    def prepare(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
    ):
        """Prepare TMA operands and the persistent tile schedule.
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(
                f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}"
            )
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16, float8, or int8 ")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_tma_smem_layout_staged,
            self.a_tma_tile,
            self.cluster_shape_mn[1],
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.schedule_tile_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along_m,
            max_active_clusters,
            self.splits,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            # MegaBake: the split-K arrival count, shared by the MMA threads
            # (slot 0), and the streamed GEMM's tag and K tiles (slots 2 and
            # 3; see stream_stages).
            split_arrival: cute.struct.MemRange[cutlass.Int32, 4]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        # Static element counts for union_shared_storage.
        self.storage_sizes = (
            self.ab_stage,
            cute.cosize(self.a_smem_layout_staged),
            cute.cosize(self.b_smem_layout_staged),
            cute.cosize(self.epi_smem_layout_staged),
        )

        return (
            (
                tma_atom_a,
                tma_tensor_a,
                tma_atom_b,
                tma_tensor_b,
                self.tiled_mma,
                self.cta_layout_mnk,
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
            ),
            (tma_atom_c, tma_tensor_c, self.epi_smem_layout_staged),
            tile_sched_params,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        tensor_bindings, epilogue, tile_sched_params = self.prepare(
            a, b, c, max_active_clusters
        )
        (
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tiled_mma,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
        ) = tensor_bindings
        tma_atom_c, tma_tensor_c, epi_smem_layout_staged = epilogue

        self.range_kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tile_sched_params,
            tiled_mma,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
        ).launch(
            grid=utils.StaticPersistentTileScheduler.get_grid_shape(
                tile_sched_params, max_active_clusters
            ),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
        return

    @cute.jit
    def run(
        self,
        work_iterator,
        tensor_bindings,
        scratch,
        epilogue,
    ):
        (
            tma_atom_a,
            mA_mkl,
            tma_atom_b,
            mB_nkl,
            tiled_mma,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
        ) = tensor_bindings
        tma_atom_c, mC_mnl, epi_smem_layout_staged = epilogue
        self.kernel(
            tma_atom_a,
            mA_mkl,
            tma_atom_b,
            mB_nkl,
            tma_atom_c,
            mC_mnl,
            tiled_mma,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
            work_iterator,
            scratch,
            cute.arch.grid_dim(),
        )

    @cute.jit
    def run_table(
        self,
        prepared,
        scratch,
        tma_descs,
        rotated_block=None,
        split_state=None,
        bias_address=None,
        c_address=None,
        epilogue_state=None,
        stream_tag=None,
    ):
        """Run one table-launched phase whose TMA descriptors live in global
        memory. ``prepared`` is this variant's ``prepare`` result; the kernel
        prologue has written ``tma_descs`` (A, B, C, second A) for this
        phase's tensors. Split-K variants take ``split_state``: (FP32
        partials, Int32 tile arrival counters, this phase's partial offset,
        its counter offset). Fused epilogues take ``epilogue_state``: a
        callable returning (the phase's epilogue kind, one tuple of extra
        tensors per epilogue), called after the mainloop. Streaming variants
        take ``stream_tag``, this phase's nonzero tag: the CTA's pipeline
        stages are already loaded when ``preload_table`` stored that tag."""
        tensor_bindings, epilogue, tile_sched_params = prepared
        (
            tma_atom_a,
            mA_mkl,
            tma_atom_b,
            mB_nkl,
            tiled_mma,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
        ) = tensor_bindings
        tma_atom_c, mC_mnl, epi_smem_layout_staged = epilogue
        self.kernel(
            tma_atom_a,
            mA_mkl,
            tma_atom_b,
            mB_nkl,
            tma_atom_c,
            mC_mnl,
            tiled_mma,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            epi_smem_layout_staged,
            tile_sched_params,
            scratch,
            cute.arch.grid_dim(),
            tma_descs,
            rotated_block,
            split_state,
            bias_address,
            c_address,
            epilogue_state,
            stream_tag,
        )

    @cute.jit
    def preload_table(
        self,
        prepared,
        scratch,
        tma_descs,
        rotated_block,
        stream_tag,
    ):
        """Load A of the leading K tiles of this CTA's first tile of a later
        table phase into the pipeline stages, and tag the stages with that
        phase's ``stream_tag`` and the number of K tiles loaded (see
        ``stream_stages``).

        Called by warp 0 when every warp of the CTA has finished its previous
        body (whose pipeline is drained), with the rotated block index the
        phase will run with. Each stage expects its full A + B bytes, so it
        completes once the phase loads B. A must not be written by the
        launch."""
        tensor_bindings, _, tile_sched_params = prepared
        (
            tma_atom_a,
            mA_mkl,
            _,
            _,
            _,
            cta_layout_mnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
        ) = tensor_bindings
        desc_a, _, _, desc_a2 = tma_descs
        storage = scratch

        mainloop_pipeline = self._mainloop_pipeline(
            storage, a_smem_layout_staged, b_smem_layout_staged, cta_layout_mnk
        )
        cute.arch.mbarrier_init_fence()

        a_tma_smem_layout_staged = a_smem_layout_staged
        if cutlass.const_expr(self.paired):
            a_tma_smem_layout_staged = self._paired_a_smem_layout_staged()
        sA_tma = storage.sA.get_tensor(
            a_tma_smem_layout_staged.outer, swizzle=a_tma_smem_layout_staged.inner
        )
        gA_mkl = cute.local_tile(mA_mkl, self.a_tma_tile, (None, None, None))
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        )
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            cluster_coord_mnk[1],
            cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape),
            cute.group_modes(sA_tma, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        block_x, block_y, _ = cute.arch.block_idx()
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, (block_x, block_y, rotated_block), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        k_tiles = self._streamed_k_tiles(cute.size(gA_mkl, mode=[3]))

        cute.nvgpu.cpasync.fence_tma_desc_acquire(desc_a)
        if cutlass.const_expr(self.paired):
            cute.nvgpu.cpasync.fence_tma_desc_acquire(desc_a2)
        if work_tile.is_valid_tile:
            tile_coord_mnl = work_tile.tile_idx
            tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            for k_tile in range(k_tiles):
                mainloop_pipeline.producer_acquire(producer_state)
                a_stage = producer_state.index
                if cutlass.const_expr(self.paired):
                    a_stage = producer_state.index * 2
                cute.copy(
                    tma_atom_a,
                    tAgA_mkl[(None, producer_state.count)],
                    tAsA[(None, a_stage)],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                    mcast_mask=0,
                    tma_desc_ptr=desc_a,
                )
                if cutlass.const_expr(self.paired):
                    cute.copy(
                        tma_atom_a,
                        tAgA_mkl[(None, producer_state.count)],
                        tAsA[(None, a_stage + 1)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            producer_state
                        ),
                        mcast_mask=0,
                        tma_desc_ptr=desc_a2,
                    )
                producer_state.advance()
            # The phase loads A itself from K tile ``k_tiles`` on.
            storage.split_arrival[STREAM_TAG_SLOT + 1] = k_tiles
            storage.split_arrival[STREAM_TAG_SLOT] = stream_tag

    def _streamed_k_tiles(self, k_tile_cnt):
        """K tiles whose A ``preload_table`` loads for a tile of ``k_tile_cnt``."""
        return cutlass.min(k_tile_cnt, min(self.ab_stage, self.stream_stages))

    def _mainloop_pipeline(
        self,
        storage,
        a_smem_layout_staged,
        b_smem_layout_staged,
        cta_layout_mnk,
        initialize: bool = True,
    ):
        """The A/B mainloop pipeline over ``storage``'s mbarriers. Without
        ``initialize`` its mbarrier init is not emitted: the caller emits it
        under a run-time condition (see ``kernel``)."""
        # Each warp will constribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = (
            mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        )
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, cute.slice_(a_smem_layout_staged, (None, None, 0))
        ) + cute.size_in_bytes(
            self.b_dtype, cute.slice_(b_smem_layout_staged, (None, None, 0))
        )
        with contextlib.nullcontext() if initialize else _without_mbarrier_init():
            return pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
                num_stages=self.ab_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, consumer_arrive_cnt
                ),
                tx_count=tma_copy_bytes,
                cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
                defer_sync=True,
            )

    @cute.kernel
    def range_kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        self.run(
            tile_sched_params,
            (
                tma_atom_a,
                mA_mkl,
                tma_atom_b,
                mB_nkl,
                tiled_mma,
                cta_layout_mnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
            ),
            storage,
            (tma_atom_c, mC_mnl, epi_smem_layout_staged),
        )

    # GPU device kernel
    @cute.jit
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        scratch,
        schedule_grid_dim,
        tma_descs=None,
        rotated_block=None,
        split_state=None,
        bias_address=None,
        c_address=None,
        epilogue_state=None,
        stream_tag=None,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param tile_sched_params: Parameters for the persistent tile scheduler
        :type tile_sched_params: utils.PersistentTileSchedulerParams
        """

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if cutlass.const_expr(self.stamp_address != 0):
            if tidx == 0:
                self._stamp(cutlass.Int32(0), 8)

        # MegaBake: phases sharing a barrier interval start their tiles on
        # different CTAs, so the scheduler sees a rotated block index.
        schedule_block = cute.arch.block_idx()
        if cutlass.const_expr(rotated_block is not None):
            block_x, block_y, block_z = schedule_block
            schedule_block = (block_x, block_y, rotated_block)

        # MegaBake: table launches pass per-phase descriptors in global
        # memory; copies then name them through tma_desc_ptr.
        desc_a = None
        desc_b = None
        desc_c = None
        desc_a2 = None
        if cutlass.const_expr(tma_descs is not None):
            desc_a, desc_b, desc_c, desc_a2 = tma_descs

        # Prefetch Tma desc
        if cutlass.const_expr(tma_descs is None):
            if warp_idx == 0:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        # The enclosing worker owns shared storage for the whole range.
        storage = scratch

        # MegaBake: a streamed phase's stages were loaded, and their
        # mbarriers initialized, by preload_table; only warp 0 initializes
        # and produces, so only it reads the tag.
        streaming = self.stream_stages > 0 and stream_tag is not None
        mainloop_pipeline = self._mainloop_pipeline(
            storage,
            a_smem_layout_staged,
            b_smem_layout_staged,
            cta_layout_mnk,
            initialize=not streaming,
        )
        if cutlass.const_expr(streaming):
            if warp_idx == 0:
                if storage.split_arrival[STREAM_TAG_SLOT] != stream_tag:
                    mainloop_pipeline.sync_object_full.mbarrier_init()
                    mainloop_pipeline.sync_object_empty.mbarrier_init()

        if cutlass.const_expr(self.splits > 1):
            # Split-K: non-owner items whose partials the MMA threads stored
            # (SPLIT_PUBLISHED_SLOT); pipeline_init_wait orders the reset.
            if tidx == 0:
                storage.split_arrival[SPLIT_PUBLISHED_SLOT] = cutlass.Int32(0)

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Generate smem tensor A/B
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # The same memory as TMA writes it (64-row half-stages when paired).
        # Layouts built in prepare belong to the host function, so the kernel
        # rebuilds the paired one.
        a_tma_smem_layout_staged = a_smem_layout_staged
        if cutlass.const_expr(self.paired):
            a_tma_smem_layout_staged = self._paired_a_smem_layout_staged()
        sA_tma = storage.sA.get_tensor(
            a_tma_smem_layout_staged.outer, swizzle=a_tma_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # Local_tile partition global tensors
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            self.a_tma_tile,
            (None, None, None),
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # Partition shared tensor for TMA load A/B
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA_tma, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        # Partition global tensor for TiledMMA_A/B/C
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        mma_warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(
            mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
        )

        # Make fragments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Cluster wait for barrier init
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        if cutlass.const_expr(self.stamp_address != 0):
            if tidx == 0:
                self._stamp(cutlass.Int32(0), 9)

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if cutlass.const_expr(self.reallocate_registers):
            if is_dma_warp_group:
                cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, schedule_block, schedule_grid_dim
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            if cutlass.const_expr(tma_descs is not None):
                cute.nvgpu.cpasync.fence_tma_desc_acquire(desc_a)
                cute.nvgpu.cpasync.fence_tma_desc_acquire(desc_b)
                if cutlass.const_expr(self.paired):
                    cute.nvgpu.cpasync.fence_tma_desc_acquire(desc_a2)
            # MegaBake: operands may have been written with generic stores by
            # other CTAs before the enclosing grid barrier; order those
            # writes before this thread's async-proxy TMA loads.
            cute.arch.fence_proxy("async.global")

            # MegaBake: K tiles of the first tile whose A preload_table loaded.
            if cutlass.const_expr(streaming):
                streamed_k_tiles = cutlass.Int32(0)
                if storage.split_arrival[STREAM_TAG_SLOT] == stream_tag:
                    streamed_k_tiles = cutlass.Int32(storage.split_arrival[STREAM_TAG_SLOT + 1])

            stamp_item = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                if cutlass.const_expr(self.stamp_address != 0):
                    if tidx == 0:
                        self._stamp(stamp_item, 0)
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()

                for k_tile in range(k_tile_cnt):
                    load_a = True
                    if cutlass.const_expr(streaming):
                        load_a = k_tile >= streamed_k_tiles
                    if load_a:
                        # Conditionally wait for AB buffer empty
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    # Slice to global/shared memref to current k_tile
                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    a_stage = mainloop_producer_state.index
                    if cutlass.const_expr(self.paired):
                        a_stage = mainloop_producer_state.index * 2
                    tAsA_pipe = tAsA[(None, a_stage)]

                    tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    if load_a:
                        # TMA load A/B
                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                            mcast_mask=a_mcast_mask,
                            tma_desc_ptr=desc_a,
                        )
                        if cutlass.const_expr(self.paired):
                            # The second A's rows of this tile, into rows 64-127.
                            cute.copy(
                                tma_atom_a,
                                tAgA_k,
                                tAsA[(None, a_stage + 1)],
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                mcast_mask=a_mcast_mask,
                                tma_desc_ptr=desc_a2,
                            )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                        mcast_mask=b_mcast_mask,
                        tma_desc_ptr=desc_b,
                    )

                    # Mainloop pipeline's producer commit is a NOP
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    if cutlass.const_expr(self.stamp_address != 0):
                        if tidx == 0 and k_tile == 0:
                            self._stamp(stamp_item, 10)
                    mainloop_producer_state.advance()

                if cutlass.const_expr(self.stamp_address != 0):
                    if tidx == 0:
                        self._stamp(stamp_item, 1)
                    stamp_item = stamp_item + 1
                if cutlass.const_expr(streaming):
                    streamed_k_tiles = cutlass.Int32(0)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)

        if cutlass.const_expr(self.splits > 1):
            if warp_idx == self.release_warp_id:
                # Every wait below is warp-uniform (all lanes poll, the warp
                # votes): a per-lane loop anywhere in the kernel makes ptxas
                # serialize every wgmma (C7519, tools/wgmma_serialization.py).
                release_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, schedule_block, schedule_grid_dim
                )
                release_tile = release_sched.initial_work_tile_info()
                partials, counters, partial_base, counter_base = split_state
                n_tiles = cute.size(gC_mnl, mode=[3])
                published = storage.split_arrival.data_ptr() + SPLIT_PUBLISHED_SLOT
                released = cutlass.Int32(0)
                while release_tile.is_valid_tile:
                    coordinate = release_tile.tile_idx
                    counter = (
                        counters.iterator + counter_base + coordinate[0] * n_tiles + coordinate[1]
                    )
                    if coordinate[2] != self.splits - 1:
                        # Acquire the MMA threads' partial stores (CTA scope),
                        # then release them at GPU scope.
                        released = released + 1
                        seen = cute.arch.load(published, cutlass.Int32, sem="acquire", scope="cta")
                        while cute.arch.vote_any_sync(seen < released):
                            seen = cute.arch.load(
                                published, cutlass.Int32, sem="acquire", scope="cta"
                            )
                        with cute.arch.elect_one():
                            cute.arch.red(
                                counter,
                                cutlass.Int32(1),
                                op="add",
                                dtype="s32",
                                sem="release",
                                scope="gpu",
                            )
                    else:
                        # The owner: wait for the other splits' partials.
                        seen = cute.arch.load(counter, cutlass.Int32, sem="acquire", scope="gpu")
                        while cute.arch.vote_any_sync(seen < self.splits - 1):
                            seen = cute.arch.load(
                                counter, cutlass.Int32, sem="acquire", scope="gpu"
                            )
                        # Ready for the next phase that uses this counter
                        # (after a grid barrier).
                        with cute.arch.elect_one():
                            cute.arch.store(counter, cutlass.Int32(0))
                        self.split_ready_barrier.arrive_and_wait()
                    release_sched.advance_to_next_work()
                    release_tile = release_sched.get_current_work()

        # MMA warp group
        if not is_dma_warp_group:
            if cutlass.const_expr(self.reallocate_registers):
                cute.arch.setmaxregister_increase(self.mma_register_requirement)
            if cutlass.const_expr(tma_descs is not None):
                if warp_idx == self.epi_store_warp_id:
                    cute.nvgpu.cpasync.fence_tma_desc_acquire(desc_c)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, schedule_block, schedule_grid_dim
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Partition for epilogue
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout,
                elem_ty_d=self.c_dtype,
                elem_ty_acc=self.acc_dtype,
            )

            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.c_layout.is_m_major_c(),
                    4,
                ),
                self.c_dtype,
            )

            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s,
                tiled_copy_C_Atom,
            )

            # (R2S, R2S_M, R2S_N, PIPE_D)
            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            )
            # (t)hread-partition for (r)egister to (s)mem copy (tRS_)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            # (R2S, R2S_M, R2S_N)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            # Allocate D registers.
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            k_pipe_mmas = 1
            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)

            # Initialize tma store pipeline
            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_threads,
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=tma_store_producer_group,
            )
            c_extent = (cute.size(mC_mnl, mode=[0]), cute.size(mC_mnl, mode=[1]))
            if cutlass.const_expr(self.direct_store):
                # This phase's C in memory: mC_mnl holds TMA coordinates of
                # the variant template's C. swapAB writes C^T, so C is
                # M-major (see build_table_descriptors).
                gC_direct = cute.local_tile(
                    cute.make_tensor(
                        cute.make_ptr(
                            self.c_dtype, c_address, cute.AddressSpace.gmem, assumed_align=16
                        ),
                        cute.make_layout(
                            (c_extent[0], c_extent[1], 1),
                            stride=(1, c_extent[0], c_extent[0] * c_extent[1]),
                        ),
                    ),
                    cute.slice_(self.tile_shape_mnk, (None, None, 0)),
                    (None, None, None),
                )

            stamp_item = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                if cutlass.const_expr(self.stamp_address != 0):
                    if tidx == self.num_dma_warp_groups * self.num_threads_per_warp_group:
                        self._stamp(stamp_item, 2)
                tile_coord_mnl = work_tile.tile_idx
                if cutlass.const_expr(self.direct_store and self.splits > 1):
                    # The split index is not a C coordinate; set after the reduction.
                    gC_mnl_slice = gC_direct[
                        (None, None, tile_coord_mnl[0], tile_coord_mnl[1], 0)
                    ]
                elif cutlass.const_expr(self.direct_store):
                    gC_mnl_slice = gC_direct[(None, None, *tile_coord_mnl)]
                else:
                    gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]

                # MAINLOOP
                mainloop_consumer_read_state.reset_count()
                mainloop_consumer_release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()

                for k_tile in range(prologue_mma_cnt):
                    # Wait for TMA copies to complete
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    if cutlass.const_expr(self.stamp_address != 0):
                        if tidx == self.num_dma_warp_groups * self.num_threads_per_warp_group:
                            self._stamp(stamp_item, 3)
                    # WGMMA
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )

                    cute.nvgpu.warpgroup.commit_group()
                    mainloop_consumer_read_state.advance()

                for k_tile in range(prologue_mma_cnt, k_tile_cnt):
                    # Wait for TMA copies to complete
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    # WGMMA
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )

                    cute.nvgpu.warpgroup.commit_group()
                    # Wait on the wgmma barrier for WGMMA to complete
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()

                cute.nvgpu.warpgroup.wait_group(0)
                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()

                if cutlass.const_expr(self.stamp_address != 0):
                    if tidx == self.num_dma_warp_groups * self.num_threads_per_warp_group:
                        self._stamp(stamp_item, 4)
                run_epilogue = True
                if cutlass.const_expr(self.splits > 1):
                    run_epilogue = self._reduce_splits(
                        accumulators,
                        tile_coord_mnl,
                        cute.size(gC_mnl, mode=[3]),
                        tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group,
                        storage.split_arrival.data_ptr(),
                        split_state,
                    )
                    # Every split's result lands in C's single L slice.
                    if cutlass.const_expr(not self.direct_store):
                        gC_mnl_slice = gC_mnl[
                            (None, None, tile_coord_mnl[0], tile_coord_mnl[1], 0)
                        ]

                if cutlass.const_expr(self.stamp_address != 0):
                    if tidx == self.num_dma_warp_groups * self.num_threads_per_warp_group:
                        self._stamp(stamp_item, 5)
                if cutlass.const_expr(self.paired):
                    if run_epilogue:
                        self._store_paired_tile(
                            tRS_rAcc,
                            thr_copy_r2s,
                            size_tRS_rD,
                            tile_coord_mnl,
                            c_address,
                            c_extent,
                            epilogue_state,
                        )
                else:
                    if run_epilogue:
                        self._store_tile(
                            accumulators,
                            gC_mnl_slice,
                            tma_atom_c,
                            sC,
                            tiled_copy_r2s,
                            tRS_sD,
                            tRS_rAcc,
                            tRS_rD,
                            tRS_rD_out,
                            size_tRS_rD,
                            tma_store_pipeline,
                            tile_sched,
                            warp_idx,
                            desc_c,
                            thr_copy_r2s,
                            tile_coord_mnl,
                            bias_address,
                            c_extent,
                            epilogue_state,
                            c_address,
                        )

                if cutlass.const_expr(self.stamp_address != 0):
                    if tidx == self.num_dma_warp_groups * self.num_threads_per_warp_group:
                        self._stamp(stamp_item, 6)
                    stamp_item = stamp_item + 1
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            if cutlass.const_expr(self.stamp_address != 0):
                if tidx == self.num_dma_warp_groups * self.num_threads_per_warp_group:
                    self._stamp(cutlass.Int32(0), 11)
            if cutlass.const_expr(not self.direct_store):
                tma_store_pipeline.producer_tail()
                # MegaBake: the tail only waits until shared memory has been
                # read. Other CTAs read these outputs after the enclosing grid
                # barrier, so wait for the global writes and order them with
                # the generic proxy before returning.
                if warp_idx == self.epi_store_warp_id:
                    cute.arch.cp_async_bulk_wait_group(0, read=False)
                    cute.arch.fence_proxy("async.global")

    @cute.jit
    def _stamp(self, item, field):
        """Write one diagnostic stamp for this CTA's work item (one thread)."""
        if cutlass.const_expr(self.stamp_address != 0):
            slot = (
                (self.stamp_tag % GEMM_STAMP_VARIANTS * GEMM_STAMP_CTAS + cute.arch.block_idx()[2])
                * GEMM_STAMP_ITEMS
                + cutlass.min(item, cutlass.Int32(GEMM_STAMP_ITEMS - 1))
            ) * GEMM_STAMP_FIELDS
            stamps = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Int32,
                    cutlass.Int64(self.stamp_address),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                cute.make_layout(
                    GEMM_STAMP_VARIANTS * GEMM_STAMP_CTAS * GEMM_STAMP_ITEMS * GEMM_STAMP_FIELDS
                ),
            )
            stamps[slot + field] = cute.arch.globaltimer_lo().to(cutlass.Int32)
            if field == 8:
                stamps[slot + 7] = cutlass.Int32(self.stamp_tag)

    @cute.jit
    def _reduce_splits(
        self,
        accumulators,
        tile_coord_mnl,
        n_tiles,
        mma_thread,
        arrival_flag,
        split_state,
    ):
        """Fixed-owner split-K. The tile's last split (the latest work item in
        every CTA's static order: L is the scheduler's slowest mode) owns it.
        Other splits store FP32 partials (element-major, so a warp's lanes
        write consecutive words) and hand them to the release warp, which
        releases them at GPU scope; the MMA threads never wait on global
        memory for them. The owner keeps its accumulators, waits on the
        release warp (which has acquired the other splits' arrivals), and
        adds their partials in split order. True for the owner, whose
        accumulators then hold the tile's sum. Every wait is on a work item
        with a smaller linear index, so the lowest unfinished item can always
        proceed."""
        partials, counters, partial_base, counter_base = split_state
        count = cute.size(accumulators)
        tile = tile_coord_mnl[0] * n_tiles + tile_coord_mnl[1]
        tile_elements = count * self.num_mma_threads
        split = tile_coord_mnl[2]
        owner = split == self.splits - 1
        if owner:
            self.split_ready_barrier.arrive_and_wait()
            first = partial_base + tile * self.splits * tile_elements + mma_thread
            for index in cutlass.range_constexpr(count):
                total = cutlass.Float32(0.0)
                for other in cutlass.range_constexpr(self.splits - 1):
                    total = total + cute.arch.load(
                        partials.iterator
                        + first
                        + other * tile_elements
                        + index * self.num_mma_threads,
                        cutlass.Float32,
                        cop="cg",
                    )
                accumulators[index] = total + accumulators[index]
        else:
            own = partial_base + (tile * self.splits + split) * tile_elements
            for index in cutlass.range_constexpr(count):
                cute.arch.store(
                    partials.iterator + own + index * self.num_mma_threads + mma_thread,
                    accumulators[index],
                    cop="cg",
                )
            # Every MMA thread's stores precede one CTA-scope release; the
            # release warp acquires it and releases at GPU scope.
            self.epilog_sync_barrier.arrive_and_wait()
            if mma_thread // 32 == 0:
                with cute.arch.elect_one():
                    cute.arch.atomic_add(
                        arrival_flag + SPLIT_PUBLISHED_SLOT,
                        cutlass.Int32(1),
                        sem="release",
                        scope="cta",
                    )
        return owner

    @cute.jit
    def _store_tile(
        self,
        accumulators,
        gC_mnl_slice,
        tma_atom_c,
        sC,
        tiled_copy_r2s,
        tRS_sD,
        tRS_rAcc,
        tRS_rD,
        tRS_rD_out,
        size_tRS_rD,
        tma_store_pipeline,
        tile_sched,
        warp_idx,
        desc_c,
        thr_copy_r2s=None,
        tile_coord_mnl=None,
        bias_address=None,
        c_extent=None,
        epilogue_state=None,
        c_address=None,
    ):
        # Epilogue
        if cutlass.const_expr(len(self.epilogues) > 0):
            epilogue_state = epilogue_state()
        tCgC_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)

        # thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            tCgC_for_tma_partition,
        )

        epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
        epi_tile_shape = tCgC_for_tma_partition.shape[1]
        epi_tile_layout = cute.make_layout(
            epi_tile_shape, stride=(epi_tile_shape[1], 1)
        )

        num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num
        if cutlass.const_expr(self.bias_features > 0 or self.direct_store):
            # MegaBake: the source side of the per-thread register-to-shared
            # copy follows the MMA accumulator layout, so it locates each D
            # register in the epilogue subtile. (The MMA slice is per
            # warpgroup, and the stmatrix destination side names the rows a
            # thread addresses, not the values it holds.)
            epi_coordinates = thr_copy_r2s.partition_S(
                cute.make_identity_tensor(self.epi_tile)
            )
        if cutlass.const_expr(self.bias_features > 0):
            bias = cute.make_tensor(
                cute.make_ptr(
                    cutlass.BFloat16, bias_address, cute.AddressSpace.gmem, assumed_align=2
                ),
                cute.make_layout(self.bias_features),
            )
        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # Copy from accumulators to D registers
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
            if cutlass.const_expr(self.bias_features > 0):
                # addmm: add the output feature's bias in FP32, then round once.
                epi_m, epi_n = epi_tile_layout.get_hier_coord(epi_idx)
                for epi_v in cutlass.range_constexpr(size_tRS_rD):
                    coordinate = epi_coordinates[epi_v]
                    if cutlass.const_expr(self.bias_along_m):
                        feature = (
                            tile_coord_mnl[0] * self.tile_shape_mnk[0]
                            + epi_m * self.epi_tile[0]
                            + coordinate[0]
                        )
                    else:
                        feature = (
                            tile_coord_mnl[1] * self.tile_shape_mnk[1]
                            + epi_n * self.epi_tile[1]
                            + coordinate[1]
                        )
                    tRS_rD[epi_v] = tRS_rD[epi_v] + bias[feature].to(cutlass.Float32)

            # Type conversion
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.c_dtype))

            if cutlass.const_expr(self.direct_store):
                # Each register to its element of C, inside C's bounds (the
                # token tile pads the token count).
                store_m, store_n = epi_tile_layout.get_hier_coord(epi_idx)
                for epi_v in cutlass.range_constexpr(size_tRS_rD):
                    coordinate = epi_coordinates[epi_v]
                    row = store_m * self.epi_tile[0] + coordinate[0]
                    column = store_n * self.epi_tile[1] + coordinate[1]
                    feature = tile_coord_mnl[0] * self.tile_shape_mnk[0] + row
                    token = tile_coord_mnl[1] * self.tile_shape_mnk[1] + column
                    if cutlass.const_expr(len(self.epilogues) == 0):
                        # C^T is M-major: element (feature, token) is at
                        # feature + token * M. A predicated store, not a branch.
                        store_bf16_if(
                            c_address
                            + (cutlass.Int64(feature) + cutlass.Int64(token) * c_extent[0]) * 2,
                            tRS_rD_out[epi_v],
                            cutlass.Int32(feature < c_extent[0] and token < c_extent[1]),
                        )
                    elif feature < c_extent[0] and token < c_extent[1]:
                        # swapAB: CUTLASS rows are the source GEMM's columns.
                        gC_mnl_slice[row, column] = self._apply_epilogue(
                            tRS_rD_out[epi_v],
                            tRS_rD[epi_v],
                            tRS_rD[epi_v],
                            token,
                            feature,
                            epilogue_state,
                        )
                continue

            # Copy from D registers to shared memory
            epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(
                tRS_sD, mode=[3]
            )
            cute.copy(
                tiled_copy_r2s,
                tRS_rD_out,
                tRS_sD[(None, None, None, epi_buffer)],
            )

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.epilog_sync_barrier.arrive_and_wait()

            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from shared memory to global memory
            if warp_idx == self.epi_store_warp_id:
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                    tma_desc_ptr=desc_c,
                )
                tma_store_pipeline.producer_commit()
                tma_store_pipeline.producer_acquire()

            self.epilog_sync_barrier.arrive_and_wait()

        if cutlass.const_expr(self.direct_store):
            # MegaBake: reconverge the MMA warpgroup after its per-element
            # stores before the next tile's wgmma.
            self.epilog_sync_barrier.arrive_and_wait()


    @cute.jit
    def _store_paired_tile(
        self,
        tRS_rAcc,
        thr_copy_r2s,
        size_tRS_rD,
        tile_coord_mnl,
        c_address,
        c_extent,
        epilogue_state,
    ):
        """Store one paired tile from registers: output feature f of the tile
        combines row f of the first product (accumulator rows 0-63) with row
        f of the second (rows 64-127) through the fused epilogue."""
        epilogue_state = epilogue_state()
        # swapAB writes C^T, so C is M-major: features are contiguous.
        out = cute.make_tensor(
            cute.make_ptr(self.c_dtype, c_address, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout(c_extent, stride=(1, c_extent[0])),
        )
        # The MMA slice is per warpgroup; the register-to-shared copy's source
        # partition locates each of this thread's registers in an epilogue
        # subtile. Subtiles are 64 rows and run M-major after N (see
        # _store_tile), so subtile (1, n) holds the second product's rows
        # matching subtile (0, n).
        epi_coordinates = thr_copy_r2s.partition_S(cute.make_identity_tensor(self.epi_tile))
        epi_columns = self.tile_shape_mnk[1] // self.epi_tile[1]
        for epi_n in cutlass.range_constexpr(epi_columns):
            for epi_v in cutlass.range_constexpr(size_tRS_rD):
                coordinate = epi_coordinates[epi_v]
                feature = tile_coord_mnl[0] * self.schedule_tile_mnk[0] + coordinate[0]
                token = (
                    tile_coord_mnl[1] * self.tile_shape_mnk[1]
                    + epi_n * self.epi_tile[1]
                    + coordinate[1]
                )
                if feature < c_extent[0] and token < c_extent[1]:
                    first = tRS_rAcc[epi_n * size_tRS_rD + epi_v]
                    out[feature, token] = self._apply_epilogue(
                        first.to(self.c_dtype),
                        first,
                        tRS_rAcc[(epi_columns + epi_n) * size_tRS_rD + epi_v],
                        token,
                        feature,
                        epilogue_state,
                    )
        # Reconverge before the next tile's wgmma (see _store_tile).
        self.epilog_sync_barrier.arrive_and_wait()

    @cute.jit
    def _apply_epilogue(
        self, plain, accumulator, accumulator_1, token, feature, epilogue_state
    ):
        """``plain`` for epilogue kind 0, else epilogue ``kind`` of this
        output element; every kind is one branch of the same body."""
        kind, tensors = epilogue_state
        value = plain
        for index in cutlass.range_constexpr(len(self.epilogues)):
            if kind == index + 1:
                value = self.epilogues[index](
                    accumulator, accumulator_1, token, feature, tensors[index]
                ).to(self.c_dtype)
        return value

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_stage = 4
        epi_bytes = c_bytes_per_stage * epi_stage

        mbar_helpers_bytes = 1024

        ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes)
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk: tuple[int, int, int],
        element_type: type[cutlass.Numeric],
        is_cooperative: bool = False,
        epi_tile_override: Optional[tuple[int, int]] = None,
    ) -> tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if is_cooperative:
            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: utils.LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: utils.LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum for the output matrix C
        :type c_layout: utils.LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = (
            a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        b_is_k_major = (
            b_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        max_active_clusters: cutlass.Constexpr,
        splits: int = 1,
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        if cutlass.const_expr(splits > 1):
            # MegaBake: C has one L slice; split-K tiles add the split mode.
            num_ctas_mnl = (num_ctas_mnl[0], num_ctas_mnl[1], splits)
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size,
            raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        valid_ab_dtypes = {
            cutlass.Float16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
            cutlass.Uint8,
            cutlass.Int8,
        }
        if a_dtype not in valid_ab_dtypes:
            is_valid = False
        if b_dtype not in valid_ab_dtypes:
            is_valid = False

        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        if a_dtype.width != b_dtype.width:
            is_valid = False
        if not a_dtype.is_same_kind(b_dtype):
            is_valid = False

        # for 8-bit types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (
            b_dtype.width == 8 and b_major != "k"
        ):
            is_valid = False

        # Define compatibility mapping between accumulator type and AB type
        acc_ab_compatibility = {
            cutlass.Float32: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {cutlass.Uint8, cutlass.Int8},
        }
        # Check compatibility between accumulator type and A type
        if a_dtype not in acc_ab_compatibility[acc_dtype]:
            is_valid = False

        # Define compatibility mapping between accumulator type and C type
        acc_c_compatibility = {
            cutlass.Float32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
        }
        # Check compatibility between accumulator type and C type
        if c_dtype not in acc_c_compatibility[acc_dtype]:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid
