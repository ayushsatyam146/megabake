"""SM100 table GEMM descriptor addressing."""

from ....platform import TensorMapManager, TensorMapUpdateMode, cute, cutlass

_TABLE_TENSORMAPS = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
_DESCRIPTORS_PER_GEMM = 4


@cute.jit
def table_descriptor(
    descriptors: cute.Tensor,
    gemm: cutlass.Int32,
    operand: cutlass.Constexpr,
    space: cutlass.Constexpr = cute.AddressSpace.generic,
):
    return _TABLE_TENSORMAPS.get_tensormap_ptr(
        descriptors.iterator + (gemm * _DESCRIPTORS_PER_GEMM + operand) * 16, space
    )


@cute.jit
def table_descriptors(descriptors: cute.Tensor, gemm: cutlass.Int32):
    return (
        table_descriptor(descriptors, gemm, 0),
        table_descriptor(descriptors, gemm, 1),
        table_descriptor(descriptors, gemm, 2),
        table_descriptor(descriptors, gemm, 3),
    )
