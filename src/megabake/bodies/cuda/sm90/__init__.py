"""Hopper (sm90a): TMA/WGMMA GEMMs, softmax attention and linear-attention decode."""

from ...interface import Generation
from .attention import AttentionBodyProvider
from .gemm import G0RangeBodyProvider, launch_options
from .gemv import GemvBodyProvider

GENERATION = Generation(
    architectures=frozenset({"sm90a"}),
    providers=(G0RangeBodyProvider(), GemvBodyProvider(), AttentionBodyProvider()),
    launch_options=launch_options,
)
