"""Blackwell bodies: SM100 CUTLASS GEMM and warp-MMA fallback."""

from ...interface import Generation
from . import table as table
from . import tuning as tuning
from .attention import AttentionBodyProvider
from .gemm import BLACKWELL_ARCHITECTURES, BlackwellGemmProvider

GENERATION = Generation(
    architectures=BLACKWELL_ARCHITECTURES,
    providers=(BlackwellGemmProvider(), AttentionBodyProvider()),
)
