"""SM100 GEMM configuration shared by selection, code generation and tuning."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PersistentGemmSettings:
    enabled: bool
    width: int
    stages: int
    cutlass_fork: bool


def persistent_gemm_settings() -> PersistentGemmSettings:
    enabled = os.environ.get("MEGABAKE_SM100_PERSISTENT", "1") == "1"
    width = 256 if os.environ.get("MEGABAKE_SM100_PERSISTENT_WIDTH", "256") == "256" else 128
    stages = 2 if os.environ.get("MEGABAKE_SM100_PERSISTENT_STAGES") == "2" else 1
    fork_default = "1" if width == 256 else "0"
    cutlass_fork = enabled and os.environ.get("MEGABAKE_SM100_CUTLASS_FORK", fork_default) == "1"
    if cutlass_fork and width != 256:
        raise ValueError("the CUTLASS persistent fork requires 256 threads")
    if enabled and width == 128 and stages != 1:
        raise ValueError("the 128-thread persistent variant requires one accumulator stage")
    return PersistentGemmSettings(enabled, width, stages, cutlass_fork)
