"""Environment checks behind `megabake doctor`."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .frontend import PINNED_TORCH, SUPPORTED_CAPABILITIES, torch_version

Status = Literal["ok", "warn", "fail"]

SUPPORTED_PYTHON = ((3, 12), (3, 14))
REQUIRED_CUDA_TOOLKIT = (13, 0)


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


def _check_python() -> Check:
    version = sys.version_info[:2]
    low, high = SUPPORTED_PYTHON
    detail = f"{sys.version.split()[0]} ({sys.executable})"
    if low <= version <= high:
        return Check("python", "ok", detail)
    return Check("python", "fail", f"{detail}; need {low[0]}.{low[1]}-{high[0]}.{high[1]}")


def _check_torch() -> Check:
    detail = torch.__version__
    if torch_version() == PINNED_TORCH:
        return Check("torch", "ok", detail)
    return Check("torch", "fail", f"{detail}; need {PINNED_TORCH[0]}.{PINNED_TORCH[1]}")


def _check_torch_cuda(required: Status) -> Check:
    if torch.version.cuda is None:
        return Check("torch CUDA build", required, "CPU-only build")
    return Check("torch CUDA build", "ok", f"CUDA {torch.version.cuda}")


def _check_gpu(required: Status) -> Check:
    if not torch.cuda.is_available():
        return Check("GPU", required, "no CUDA device visible")
    index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    detail = f"{torch.cuda.get_device_name(index)} (sm_{capability[0]}{capability[1]})"
    if capability in SUPPORTED_CAPABILITIES:
        return Check("GPU", "ok", detail)
    supported = ", ".join(f"sm_{major}{minor}" for major, minor in sorted(SUPPORTED_CAPABILITIES))
    return Check("GPU", required, f"{detail}; megakernels need one of {supported}")


def _find_nvcc() -> str | None:
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("nvcc")


def _check_nvcc(required: Status) -> Check:
    nvcc = _find_nvcc()
    if nvcc is None:
        return Check("nvcc", required, "not found (set CUDA_HOME or add it to PATH)")
    try:
        output = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return Check("nvcc", required, f"{nvcc} failed to run: {error}")
    match = re.search(r"release (\d+)\.(\d+)", output)
    if match is None:
        return Check("nvcc", required, f"{nvcc}: cannot parse version")
    version = (int(match.group(1)), int(match.group(2)))
    detail = f"{version[0]}.{version[1]} ({nvcc})"
    if version == REQUIRED_CUDA_TOOLKIT:
        return Check("nvcc", "ok", detail)
    need = f"{REQUIRED_CUDA_TOOLKIT[0]}.{REQUIRED_CUDA_TOOLKIT[1]}"
    return Check("nvcc", required, f"{detail}; need {need}")


def _check_cutlass(required: Status) -> Check:
    from . import platform

    if platform.HAS_CUTLASS:
        version = getattr(platform.cutlass, "__version__", "unknown version")
        return Check("CUTLASS DSL", "ok", str(version))
    return Check(
        "CUTLASS DSL", required, "not installed (Linux only; installed with megabake there)"
    )


def run_checks(*, require_cuda: bool = False) -> list[Check]:
    """Check the environment. CUDA checks warn unless `require_cuda` is set."""
    required: Status = "fail" if require_cuda else "warn"
    return [
        _check_python(),
        _check_torch(),
        _check_torch_cuda(required),
        _check_gpu(required),
        _check_nvcc(required),
        _check_cutlass(required),
    ]


def format_checks(checks: list[Check]) -> str:
    width = max(len(check.name) for check in checks)
    return "\n".join(
        f"[{check.status:>4}] {check.name:<{width}}  {check.detail}" for check in checks
    )
