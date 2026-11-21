from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
GPU_TESTS = Path(__file__).resolve().parent / "gpu"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from megabake.platform import HAS_CUDA, HAS_CUTLASS


def pytest_ignore_collect(collection_path: Path, *args, **kwargs) -> bool | None:
    if collection_path.is_relative_to(GPU_TESTS) and (
        not HAS_CUDA or not HAS_CUTLASS or not torch.cuda.is_available()
    ):
        return True
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if Path(str(item.path)).is_relative_to(GPU_TESTS):
            item.add_marker(pytest.mark.gpu)
