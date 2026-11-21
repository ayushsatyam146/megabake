"""megabake.platform
=================
Platform and hardware driver abstractions (CUDA, CUTLASS, and future backends).
Follows PyTorch Inductor and vLLM conventions for optional backend handling.
"""

from __future__ import annotations

from typing import Any

# Availability flags
HAS_CUDA: bool = False
HAS_CUTLASS: bool = False


class _UnavailableBackend(type):
    """Metaclass that allows safe class definitions and type annotations."""

    def __getattr__(cls, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _UnavailableBackendProxy(f"{cls.__name__}.{name}")


class _UnavailableBackendProxy(metaclass=_UnavailableBackend):
    """Proxy object that safely ignores decorators and type lookups on CPU."""

    def __init__(self, name: str = "backend", *args: Any, **kwargs: Any) -> None:
        self._name = name

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _UnavailableBackendProxy(f"{self._name}.{name}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # If used as a decorator (e.g. @cute.jit), return the original function unchanged
        if args and callable(args[0]):
            return args[0]
        return _UnavailableBackendProxy(f"{self._name}()")


# Import hardware drivers or fall back to clean proxies
try:
    import cuda.bindings.driver as cuda

    HAS_CUDA = True
except ImportError:
    cuda = _UnavailableBackendProxy("cuda")

try:
    import cutlass
    import cutlass.utils.hopper_helpers as sm90_utils
    from cutlass import cute, pipeline, utils
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import from_dlpack
    from cutlass.cutlass_dsl import dsl_user_op
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
    from cutlass.pipeline.helpers import MbarrierArray
    from cutlass.tensor_utils import TensorMapManager, TensorMapUpdateMode

    HAS_CUTLASS = True
except ImportError:
    cutlass = _UnavailableBackendProxy("cutlass")
    cute = _UnavailableBackendProxy("cute")
    sm90_utils = _UnavailableBackendProxy("sm90_utils")
    pipeline = _UnavailableBackendProxy("pipeline")
    utils = _UnavailableBackendProxy("utils")
    dsl_user_op = _UnavailableBackendProxy("dsl_user_op")
    from_dlpack = _UnavailableBackendProxy("from_dlpack")
    pipeline_init_arrive = None
    pipeline_init_wait = None
    TensorMapManager = _UnavailableBackendProxy("TensorMapManager")
    TensorMapUpdateMode = _UnavailableBackendProxy("TensorMapUpdateMode")
    llvm = _UnavailableBackendProxy("llvm")
    ir = _UnavailableBackendProxy("ir")
    MbarrierArray = _UnavailableBackendProxy("MbarrierArray")


def require_cuda_and_cutlass(feature: str = "This operation") -> None:
    """Raise a helpful error if physical CUDA execution is attempted without hardware."""
    if not (HAS_CUDA and HAS_CUTLASS):
        raise RuntimeError(
            f"{feature} requires an NVIDIA GPU and the CUTLASS compiler toolchain.\n"
            f"If you are on an NVIDIA machine, install them via: uv sync --extra cuda"
        )


__all__ = [
    "HAS_CUDA",
    "HAS_CUTLASS",
    "MbarrierArray",
    "TensorMapManager",
    "TensorMapUpdateMode",
    "cuda",
    "cute",
    "cutlass",
    "dsl_user_op",
    "from_dlpack",
    "ir",
    "llvm",
    "pipeline",
    "pipeline_init_arrive",
    "pipeline_init_wait",
    "require_cuda_and_cutlass",
    "sm90_utils",
    "utils",
]
