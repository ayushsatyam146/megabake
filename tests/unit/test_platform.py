from __future__ import annotations

import pytest

from megabake import platform


def test_backend_availability_flags_are_boolean() -> None:
    assert isinstance(platform.HAS_CUDA, bool)
    assert isinstance(platform.HAS_CUTLASS, bool)


def test_unavailable_backend_proxy_preserves_decorated_callable() -> None:
    proxy = platform._UnavailableBackendProxy("cute.jit")

    def function() -> int:
        return 1

    assert proxy(function) is function


def test_require_cuda_and_cutlass_explains_optional_extra(monkeypatch) -> None:
    monkeypatch.setattr(platform, "HAS_CUDA", False)
    monkeypatch.setattr(platform, "HAS_CUTLASS", False)

    with pytest.raises(RuntimeError, match=r"uv sync --extra cuda"):
        platform.require_cuda_and_cutlass("kernel compilation")
