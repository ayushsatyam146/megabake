from __future__ import annotations

import pytest
import torch

from megabake import make_backend
from megabake.fx_handler.transformers import StatefulAttention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="task 28 requires an SM90 CUDA device",
)


def test_task28_compiles_prefill_then_two_advancing_decode_calls(tmp_path):
    torch.manual_seed(28)
    device = torch.device("cuda")
    source = torch.randn(1, 1, 2, 8, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 2, 8, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 2, 8, device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(1, 3, 2, 8, device=device, dtype=torch.bfloat16)
    value_cache = torch.randn(1, 3, 2, 8, device=device, dtype=torch.bfloat16)
    valid_length = torch.tensor([1], device=device, dtype=torch.int32)
    inputs = (source, key, value, key_cache, value_cache, valid_length)

    expected_key_cache = key_cache.clone()
    expected_value_cache = value_cache.clone()
    expected_length = valid_length.clone()
    expected_inputs = (
        source,
        key,
        value,
        expected_key_cache,
        expected_value_cache,
        expected_length,
    )

    model = StatefulAttention().eval()
    backend = make_backend(
        mode="decode",
        compile_kernel=True,
        cache=False,
        dump_dir=tmp_path,
        scheduler="phase",
    )
    optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)

    with torch.no_grad():
        first = optimized(*inputs)
        second = optimized(source, key * 0.5, value * 0.5, *inputs[3:])
        expected_first = model(*expected_inputs)
        expected_second = model(source, key * 0.5, value * 0.5, *expected_inputs[3:])

    torch.testing.assert_close(first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(second, expected_second, rtol=0, atol=0)
    torch.testing.assert_close(key_cache, expected_key_cache, rtol=0, atol=0)
    torch.testing.assert_close(value_cache, expected_value_cache, rtol=0, atol=0)
    torch.testing.assert_close(valid_length, expected_length, rtol=0, atol=0)
    assert int(valid_length.item()) == 3

    compiled = backend.megabake_last_compiled
    assert compiled.plan.selection.selections[0].body.name == ("attention_stateful_decode_rows")
    assert compiled.plan.storage.state[0].capacity == 3
    assert compiled.plan.storage.state[0].lifetime == "boundary_state"
    assert compiled.generated.source_path.exists()
