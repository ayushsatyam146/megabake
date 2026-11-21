from __future__ import annotations

import torch

from benchmarks.workloads import transformer_block


def test_transformer_block_workload_covers_prefill_and_decode():
    prefill_model, prefill_inputs = transformer_block("prefill", 1, 2, device="cpu")
    assert prefill_model.training is False
    assert prefill_inputs[0].shape == (1, 2, 576)
    assert prefill_inputs[-1].shape == (1, 1, 2, 2)

    decode_model, decode_inputs = transformer_block("decode", 1, 1, device="cpu")
    with torch.no_grad():
        outputs = decode_model(*decode_inputs)
    assert outputs[0].shape == (1, 1, 576)
    assert int(decode_inputs[-1].item()) == 1
