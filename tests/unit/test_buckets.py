import pytest
import torch

from megabake.buckets import PaddingDependence, bucketed


class CausalMix(torch.nn.Module):
    """Per-token work and a causal prefix sum: padding after the real tokens
    cannot reach them."""

    def forward(self, tokens, mask):
        hidden = tokens.float() * 0.5 + mask.float()
        return {"logits": torch.cumsum(hidden, dim=1)}


class SequenceSum(torch.nn.Module):
    """Every position reads every other: padding changes the outputs."""

    def forward(self, tokens, mask):
        hidden = tokens.float() + mask.float()
        return hidden + hidden.sum(dim=1, keepdim=True)


def _call(module, length, **options):
    tokens = torch.arange(length).view(1, length)
    mask = torch.ones_like(tokens)
    wrapper = bucketed(
        module,
        backend="eager",
        sequence_args={"tokens": 1, "mask": 1},
        pad_values={"tokens": 0, "mask": 0},
        **options,
    )
    return wrapper, wrapper(tokens=tokens, mask=mask), module(tokens, mask)


def test_bucketed_pads_to_the_bucket_and_cuts_outputs_back():
    wrapper, result, expected = _call(CausalMix(), 5, buckets=(4, 8, 16))

    assert wrapper.bucket_for(5) == 8
    assert result["logits"].shape == (1, 5)
    torch.testing.assert_close(result["logits"], expected["logits"])
    assert wrapper.verified == {8}


def test_bucketed_rejects_outputs_that_depend_on_padding():
    with pytest.raises(PaddingDependence):
        _call(SequenceSum(), 5, buckets=(8,))


class SequenceLength(CausalMix):
    """An output that reads the padded length itself."""

    def forward(self, tokens, mask):
        return {**super().forward(tokens, mask), "length": torch.tensor(tokens.shape[1])}


def test_bucketed_rejects_outputs_that_read_the_padded_shape():
    with pytest.raises(PaddingDependence, match="differs from unpadded"):
        _call(SequenceLength(), 5, buckets=(8,))


def test_bucketed_rejects_lengths_past_the_largest_bucket():
    wrapper = bucketed(CausalMix(), backend="eager", sequence_args={"tokens": 1}, buckets=(4,))
    with pytest.raises(ValueError, match="largest bucket"):
        wrapper.bucket_for(5)
