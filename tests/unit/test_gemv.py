"""The batch-1 decode GEMV body (bodies/cuda/sm90/gemv.py): which GEMMs it
takes, how it tiles them, and what source it emits. CPU only."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from megabake.bodies.cuda.sm90.gemv import GemvBodyProvider, emit_gemv_body, gemv_config, gemv_rejection, gemv_rows
from megabake.bodies.interface import SelectionPolicy, TargetFacts
from megabake.ir import Accumulator, Cast, IndexedLoad, IterationDomain, TensorProgram, TensorStore


def _target(sms: int = 132) -> TargetFacts:
    return TargetFacts("sm90a", 32, 1024, 232_448, 255, sms, True, True)


def _policy(mode: str = "decode") -> SelectionPolicy:
    return SelectionPolicy(mode=mode, table_launch=True)


def _program(rows: int, columns: int, k: int, dtype: str = "torch.bfloat16") -> TensorProgram:
    return TensorProgram(
        family="contraction", source_target="mm", decomposition="",
        iteration_domains=(IterationDomain("o0", rows), IterationDomain("o1", columns), IterationDomain("r0", k)),
        reduction_axes=("r0",),
        loads=(
            IndexedLoad("load:0", "a", ("o0", "r0"), (k, 1), dtype, (rows, k)),
            IndexedLoad("load:1", "b", ("r0", "o1"), (1, k), dtype, (k, columns)),
        ),
        init=None, combine=None,
        post=Cast(Accumulator("torch.float32"), "torch.bfloat16", "round_nearest_even"),
        accumulator_dtype="torch.float32",
        store=TensorStore("c", (rows, columns), "torch.bfloat16", (columns, 1), True),
        predicates=(),
    )


def _region(program: TensorProgram):
    return SimpleNamespace(id="region:mm", source_op_ids=("op:mm",), tensor_programs=(program,))


def test_gemv_takes_narrow_one_row_decode_gemms():
    # Llama-3.2-1B's q, k/v, o and down projections on a 132-SM H200.
    for columns, k in ((2048, 2048), (512, 2048), (2048, 8192)):
        assert gemv_rejection(_program(1, columns, k), _target(), _policy()) is None


def test_gemv_leaves_wide_prefill_and_non_bf16_gemms_alone():
    assert "fill half the SMs" in gemv_rejection(_program(1, 128256, 2048), _target(), _policy())
    # Beam DFlash2's 5,120-wide projections: 80 of 132 SMs already busy.
    assert "fill half the SMs" in gemv_rejection(_program(1, 5120, 5120), _target(), _policy())
    assert "decode" in gemv_rejection(_program(1, 2048, 2048), _target(), _policy("prefill"))
    assert "one output row" in gemv_rejection(_program(16, 2048, 2048), _target(), _policy())
    assert "BF16" in gemv_rejection(_program(1, 2048, 2048, "torch.float16"), _target(), _policy())
    assert "multiple of 8" in gemv_rejection(_program(1, 2048, 2044), _target(), _policy())


def test_gemv_tiles_one_round_over_the_sms():
    assert gemv_rows(2048, 132) == 16
    assert gemv_rows(512, 132) == 8  # the minimum
    assert gemv_rows(2048, 60) == 24  # the maximum
    assert all(gemv_rows(columns, 132) % 4 == 0 for columns in (576, 896, 1536, 2048))


def test_gemv_is_on_unless_turned_off(monkeypatch):
    region = _region(_program(1, 2048, 2048))
    monkeypatch.setenv("MEGABAKE_GEMV", "0")
    assert not GemvBodyProvider().get_candidates(region, _target(), _policy()).configurations
    monkeypatch.delenv("MEGABAKE_GEMV")
    (config,) = GemvBodyProvider().get_candidates(region, _target(), _policy()).configurations
    assert config.provider == "gemv" and config.tiles.cta_shape == (1, 16)


@pytest.mark.parametrize("k", [576, 2048, 8192])
def test_gemv_source_parses_and_covers_every_column(k):
    program = _program(1, 2048, k)
    generated = emit_gemv_body(program, gemv_config(_region(program), program, 132))
    ast.parse(generated.source)
    assert generated.tile_count * 16 >= 2048 > (generated.tile_count - 1) * 16
    assert generated.scratch_elements == (256 // 32) * 16
    assert "def run(" in generated.source and "warp_reduction_sum" in generated.source
