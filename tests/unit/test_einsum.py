from __future__ import annotations

import torch

from megabake.einsum import (
    attention_scores,
    contraction,
    einsum_view,
    footprint_bytes,
    hbl_tile,
    tile_fetches,
    traffic_bytes,
    traffic_lower_bound_bytes,
)
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.regions import form_regions


def _gemm(m: int, k: int, n: int):
    return contraction({"m": m, "k": k, "n": n}, {"A": "mk", "B": "kn"}, "mn")


def test_view_of_a_captured_gemm_has_its_ranks_reuse_and_flops():
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    inputs = (torch.randn(4, 128, dtype=torch.bfloat16), torch.randn(256, 128, dtype=torch.bfloat16))
    prepared = prepare_fx_graph(torch.export.export(Model(), inputs).graph_module, inputs)
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    program = next(
        region.tensor_programs[0]
        for region in graph.regions
        if region.tensor_programs and region.tensor_programs[0].family == "contraction"
    )
    view = einsum_view(program)
    assert sorted(view.ranks.values()) == [4, 128, 256]
    assert view.points == 4 * 128 * 256
    assert view.flops == 2 * 4 * 128 * 256
    assert not view.opaque
    activation, weight = sorted(view.inputs, key=lambda tensor: tensor.elements)
    # The activation does not depend on the output column, the weight not on the row.
    assert {view.ranks[rank] for rank in view.reused_along(activation.name)} == {256}
    assert {view.ranks[rank] for rank in view.reused_along(weight.name)} == {4}
    assert view.bytes_read() == (4 * 128 + 128 * 256) * 2
    assert view.bytes_written() == 4 * 256 * 2


def test_batch_one_decode_has_no_tiling_freedom():
    # C[1,n] += A[1,k]·B[k,n]: every weight is read once whatever the tile.
    view = _gemm(1, 2048, 8192)
    weights = 2048 * 8192 * 2
    for tile in ({"n": 64}, {"n": 16}, {"n": 8192}, {"n": 64, "k": 64}):
        assert tile_fetches(view, "B", tile, "mnk") * _tile(view, "B", tile) == weights
    assert traffic_lower_bound_bytes(view, 200_000) == view.bytes_read()


def _tile(view, name, tile):
    from megabake.einsum import tile_bytes

    return tile_bytes(view, name, tile)


def test_loop_order_decides_refetches():
    view = _gemm(256, 256, 256)
    tile = {"m": 64, "k": 256, "n": 64}
    # n outside m: each activation tile is fetched again for every n tile.
    assert tile_fetches(view, "A", tile, "nmk") == 4 * 4
    # m outside n: the activation tile stays while n advances.
    assert tile_fetches(view, "A", tile, "mnk") == 4
    # The weight pays instead: one of the two is always refetched.
    assert tile_fetches(view, "B", tile, "mnk") == 4 * 4
    assert traffic_bytes(view, tile, "mnk") == (4 + 16) * 64 * 256 * 2
    assert footprint_bytes(view, tile) == (64 * 256 + 256 * 64 + 64 * 64) * 2


def test_kv_group_packing_follows_from_the_index_sets():
    # 3 KV heads, 3 query heads each, 1,000 keys, head_dim 64.
    view = attention_scores(kv_heads=3, group_size=3, keys=1000, head_dim=64)
    assert view.reused_along("K") == frozenset({"j"})
    keys_once = 3 * 1000 * 64 * 2
    # A row per query head, each row on its own CTA: every head reads the
    # group's keys ...
    per_head = {"g": 1, "j": 1}
    assert tile_fetches(view, "K", per_head, "gjtd", spatial="gj") * _tile(view, "K", per_head) == 3 * keys_once
    # ... a row per KV group reads them once.
    per_group = {"g": 1, "j": 3}
    assert tile_fetches(view, "K", per_group, "gjtd", spatial="g") * _tile(view, "K", per_group) == keys_once


def test_hbl_tile_is_square_for_matmul_and_respects_rank_sizes():
    view = _gemm(4096, 4096, 4096)
    tile = hbl_tile(view, 1 << 20)
    assert tile == {"m": 1024, "k": 1024, "n": 1024}
    # A one-row GEMM cannot spend its buffer on m.
    assert hbl_tile(_gemm(1, 4096, 4096), 1 << 20)["m"] == 1


def test_lower_bound_grows_as_the_buffer_shrinks_only_when_reuse_exists():
    square = _gemm(4096, 4096, 4096)
    small, large = (traffic_lower_bound_bytes(square, size) for size in (1 << 16, 1 << 24))
    assert small > large >= square.bytes_read()
    decode = _gemm(1, 4096, 4096)
    assert traffic_lower_bound_bytes(decode, 1 << 16) == decode.bytes_read()
