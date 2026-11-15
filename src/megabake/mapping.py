"""What the planner decides, as data.

A plan has two layers of decisions:

- ``RegionMapping``: how one region runs on a leaf. This is a LoopTree point
  restricted to what its body can do: the tile each rank gets at the leaf,
  the loop order above it, which ranks are split across leaves, and where
  each tensor's tile is held.
- ``KernelMapping``: what only a single kernel has. How each interval hands
  on its results, what the producer stages ahead of need, and what the cost
  model expects of the whole plan.

Two partial mappings can be joined along a shared tensor only when their
``EdgeFormat``s agree (the compatibility key of Andrulis et al., 2026).

``KernelPlan.mapping`` carries a ``KernelMapping``; codegen reads its staging
decision and reports print the rest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import ceil, prod
from typing import Any

from .einsum import EinsumView, einsum_view, footprint_bytes, traffic_bytes


@dataclass(frozen=True)
class EdgeFormat:
    """How a tensor crosses from its producer to its consumer: the level
    that holds it between them, its dtype and its memory layout. Producer and
    consumer must agree on all three, or a conversion body sits between."""

    level: str
    dtype: str
    strides: tuple[int, ...]


@dataclass(frozen=True)
class RegionMapping:
    region_id: str
    body: str
    view: EinsumView
    tile: Mapping[str, int]  # rank -> tile size at the leaf
    order: tuple[str, ...]  # tile loops, outermost first
    spatial: tuple[str, ...]  # ranks whose tiles are dealt to different leaves
    splits: int = 1  # ways the reduction is split across leaves
    stages: int = 1  # pipeline depth
    # Innermost level that holds a tile of each tensor.
    placement: Mapping[str, str] = field(default_factory=dict)

    @property
    def items(self) -> int:
        """Work items dealt to leaves."""
        return self.splits * prod(
            ceil(self.view.ranks[rank] / self.tile[rank]) for rank in self.spatial
        )

    @property
    def leaf_bytes(self) -> int:
        """Bytes one leaf holds while it runs an item."""
        return footprint_bytes(self.view, self.tile)

    @property
    def traffic_bytes(self) -> int:
        """Bytes the inputs move into the leaf under this tiling."""
        return traffic_bytes(self.view, self.tile, self.order, self.spatial)


@dataclass(frozen=True)
class StagePlan:
    """What the producer moves ahead of need, per leaf.

    ``ring_pages`` weight pages of ``page_bytes`` stay staged in the leaf's
    software-managed memory."""

    producer: bool = False
    ring_pages: int = 0
    page_bytes: int = 0
    reason: str = ""

    @property
    def bytes_per_leaf(self) -> int:
        return self.ring_pages * self.page_bytes


@dataclass(frozen=True)
class KernelMapping:
    regions: tuple[RegionMapping, ...]
    staging: StagePlan
    # (producer region, consumer region, value) -> format of the value between them.
    edges: Mapping[tuple[str, str, str], EdgeFormat] = field(default_factory=dict)
    device: str = ""
    # costmodel.PlanEstimate of the plan at ``kv_valid_length``.
    estimate: Any = None
    kv_valid_length: int | None = None
    # The DeviceTree the plan was mapped on.
    tree: Any = None

    def region(self, region_id: str) -> RegionMapping:
        return next(region for region in self.regions if region.region_id == region_id)

    def report(self) -> dict[str, Any]:
        """A readable plan point."""
        classes: dict[tuple[Any, ...], int] = {}
        for region in self.regions:
            key = (
                region.body,
                tuple(sorted(region.view.ranks.items())),
                tuple(sorted(region.tile.items())),
                region.splits,
            )
            classes[key] = classes.get(key, 0) + 1
        return {
            "device": self.device,
            "staging": {
                "producer": self.staging.producer,
                "ring_pages": self.staging.ring_pages,
                "bytes_per_leaf": self.staging.bytes_per_leaf,
                "reason": self.staging.reason,
            },
            "predicted_us": None if self.estimate is None else self.estimate.total_s * 1e6,
            "bounds_us": None
            if self.estimate is None
            else {
                name: getattr(self.estimate.bounds, name) * 1e6
                for name in ("b0", "b1", "b2", "b3")
            },
            "shape_classes": [
                {"body": body, "ranks": dict(ranks), "tile": dict(tile), "splits": splits, "count": count}
                for (body, ranks, tile, splits), count in classes.items()
            ],
        }


def region_mapping(selected: Any, level: str) -> RegionMapping:
    """The mapping a selected body implements, read from its contracts.

    A body's CTA tile is in output coordinates; ranks it does not tile are
    held whole (a GEMM streams its reduction rank through the pipeline). The
    loop order puts the output ranks outside the reduced ones, as every body
    here iterates output tiles and reduces inside them."""
    program = selected.program
    body = selected.body
    view = einsum_view(program)
    outputs = [d.axis for d in program.iteration_domains if d.axis not in program.reduction_axes]
    tile = dict(view.ranks)
    shape = body.tiles.cta_shape
    if len(shape) == len(outputs) and body.tiles.work_partition != "one output element per thread":
        tile.update({rank: min(size, view.ranks[rank]) for rank, size in zip(outputs, shape)})
    spatial = tuple(rank for rank in outputs if tile[rank] < view.ranks[rank])
    splits = 1
    if body.provider == "g0_range":
        from .bodies.cuda.sm90.gemm import split_k

        splits = split_k(body)
    return RegionMapping(
        region_id=selected.region_id,
        body=body.name,
        view=view,
        tile=tile,
        order=(*outputs, *program.reduction_axes),
        spatial=spatial,
        splits=splits,
        stages=body.pipeline.stage_count,
        placement={tensor.name: level for tensor in view.tensors},
    )


def edge_formats(selection: Any, level: str) -> dict[tuple[str, str, str], EdgeFormat]:
    """The format of every value one selected body hands to another: the
    producer's output layout, held at ``level`` (global memory today; a body
    that keeps an intermediate in registers or shared memory fuses it and
    has no edge)."""
    producers = {
        selected.program.store.value_id: selected for selected in selection.selections
    }
    edges: dict[tuple[str, str, str], EdgeFormat] = {}
    for consumer in selection.selections:
        for value_id in consumer.inputs:
            producer = producers.get(value_id)
            if producer is None or producer is consumer:
                continue
            layout = producer.body.layouts.outputs.get(value_id)
            if layout is None:
                continue
            edges[(producer.region_id, consumer.region_id, value_id)] = EdgeFormat(
                level, layout.dtype, tuple(layout.strides)
            )
    return edges
