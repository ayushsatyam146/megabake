"""The staging planner: what the producer loads ahead of need.

In one persistent kernel the root edge (HBM) idles whenever the leaves do
work that moves few bytes: attention, norms, barriers. A producer that runs
ahead can use that time, but only as far as the leaf's software-managed
memory holds the bytes. This module sizes the run-ahead from the plan and the
device tree instead of a fixed page count:

    pages = min(what fits,
                what the body keeps in flight to stream at one leaf's rate
                + what the idle time before a weight-heavy interval can fill,
                  at most what that interval's busiest leaf takes)


Whether a kernel can have a producer at all is the body's business (codegen
knows which GEMM variants take pages); this module decides how deep it runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

from .costmodel import Interval, interval_time
from .device import DeviceTree
from .mapping import StagePlan

# Below this many pages the producer cannot run ahead of its consumer (X1.1).
MIN_PAGES = 8


def _stream_bytes_per_leaf(interval: Interval) -> float:
    """Bytes the busiest leaf streams through pages in an interval."""
    per_leaf = [0.0] * interval.workers
    offset = 0
    for work in interval.work:
        if work.every_worker:
            continue
        if work.paged:
            for item in range(work.items):
                per_leaf[(offset + item) % interval.workers] += work.item_bytes
        offset += work.items
    return max(per_leaf, default=0.0)


def plan_staging(
    intervals: Sequence[Interval],
    tree: DeviceTree,
    *,
    free_bytes: int,
    page_bytes: int,
    staged_bytes_per_page: int,
) -> StagePlan:
    """Size the producer's page ring for a plan.

    ``free_bytes`` is the leaf memory left beside the other bodies' scratch,
    ``page_bytes`` what one page occupies there, and ``staged_bytes_per_page``
    the root-edge bytes it holds (a page also carries an activation slot the
    producer does not fill)."""
    if not intervals:
        return StagePlan(reason="no intervals")
    workers = intervals[0].workers
    fit = free_bytes // page_bytes
    streams = [_stream_bytes_per_leaf(interval) for interval in intervals]
    if not any(streams):
        return StagePlan(reason="no interval streams weights through pages")
    # The idle time before each streaming interval: the non-streaming
    # intervals that precede it, plus one synchronization.
    share = tree.read_share(workers)
    idle_s = 0.0
    wanted = 0.0
    for interval, stream in zip(intervals, streams):
        time = interval_time(interval, tree)
        if stream:
            # Run ahead by what the idle time before the interval can fill,
            # at most what the interval's busiest leaf will take.
            wanted = max(wanted, min((idle_s + time.sync_s) * share, stream))
            idle_s = 0.0
        else:
            idle_s += time.total_s
    # The ring is also the body's pipeline: to stream at the rate one leaf
    # can reach, it keeps that rate times the copy's latency in flight.
    in_flight = tree.read_share(1) * tree.mechanism().latency_s
    pages = ceil((in_flight + wanted) / staged_bytes_per_page)
    pages += pages % 2  # paired stages take two pages
    pages = min(pages, fit - fit % 2)
    if pages < MIN_PAGES:
        if fit < MIN_PAGES:
            return StagePlan(reason=f"only {fit} pages fit beside the other bodies' scratch")
        pages = MIN_PAGES
    staged = pages * staged_bytes_per_page
    return StagePlan(
        producer=True,
        ring_pages=pages,
        page_bytes=page_bytes,
        reason=(
            f"{in_flight / 1024:.0f} KiB in flight + {wanted / 1024:.0f} KiB run-ahead per leaf, "
            f"{fit} pages fit, "
            f"staging depth {tree.staging_depth_s(reserved_bytes=tree.leaf.capacity_bytes - staged) * 1e6:.1f} us"
        ),
    )
