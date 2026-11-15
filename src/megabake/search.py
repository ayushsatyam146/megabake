"""Choosing among mappings: prune, then join.

The planner's choices come in *shape classes*. A decoder repeats a few
(body, shape) classes many times, so each class is decided once. A class
offers ``Candidate``s; the kernel needs one per class.

Classes are not independent: every body of one kernel shares the leaf's
software-managed memory and its registers, so the fastest candidate of one
class can make the kernel illegal or force a slower choice elsewhere. The
search therefore keeps, per class, the candidates no other candidate beats
on time *and* on every shared reservation (a Pareto set), and joins classes
with a small dynamic program that keeps Pareto-optimal partial kernels. This
is the pruning rule of the Fast and Fusiest mapper (Andrulis et al., 2026);
at this size it is exact.

A candidate that is only slightly faster than the class's default is not
taken: the model's error is larger than that (``MARGIN``).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

# A candidate must beat the default by this fraction of the default's time
# to be chosen. The cost model ranks plans within about this error.
MARGIN = 0.05


@dataclass(frozen=True)
class Candidate:
    choice: Hashable  # what the body is told, e.g. a split count
    time_s: float
    # Shared reservations this choice needs at the leaf (bytes, registers):
    # the kernel needs the largest any body asks for.
    reservations: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ShapeClass:
    """Regions that run the same body on the same shape. The first candidate
    is the default: what the body does when nothing is decided."""

    key: Hashable
    count: int
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class Joined:
    time_s: float
    reservations: Mapping[str, float]
    choices: Mapping[Hashable, Hashable]  # class key -> chosen candidate


def _dominates(a_time: float, a_res: Mapping[str, float], b_time: float, b_res: Mapping[str, float]) -> bool:
    keys = set(a_res) | set(b_res)
    return a_time <= b_time and all(a_res.get(key, 0.0) <= b_res.get(key, 0.0) for key in keys)


def pareto(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    """Candidates that no other one beats on time and every reservation.
    Among equals the earlier one stays."""
    kept: list[Candidate] = []
    for index, candidate in enumerate(candidates):
        beaten = any(
            _dominates(other.time_s, other.reservations, candidate.time_s, candidate.reservations)
            and (
                other_index < index
                or other.time_s < candidate.time_s
                or dict(other.reservations) != dict(candidate.reservations)
            )
            for other_index, other in enumerate(candidates)
            if other_index != index
        )
        if not beaten:
            kept.append(candidate)
    return tuple(kept)


def _with_margin(shape: ShapeClass, margin: float) -> tuple[Candidate, ...]:
    """Drop candidates that do not beat the default by ``margin``."""
    default = shape.candidates[0]
    return (default,) + tuple(
        candidate
        for candidate in shape.candidates[1:]
        if candidate.time_s < default.time_s * (1.0 - margin)
    )


def join(
    classes: Sequence[ShapeClass],
    capacity: Mapping[str, float] | None = None,
    margin: float = MARGIN,
) -> Joined | None:
    """The fastest kernel that takes one candidate per class and keeps every
    shared reservation within ``capacity``. None if no combination fits."""
    capacity = capacity or {}
    partial: list[Joined] = [Joined(0.0, {}, {})]
    for shape in classes:
        options = pareto(_with_margin(shape, margin))
        grown: list[Joined] = []
        for state in partial:
            for candidate in options:
                reservations = {
                    key: max(state.reservations.get(key, 0.0), candidate.reservations.get(key, 0.0))
                    for key in set(state.reservations) | set(candidate.reservations)
                }
                if any(reservations.get(key, 0.0) > limit for key, limit in capacity.items()):
                    continue
                grown.append(
                    Joined(
                        state.time_s + shape.count * candidate.time_s,
                        reservations,
                        {**state.choices, shape.key: candidate.choice},
                    )
                )
        # Keep the partial kernels nothing else beats on time and reservations.
        partial = [
            state
            for index, state in enumerate(grown)
            if not any(
                _dominates(other.time_s, other.reservations, state.time_s, state.reservations)
                and (other_index < index or other.time_s < state.time_s)
                for other_index, other in enumerate(grown)
                if other_index != index
            )
        ]
        if not partial:
            return None
    return min(partial, key=lambda state: state.time_s)


def best_of(alternatives: Mapping[Hashable, Joined | None], margin: float = MARGIN) -> Hashable:
    """Among launch-wide alternatives (the first is the default), the one
    whose joined kernel is fastest by more than ``margin``."""
    names = list(alternatives)
    default = names[0]
    best = default
    for name in names[1:]:
        joined = alternatives[name]
        current = alternatives[best]
        if joined is None:
            continue
        if current is None or joined.time_s < current.time_s * (1.0 - margin):
            best = name
    return best
