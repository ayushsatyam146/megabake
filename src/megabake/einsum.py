"""An einsum view of tensor programs: which ranks index which tensor.

A ``TensorProgram`` already is an extended einsum: its iteration domains are
ranks, its reduction axes are the contracted ranks, and each load names the
ranks that index it. ``EinsumView`` keeps only that, so reuse, traffic and
lower bounds follow from index sets and not from operator names:

- a tensor is *reused* along every rank that does not index it;
- with a tile per rank and a loop order, the bytes fetched from the parent
  level follow from which loops sit outside a tensor's own ranks;
- the HBL linear program bounds the traffic any tiling needs for a given
  buffer size (Christ, Demmel, Knight, Scanlan, Yelick, 2013).

Data-dependent accesses (gathers) are opaque: they have no reuse to plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import combinations
from math import ceil, log, prod

from .graph import DTYPE_BYTES
from .ir import Accumulator, AxisAffine, DataIndex, TensorProgram


@dataclass(frozen=True)
class EinsumTensor:
    name: str
    ranks: frozenset[str]
    item_bytes: int
    # Elements of the whole tensor that the einsum touches.
    elements: int
    output: bool = False


@dataclass(frozen=True)
class EinsumView:
    ranks: Mapping[str, int]  # rank -> size
    tensors: tuple[EinsumTensor, ...]
    reduced: frozenset[str]
    # Multiply-accumulate chains evaluated per iteration point (2 for a paired
    # contraction, 0 for work that is not a contraction).
    accumulators: int = 0
    opaque: bool = False

    @property
    def inputs(self) -> tuple[EinsumTensor, ...]:
        return tuple(tensor for tensor in self.tensors if not tensor.output)

    @property
    def output(self) -> EinsumTensor:
        return next(tensor for tensor in self.tensors if tensor.output)

    @property
    def points(self) -> int:
        """Iteration points: one per combination of rank values."""
        return prod(self.ranks.values())

    @property
    def flops(self) -> int:
        return 2 * self.points * self.accumulators

    def tensor(self, name: str) -> EinsumTensor:
        return next(tensor for tensor in self.tensors if tensor.name == name)

    def reused_along(self, name: str) -> frozenset[str]:
        """Ranks a tensor does not depend on: one tile serves every value."""
        return frozenset(self.ranks) - self.tensor(name).ranks

    def bytes_read(self) -> int:
        """Bytes of the distinct inputs, each read once."""
        return sum(tensor.elements * tensor.item_bytes for tensor in self.inputs)

    def bytes_written(self) -> int:
        return self.output.elements * self.output.item_bytes


def _item_bytes(dtype: str) -> int:
    return DTYPE_BYTES.get(dtype, 0)


def _ranks_of(axes: Sequence[object]) -> frozenset[str]:
    names: set[str] = set()
    for axis in axes:
        if isinstance(axis, str):
            names.add(axis)
        elif isinstance(axis, AxisAffine):
            names.update(term.axis for term in axis.terms)
    return frozenset(names)


def _accumulators(program: TensorProgram) -> int:
    """Accumulators a contraction's post expression reads (1 if none named)."""
    if program.family != "contraction":
        return 0
    found: set[int] = set()
    stack = [program.post, program.combine]
    while stack:
        node = stack.pop()
        if isinstance(node, Accumulator):
            found.add(node.index)
        for name in ("operand", "lhs", "rhs", "condition", "true_value", "false_value"):
            child = getattr(node, name, None)
            if child is not None and not isinstance(child, (str, int, float)):
                stack.append(child)
    return max(1, len(found))


def einsum_view(program: TensorProgram) -> EinsumView:
    """The view of one tensor program. Loads of the same value with the same
    ranks are one tensor (a norm reads its row twice, once per role)."""
    ranks = {domain.axis: domain.size for domain in program.iteration_domains}
    tensors: dict[tuple[str, frozenset[str]], EinsumTensor] = {}
    opaque = False
    for load in program.loads:
        if any(isinstance(axis, DataIndex) for axis in load.axes):
            opaque = True
        indexed = _ranks_of(load.axes)
        dense = prod(ranks[rank] for rank in indexed)
        # An affine window (stride, group map) can touch fewer elements than
        # the product of its ranks; the source shape bounds it.
        elements = min(dense, prod(load.source_shape)) if load.source_shape else dense
        key = (load.value_id, indexed)
        if key not in tensors:
            tensors[key] = EinsumTensor(
                load.value_id, indexed, _item_bytes(load.dtype), max(1, elements)
            )
    output_ranks = frozenset(ranks) - frozenset(program.reduction_axes)
    output = EinsumTensor(
        program.store.value_id,
        output_ranks,
        _item_bytes(program.store.dtype),
        max(1, prod(program.store.shape)),
        output=True,
    )
    return EinsumView(
        ranks=ranks,
        tensors=(*tensors.values(), output),
        reduced=frozenset(program.reduction_axes),
        accumulators=_accumulators(program),
        opaque=opaque,
    )


def contraction(
    ranks: Mapping[str, int],
    inputs: Mapping[str, Sequence[str]],
    output: Sequence[str],
    item_bytes: int = 2,
) -> EinsumView:
    """An einsum written by hand: ``contraction({"m": 4, "k": 8, "n": 8},
    {"A": "mk", "B": "kn"}, "mn")``."""
    tensors = tuple(
        EinsumTensor(name, frozenset(indexed), item_bytes, prod(ranks[r] for r in indexed))
        for name, indexed in inputs.items()
    )
    result = EinsumTensor(
        "out", frozenset(output), item_bytes, prod(ranks[r] for r in output), output=True
    )
    return EinsumView(
        ranks=dict(ranks),
        tensors=(*tensors, result),
        reduced=frozenset(ranks) - frozenset(output),
        accumulators=1,
    )


def attention_scores(
    kv_heads: int, group_size: int, keys: int, head_dim: int, item_bytes: int = 2
) -> EinsumView:
    """Decode attention scores as an einsum: ``S[g,j,t] = Q[g,j,d]·K[g,t,d]``
    with ``g`` KV heads, ``j`` query heads per KV head, ``t`` keys. The value
    sum ``O[g,j,d] = P[g,j,t]·V[g,t,d]`` has the same index structure.

    ``K`` has no ``j``: a KV group's query heads can share one read of it."""
    return contraction(
        {"g": kv_heads, "j": group_size, "t": keys, "d": head_dim},
        {"Q": "gjd", "K": "gtd"},
        "gjt",
        item_bytes,
    )


# --------------------------------------------------------------------------
# Traffic of a tiling


def tile_bytes(view: EinsumView, name: str, tile: Mapping[str, int]) -> int:
    """Footprint of one tile of a tensor."""
    tensor = view.tensor(name)
    return tensor.item_bytes * prod(min(tile.get(r, view.ranks[r]), view.ranks[r]) for r in tensor.ranks)


def tile_fetches(
    view: EinsumView,
    name: str,
    tile: Mapping[str, int],
    order: Sequence[str],
    spatial: Sequence[str] = (),
) -> int:
    """How many tiles of a tensor are fetched when a buffer holds one tile
    of it.

    ``order`` lists the tile loops from outermost to innermost. A tensor's
    tile changes with its own ranks' loops, and it is fetched again for every
    iteration of a loop it does not depend on that sits *outside* one of its
    own loops. Loops inside all of its ranks reuse the tile.

    ``spatial`` ranks are dealt to different leaves, which share no buffer:
    each leaf fetches its own copy, whatever the order."""
    tensor = view.tensor(name)
    steps = {rank: ceil(view.ranks[rank] / min(tile.get(rank, size), size))
             for rank, size in view.ranks.items()}
    # A rank whose tile spans it has one step: no loop.
    own = [index for index, rank in enumerate(order) if rank in tensor.ranks and steps[rank] > 1]
    innermost_own = max(own, default=-1)
    count = 1
    for index, rank in enumerate(order):
        if rank in tensor.ranks or index < innermost_own or rank in spatial:
            count *= steps[rank]
    return count


def traffic_bytes(
    view: EinsumView,
    tile: Mapping[str, int],
    order: Sequence[str],
    spatial: Sequence[str] = (),
) -> int:
    """Bytes the inputs move from the parent level under a tiling."""
    return sum(
        tile_fetches(view, tensor.name, tile, order, spatial) * tile_bytes(view, tensor.name, tile)
        for tensor in view.inputs
    )


def footprint_bytes(view: EinsumView, tile: Mapping[str, int]) -> int:
    """Bytes of one tile of every tensor: what the level must hold."""
    return sum(tile_bytes(view, tensor.name, tile) for tensor in view.tensors)


# --------------------------------------------------------------------------
# Lower bound (HBL)


def _maximize(objective: Sequence[int], rows: Sequence[Sequence[int]], limits: Sequence[Fraction]):
    """max objective·x subject to rows·x <= limits and x >= 0, by vertex
    enumeration with exact fractions. Fine for the handful of ranks an
    einsum has."""
    count = len(objective)
    constraints = [(list(row), limit) for row, limit in zip(rows, limits)]
    constraints += [([-1 if i == j else 0 for j in range(count)], Fraction(0)) for i in range(count)]
    best = None
    for chosen in combinations(range(len(constraints)), count):
        matrix = [constraints[index][0][:] + [constraints[index][1]] for index in chosen]
        solution = _solve(matrix)
        if solution is None:
            continue
        if all(
            sum(a * x for a, x in zip(row, solution)) <= limit
            for row, limit in constraints
        ):
            value = sum(c * x for c, x in zip(objective, solution))
            if best is None or value > best[0]:
                best = (value, solution)
    return best


def _solve(matrix: list[list[Fraction]]) -> list[Fraction] | None:
    """Gaussian elimination on an augmented square system; None if singular."""
    size = len(matrix)
    matrix = [[Fraction(value) for value in row] for row in matrix]
    for column in range(size):
        pivot = next((r for r in range(column, size) if matrix[r][column] != 0), None)
        if pivot is None:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [value / scale for value in matrix[column]]
        for row in range(size):
            if row != column and matrix[row][column] != 0:
                factor = matrix[row][column]
                matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[column])]
    return [row[size] for row in matrix]


def hbl_tile(view: EinsumView, buffer_elements: int) -> dict[str, int]:
    """The tile shape that does the most iteration points while every tensor's
    tile fits in ``buffer_elements``: rank ``i`` gets ``buffer ** x_i`` where
    ``x`` maximizes ``Σ x_i`` subject to ``Σ_{i in tensor} x_i <= 1`` for each
    tensor and ``x_i <= log(size_i) / log(buffer)``. This is the dual HBL
    program with loop bounds (Dinh and Demmel, 2020)."""
    return dict(
        _hbl_tile(
            tuple(view.ranks.items()),
            tuple(tuple(sorted(tensor.ranks)) for tensor in view.tensors),
            buffer_elements,
        )
    )


@cache
def _hbl_tile(
    ranks: tuple[tuple[str, int], ...],
    tensors: tuple[tuple[str, ...], ...],
    buffer_elements: int,
) -> tuple[tuple[str, int], ...]:
    # A model repeats a few shapes once per layer: solved once each.
    names = [name for name, _ in ranks]
    sizes = dict(ranks)
    if buffer_elements < 2:
        return tuple((name, 1) for name in names)
    rows = [[1 if name in indexed else 0 for name in names] for indexed in tensors]
    limits = [Fraction(1)] * len(rows)
    for index, name in enumerate(names):
        rows.append([1 if other == index else 0 for other in range(len(names))])
        bound = log(sizes[name]) / log(buffer_elements)
        limits.append(Fraction(min(bound, 1.0)).limit_denominator(10_000))
    best = _maximize([1] * len(names), rows, limits)
    exponents = best[1] if best else [Fraction(0)] * len(names)
    return tuple(
        (name, max(1, min(sizes[name], round(buffer_elements ** float(x)))))
        for name, x in zip(names, exponents)
    )


def traffic_lower_bound_bytes(view: EinsumView, buffer_bytes: int) -> int:
    """Bytes any execution of the einsum moves into a buffer of
    ``buffer_bytes``.

    Every input is read at least once. Beyond that, a stretch of execution
    that moves ``M`` elements has at most ``2M`` available, so it does at
    most ``F(2M)`` iteration points, ``F`` being the HBL tile volume; the
    einsum needs ``floor(points / F(2M))`` such stretches."""
    read_once = view.bytes_read()
    item = max((tensor.item_bytes for tensor in view.tensors), default=1) or 1
    elements = buffer_bytes // item
    if view.opaque or elements < 2:
        return read_once
    tile = hbl_tile(view, 2 * elements)
    volume = prod(tile.values())
    segments = view.points // max(1, volume)
    return max(read_once, segments * elements * item)
