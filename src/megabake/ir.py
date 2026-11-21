"""MegaBake's tensor-program IR: the per-element programs lowering produces,
regions fuse, and device bodies implement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IterationDomain:
    axis: str
    size: int


@dataclass(frozen=True)
class AxisPredicate:
    axis: str
    upper_bound: int


@dataclass(frozen=True)
class AxisTerm:
    axis: str
    scale: int = 1
    divisor: int = 1
    modulus: int = 0


@dataclass(frozen=True)
class AxisAffine:
    terms: tuple[AxisTerm, ...]
    offset: int = 0


@dataclass(frozen=True)
class DataIndex:
    load_id: str


@dataclass(frozen=True)
class IndexedLoad:
    id: str
    value_id: str
    axes: tuple[str | int | AxisAffine | DataIndex, ...]
    strides: tuple[int, ...]
    dtype: str
    source_shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TensorStore:
    value_id: str
    shape: tuple[int, ...]
    dtype: str
    strides: tuple[int, ...]
    owns_storage: bool


@dataclass(frozen=True)
class Constant:
    value: int | float | bool
    dtype: str
    weak_type: bool = False


@dataclass(frozen=True)
class Load:
    load_id: str
    dtype: str


@dataclass(frozen=True)
class Accumulator:
    dtype: str
    # Which accumulator of a paired contraction: 0 multiplies the first right
    # operand (load 1), 1 the second (load 2) with the same left operand.
    index: int = 0


@dataclass(frozen=True)
class Unary:
    operator: str
    operand: ScalarExpression
    dtype: str


@dataclass(frozen=True)
class AxisIndex:
    axis: str
    dtype: str
    start: int = 0
    step: int = 1


@dataclass(frozen=True)
class Binary:
    operator: str
    lhs: ScalarExpression
    rhs: ScalarExpression
    dtype: str


@dataclass(frozen=True)
class Where:
    condition: ScalarExpression
    true_value: ScalarExpression
    false_value: ScalarExpression
    dtype: str


@dataclass(frozen=True)
class Cast:
    operand: ScalarExpression
    dtype: str
    rounding: str


ScalarExpression = Constant | Load | Accumulator | Unary | AxisIndex | Binary | Where | Cast

AxisExpression = str | int | AxisAffine | DataIndex


@dataclass(frozen=True)
class TensorProgram:
    family: str
    source_target: str
    decomposition: str
    iteration_domains: tuple[IterationDomain, ...]
    reduction_axes: tuple[str, ...]
    loads: tuple[IndexedLoad, ...]
    init: Constant | None
    combine: ScalarExpression
    post: ScalarExpression | None
    accumulator_dtype: str | None
    store: TensorStore
    predicates: tuple[AxisPredicate, ...]
    state_accesses: tuple[KVCacheAccess, ...] = ()


@dataclass(frozen=True)
class KVCacheLayout:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class KVCacheAccess:
    key_cache_value_id: str
    value_cache_value_id: str
    valid_length_value_id: str
    key_cache_load_id: str
    value_cache_load_id: str
    new_key_load_id: str
    new_value_load_id: str
    valid_length_load_id: str
    capacity: int
    query_length: int
    key_layout: KVCacheLayout
    value_layout: KVCacheLayout
    storage_identity: tuple[str, str]
    read_range: tuple[int | str, int | str] = (0, "valid_length")
    write_range: tuple[int | str, int | str] = (
        "valid_length",
        "valid_length + query_length",
    )
    source_order: tuple[str, ...] = (
        "read_valid_length",
        "read_cache_prefix",
        "read_new_key_value",
        "write_key_cache",
        "write_value_cache",
        "write_valid_length",
    )
    storage_class: str = "bounded_contiguous"
    valid_length_specialized: bool = False


@dataclass(frozen=True)
class CanonicalProgram:
    family: str
    source_target: str
    decomposition: str
    result_roles: tuple[str, ...]
    semantics: Mapping[str, Any]


class UnsupportedLowering(ValueError):
    pass
