from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import operator
from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch.fx.operator_schemas import normalize_function
from torch.utils._pytree import tree_flatten, tree_unflatten

if TYPE_CHECKING:
    from .ops import Semantics


_MISSING = object()

# Bytes per element of each supported dtype, by its ``str(torch.dtype)``.
DTYPE_BYTES = {
    "torch.bool": 1,
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.int16": 2,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
}


@dataclass(frozen=True)
class StorageRelation:
    storage_id: str | None
    aliases: tuple[str, ...] = ()
    unknown: bool = False


@dataclass(frozen=True)
class Value:
    id: str
    name: str
    source_fx_id: str
    shape: tuple[int, ...] | None
    dtype: str | None
    device: str | None
    strides: tuple[int, ...] | None
    storage_relation: StorageRelation
    kind: str
    binding_name: str | None = None
    type_name: str | None = None
    producer: str | None = None
    users: tuple[str, ...] = ()
    storage_offset: int = 0


@dataclass(frozen=True)
class Op:
    id: str
    source_fx_ids: tuple[str, ...]
    operands: tuple[str, ...]
    results: tuple[str, ...]
    attributes: dict[str, Any]
    tensor_program: Any | None
    effects: dict[str, Any]
    numerical_rules: dict[str, Any]
    source_location: str | None
    result_reconstruction: Any


@dataclass(frozen=True)
class Region:
    id: str
    source_op_ids: tuple[str, ...]
    external_inputs: tuple[str, ...]
    live_outputs: tuple[str, ...]
    semantics: Semantics | None = None
    tensor_programs: tuple[Any, ...] = ()
    fusion_candidates: tuple[Any, ...] = ()


@dataclass(frozen=True)
class RegionCandidate:
    id: str
    source_op_ids: tuple[str, ...]
    external_inputs: tuple[str, ...]
    live_outputs: tuple[str, ...]
    tensor_program: Any | None
    semantics: Semantics | None = None
    cost_bytes: int = 0
    rejections: tuple[str, ...] = ()

    @property
    def executable(self) -> bool:
        return self.tensor_program is not None and not self.rejections


@dataclass(frozen=True)
class ComputeGraph:
    values: dict[str, Value]
    ops: dict[str, Op]
    regions: tuple[Region, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    output_reconstruction: Any
    output_tree_spec: Any
    runtime_context: Any
    source_nodes: dict[str, torch.fx.Node]


def _source_id(node: torch.fx.Node) -> str:
    return f"fx:{node.name}"


def _value_id(node: torch.fx.Node, result_index: int, result_count: int) -> str:
    suffix = "" if result_count == 1 else f".{result_index}"
    return f"value:{node.name}{suffix}"


def _nodes_in(value: Any) -> list[torch.fx.Node]:
    if isinstance(value, torch.fx.Node):
        return [value]
    if isinstance(value, (tuple, list)):
        nodes: list[torch.fx.Node] = []
        for item in value:
            nodes.extend(_nodes_in(item))
        return nodes
    if isinstance(value, Mapping):
        nodes = []
        for item in value.values():
            nodes.extend(_nodes_in(item))
        return nodes
    return []


def _tensor_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        shape = _static_dimensions(value.shape, "tensor shape")
        strides = _static_dimensions(value.stride(), "tensor strides")
        return {
            "shape": shape,
            "dtype": str(value.dtype),
            "device": str(value.device),
            "strides": strides,
            "storage_offset": int(value.storage_offset()),
            "type_name": "Tensor",
        }
    return {
        "shape": None,
        "dtype": None,
        "device": None,
        "strides": None,
        "type_name": type(value).__name__,
    }


def _spec_metadata(spec: Any) -> dict[str, Any]:
    shape = (
        _static_dimensions(spec.shape, "input shape") if spec.shape is not None else None
    )
    strides = (
        _static_dimensions(spec.stride, "input strides")
        if spec.stride is not None
        else None
    )
    return {
        "shape": shape,
        "dtype": spec.dtype,
        "device": spec.device,
        "strides": strides,
        "type_name": "Tensor" if spec.shape is not None else None,
    }


def _static_dimensions(dimensions: Any, description: str) -> tuple[int, ...]:
    if any(not isinstance(dimension, int) for dimension in dimensions):
        raise ValueError(f"Symbolic {description} is unsupported")
    return tuple(int(dimension) for dimension in dimensions)


def _metadata(node: torch.fx.Node, graph: torch.fx.GraphModule, spec: Any) -> Any:
    if "val" in node.meta:
        return node.meta["val"]
    if spec is not None:
        return spec
    if node.op == "get_attr":
        try:
            return graph.get_attr(node.target)
        except AttributeError:
            pass
    return None


def _placeholder_metadata(
    node: torch.fx.Node, graph: torch.fx.GraphModule, spec: Any
) -> dict[str, Any]:
    if "val" in node.meta:
        return _tensor_metadata(node.meta["val"])
    if spec is not None:
        return _spec_metadata(spec)
    return _tensor_metadata(None)


def _schema_alias_sources(
    node: torch.fx.Node, node_values: Mapping[torch.fx.Node, tuple[str, ...]]
) -> list[tuple[frozenset[str], tuple[str, ...]]] | None:
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return None
    try:
        normalized = normalize_function(
            node.target,
            node.args,
            node.kwargs,
            normalize_to_only_use_kwargs=True,
        )
    except (TypeError, ValueError):
        return None
    if normalized is None:
        return None
    sources: list[tuple[frozenset[str], tuple[str, ...]]] = []
    for argument in schema.arguments:
        if argument.alias_info is None:
            continue
        argument_name = "input" if argument.name == "self" else argument.name
        value = normalized.kwargs.get(argument_name, _MISSING)
        if value is _MISSING:
            continue
        operand_nodes = _nodes_in(value)
        if not operand_nodes:
            continue
        operand_values = tuple(
            value_id
            for operand_node in operand_nodes
            for value_id in node_values[operand_node]
        )
        sources.append((frozenset(argument.alias_info.before_set), operand_values))
    return sources


def _storage_relation(
    node: torch.fx.Node,
    result_index: int,
    values: Mapping[str, Value],
    node_values: Mapping[torch.fx.Node, tuple[str, ...]],
    value_id: str,
) -> StorageRelation:
    if node.target is operator.getitem:
        source_node, result_index = node.args
        source_id = node_values[source_node][result_index]
        source = values[source_id].storage_relation
        return StorageRelation(
            storage_id=source.storage_id or source_id,
            aliases=(source_id,),
            unknown=source.unknown,
        )
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        operand_ids = tuple(
            value_id
            for operand_node in _nodes_in((node.args, node.kwargs))
            for value_id in node_values[operand_node]
        )
        return StorageRelation(
            storage_id=None, aliases=operand_ids, unknown=True
        )
    if result_index >= len(schema.returns):
        return StorageRelation(storage_id=value_id)
    return_alias = schema.returns[result_index].alias_info
    if return_alias is None:
        return StorageRelation(storage_id=value_id)
    return_symbols = frozenset(return_alias.before_set)
    alias_sources = _schema_alias_sources(node, node_values) or []
    aliases = tuple(
        source_value_id
        for source_symbols, source_value_ids in alias_sources
        if source_symbols & return_symbols
        for source_value_id in source_value_ids
    )
    if not aliases:
        return StorageRelation(storage_id=value_id)
    if len(aliases) == 1:
        source = values[aliases[0]].storage_relation
        if source.unknown:
            return StorageRelation(
                storage_id=None, aliases=aliases, unknown=True
            )
        return StorageRelation(
            storage_id=source.storage_id or aliases[0],
            aliases=aliases,
        )
    return StorageRelation(storage_id=None, aliases=aliases, unknown=True)


def _effects(node: torch.fx.Node) -> dict[str, Any]:
    if node.op == "get_attr":
        return {"unknown": False, "mutates_inputs": False}
    if node.target is operator.getitem:
        return {"unknown": False, "mutates_inputs": False}
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return {"unknown": True, "may_mutate_inputs": True}
    return {"unknown": False, "mutates_inputs": bool(schema.is_mutable)}


def _with_mutated_operands(
    effects: dict[str, Any],
    node: torch.fx.Node,
    node_values: Mapping[torch.fx.Node, tuple[str, ...]],
) -> dict[str, Any]:
    """Add ``mutated_operands``, the values a mutable op's schema says it
    writes, when the op names them (``megabake::linear_attention`` writes
    only its state; its weights stay read-only)."""
    schema = getattr(node.target, "_schema", None)
    if not effects.get("mutates_inputs") or schema is None:
        return effects
    mutated: list[str] = []
    for index, argument in enumerate(schema.arguments):
        if argument.alias_info is None or not argument.alias_info.is_write:
            continue
        value = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
        if isinstance(value, torch.fx.Node):
            mutated.extend(item for item in node_values[value] if item not in mutated)
    return {**effects, "mutated_operands": tuple(mutated)}


def _operand_ids(
    node: torch.fx.Node, node_values: Mapping[torch.fx.Node, tuple[str, ...]]
) -> tuple[str, ...]:
    operand_ids: list[str] = []
    for operand_node in _nodes_in((node.args, node.kwargs)):
        for value_id in node_values[operand_node]:
            if value_id not in operand_ids:
                operand_ids.append(value_id)
    return tuple(operand_ids)


def _output_value_ids(
    leaves: list[Any],
    node_values: Mapping[torch.fx.Node, tuple[str, ...]],
    values: dict[str, Value],
) -> tuple[str, ...]:
    output_ids: list[str] = []
    for leaf in leaves:
        if isinstance(leaf, torch.fx.Node):
            output_ids.extend(node_values[leaf])
            continue
        value_id = f"value:output:{len(values)}"
        while value_id in values:
            value_id += ".constant"
        metadata = _tensor_metadata(leaf)
        values[value_id] = Value(
            id=value_id,
            name=value_id,
            source_fx_id="fx:output",
            shape=metadata["shape"],
            dtype=metadata["dtype"],
            device=metadata["device"],
            strides=metadata["strides"],
            storage_relation=StorageRelation(storage_id=value_id),
            kind="constant",
            type_name=metadata["type_name"],
        )
        output_ids.append(value_id)
    return tuple(output_ids)


def _is_self_copy(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops.aten.copy_.default
        and len(node.args) >= 2
        and node.args[0] is node.args[1]
        and not node.kwargs.get("non_blocking", False)
    )


def import_fx(graph: torch.fx.GraphModule, context: Any) -> ComputeGraph:
    graph.graph.lint()
    values: dict[str, Value] = {}
    ops: dict[str, Op] = {}
    node_values: dict[torch.fx.Node, tuple[str, ...]] = {}
    source_nodes: dict[str, torch.fx.Node] = {}
    inputs: list[str] = []
    input_specs = list(getattr(context, "input_specs", []))
    input_order = list(getattr(context, "input_order", []))

    for node in graph.graph.nodes:
        source_id = _source_id(node)
        if source_id in source_nodes:
            raise ValueError(f"Duplicate FX source ID {source_id}")
        source_nodes[source_id] = node

        if node.op == "placeholder":
            input_index = len(inputs)
            spec = input_specs[input_index] if input_index < len(input_specs) else None
            metadata = _placeholder_metadata(node, graph, spec)
            kind = getattr(spec, "kind", None) or "runtime"
            value_id = _value_id(node, 0, 1)
            binding_name = (
                input_order[input_index]
                if input_index < len(input_order)
                else node.name
            )
            value = Value(
                id=value_id,
                name=node.name,
                source_fx_id=source_id,
                shape=metadata["shape"],
                dtype=metadata["dtype"],
                device=metadata["device"],
                strides=metadata["strides"],
                storage_relation=StorageRelation(storage_id=value_id),
                kind=kind,
                binding_name=binding_name,
                type_name=metadata["type_name"],
            )
            values[value_id] = value
            node_values[node] = (value_id,)
            inputs.append(value_id)
            continue

        if node.op == "output":
            continue

        if _is_self_copy(node):
            # Reinplacing leaves copy_(x, x) after an in-place state update;
            # it writes nothing and returns its destination.
            node_values[node] = node_values[node.args[0]]
            continue

        metadata = _metadata(node, graph, None)
        result_leaves, result_spec = tree_flatten(metadata)
        result_count = len(result_leaves)
        result_ids = tuple(
            _value_id(node, index, result_count)
            for index in range(result_count)
        )
        result_reconstruction = tree_unflatten(result_ids, result_spec)
        node_values[node] = result_ids
        operand_ids = _operand_ids(node, node_values)

        for index, (value_id, result) in enumerate(zip(result_ids, result_leaves)):
            result_metadata = _tensor_metadata(result)
            values[value_id] = Value(
                id=value_id,
                name=f"{node.name}{'' if result_count == 1 else f'.{index}'}",
                source_fx_id=source_id,
                shape=result_metadata["shape"],
                dtype=result_metadata["dtype"],
                device=result_metadata["device"],
                strides=result_metadata["strides"],
                storage_relation=_storage_relation(
                    node, index, values, node_values, value_id
                ),
                kind="attribute" if node.op == "get_attr" else "result",
                binding_name=node.target if node.op == "get_attr" else None,
                type_name=result_metadata["type_name"],
                producer=f"op:{node.name}",
                storage_offset=result_metadata.get("storage_offset", 0),
            )

        op_id = f"op:{node.name}"
        ops[op_id] = Op(
            id=op_id,
            source_fx_ids=(source_id,),
            operands=operand_ids,
            results=result_ids,
            attributes={
                "target": str(node.target),
                "fx_args": node.args,
                "fx_kwargs": node.kwargs,
            },
            tensor_program=None,
            effects=_with_mutated_operands(_effects(node), node, node_values),
            numerical_rules={"known": False, "reason": "task-03 lowering pending"},
            source_location=getattr(node, "stack_trace", None),
            result_reconstruction=result_reconstruction,
        )
        for operand_id in operand_ids:
            operand = values[operand_id]
            values[operand_id] = replace(
                operand, users=tuple((*operand.users, op_id))
            )

    output_node = next(node for node in graph.graph.nodes if node.op == "output")
    output_leaves, source_output_spec = tree_flatten(output_node.args[0])
    outputs = _output_value_ids(output_leaves, node_values, values)
    output_tree_spec = (
        getattr(context, "runtime_output_tree_spec", None)
        or getattr(context, "output_tree_spec", None)
        or source_output_spec
    )
    output_reconstruction = tree_unflatten(list(outputs), output_tree_spec)
    boundary_strides = output_node.meta.get("original_output_strides")
    if boundary_strides is not None:
        flat_strides = list(boundary_strides)
        if len(flat_strides) != len(outputs):
            raise ValueError("Boundary stride inventory does not match outputs")
        for value_id, strides in zip(outputs, flat_strides):
            value = values[value_id]
            if strides is not None and value.shape is not None:
                values[value_id] = replace(
                    value, strides=tuple(int(stride) for stride in strides)
                )

    result = ComputeGraph(
        values=values,
        ops=ops,
        regions=(),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        output_reconstruction=output_reconstruction,
        output_tree_spec=output_tree_spec,
        runtime_context=context,
        source_nodes=source_nodes,
    )
    verify_compute_graph(result)
    return result


def dump_compute_graph(graph: ComputeGraph, path: str | Path) -> Path:
    lines = [
        "# ComputeGraph",
        f"values={len(graph.values)} operations={len(graph.ops)} "
        f"inputs={len(graph.inputs)} outputs={len(graph.outputs)}",
        "",
        "## Values",
    ]
    for value in graph.values.values():
        storage = value.storage_relation
        lines.append(
            f"{value.id} name={value.name} kind={value.kind} "
            f"shape={value.shape} dtype={value.dtype} device={value.device} "
            f"strides={value.strides} storage_id={storage.storage_id} "
            f"aliases={storage.aliases} storage_unknown={storage.unknown} "
            f"producer={value.producer} users={list(value.users)}"
        )

    lines.extend(["", "## Operations"])
    for operation in graph.ops.values():
        lines.append(
            f"{operation.id} source={list(operation.source_fx_ids)} "
            f"operands={list(operation.operands)} "
            f"results={list(operation.results)} "
            f"target={operation.attributes.get('target')} "
            f"effects={operation.effects} "
            f"tensor_program={operation.tensor_program!r}"
        )

    lines.extend(
        [
            "",
            "## Inputs",
            *[str(value_id) for value_id in graph.inputs],
            "",
            "## Outputs",
            *[str(value_id) for value_id in graph.outputs],
            "",
            "## Output Reconstruction",
            repr(graph.output_reconstruction),
            "",
        ]
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _check_cycle(ops: Mapping[str, Op], values: Mapping[str, Value]) -> None:
    adjacency: dict[str, set[str]] = {}
    for op_id, op in ops.items():
        adjacency[op_id] = {
            values[operand_id].producer
            for operand_id in op.operands
            if values[operand_id].producer is not None
        }
    state: dict[str, int] = {}

    def visit(op_id: str) -> None:
        mark = state.get(op_id, 0)
        if mark == 1:
            raise ValueError(f"Compute graph contains a cycle at {op_id}")
        if mark == 2:
            return
        state[op_id] = 1
        for dependency in adjacency[op_id]:
            visit(dependency)
        state[op_id] = 2

    for op_id in ops:
        visit(op_id)


def verify_compute_graph(graph: ComputeGraph) -> None:
    for value_id, value in graph.values.items():
        if value.id != value_id:
            raise ValueError(f"Value ID mismatch for {value_id}")
        tensor_fields = (
            value.shape,
            value.dtype,
            value.device,
            value.strides,
        )
        if any(field is not None for field in tensor_fields):
            if any(field is None for field in tensor_fields):
                raise ValueError(f"Incomplete tensor metadata for {value_id}")
            if len(value.shape) != len(value.strides):
                raise ValueError(f"Shape and stride rank mismatch for {value_id}")
            if any(dim < 0 for dim in value.shape):
                raise ValueError(f"Negative dimension for {value_id}")
        if value.producer is not None:
            producer = graph.ops.get(value.producer)
            if producer is None or value_id not in producer.results:
                raise ValueError(f"Invalid producer for {value_id}")
        for user in value.users:
            user_op = graph.ops.get(user)
            if user_op is None or value_id not in user_op.operands:
                raise ValueError(f"Invalid user {user} for {value_id}")
        if value.kind == "parameter" and value.binding_name is None:
            raise ValueError(f"Parameter {value_id} has no live binding")

    for op_id, op in graph.ops.items():
        if op.id != op_id:
            raise ValueError(f"Operation ID mismatch for {op_id}")
        for operand_id in op.operands:
            if operand_id not in graph.values:
                raise ValueError(f"Missing operand {operand_id} in {op_id}")
        for result_id in op.results:
            if result_id not in graph.values:
                raise ValueError(f"Missing result {result_id} in {op_id}")
        for operand_id in op.operands:
            if op_id not in graph.values[operand_id].users:
                raise ValueError(f"Missing user {op_id} for {operand_id}")
        for source_id in op.source_fx_ids:
            if source_id not in graph.source_nodes:
                raise ValueError(f"Missing source FX node {source_id}")

    for input_id in graph.inputs:
        if input_id not in graph.values:
            raise ValueError(f"Invalid graph input {input_id}")
    for output_id in graph.outputs:
        if output_id not in graph.values:
            raise ValueError(f"Invalid graph output {output_id}")
    reconstructed = tree_unflatten(list(graph.outputs), graph.output_tree_spec)
    if reconstructed != graph.output_reconstruction:
        raise ValueError("Output reconstruction does not match flattened outputs")

    input_bindings = tuple(
        graph.values[input_id].binding_name for input_id in graph.inputs
    )
    if input_bindings != tuple(getattr(graph.runtime_context, "input_order", ())):
        raise ValueError("Input order does not match the capture context")
    operation_source_ids = {
        source_id
        for op in graph.ops.values()
        for source_id in op.source_fx_ids
    }
    for source_id, node in graph.source_nodes.items():
        if node.op in {"get_attr", "call_function", "call_method", "call_module"}:
            if source_id not in operation_source_ids and not _is_self_copy(node):
                raise ValueError(f"FX operation {source_id} was not imported")

    _check_cycle(graph.ops, graph.values)
