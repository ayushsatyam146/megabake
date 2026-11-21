"""Hoist parameter-only work out of the captured graph.

A decode step recomputes, on every call, values that depend only on the
model's parameters: Gemma's ``1 + weight`` RMSNorm scales and their casts,
Qwen3.5's ``-exp(A_log)``, any ``weight.float()``. In a megakernel each of
these is a phase of its own (Gemma-3-270m: 75 phases, about 80 us per
token on a full H200). This pass moves them out of the graph: every
parameter-only node whose result feeds runtime work and which computes
something (not a pure view) becomes a new trailing placeholder, whose value
an extracted subgraph computes from the parameters. The values are cached
and recomputed when a source parameter's storage or version changes.

Generic by construction: it reads only the FX graph and which placeholders
are ``nn.Parameter``s. It hoists nothing with side effects (in-place,
random or copy targets) and stops at ``MEGABAKE_HOIST_MAX_BYTES`` of
derived tensors. ``MEGABAKE_HOIST_PARAMS=0`` turns it off.

A CUDA graph captured around the compiled step replays the cached values
without calling back into Python, so parameters changed in place between
replays are not seen there (as with Inductor's freezing).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Sequence

import torch
import torch.fx

# Targets that only reinterpret their input: hoisting them saves no work and
# would copy the parameter.
_VIEW_NAMES = frozenset(
    {
        "view", "reshape", "permute", "transpose", "t", "expand", "expand_as", "unsqueeze",
        "squeeze", "flatten", "unflatten", "select", "narrow", "contiguous", "view_as",
        "detach", "__getitem__", "chunk", "split", "unbind", "movedim", "swapaxes",
        "_unsafe_view", "slice", "alias", "as_strided", "clone", "split_with_sizes",
    }
)
# Targets with effects or nondeterminism, never hoisted.
_UNSAFE_NAMES = frozenset(
    {"copy_", "set_", "dropout", "rand", "randn", "rand_like", "randn_like", "randint", "bernoulli",
     "multinomial", "normal_", "uniform_", "manual_seed", "item", "tolist", "numpy", "cpu", "cuda", "to_dense"}
)


def hoisting_enabled() -> bool:
    return os.environ.get("MEGABAKE_HOIST_PARAMS", "1") == "1"


def _max_bytes() -> int:
    return int(os.environ.get("MEGABAKE_HOIST_MAX_BYTES", 256 << 20))


def _target_name(node: torch.fx.Node) -> str:
    target = node.target
    if isinstance(target, str):
        return target
    packet = getattr(target, "overloadpacket", None)  # aten.add.Tensor -> add
    if packet is not None:
        return packet.__name__
    return getattr(target, "__name__", str(target)).split(".")[-1]


def _example(node: torch.fx.Node) -> Any:
    return node.meta.get("example_value", node.meta.get("val"))


def _safe(node: torch.fx.Node) -> bool:
    if node.op not in {"call_function", "call_method"}:
        return False
    name = _target_name(node)
    if name.endswith("_") and not name.startswith("__") or name in _UNSAFE_NAMES:
        return False
    if node.kwargs.get("out") is not None:
        return False
    return isinstance(_example(node), torch.Tensor)


def _is_view(node: torch.fx.Node) -> bool:
    name = _target_name(node)
    if name in _VIEW_NAMES:
        return True
    return False


@dataclass
class HoistedParameters:
    """The rewritten graph's extra inputs: ``derive(*args)`` computes them
    from the original call's arguments."""

    graph: torch.fx.GraphModule
    derive_module: torch.fx.GraphModule
    parameter_positions: tuple[int, ...]
    examples: tuple[torch.Tensor, ...]
    hoisted_nodes: int

    def derive(self, args: Sequence[Any]) -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            result = self.derive_module(*(args[position] for position in self.parameter_positions))
        return tuple(result) if isinstance(result, (tuple, list)) else (result,)


def hoist_parameter_work(
    graph: torch.fx.GraphModule, example_inputs: Sequence[Any], parameter_names: set[str]
) -> HoistedParameters | None:
    """Rewrite ``graph`` with its parameter-only work as trailing inputs, or
    None when there is nothing worth hoisting."""
    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(example_inputs):
        return None
    parameter_nodes = {
        node
        for node, value in zip(placeholders, example_inputs, strict=True)
        if node.name in parameter_names and isinstance(value, torch.Tensor)
    }
    if not parameter_nodes:
        return None

    # Parameter-only nodes: safe, every tensor input parameter-only.
    constant: set[torch.fx.Node] = set(parameter_nodes)
    for node in graph.graph.nodes:
        if node in constant or not _safe(node):
            continue
        inputs = node.all_input_nodes
        if inputs and all(item in constant for item in inputs):
            constant.add(node)

    def computes(node: torch.fx.Node) -> bool:
        """Whether ``node``'s value is more than a view of a parameter."""
        if node in parameter_nodes:
            return False
        if not _is_view(node):
            return True
        return any(computes(item) for item in node.all_input_nodes)

    frontier = [
        node
        for node in graph.graph.nodes
        if node in constant
        and node not in parameter_nodes
        and any(user not in constant for user in node.users)
        and computes(node)
    ]
    budget = _max_bytes()
    chosen: list[torch.fx.Node] = []
    for node in frontier:
        value = _example(node)
        size = value.numel() * value.element_size()
        if size <= budget:
            chosen.append(node)
            budget -= size
    if not chosen:
        return None

    # The derive module: the chosen nodes' ancestors, on the parameters they read.
    needed: set[torch.fx.Node] = set()
    stack = list(chosen)
    while stack:
        node = stack.pop()
        if node in needed:
            continue
        needed.add(node)
        stack.extend(node.all_input_nodes)
    sources = [node for node in placeholders if node in needed]
    derive_graph = torch.fx.Graph()
    env: dict[torch.fx.Node, torch.fx.Node] = {}
    for node in sources:
        env[node] = derive_graph.placeholder(node.name)
    for node in graph.graph.nodes:
        if node in needed and node not in env:
            env[node] = derive_graph.node_copy(node, lambda item: env[item])
    derive_graph.output(tuple(env[node] for node in chosen))
    derive_module = torch.fx.GraphModule(graph, derive_graph)

    # The rewritten graph: each chosen node becomes a trailing placeholder.
    rewritten = torch.fx.GraphModule(graph, graph.graph.__deepcopy__())
    copies = {node.name: node for node in rewritten.graph.nodes}
    last = [node for node in rewritten.graph.nodes if node.op == "placeholder"][-1]
    for index, node in enumerate(chosen):
        copy = copies[node.name]
        with rewritten.graph.inserting_after(last):
            placeholder = rewritten.graph.placeholder(f"megabake_hoisted_{index}")
        placeholder.meta = dict(copy.meta)
        last = placeholder
        copy.replace_all_uses_with(placeholder)
    rewritten.graph.eliminate_dead_code()
    rewritten.recompile()

    positions = tuple(placeholders.index(node) for node in sources)
    hoisted = HoistedParameters(rewritten, derive_module, positions, (), len(needed) - len(sources))
    hoisted.examples = hoisted.derive(example_inputs)
    return hoisted


class WithHoistedParameters:
    """``forward`` called with the derived inputs appended; the derived values
    are cached and recomputed when a source parameter's storage or version
    changes. Attributes not set here are ``forward``'s."""

    def __init__(
        self, forward: Callable[..., Any], hoisted: HoistedParameters, example_inputs: Sequence[Any]
    ):
        self.megabake_inner = forward
        self.megabake_hoisted = hoisted
        # The examples were derived from the compile-time parameters.
        self._key = self._source_key(example_inputs)
        self._values: tuple[torch.Tensor, ...] = hoisted.examples

    def _source_key(self, args: Sequence[Any]) -> tuple:
        return tuple(
            (args[position].data_ptr(), args[position]._version)
            for position in self.megabake_hoisted.parameter_positions
        )

    def __call__(self, *args: Any) -> Any:
        key = self._source_key(args)
        if key != self._key:
            self._values = self.megabake_hoisted.derive(args)
            self._key = key
        return self.megabake_inner(*args, *self._values)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("megabake_inner") or name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.megabake_inner, name)


def hoist_prepared(prepared: Any) -> tuple[Any, HoistedParameters | None]:
    """``prepared`` (a frontend.PreparedFX) with its post-grad graph's
    parameter-only work moved to trailing parameter inputs, and the hoisting
    record; ``prepared`` unchanged and None when nothing is worth hoisting.

    The post-grad graph's placeholders line up with ``example_inputs`` and
    with the capture context's input specs; the derived inputs are appended
    to both as parameters. The reference forward keeps running the original
    program on the original inputs."""
    from dataclasses import replace

    from .frontend import TensorInputSpec

    context = prepared.context
    placeholders = [node for node in prepared.graph.graph.nodes if node.op == "placeholder"]
    parameter_names = {
        node.name
        for index, (node, value) in enumerate(zip(placeholders, prepared.example_inputs, strict=False))
        if isinstance(value, torch.nn.Parameter)
        or (index < len(context.input_specs) and context.input_specs[index].kind == "parameter")
    }
    hoisted = hoist_parameter_work(prepared.graph, prepared.example_inputs, parameter_names)
    if hoisted is None:
        return prepared, None
    names = [node.name for node in hoisted.graph.graph.nodes if node.op == "placeholder"][len(placeholders):]
    specs = [
        TensorInputSpec(
            name=name, kind="parameter", shape=tuple(value.shape), dtype=str(value.dtype),
            device=str(value.device), stride=tuple(value.stride()), requires_grad=False,
        )
        for name, value in zip(names, hoisted.examples, strict=True)
    ]
    context = replace(
        context,
        input_order=[*context.input_order, *names],
        input_specs=[*context.input_specs, *specs],
        input_kinds={**context.input_kinds, **{name: "parameter" for name in names}},
        parameter_names={*context.parameter_names, *names},
    )
    original = prepared.forward
    count = len(placeholders)

    def forward(*args: Any) -> Any:
        return original(*args[:count])

    return (
        replace(
            prepared,
            graph=hoisted.graph,
            context=context,
            forward=forward,
            example_inputs=(*prepared.example_inputs, *hoisted.examples),
        ),
        hoisted,
    )
