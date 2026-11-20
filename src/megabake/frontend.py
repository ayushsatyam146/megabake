from __future__ import annotations

import hashlib
import operator
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import torch
from torch.utils import _pytree as pytree

from .diagnostics import CompilerError
from .fx_handler.utils import dump_graph

PINNED_TORCH = (2, 14)
SUPPORTED_CAPABILITIES = frozenset({(9, 0), (10, 0), (10, 3), (12, 0), (12, 1)})


@dataclass
class TensorInputSpec:
    name: str
    kind: str
    shape: tuple[int, ...] | None
    dtype: str | None
    device: str | None
    stride: tuple[int, ...] | None
    requires_grad: bool | None


@dataclass
class CaptureRecord:
    phase: str
    cache_state: str
    graph_hash: str


@dataclass
class CaptureContext:
    mode: str
    fallback: str
    input_order: list[str]
    input_specs: list[TensorInputSpec]
    input_kinds: dict[str, str]
    parameter_names: set[str]
    output_structure: str
    output_count: int
    num_model_outputs: int
    prepared_graph_hash: str
    output_tree_spec: Any = None
    runtime_output_tree_spec: Any = None
    requirements: dict[str, Any] = field(default_factory=dict)
    phase: str = "prepared"
    cache_state: str = "miss"
    graph_hash: str = ""
    graph: Any = None
    captures: list[CaptureRecord] = field(default_factory=list)

    def record(self, graph: Any, phase: str, cache_state: str) -> None:
        graph_hash = _graph_hash(graph)
        self.graph = graph
        self.phase = phase
        self.cache_state = cache_state
        self.graph_hash = graph_hash
        self.captures.append(CaptureRecord(phase, cache_state, graph_hash))


@dataclass
class CaptureCacheEntry:
    prepared_graph_hash: str
    post_grad_graph: Any
    fx_forward: Callable[..., Any]


@dataclass
class CaptureCache:
    entries: dict[str, CaptureCacheEntry] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> CaptureCacheEntry | None:
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return entry

    def put(self, key: str, entry: CaptureCacheEntry) -> None:
        self.entries[key] = entry


@dataclass
class PreparedFX:
    graph: Any
    context: CaptureContext
    forward: Callable[..., Any]
    example_inputs: tuple[Any, ...] = ()


_SPLITS = frozenset({torch.ops.aten.split.Tensor, torch.ops.aten.split_with_sizes.default})


def split_to_slices(graph_module: Any) -> None:
    """Rewrite ``split(x)[i]`` as ``slice(x, dim, start, end)``.

    A split returns a tuple of views, which one tensor program cannot store;
    each projection is a plain slice view. Inductor lowers splits the same way.
    """
    graph = graph_module.graph
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in _SPLITS:
            continue
        source, sizes = node.args[0], node.args[1]
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        length = source.meta["val"].shape[dim]
        if node.target is torch.ops.aten.split.Tensor:
            sizes = [min(sizes, length - start) for start in range(0, length, sizes)]
        starts = [sum(sizes[:index]) for index in range(len(sizes))]
        users = list(node.users)
        if not all(user.target is operator.getitem for user in users):
            continue
        for user in users:
            index = user.args[1]
            with graph.inserting_before(user):
                piece = graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    (source, dim, starts[index], starts[index] + sizes[index]),
                )
            piece.meta = dict(user.meta)
            user.replace_all_uses_with(piece)
            graph.erase_node(user)
        graph.erase_node(node)
    graph.lint()


def eliminate_common_subexpressions(graph_module: Any) -> None:
    """Merge pure ATen nodes that apply one op to the same arguments.

    Models often recompute one value per layer (RoPE cosines and sines from
    the positions, say); each copy would cost a kernel phase. Random ops,
    mutating ops, and nodes whose result a mutating op writes are left alone;
    views merge too, so that the computations reading them can.
    """
    graph = graph_module.graph
    written = {
        argument
        for node in graph.nodes
        if node.op == "call_function" and _is_mutating(node.target)
        for argument in node.all_input_nodes
    }
    seen: dict[Any, Any] = {}
    for node in list(graph.nodes):
        target = node.target
        if (
            node.op != "call_function"
            or not isinstance(target, torch._ops.OpOverload)
            or _is_mutating(target)
            or torch.Tag.nondeterministic_seeded in target.tags
            or node in written
        ):
            continue
        key = _cse_key(target, node.args, node.kwargs)
        if key is None:
            continue
        original = seen.setdefault(key, node)
        if original is not node:
            node.replace_all_uses_with(original)
            graph.erase_node(node)
    graph.lint()


def _is_mutating(target: Any) -> bool:
    schema = getattr(target, "_schema", None)
    return schema is not None and schema.is_mutable


def _cse_key(target: Any, args: Any, kwargs: Any) -> Any:
    """A hashable key for ``target(*args, **kwargs)``, or None."""

    def freeze(value: Any) -> Any:
        if isinstance(value, torch.fx.Node):
            return ("node", value)
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, *(freeze(item) for item in value))
        if isinstance(value, dict):
            return ("dict", *sorted((key, freeze(item)) for key, item in value.items()))
        if value is None or isinstance(
            value,
            (bool, int, float, str, torch.dtype, torch.device, torch.layout, torch.memory_format),
        ):
            # Distinguish 1 from 1.0 and True.
            return (type(value).__name__, value)
        raise TypeError

    try:
        key = (target, freeze(args), freeze(kwargs))
        hash(key)
    except TypeError:
        return None
    return key


def torch_version() -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if match is None:
        raise RuntimeError(f"Cannot parse PyTorch version {torch.__version__!r}")
    return int(match.group(1)), int(match.group(2))


def _check_torch_version() -> None:
    if torch_version() != PINNED_TORCH:
        raise RuntimeError(
            f"MegaBake supports PyTorch {PINNED_TORCH[0]}.{PINNED_TORCH[1]}; "
            f"found {torch.__version__}"
        )


def _torch_internals() -> SimpleNamespace:
    from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_func
    from torch._inductor import config as inductor_config
    from torch._inductor.compile_fx import (
        _recursive_post_grad_passes,
        _recursive_record_original_output_strides,
        fake_tensor_prop,
        get_cuda_device_context,
        graph_returns_tuple,
        make_graph_return_tuple,
        partition_fn,
        run_pre_grad_passes,
    )
    from torch._inductor.decomposition import select_decomp_table
    from torch._inductor.fx_passes.post_grad import view_to_reshape
    from torch._inductor.virtualized import V

    return SimpleNamespace(
        aot_module_simplified=aot_module_simplified,
        make_boxed_func=make_boxed_func,
        recursive_post_grad_passes=_recursive_post_grad_passes,
        record_original_output_strides=_recursive_record_original_output_strides,
        fake_tensor_prop=fake_tensor_prop,
        cuda_device_context=get_cuda_device_context,
        graph_returns_tuple=graph_returns_tuple,
        make_graph_return_tuple=make_graph_return_tuple,
        partition_fn=partition_fn,
        run_pre_grad_passes=run_pre_grad_passes,
        inductor_config=inductor_config,
        select_decomp_table=select_decomp_table,
        view_to_reshape=view_to_reshape,
        V=V,
    )


def _input_fingerprint(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return (
            f"tensor:{tuple(value.shape)}:{value.dtype}:{value.device}:"
            f"{tuple(value.stride())}:{value.requires_grad}"
        )
    return f"{type(value).__name__}:{value!r}"


def _graph_hash(graph: Any) -> str:
    graph_text = getattr(graph, "code", None)
    if graph_text is None:
        graph_text = str(graph)
    return hashlib.sha256(str(graph_text).encode("utf-8")).hexdigest()


def _capture_key(graph: Any, example_inputs: Sequence[Any]) -> str:
    input_text = "|".join(_input_fingerprint(value) for value in example_inputs)
    return hashlib.sha256((_graph_hash(graph) + "\0" + input_text).encode("utf-8")).hexdigest()


def _placeholder_specs(
    graph: Any,
    example_inputs: Sequence[Any],
    input_kinds: Mapping[str, str] | None,
) -> tuple[list[str], list[TensorInputSpec], dict[str, str], set[str]]:
    kinds = dict(input_kinds or {})
    names: list[str] = []
    specs: list[TensorInputSpec] = []
    parameter_names: set[str] = set()
    input_index = 0
    for node in graph.graph.nodes:
        if node.op != "placeholder":
            continue
        names.append(node.name)
        kind = kinds.get(node.name, "runtime")
        value = example_inputs[input_index] if input_index < len(example_inputs) else None
        if isinstance(value, torch.Tensor):
            spec = TensorInputSpec(
                name=node.name,
                kind=kind,
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                stride=tuple(value.stride()),
                requires_grad=value.requires_grad,
            )
        else:
            spec = TensorInputSpec(
                name=node.name,
                kind=kind,
                shape=None,
                dtype=None,
                device=None,
                stride=None,
                requires_grad=None,
            )
        specs.append(spec)
        if kind == "parameter":
            parameter_names.add(node.name)
        input_index += 1
    return names, specs, kinds, parameter_names


def _output_structure(graph: Any) -> tuple[str, int, Any]:
    output_node = next(node for node in graph.graph.nodes if node.op == "output")
    output_values, output_spec = pytree.tree_flatten(output_node.args[0])
    return str(output_spec), len(output_values), output_spec


def _make_context(
    graph: Any,
    example_inputs: Sequence[Any],
    options: Mapping[str, Any],
    input_kinds: Mapping[str, str] | None,
    parameter_names: set[str] | None = None,
) -> CaptureContext:
    names, specs, kinds, inferred_parameters = _placeholder_specs(
        graph, example_inputs, input_kinds
    )
    output_structure, output_count, output_tree_spec = _output_structure(graph)
    return CaptureContext(
        mode=options["mode"],
        fallback=options["fallback"],
        input_order=names,
        input_specs=specs,
        input_kinds=kinds,
        parameter_names=(inferred_parameters if parameter_names is None else parameter_names),
        output_structure=output_structure,
        output_count=output_count,
        output_tree_spec=output_tree_spec,
        num_model_outputs=output_count,
        prepared_graph_hash=_graph_hash(graph),
        requirements={
            "inference_only": True,
            "preserve_argument_wrappers": True,
            "preserve_output_wrappers": True,
            "preserve_mutation_and_alias_semantics": True,
            "execution": "reference",
        },
    )


def _validate_options(options: Mapping[str, Any]) -> dict[str, Any]:
    mode = options.get("mode", "prefill")
    fallback = options.get("fallback", "error")
    if mode not in {"prefill", "decode"}:
        raise ValueError(f"Unsupported mode {mode!r}; expected 'prefill' or 'decode'")
    if fallback not in {"error", "inductor"}:
        raise ValueError(f"Unsupported fallback {fallback!r}; expected 'error' or 'inductor'")
    if not isinstance(options.get("cache", True), bool):
        raise TypeError("cache must be a boolean")
    model_name = options.get("model_name", "model")
    if not isinstance(model_name, str) or not model_name:
        raise TypeError("model_name must be a non-empty string")
    dump_dir = options.get("dump_dir")
    if dump_dir is not None and not isinstance(dump_dir, (str, Path)):
        raise TypeError("dump_dir must be a path or None")
    compile_kernel = options.get("compile_kernel", None)
    if compile_kernel not in {None, True, False}:
        raise TypeError("compile_kernel must be True, False, or None for automatic")
    scheduler = options.get("scheduler", "auto")
    if scheduler not in {"phase", "region-events", "auto"}:
        raise ValueError("scheduler must be 'phase', 'region-events', or 'auto'")
    return {
        "mode": mode,
        "fallback": fallback,
        "cache": options.get("cache", True),
        "model_name": model_name,
        "dump_dir": dump_dir,
        "compile_kernel": compile_kernel,
        "scheduler": scheduler,
    }


def _safe_model_name(model_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name).strip("._")
    return safe_name or "model"


def _infer_parameter_bindings(
    graph: Any,
    example_inputs: Sequence[Any],
    input_kinds: Mapping[str, str] | None,
    parameter_names: set[str] | None,
) -> tuple[Mapping[str, str] | None, set[str] | None]:
    placeholder_names = [node.name for node in graph.graph.nodes if node.op == "placeholder"]
    inferred_kinds = dict(input_kinds or {})
    inferred_parameters = set(parameter_names or ())
    if parameter_names is None:
        for name, value in zip(placeholder_names, example_inputs, strict=False):
            if isinstance(value, torch.nn.Parameter):
                inferred_parameters.add(name)
    if input_kinds is None:
        for name in placeholder_names:
            inferred_kinds[name] = "parameter" if name in inferred_parameters else "runtime"
    return inferred_kinds, inferred_parameters


def _reject_symbolic_shapes(graph: Any) -> None:
    for node in graph.graph.nodes:
        leaves, _ = pytree.tree_flatten(node.meta.get("val"))
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor) and any(
                not isinstance(dimension, int) for dimension in leaf.shape
            ):
                raise ValueError("Symbolic tensor shape is unsupported")


def _prepare_fx_graph(
    graph: Any,
    example_inputs: Sequence[Any],
    options: dict[str, Any],
    cache: CaptureCache,
    contexts: list[CaptureContext] | None = None,
    input_kinds: Mapping[str, str] | None = None,
    parameter_names: set[str] | None = None,
    output_tree_spec: Any = None,
    model_name: str | None = None,
) -> PreparedFX:
    internals = _torch_internals()
    _reject_symbolic_shapes(graph)
    dump_directory = (
        Path(options["dump_dir"]) / _safe_model_name(model_name or options["model_name"])
        if options["dump_dir"] is not None
        else None
    )
    dump_graph("input", graph, dump_directory, "Initial FX graph received by MegaBake.")
    if not internals.graph_returns_tuple(graph):
        internals.make_graph_return_tuple(graph, example_inputs, lambda value, _: value)
    dump_graph("normalized", graph, dump_directory, "Normalized FX graph return.")
    context = _make_context(graph, example_inputs, options, input_kinds, parameter_names)
    if contexts is not None:
        contexts.append(context)
    if output_tree_spec is not None:
        context.runtime_output_tree_spec = output_tree_spec
    final_graphs: list[Any] = []

    def pre_grad_callback(pre_grad_graph: Any, inputs: Sequence[Any]) -> Any:
        transformed = internals.run_pre_grad_passes(pre_grad_graph, inputs)
        dump_graph("pre_grad", transformed, dump_directory)
        context.captures.append(CaptureRecord("pre_grad", "miss", _graph_hash(transformed)))
        return transformed

    def prepare_aot_graph(prepared_graph: Any, inputs: Sequence[Any]) -> Callable[..., Any]:
        dump_graph(
            "aot",
            prepared_graph,
            dump_directory,
            "AOTAutograd inference FX graph after decomposition.",
        )
        key = _capture_key(prepared_graph, inputs)
        cache_state = "disabled"
        if options["cache"]:
            entry = cache.get(key)
            cache_state = "hit" if entry is not None else "miss"
            dump_graph(
                "cache",
                prepared_graph,
                dump_directory,
                f"cache_state={cache_state}\nkey={key}",
            )
            if entry is not None:
                dump_graph("prepared", prepared_graph, dump_directory)
                final_graphs.append(entry.post_grad_graph)
                dump_graph(
                    "post_grad",
                    entry.post_grad_graph,
                    dump_directory,
                    "Cache hit; reusing the transformed post-grad FX graph.",
                )
                context.record(entry.post_grad_graph, "post_grad", "hit")
                return entry.fx_forward
        else:
            cache.misses += 1
            dump_graph(
                "cache",
                prepared_graph,
                dump_directory,
                "cache_state=disabled",
            )

        dump_graph("prepared", prepared_graph, dump_directory)
        context.record(prepared_graph, "prepared", "miss")
        internals.view_to_reshape(prepared_graph)
        fake_mode = internals.fake_tensor_prop(prepared_graph, inputs)
        internals.record_original_output_strides(prepared_graph)
        with internals.V.set_fake_mode(fake_mode), internals.cuda_device_context(prepared_graph):
            internals.recursive_post_grad_passes(prepared_graph, is_inference=True)
        split_to_slices(prepared_graph)
        eliminate_common_subexpressions(prepared_graph)
        prepared_graph.recompile()
        final_graphs.append(prepared_graph)
        dump_graph("post_grad", prepared_graph, dump_directory)
        context.record(prepared_graph, "post_grad", "miss")
        fx_forward = internals.make_boxed_func(prepared_graph.forward)
        if options["cache"]:
            cache.put(
                key,
                CaptureCacheEntry(
                    prepared_graph_hash=context.prepared_graph_hash,
                    post_grad_graph=prepared_graph,
                    fx_forward=fx_forward,
                ),
            )
        return fx_forward

    with torch.no_grad(), internals.inductor_config.patch({"fx_graph_cache": False}):
        aot_forward = internals.aot_module_simplified(
            graph,
            example_inputs,
            fw_compiler=prepare_aot_graph,
            bw_compiler=None,
            inference_compiler=prepare_aot_graph,
            partition_fn=internals.partition_fn,
            decompositions=internals.select_decomp_table(),
            keep_inference_input_mutations=True,
            pre_grad_passes=pre_grad_callback,
            compile_region_name="megabake",
        )
    if not final_graphs:
        raise RuntimeError("AOT did not produce an FX graph")

    def fx_forward(*args: Any, **kwargs: Any) -> Any:
        output = aot_forward(*args, **kwargs)
        output_spec = context.runtime_output_tree_spec or context.output_tree_spec
        return pytree.tree_unflatten(output, output_spec)

    return PreparedFX(
        graph=final_graphs[-1],
        context=context,
        forward=fx_forward,
        example_inputs=tuple(example_inputs),
    )


def make_backend(**options: Any) -> Callable[..., Any]:
    _check_torch_version()
    validated_options = _validate_options(options)
    cache = CaptureCache()
    plan_cache = None
    if validated_options["cache"]:
        from .cache import PlanCache

        plan_cache = PlanCache()
    contexts: list[CaptureContext] = []

    def prepare(
        graph: Any,
        example_inputs: Sequence[Any],
        *,
        input_kinds: Mapping[str, str] | None = None,
        parameter_names: set[str] | None = None,
        output_tree_spec: Any = None,
        model_name: str | None = None,
    ) -> PreparedFX:
        return _prepare_fx_graph(
            graph,
            example_inputs,
            validated_options,
            cache,
            contexts,
            input_kinds=input_kinds,
            parameter_names=parameter_names,
            output_tree_spec=output_tree_spec,
            model_name=model_name,
        )

    def backend(
        graph: Any,
        example_inputs: Sequence[Any],
        *,
        input_kinds: Mapping[str, str] | None = None,
        parameter_names: set[str] | None = None,
        **dynamo_kwargs: Any,
    ) -> Callable[..., Any]:
        if dynamo_kwargs:
            unexpected = ", ".join(sorted(dynamo_kwargs))
            raise TypeError(f"Unsupported Dynamo backend options: {unexpected}")
        if torch.is_grad_enabled() and any(
            isinstance(value, torch.Tensor) and value.requires_grad for value in example_inputs
        ):
            raise RuntimeError("MegaBake supports inference only")
        input_kinds, parameter_names = _infer_parameter_bindings(
            graph, example_inputs, input_kinds, parameter_names
        )
        prepared = prepare(
            graph,
            example_inputs,
            input_kinds=input_kinds,
            parameter_names=parameter_names,
        )
        backend.megabake_last_prepared = prepared
        compile_kernel = validated_options["compile_kernel"]
        should_compile = compile_kernel is True or (
            compile_kernel is None and _can_auto_compile(example_inputs)
        )
        compiled = None
        if should_compile:
            from .compiler import compile_inductor_fallback, compile_post_grad

            backend.megabake_last_compiled = None
            backend.megabake_last_compile_us = None
            backend.megabake_last_trace = None
            backend.megabake_last_callable = None
            try:
                compiled = compile_post_grad(
                    prepared,
                    output_dir=(
                        Path(validated_options["dump_dir"]) / "generated"
                        if validated_options["dump_dir"] is not None
                        else None
                    ),
                    scheduler=validated_options["scheduler"],
                    cache=plan_cache,
                )
            except CompilerError as error:
                if validated_options["fallback"] != "inductor" or not error.safe_for_fallback:
                    raise
                forward = compile_inductor_fallback(prepared, str(error))
                backend.megabake_last_fallback = forward
                forward.megabake_contexts = contexts
                forward.megabake_cache = cache
                forward.megabake_plan_cache = plan_cache
                return forward
            forward = compiled
            prepared.context.requirements["execution"] = "megakernel"
            backend.megabake_last_fallback = None
        else:
            forward = prepared.forward
            backend.megabake_last_fallback = None
        forward.megabake_contexts = contexts
        forward.megabake_cache = cache
        forward.megabake_plan_cache = plan_cache
        if compiled is not None:
            backend.megabake_last_compiled = compiled.megabake_compiled
            backend.megabake_last_compile_us = forward.megabake_compile_us
            backend.megabake_last_trace = forward.megabake_trace
            backend.megabake_last_callable = forward
        return forward

    backend.megabake_cache = cache
    backend.megabake_plan_cache = plan_cache
    backend.megabake_contexts = contexts
    backend.megabake_options = MappingProxyType(validated_options)
    backend.megabake_prepare = prepare
    backend.__name__ = "megabake_backend"
    return backend


def _can_auto_compile(example_inputs: Sequence[Any]) -> bool:
    tensors = [value for value in example_inputs if isinstance(value, torch.Tensor)]
    if not tensors or any(tensor.device.type != "cuda" for tensor in tensors):
        return False
    try:
        capability = torch.cuda.get_device_capability(tensors[0].device)
    except Exception:  # noqa: BLE001  # Preserve the diagnostic/fallback boundary for arbitrary backend failures.
        return False
    return capability in SUPPORTED_CAPABILITIES


def prepare_fx_graph(
    graph: Any,
    example_inputs: Sequence[Any],
    *,
    backend: Callable[..., Any] | None = None,
    input_kinds: Mapping[str, str] | None = None,
    parameter_names: set[str] | None = None,
    output_tree_spec: Any = None,
    model_name: str | None = None,
    **options: Any,
) -> PreparedFX:
    if backend is not None and options:
        raise TypeError("Pass either backend or backend options, not both")
    if backend is None:
        backend = make_backend(**options)
    prepare = getattr(backend, "megabake_prepare", None)
    if prepare is None:
        raise TypeError("backend must be created by megabake.make_backend")
    return prepare(
        graph,
        example_inputs,
        input_kinds=input_kinds,
        parameter_names=parameter_names,
        output_tree_spec=output_tree_spec,
        model_name=model_name,
    )
