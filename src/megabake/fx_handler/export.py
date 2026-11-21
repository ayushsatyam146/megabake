from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.export.graph_signature import InputKind

from ..frontend import CaptureContext, prepare_fx_graph


@dataclass
class ExportedCapture:
    exported_program: Any
    model: torch.nn.Module | None
    graph: Any
    context: CaptureContext
    forward: Callable[..., Any]
    input_defaults: dict[str, Any] | None = None

    def binding_values(
        self,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        return _binding_values(
            self.exported_program,
            self.model,
            args,
            dict(kwargs or {}),
            self.input_defaults or {},
        )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*self.binding_values(args, kwargs))


def _user_input(
    name: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> Any:
    if name in kwargs:
        return kwargs.pop(name)
    if args:
        return args[0]
    if name in defaults:
        return defaults[name]
    raise TypeError(f"Missing exported user input {name!r}")


def _binding_values(
    exported: Any,
    model: torch.nn.Module | None,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    defaults: Mapping[str, Any] | None = None,
) -> list[Any]:
    defaults = defaults or {}
    state_dict = model.state_dict() if model is not None else exported.state_dict
    values: list[Any] = []
    remaining_args = list(args)
    for spec in exported.graph_signature.input_specs:
        if spec.kind == InputKind.PARAMETER:
            values.append(state_dict[spec.target])
        elif spec.kind == InputKind.BUFFER:
            if spec.target in state_dict:
                values.append(state_dict[spec.target])
            elif model is not None:
                values.append(model.get_buffer(spec.target))
            else:
                raise KeyError(f"Missing exported buffer {spec.target!r}")
        elif spec.kind == InputKind.CONSTANT_TENSOR:
            values.append(exported.constants[spec.target])
        elif spec.kind == InputKind.USER_INPUT:
            values.append(_user_input(spec.arg.name, remaining_args, kwargs, defaults))
            if remaining_args:
                remaining_args.pop(0)
        else:
            raise RuntimeError(f"Unsupported exported input kind: {spec.kind}")
    if remaining_args or kwargs:
        raise TypeError("Unused arguments supplied to an exported FX graph")
    return values


def capture_exported_program(
    exported: Any,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    model: torch.nn.Module | None = None,
    backend: Callable[..., Any] | None = None,
    model_name: str | None = None,
    input_defaults: Mapping[str, Any] | None = None,
    **backend_options: Any,
) -> ExportedCapture:
    example_inputs = _binding_values(exported, model, args, dict(kwargs or {}), input_defaults)
    graph = copy.deepcopy(exported.graph_module)
    input_kinds: dict[str, str] = {}
    parameter_names: set[str] = set()
    for spec in exported.graph_signature.input_specs:
        input_kinds[spec.arg.name] = spec.kind.name.lower()
        if spec.kind == InputKind.PARAMETER:
            parameter_names.add(spec.target)
    prepared = prepare_fx_graph(
        graph,
        example_inputs,
        backend=backend,
        input_kinds=input_kinds,
        parameter_names=parameter_names,
        output_tree_spec=exported.call_spec.out_spec,
        model_name=model_name or (type(model).__name__ if model is not None else None),
        **backend_options,
    )
    context = prepared.context
    context.runtime_output_tree_spec = getattr(
        exported.call_spec, "out_spec", context.output_tree_spec
    )
    return ExportedCapture(
        exported_program=exported,
        model=model,
        graph=prepared.graph,
        context=context,
        forward=prepared.forward,
        input_defaults=dict(input_defaults or {}),
    )
