from __future__ import annotations

import argparse
import importlib
import json
import re
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from .doctor import format_checks, run_checks
from .frontend import make_backend
from .fx_handler.export import capture_exported_program
from .fx_handler.transformers import capture_transformer
from .graph import dump_compute_graph, import_fx
from .lowering import canonicalize_semantics, dump_lowering_graph
from .regions import dump_region_graph, form_regions


def _backend(args: argparse.Namespace) -> Any:
    return make_backend(
        mode=getattr(args, "mode", "prefill"),
        fallback=getattr(args, "fallback", "error"),
        cache=not getattr(args, "no_cache", False),
        dump_dir=getattr(args, "dump_dir", "fx_traces"),
        model_name=getattr(args, "model_name", "model"),
    )


def _summary(result: Any) -> str:
    if isinstance(result, torch.Tensor):
        return f"tensor(shape={tuple(result.shape)}, dtype={result.dtype}, device={result.device})"
    if isinstance(result, (tuple, list)):
        return f"{type(result).__name__}({', '.join(_summary(value) for value in result)})"
    return type(result).__name__


def _run_demo(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8, device=device),
        torch.nn.GELU(),
        torch.nn.Linear(8, 3, device=device),
    ).eval()
    inputs = (torch.randn(5, 4, device=device),)
    backend = _backend(args)
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    with torch.no_grad():
        expected = model(*inputs)
        actual = compiled(*inputs)
    torch.testing.assert_close(actual, expected)
    print(f"demo: PASS {_summary(actual)}")
    print(
        f"capture: phase={backend.megabake_contexts[-1].phase} "
        f"cache_state={backend.megabake_contexts[-1].cache_state}"
    )
    return 0


def _run_transformers(args: argparse.Namespace) -> int:
    device = args.device
    input_ids = torch.tensor([args.input_ids], device=device)
    backend = _backend(args)
    capture = capture_transformer(
        args.model_name,
        input_ids,
        revision=args.revision,
        device=device,
        backend=backend,
    )
    output = capture.run(input_ids)
    compute_graph_path = None
    lowering_graph_path = None
    region_graph_path = None
    if (
        getattr(args, "dump_compute_graph", False)
        or getattr(args, "dump_lowering_graph", False)
        or getattr(args, "dump_region_graph", False)
    ):
        compute_graph = import_fx(capture.graph, capture.context)
        model_name = args.model_name
        safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name).strip("._")
        output_directory = Path(args.dump_dir or "fx_traces") / (safe_model_name or "model")
        if getattr(args, "dump_compute_graph", False):
            compute_graph_path = dump_compute_graph(
                compute_graph, output_directory / "compute_graph.txt"
            )
        if getattr(args, "dump_lowering_graph", False):
            canonical_graph = canonicalize_semantics(compute_graph)
            lowering_graph_path = dump_lowering_graph(
                canonical_graph, output_directory / "lowering_graph.txt"
            )
        if getattr(args, "dump_region_graph", False):
            region_graph = form_regions(canonicalize_semantics(compute_graph))
            region_graph_path = dump_region_graph(
                region_graph, output_directory / "region_graph.txt"
            )
    print(f"transformers: PASS {_summary(output)}")
    print(
        f"capture: phase={capture.context.phase} "
        f"cache_state={capture.context.cache_state} "
        f"inputs={len(capture.context.input_order)}"
    )
    if compute_graph_path is not None:
        print(f"compute graph: {compute_graph_path}")
    if lowering_graph_path is not None:
        print(f"lowering graph: {lowering_graph_path}")
    if region_graph_path is not None:
        print(f"region graph: {region_graph_path}")
    return 0


def _json_value(value: str) -> Any:
    """Parse a JSON argument; a list becomes a tensor, since torch.export traces tensors."""
    parsed = json.loads(value)
    return torch.tensor(parsed) if isinstance(parsed, list) else parsed


def _run_export(args: argparse.Namespace) -> int:
    module_name, _, attribute = args.callable.partition(":")
    if not module_name or not attribute:
        raise ValueError("--callable must use module:attribute syntax")
    callable_obj = getattr(importlib.import_module(module_name), attribute)
    args_values = [_json_value(value) for value in args.arg]
    kwargs = dict(item.split("=", 1) for item in args.kwarg)
    kwargs = {name: _json_value(value) for name, value in kwargs.items()}
    if args.model_name is None:
        args.model_name = args.callable
    backend = _backend(args)
    with torch.no_grad():
        exported = torch.export.export(callable_obj, tuple(args_values), kwargs)
        capture = capture_exported_program(
            exported,
            args=args_values,
            kwargs=kwargs,
            backend=backend,
        )
        output = capture.run(*args_values, **kwargs)
    print(f"export: PASS {_summary(output)}")
    print(
        f"capture: phase={capture.context.phase} "
        f"cache_state={capture.context.cache_state} "
        f"inputs={len(capture.context.input_order)}"
    )
    return 0


def _run_megakernel(args: argparse.Namespace) -> int:
    """Compile a Hugging Face causal LM prefill into one megakernel artifact."""
    import dataclasses
    import time

    from transformers import AutoModelForCausalLM

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name,
            revision=args.revision,
            attn_implementation="eager",
            dtype=getattr(torch, args.dtype),
        )
        .to("cuda")
        .eval()
    )
    model.config.use_cache = False
    input_ids = torch.tensor([args.input_ids], device="cuda")
    attention_mask = torch.ones_like(input_ids)

    backend = make_backend(
        mode="prefill",
        compile_kernel=True,
        cache=False,
        dump_dir=output_dir,
        scheduler="phase",
    )
    compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    started = time.perf_counter()
    with torch.no_grad():
        actual = compiled_model(input_ids, attention_mask=attention_mask).logits
        compile_seconds = time.perf_counter() - started
        expected = model(input_ids, attention_mask=attention_mask).logits
    relative_error = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()

    compiled = backend.megabake_last_compiled
    kernel_object = None
    dump_to_object = getattr(compiled.compiled, "dump_to_object", None)
    if callable(dump_to_object):
        kernel_object = output_dir / "megakernel.o"
        kernel_object.write_bytes(dump_to_object("megabake_kernel"))
    summary = {
        "model": args.model_name,
        "revision": args.revision,
        "mode": "prefill",
        "input_ids": args.input_ids,
        "dtype": args.dtype,
        "generated_source": str(compiled.generated.source_path),
        "source_map": str(compiled.generated.source_map_path),
        "kernel_object": None if kernel_object is None else str(kernel_object),
        "phases": len(compiled.plan.phases),
        "graph_operations": len(compiled.plan.graph.ops),
        "launch_policy": compiled.generated.launch.launch_policy,
        "resources": dataclasses.asdict(compiled.resources),
        "compile_and_first_call_seconds": round(compile_seconds, 2),
        "recurring_gpu_operations": backend.megabake_last_trace[-1].recurring_gpu_operations,
        "relative_l2_error_vs_eager": relative_error,
        "argmax_matches_eager": bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(
        f"megakernel: {summary['phases']} phases, "
        f"{compiled.resources.body_variant_count} body variants, "
        f"relative error {relative_error:.4f} vs eager, "
        f"{compile_seconds:.1f} s"
    )
    print(f"generated source: {summary['generated_source']}")
    if kernel_object is not None:
        print(f"kernel object: {kernel_object}")
    print(f"summary: {summary_path}")
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    checks = run_checks(require_cuda=args.cuda)
    print(format_checks(checks))
    return 1 if any(check.status == "fail" for check in checks) else 0


def _package_version() -> str:
    try:
        return version("megabake")
    except PackageNotFoundError:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode", choices=("prefill", "decode"), default=argparse.SUPPRESS)
    common.add_argument("--fallback", choices=("error", "inductor"), default=argparse.SUPPRESS)
    common.add_argument("--no-cache", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--dump-dir", default=argparse.SUPPRESS)
    parser = argparse.ArgumentParser(prog="megabake", parents=[common])
    parser.add_argument("--version", action="version", version=f"megabake {_package_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", parents=[common])
    demo.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    demo.add_argument("--model-name", default="demo")
    demo.set_defaults(handler=_run_demo)

    transformers_parser = subparsers.add_parser("transformers", parents=[common])
    transformers_parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM-135M")
    transformers_parser.add_argument("--revision")
    transformers_parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    transformers_parser.add_argument(
        "--input-ids",
        type=lambda value: [int(item) for item in value.split(",")],
        default="1,2,3,4",
    )
    transformers_parser.add_argument("--dump-compute-graph", action="store_true", default=False)
    transformers_parser.add_argument("--dump-lowering-graph", action="store_true", default=False)
    transformers_parser.add_argument("--dump-region-graph", action="store_true", default=False)
    transformers_parser.set_defaults(handler=_run_transformers)

    megakernel_parser = subparsers.add_parser(
        "megakernel", help="compile a Hugging Face causal LM prefill into one kernel"
    )
    megakernel_parser.add_argument("--model-name", required=True)
    megakernel_parser.add_argument("--revision")
    megakernel_parser.add_argument(
        "--input-ids",
        type=lambda value: [int(item) for item in value.split(",")],
        default="1,504,3458,314",
    )
    megakernel_parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    megakernel_parser.add_argument("--out", default="artifacts/megakernel")
    megakernel_parser.set_defaults(handler=_run_megakernel)

    export_parser = subparsers.add_parser("export", parents=[common])
    export_parser.add_argument("--callable", required=True)
    export_parser.add_argument("--model-name")
    export_parser.add_argument("--arg", action="append", default=[], metavar="JSON")
    export_parser.add_argument("--kwarg", action="append", default=[], metavar="NAME=JSON")
    export_parser.set_defaults(handler=_run_export)

    doctor_parser = subparsers.add_parser(
        "doctor", help="check Python, PyTorch, GPU and CUDA toolkit setup"
    )
    doctor_parser.add_argument(
        "--cuda", action="store_true", help="treat missing CUDA pieces as failures, not warnings"
    )
    doctor_parser.set_defaults(handler=_run_doctor)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
