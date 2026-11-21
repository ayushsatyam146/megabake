"""Read and measure device-specific choices for SM100 GEMM shapes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _device_details(device_index: int) -> dict:
    import torch
    from cuda.bindings import driver

    properties = torch.cuda.get_device_properties(device_index)
    root = Path(__file__).resolve().parents[3]
    repository = root.parents[1]
    files = (
        Path(__file__).with_name("tcgen_gemm.py"),
        Path(__file__).with_name("descriptors.py"),
        Path(__file__).with_name("persistent_gemm.py"),
        Path(__file__).with_name("cutlass_gemm.py"),
        Path(__file__).with_name("config.py"),
        Path(__file__).with_name("gemm.py"),
        Path(__file__).with_name("table.py"),
        Path(__file__).with_name("__init__.py"),
        root / "bodies/interface.py",
        root / "bodies/catalog.py",
        Path(__file__).with_name("attention.py"),
        *(
            root / name
            for name in (
                "codegen.py",
                "compiler.py",
                "graph.py",
                "ir.py",
                "ops/gemm.py",
                "ops/attention.py",
                "plan.py",
                "planner.py",
                "platform.py",
                "regions.py",
                "schedule.py",
            )
        ),
        repository / "benchmarks/full_model.py",
        repository / "benchmarks/stateful_models.py",
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    try:
        cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        cutlass_version = None
    status, driver_version = driver.cuDriverGetVersion()
    return {
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "multiprocessor_count": properties.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass_dsl": cutlass_version,
        "driver": int(driver_version) if int(status) == 0 else None,
        "implementation_sha256": digest.hexdigest(),
    }


def device_fingerprint(device_index: int) -> dict:
    from .attention import DECODE_SECOND_WAVE_KEYS, _prefill_value_block
    from .config import persistent_gemm_settings

    token_setting = os.environ.get("MEGABAKE_SM100_TOKEN_TILE", "8")
    settings = persistent_gemm_settings()
    return {
        **_device_details(device_index),
        "fused_epilogues": os.environ.get("MEGABAKE_SM100_EPILOGUE", "1") == "1",
        "matvec": os.environ.get("MEGABAKE_SM100_MATVEC") == "1",
        "paired_matvec": os.environ.get("MEGABAKE_SM100_PAIRED_MATVEC") == "1",
        "persistent_gemm": settings.enabled,
        "persistent_cutlass_fork": settings.cutlass_fork,
        "persistent_width": settings.width if settings.enabled else 128,
        "persistent_acc_stages": settings.stages if settings.enabled else 1,
        "decode_second_wave_keys": DECODE_SECOND_WAVE_KEYS,
        "token_tile": int(token_setting) if token_setting in {"8", "16", "32"} else 8,
        "prefill_value_block": _prefill_value_block(),
    }


@lru_cache(maxsize=16)
def _read(path: str, mtime_ns: int, size: int) -> dict:
    del mtime_ns, size
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError("SM100 tuning file must use schema 1")
    if not isinstance(data.get("choices"), dict):
        raise ValueError("SM100 tuning file has no choices mapping")  # noqa: TRY004
    return data


def choice_for_shape(target, rows: int, columns: int, reduction: int):
    path = os.environ.get("MEGABAKE_SM100_TUNING_FILE")
    if not path:
        return None
    import torch

    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    data = _read(str(resolved), stat.st_mtime_ns, stat.st_size)
    fingerprint = device_fingerprint(torch.cuda.current_device())
    if data.get("fingerprint") != fingerprint:
        raise ValueError(
            f"SM100 tuning file {resolved} does not match this device or implementation"
        )
    if (
        target.architecture != "sm100"
        or target.multiprocessor_count != fingerprint["multiprocessor_count"]
    ):
        raise ValueError("SM100 tuning file does not match the compilation target")
    choice = data["choices"].get(f"{rows}:{columns}:{reduction}")
    if choice is None:
        return None
    if (
        not isinstance(choice, dict)
        or choice.get("tile") not in (64, 128)
        or choice.get("splits") not in (1, 2, 4, 8)
    ):
        raise ValueError(f"invalid SM100 tuning choice for {(rows, columns, reduction)}")
    tile, splits = choice["tile"], choice["splits"]
    if columns % tile or reduction % (64 * splits):
        raise ValueError(f"incompatible SM100 tuning choice for {(rows, columns, reduction)}")
    return tile, splits


def tune_gemm(args) -> None:
    """Measure GEMM choices; validate the resulting file on the full model."""
    import gc
    import statistics
    import tempfile

    import torch

    from megabake import compiler, graphed, make_backend

    from .gemm import split_k

    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    def kernel_samples(function, count: int, cache_scrub: torch.Tensor | None) -> list[float]:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
            for _ in range(count):
                if cache_scrub is not None:
                    # A full model streams many different weight matrices. Reusing
                    # a single small GEMM leaves its weights unrealistically hot.
                    cache_scrub.sum()
                function()
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            profile.export_chrome_trace(str(trace_path))
            trace = json.loads(trace_path.read_text())
        samples = [
            float(event["dur"])
            for event in trace["traceEvents"]
            if event.get("cat") == "kernel" and "megabake_kernel" in event.get("name", "")
        ]
        if len(samples) != count:
            raise RuntimeError(f"expected {count} megakernel samples, got {len(samples)}")
        return samples

    if args.tune_rows < 1 or args.tune_rows > 128 or args.samples < 5 or args.tune_cache_mib < 0:
        raise ValueError("rows must be 1–128, samples at least 5, and cache size nonnegative")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise SystemExit("An SM100 CUDA device is required")
    shapes = []
    for shape in args.tune_shapes:
        try:
            columns, reduction = map(int, shape.lower().split("x"))
            if columns < 128 or columns % 64 or reduction < 64 or reduction % 64:
                raise ValueError
            shapes.append((columns, reduction))
        except ValueError as error:
            raise ValueError(f"invalid N x K shape: {shape}") from error

    fingerprint = device_fingerprint(torch.cuda.current_device())
    os.environ.pop("MEGABAKE_SM100_TUNING_FILE", None)
    os.environ["MEGABAKE_SM100_TCGEN"] = "1"
    os.environ["MEGABAKE_SM100_TCGEN_DECODE"] = "1"
    os.environ["MEGABAKE_SM100_TCGEN_MAX_ROWS"] = "128"
    os.environ["MEGABAKE_SM100_EPILOGUE"] = "0"
    compiler._TABLE_LAUNCH_OPS = 0
    torch.manual_seed(771)
    report = {
        "schema": 1,
        "fingerprint": fingerprint,
        "cache_scrub_bytes": args.tune_cache_mib * 1024 * 1024,
        "note": "Isolated tuning disables fused epilogues; validate choices on the full model.",
        "choices": {},
        "measurements": {},
    }
    cache_scrub = (
        torch.zeros(args.tune_cache_mib * 1024 * 1024 // 2, device="cuda", dtype=torch.float16)
        if args.tune_cache_mib
        else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for columns, reduction in shapes:
        source = torch.randn(args.tune_rows, reduction, device="cuda", dtype=torch.bfloat16) / 16
        weight = torch.randn(columns, reduction, device="cuda", dtype=torch.bfloat16) / 16
        with torch.no_grad():
            reference = Gemm()(source, weight).float()
        records, seen = [], set()
        for tile in (64, 128):
            if columns % tile:
                continue
            for cap in map(int, args.tune_split_limits):
                os.environ["MEGABAKE_SM100_TCGEN_TILE"] = str(tile)
                os.environ["MEGABAKE_SM100_SPLIT_K"] = str(cap)
                torch._dynamo.reset()
                compiled = graph = backend = None
                try:
                    backend = make_backend(
                        mode="decode" if args.tune_rows == 1 else "prefill",
                        compile_kernel=True,
                        cache=False,
                        scheduler="phase",
                    )
                    compiled = torch.compile(
                        Gemm().eval(), backend=backend, fullgraph=True, dynamic=False
                    )
                    with torch.no_grad():
                        output = compiled(source, weight).float()
                        phase = next(
                            p
                            for p in backend.megabake_last_compiled.plan.phases
                            if p.body.provider == "sm100_tma"
                        )
                        actual_split = split_k(phase.body)
                        if (tile, actual_split) in seen:
                            continue
                        seen.add((tile, actual_split))
                        error = ((output - reference).norm() / reference.norm()).item()
                        if not torch.isfinite(output).all().item() or error > 0.02:
                            raise RuntimeError(f"relative L2 error {error:.6f} exceeds 0.02")
                        graph = graphed(compiled, source, weight)
                        for _ in range(5):
                            graph(source, weight)
                        torch.cuda.synchronize()
                        samples = kernel_samples(
                            lambda graph=graph, source=source, weight=weight: graph(source, weight),
                            args.samples,
                            cache_scrub,
                        )
                    record = {
                        "tile": tile,
                        "splits": actual_split,
                        "median_kernel_us": statistics.median(samples),
                        "relative_l2_vs_eager": error,
                        "samples_us": samples,
                    }
                    records.append(record)
                    print(f"{args.tune_rows}x{columns}x{reduction}: {record}", flush=True)
                except Exception as error:  # noqa: BLE001 - keep measuring other candidates
                    records.append(
                        {
                            "tile": tile,
                            "split_limit": cap,
                            "error": f"{type(error).__name__}: {error}"[:1000],
                        }
                    )
                    print(f"{args.tune_rows}x{columns}x{reduction}: {records[-1]}", flush=True)
                finally:
                    del graph, compiled, backend
                    gc.collect()
                    torch.cuda.empty_cache()
        valid = [record for record in records if "median_kernel_us" in record]
        key = f"{args.tune_rows}:{columns}:{reduction}"
        report["measurements"][key] = records
        if valid:
            best = min(valid, key=lambda record: record["median_kernel_us"])
            report["choices"][key] = {"tile": best["tile"], "splits": best["splits"]}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del source, weight, reference
        torch.cuda.empty_cache()
    if len(report["choices"]) != len(shapes):
        raise SystemExit("At least one shape had no valid measured choice")


def tune_model(args, profiler: Path) -> None:
    """Sweep table GEMM choices using existing full-model phase profiles."""
    import collections
    import subprocess
    import sys

    import torch

    if args.out is None:
        raise ValueError("--out is required for model tuning")
    if args.calls < 1 or args.profile_timeout < 1 or args.capacity <= args.calls + 4:
        raise ValueError(
            "calls and profile timeout must be positive and capacity must exceed calls + 4"
        )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise SystemExit("An SM100 CUDA device is required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    runs_dir = args.out.parent / f"{args.out.stem}-profiles"
    runs_dir.mkdir(exist_ok=True)
    measurements = collections.defaultdict(list)
    failures = []
    for tile in args.tiles:
        for split_limit in args.split_limits:
            label = f"tile{tile}-split{split_limit}"
            result = runs_dir / f"{label}.json"
            log = runs_dir / f"{label}.log"
            env = os.environ.copy()
            env.pop("MEGABAKE_SM100_TUNING_FILE", None)
            env.update(
                MEGABAKE_SM100_TCGEN="1",
                MEGABAKE_SM100_TCGEN_TILE=tile,
                MEGABAKE_SM100_SPLIT_K=split_limit,
            )
            command = [
                sys.executable,
                str(profiler),
                "--model",
                args.model,
                "--mode",
                "decode",
                "--capacity",
                str(args.capacity),
                "--calls",
                str(args.calls),
                "--cta",
                "--out",
                str(result),
            ]
            if args.revision:
                command.extend(("--revision", args.revision))
            command.extend(("--attention", args.attention))
            print(f"Profiling {label}", flush=True)
            try:
                with log.open("w") as handle:
                    run = subprocess.run(
                        command,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=args.profile_timeout,
                    )
            except subprocess.TimeoutExpired:
                failures.append({"candidate": label, "error": "timeout", "log": str(log)})
                print(f"{label} timed out; see {log}", flush=True)
                continue
            if run.returncode:
                failures.append({"candidate": label, "exit_code": run.returncode, "log": str(log)})
                print(f"{label} failed; see {log}", flush=True)
                continue
            profile = json.loads(result.read_text())
            grouped = collections.defaultdict(list)
            for row in profile["cta"]["phases"]:
                if "gemm_shape" in row:
                    key = (tuple(row["gemm_shape"]), row["feature_tile"], row["split_k"])
                    grouped[key].append(row["us"])
            for (shape, feature_tile, splits), times in grouped.items():
                key = ":".join(map(str, shape))
                measurements[key].append(
                    {
                        "tile": feature_tile,
                        "splits": splits,
                        "mean_critical_us": sum(times) / len(times),
                        "occurrences": len(times),
                        "profile": str(result),
                    }
                )
    choices = {}
    for shape, records in measurements.items():
        best = min(records, key=lambda record: record["mean_critical_us"])
        choices[shape] = {"tile": best["tile"], "splits": best["splits"]}
    report = {
        "schema": 1,
        "fingerprint": device_fingerprint(torch.cuda.current_device()),
        "model": args.model,
        "method": "instrumented full-model decode phases",
        "note": "Validate combined choices with full-model performance and correctness measurements.",
        "choices": choices,
        "measurements": dict(measurements),
        "failures": failures,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"choices": choices, "failures": failures}, indent=2), flush=True)
    if not choices:
        raise SystemExit("No SM100 GEMM phases were measured")
