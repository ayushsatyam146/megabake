from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch


def event_samples(function: Callable[[], object], count: int = 32) -> dict[str, object]:
    complete_samples_us = []
    event_samples_us = []
    for _ in range(count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start_host = time.perf_counter()
        start.record()
        function()
        end.record()
        end.synchronize()
        complete_samples_us.append((time.perf_counter() - start_host) * 1_000_000)
        event_samples_us.append(start.elapsed_time(end) * 1_000)
    return {
        "complete_call_us": statistics.median(complete_samples_us),
        "device_event_span_us": statistics.median(event_samples_us),
        "complete_call_samples_us": complete_samples_us,
        "device_event_span_samples_us": event_samples_us,
    }


def kernel_trace(function: Callable[[], object], path: Path) -> dict[str, object]:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        function()
        torch.cuda.synchronize()
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(path))
    trace = json.loads(path.read_text(encoding="utf-8"))
    kernels = [event for event in trace["traceEvents"] if event.get("cat") == "kernel"]
    launch_events = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") in {"cuda_runtime", "cuda_driver"}
        and "LaunchKernel" in event.get("name", "")
    ]
    kernel_starts = [float(event["ts"]) for event in kernels]
    kernel_ends = [float(event["ts"]) + float(event["dur"]) for event in kernels]
    first_launch = min(
        (float(event["ts"]) for event in launch_events),
        default=min(kernel_starts, default=0.0),
    )
    last_kernel_end = max(kernel_ends, default=0.0)
    return {
        "kernel_names": [event["name"] for event in kernels],
        "launch_count": len(kernels),
        "launch_event_count": len(launch_events),
        "kernel_duration_us": sum(float(event["dur"]) for event in kernels),
        "kernel_span_us": (max(kernel_ends) - min(kernel_starts) if kernel_starts else 0.0),
        "first_launch_to_last_kernel_us": last_kernel_end - first_launch,
        "trace_path": str(path),
    }
