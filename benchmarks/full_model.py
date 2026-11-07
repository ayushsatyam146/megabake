"""Time Hugging Face causal LMs: eager, Inductor, Inductor with CUDA graphs, and MegaBake.

Prefill runs a four-token prompt (--prompt-tokens); decode advances a stateful KV cache one
token per call. Each variant reports the median complete call (host and GPU,
measured with CUDA events around the call), the profiled kernel time and launch
count of one call, and its first-call (compile) time. The default model is the
pinned SmolLM-135M checkpoint; pass several `--model`s to write one report per
model under `--out/<org>--<name>`. A variant that fails is recorded with its
error and the run continues.

    .venv/bin/python benchmarks/full_model.py --out artifacts/full_model
    .venv/bin/python benchmarks/full_model.py --modes decode --capacity 256
    .venv/bin/python benchmarks/full_model.py --model Qwen/Qwen3.5-0.8B google/gemma-3-270m
    .venv/bin/python benchmarks/full_model.py --modes decode --decode-contexts 512 2048
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # This checkout's sources, not whatever ``megabake`` the environment
    # installed (an editable install points at one tree only).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from megabake import graphed, make_backend  # noqa: E402
from benchmarks.measure import kernel_trace  # noqa: E402
from benchmarks.stateful_models import StatefulCausalLM, layer_types, text_config  # noqa: E402
from megabake.fx_handler.hf_attention import (  # noqa: E402
    ATTENTION_IMPLEMENTATION,
    DecodeStep,
    register_transformers_attention,
)
from tests.fixtures import SMOLLM_MODEL_NAME, SMOLLM_REVISION, StatefulSmolLM  # noqa: E402
from megabake.ops.attention import kv_attention  # noqa: E402

PROMPT = [1, 504, 3458, 314]
VARIANTS = (
    "eager", "inductor", "inductor_cudagraphs", "inductor_graphed", "inductor_autotune_graphed",
    "megabake", "megabake_graphed",
)
# Inductor at its strongest for decode: autotuned kernels, captured in a CUDA
# graph by hand (its own CUDA graphs skip steps that mutate the KV state).
AUTOTUNE_MODE = "max-autotune-no-cudagraphs"


def _load(name: str, revision: str | None, attention: str = "eager") -> torch.nn.Module:
    """``attention="megabake"`` declares attention to MegaBake through the
    transformers attention interface instead of leaving it to pattern matching."""
    from transformers import AutoModelForCausalLM

    if attention == ATTENTION_IMPLEMENTATION:
        register_transformers_attention()
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        attn_implementation=attention,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    text_config(model).use_cache = False
    return model.to("cuda").eval()


def _time_calls(function, count: int) -> dict[str, object]:
    host, device = [], []
    for _ in range(count):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        function()
        end.record()
        end.synchronize()
        host.append((time.perf_counter() - started) * 1e6)
        device.append(start.elapsed_time(end) * 1e3)
    return {
        "complete_call_us": statistics.median(host),
        "device_span_us": statistics.median(device),
        "complete_call_samples_us": host,
        "device_span_samples_us": device,
    }


def _pipelined_calls(calls) -> float:
    """Mean time per call when ``calls`` are issued back to back with one sync."""
    # A full collection over the compilers' objects takes ~0.6 s; inside the
    # window it would add ~15 ms per call to whichever variant it lands in.
    gc.collect()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for call in calls:
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1e6 / len(calls)


def _measure(
    name: str, build, step, args: argparse.Namespace, traces: Path, pipelined=None
) -> dict:
    torch._dynamo.reset()
    function = build()
    started = time.perf_counter()
    step(function)
    torch.cuda.synchronize()
    first_call = time.perf_counter() - started
    for _ in range(args.warmup):
        step(function)
    entry = _time_calls(lambda: step(function), args.samples)
    if pipelined is not None:
        # Back-to-back calls, as a decode loop with on-device sampling issues
        # them: host work of one call overlaps the previous call's kernels.
        entry["pipelined_call_us"] = _pipelined_calls(pipelined(function))
    trace = kernel_trace(lambda: step(function), traces / f"{name}.json")
    entry.update(
        first_call_s=first_call,
        launch_count=trace["launch_count"],
        kernel_duration_us=trace["kernel_duration_us"],
    )
    summary = {key: value for key, value in entry.items() if not key.endswith("samples_us")}
    print(f"{name}: {json.dumps(summary)}", flush=True)
    return entry


def _measure_or_error(
    name: str, build, step, args: argparse.Namespace, traces: Path, pipelined=None
) -> dict:
    try:
        return _measure(name, build, step, args, traces, pipelined)
    except Exception as error:  # noqa: BLE001 - keep benchmarking the other variants
        traceback.print_exc()
        print(f"{name}: FAILED {type(error).__name__}: {error}", flush=True)
        return {"error": f"{type(error).__name__}: {error}"}


def _megabake(mode: str, out: Path):
    return make_backend(
        mode=mode, compile_kernel=True, cache=False, dump_dir=out / "dumps" / mode, scheduler="phase"
    )


def bench_prefill(model, args) -> dict:
    prompt = (PROMPT * -(-args.prompt_tokens // len(PROMPT)))[: args.prompt_tokens]
    ids = torch.tensor([prompt], device="cuda")
    mask = torch.ones_like(ids)
    variants = {
        "eager": lambda: model,
        "inductor": lambda: torch.compile(model, fullgraph=True, dynamic=False),
        # CUDA Graph replay through Inductor: the strongest launch-overhead baseline.
        "inductor_cudagraphs": lambda: torch.compile(
            model, mode="reduce-overhead", fullgraph=True, dynamic=False
        ),
        "megabake": lambda: torch.compile(
            model, backend=_megabake("prefill", args.out), fullgraph=True, dynamic=False
        ),
    }
    with torch.no_grad():
        return {
            "prompt_tokens": len(prompt),
            **{
                name: _measure_or_error(
                    f"prefill_{name}",
                    build,
                    lambda function: function(ids, attention_mask=mask),
                    args,
                    args.out / "traces",
                )
                for name, build in variants.items()
                if name in args.variants
            },
        }


def decode_module(model, capacity: int, batch: int = 1) -> tuple[torch.nn.Module, list]:
    """The one-token decode wrapper and the shapes and dtypes of its state tensors.

    With ``batch`` > 1 every per-sequence state tensor (KV caches, conv and
    recurrent state) gets that leading dimension; the int32 valid lengths stay
    ``[1]``, shared by the batch, since every sequence decodes in lockstep.
    """
    stateful, state_shapes = _decode_module(model, capacity)
    if batch == 1:
        return stateful, state_shapes
    return stateful, [
        [((batch, *shape[1:]) if len(shape) > 1 else shape, dtype) for shape, dtype in layer]
        for layer in state_shapes
    ]


def _decode_module(model, capacity: int) -> tuple[torch.nn.Module, list]:
    if model.config.model_type == "llama":
        # Keep the hand-written Llama wrapper the tests and published numbers use.
        config = model.config
        layers = config.num_hidden_layers
        shape = (
            1,
            capacity,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
        )
        state_shapes = [[(shape, torch.bfloat16)]] * 2 * layers + [[((1,), torch.int32)]] * layers
        return StatefulSmolLM(model).eval(), state_shapes
    if set(layer_types(model)) <= {"full_attention", "sliding_attention"}:
        # Any Hugging Face attention model: declared attention over explicit KV state.
        step = DecodeStep(model, capacity).eval()
        return step, step.state_shapes()
    # Hybrid token mixers (linear attention) are not declared ops yet.
    stateful = StatefulCausalLM(model).eval()
    return stateful, stateful.state_shapes(capacity)


def _static_kv_attention(
    source, key, value, key_cache, value_cache, valid_length, scale=0.0, window=0, softcap=0.0
):
    """``megabake::kv_attention`` without host synchronization.

    The reference reads the valid length on the host to slice the cache, which
    no CUDA graph can capture. This version writes the new entry at a device
    index and attends over the whole capacity with positions past the valid
    length masked out, as static-cache decoders (vLLM, Hugging Face's static
    cache) do.
    """
    index = valid_length.long()
    key_cache.index_copy_(1, index, key)
    value_cache.index_copy_(1, index, value)
    valid_length.add_(1)
    query = source.permute(0, 2, 1, 3)
    keys = key_cache.permute(0, 2, 1, 3)
    values = value_cache.permute(0, 2, 1, 3)
    if query.shape[1] != keys.shape[1]:
        group_size = query.shape[1] // keys.shape[1]
        keys = keys.repeat_interleave(group_size, dim=1)
        values = values.repeat_interleave(group_size, dim=1)
    scores = (query @ keys.transpose(-2, -1)).float() * (scale or source.shape[-1] ** -0.5)
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    positions = torch.arange(key_cache.shape[1], device=source.device)
    valid = positions < valid_length
    if window:
        valid &= positions >= valid_length - window
    scores = scores.masked_fill(~valid, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1).to(source.dtype)
    output = probabilities @ values
    return output.permute(0, 2, 1, 3).reshape(
        source.shape[0] * source.shape[2], source.shape[1], source.shape[3]
    )


@contextlib.contextmanager
def _graphable_kv_attention():
    """Run ``megabake::kv_attention`` calls through ``_static_kv_attention``."""
    kernels = kv_attention._backend_fns
    reference = kernels[None]
    kernels[None] = _static_kv_attention
    try:
        yield
    finally:
        kernels[None] = reference


class GraphedStep:
    """One decode step of ``function`` captured in a CUDA graph by hand.

    Inductor's own CUDA graphs skip decode because the KV state tensors are
    mutated inputs. Here they are the graph's static inputs: every call must
    pass the same state tensors, and the call copies the new token position
    into the captured buffer before replaying, so the caches advance in
    place as they do for MegaBake. Attention runs as ``_static_kv_attention``
    so nothing reads the device on the host. Warm-up and capture run on a
    side stream; the state is zeroed again afterwards so the first replay is
    position 0.
    """

    def __init__(self, function, token, state, warmup: int = 3):
        self.state = state
        self.position = torch.zeros((1, 1), dtype=torch.long, device="cuda")
        with _graphable_kv_attention():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    function(token, self.position, *state)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = function(token, self.position, *state)
        for tensor in state:
            tensor.zero_()
        torch.cuda.synchronize()

    def __call__(self, token, position_ids, *state):
        if any(given is not captured for given, captured in zip(state, self.state, strict=True)):
            raise ValueError("a graphed decode step must reuse its captured state tensors")
        self.position.copy_(position_ids)
        self.graph.replay()
        return self.output


def bench_decode(model, args) -> dict:
    stateful, state_shapes = decode_module(model, args.capacity, args.batch)
    token = torch.full((args.batch, 1), 504, device="cuda")
    variants = {
        "eager": lambda: stateful,
        "inductor": lambda: torch.compile(stateful, fullgraph=True, dynamic=False),
        "inductor_cudagraphs": lambda: torch.compile(
            stateful, mode="reduce-overhead", fullgraph=True, dynamic=False
        ),
        # Built per run below: the capture binds that run's state tensors.
        "inductor_graphed": None,
        "inductor_autotune_graphed": None,
        "megabake": lambda: torch.compile(
            stateful, backend=_megabake("decode", args.out), fullgraph=True, dynamic=False
        ),
        "megabake_graphed": None,
    }
    calls = 1 + args.warmup + 2 * args.samples + 1
    if calls > args.capacity:
        raise SystemExit(f"--capacity must be at least {calls} for this many calls")
    results: dict[str, object] = {"capacity": args.capacity, "batch": args.batch}
    with torch.no_grad():
        for name, build in variants.items():
            if name not in args.variants:
                continue
            state = [
                torch.zeros(shape, device="cuda", dtype=dtype)
                for layer in state_shapes
                for shape, dtype in layer
            ]
            position = [0]
            if name == "inductor_graphed":
                build = lambda state=state: GraphedStep(  # noqa: E731
                    torch.compile(stateful, fullgraph=True, dynamic=False), token, state
                )
            if name == "inductor_autotune_graphed":
                build = lambda state=state: GraphedStep(  # noqa: E731
                    torch.compile(stateful, mode=AUTOTUNE_MODE, fullgraph=True, dynamic=False), token, state
                )
            if name == "megabake_graphed":
                build = lambda state=state: graphed(  # noqa: E731
                    torch.compile(
                        stateful, backend=_megabake("decode", args.out), fullgraph=True, dynamic=False
                    ),
                    token,
                    torch.zeros((1, 1), dtype=torch.long, device="cuda"),
                    *state,
                )

            def step(function, state=state, position=position):
                position_ids = torch.tensor([[position[0]]], device="cuda")
                position[0] += 1
                return function(token, position_ids, *state)

            def pipelined_steps(function, state=state, position=position):
                # Positions made on the device ahead of time: a pageable host
                # copy would synchronize the stream on every call.
                # Separate tensors: views at different storage offsets can
                # trip alignment guards and recompile.
                ahead = [
                    row.clone()
                    for row in torch.arange(
                        position[0], position[0] + args.samples, device="cuda"
                    ).view(-1, 1, 1)
                ]
                position[0] += args.samples
                return [lambda row=row: function(token, row, *state) for row in ahead]

            results[name] = _measure_or_error(
                f"decode_{name}", build, step, args, args.out / "traces", pipelined_steps
            )
            results[name]["final_position"] = position[0]
            torch.cuda.empty_cache()
    return results


def bench_decode_contexts(model, args) -> dict:
    """Decode with ``--decode-contexts`` tokens already cached.

    Every wrapper keeps an int32 valid length per attention layer; setting
    them (and the positions) to L puts the call at context L. Cache contents
    stay zero: decode time does not depend on them. Each variant compiles
    once at the capacity of the largest context and is timed at every L.
    """
    contexts = sorted(args.decode_contexts)
    capacity = contexts[-1] + 2 * args.samples + 8
    stateful, state_shapes = decode_module(model, capacity, args.batch)
    token = torch.full((args.batch, 1), 504, device="cuda")
    variants = {
        "eager": lambda: stateful,
        "inductor": lambda: torch.compile(stateful, fullgraph=True, dynamic=False),
        "inductor_cudagraphs": lambda: torch.compile(
            stateful, mode="reduce-overhead", fullgraph=True, dynamic=False
        ),
        "megabake": lambda: torch.compile(
            stateful, backend=_megabake("decode", args.out), fullgraph=True, dynamic=False
        ),
        # Built per variant below: the capture binds that variant's state tensors.
        "inductor_graphed": None,
        "inductor_autotune_graphed": None,
        "megabake_graphed": None,
    }
    results: dict[str, object] = {"capacity": capacity, "batch": args.batch, "contexts": contexts}
    with torch.no_grad():
        for name, build in variants.items():
            if name not in args.variants:
                continue
            try:
                torch._dynamo.reset()
                state = [
                    torch.zeros(shape, device="cuda", dtype=dtype)
                    for layer in state_shapes
                    for shape, dtype in layer
                ]
                lengths = [t for t in state if t.dtype == torch.int32 and t.numel() == 1]
                if name == "inductor_graphed":
                    function = GraphedStep(
                        torch.compile(stateful, fullgraph=True, dynamic=False), token, state
                    )
                elif name == "inductor_autotune_graphed":
                    function = GraphedStep(
                        torch.compile(stateful, mode=AUTOTUNE_MODE, fullgraph=True, dynamic=False),
                        token,
                        state,
                    )
                elif name == "megabake_graphed":
                    function = graphed(
                        torch.compile(
                            stateful, backend=_megabake("decode", args.out), fullgraph=True, dynamic=False
                        ),
                        token,
                        torch.tensor([[0]], device="cuda"),
                        *state,
                    )
                else:
                    function = build()
                function(token, torch.tensor([[0]], device="cuda"), *state)
                rows = {}
                for context in contexts:
                    # Separate device tensors made before timing: a host copy
                    # inside the loop would synchronize every call.
                    positions = [
                        torch.tensor([[context + offset]], device="cuda")
                        for offset in range(2 * args.samples + 4)
                    ]
                    for length in lengths:
                        length.fill_(context)
                    for position in positions[:4]:
                        function(token, position, *state)
                    torch.cuda.synchronize()
                    complete = []
                    for position in positions[4 : 4 + args.samples]:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        function(token, position, *state)
                        torch.cuda.synchronize()
                        complete.append((time.perf_counter() - started) * 1e6)
                    pipelined = _pipelined_calls(
                        [
                            lambda position=position: function(token, position, *state)
                            for position in positions[4 + args.samples :]
                        ]
                    )
                    rows[str(context)] = {
                        "complete_call_us": statistics.median(complete),
                        "pipelined_call_us": pipelined,
                    }
                results[name] = rows
                summary = {
                    context: round(row["pipelined_call_us"]) for context, row in rows.items()
                }
                print(f"decode_contexts_{name} pipelined_call_us: {json.dumps(summary)}", flush=True)
            except Exception as error:  # noqa: BLE001 - keep benchmarking the other variants
                traceback.print_exc()
                print(f"decode_contexts_{name}: FAILED {type(error).__name__}: {error}", flush=True)
                results[name] = {"error": f"{type(error).__name__}: {error}"}
            torch.cuda.empty_cache()
    return results


def _bench_model(name: str, revision: str | None, args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"== {name}@{revision or 'main'} -> {args.out}", flush=True)
    model = _load(name, revision, args.attention)
    report: dict[str, object] = {
        "model": f"{name}@{revision or 'main'}",
        "attention": args.attention,
        "layer_types": sorted(set(layer_types(model))),
        "gpu": torch.cuda.get_device_name(),
        "samples": args.samples,
    }
    for mode in args.modes:
        report[mode] = (bench_prefill if mode == "prefill" else bench_decode)(model, args)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.decode_contexts:
        report["decode_contexts"] = bench_decode_contexts(model, args)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out / 'report.json'}")
    del model
    torch._dynamo.reset()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", nargs="+", default=[SMOLLM_MODEL_NAME], help="Hugging Face causal LM id(s)"
    )
    parser.add_argument(
        "--revision", help=f"checkpoint revision (default: pinned for {SMOLLM_MODEL_NAME}, else main)"
    )
    parser.add_argument(
        "--attention",
        choices=("eager", ATTENTION_IMPLEMENTATION),
        default="eager",
        help="prefill attention: eager (pattern-matched by MegaBake) or the declared megabake op",
    )
    parser.add_argument("--modes", nargs="*", choices=("prefill", "decode"), default=["prefill", "decode"])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="variants to time (default: all)",
    )
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument(
        "--prompt-tokens", type=int, default=len(PROMPT), help="prefill prompt length"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--capacity", type=int, default=128, help="decode KV cache capacity")
    parser.add_argument("--batch", type=int, default=1, help="decode batch size (sequences in lockstep)")
    parser.add_argument(
        "--decode-contexts",
        type=int,
        nargs="*",
        default=[],
        help="also time decode with this many tokens already cached (e.g. 512 2048)",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/full_model"))
    args = parser.parse_args()
    root = args.out
    for name in args.model:
        revision = args.revision or (SMOLLM_REVISION if name == SMOLLM_MODEL_NAME else None)
        args.out = root / name.replace("/", "--") if len(args.model) > 1 else root
        _bench_model(name, revision, args)


if __name__ == "__main__":
    main()
