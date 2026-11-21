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
    .venv/bin/python benchmarks/full_model.py --modes decode-audit --model Qwen/Qwen2.5-0.5B-Instruct --capacity 192 --warmup 5 --samples 24

The Inductor decode baseline is ``inductor_hf``: the unmodified Hugging Face
model, its own attention and a static cache, compiled end to end with CUDA
graphs. The other ``inductor*`` decode variants compile MegaBake's decode
wrapper, whose attention is an opaque custom op to Inductor; they are kept as
diagnostics only (see ``WRAPPED_INDUCTOR``).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import json
import os
import statistics
import subprocess
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

from benchmarks.measure import kernel_trace
from benchmarks.stateful_models import StatefulCausalLM, layer_types, text_config
from megabake import graphed, make_backend
from megabake.fx_handler.hf_attention import (
    ATTENTION_IMPLEMENTATION,
    DecodeStep,
    mixer_of,
    register_transformers_attention,
)
from megabake.ops.attention import kv_attention
from tests.fixtures import SMOLLM_MODEL_NAME, SMOLLM_REVISION, StatefulSmolLM

PROMPT = [1, 504, 3458, 314]
VARIANTS = (
    "eager", "inductor", "inductor_cudagraphs", "inductor_graphed", "inductor_autotune_graphed",
    "inductor_hf", "megabake", "megabake_graphed",
)
# Decode variants that compile MegaBake's own decode wrapper (``decode_module``),
# whose attention is the ``megabake::kv_attention`` custom op. Inductor cannot
# see into that op: it calls it as one opaque kernel, and graphed runs replace
# it with ``_static_kv_attention`` (eager, FP32 scores over the whole cache
# capacity). They are diagnostics, not baselines. ``inductor_hf`` is the
# baseline: the unmodified Hugging Face model with its own attention and a
# static KV cache, compiled by Inductor end to end.
WRAPPED_INDUCTOR = ("inductor", "inductor_cudagraphs", "inductor_graphed", "inductor_autotune_graphed")
# Inductor at its strongest for decode: autotuned kernels, captured in a CUDA
# graph by hand (its own CUDA graphs skip steps that mutate the KV state).
AUTOTUNE_MODE = "max-autotune-no-cudagraphs"


def _load(name: str, revision: str | None, attention: str = "eager") -> torch.nn.Module:
    """``attention="megabake"`` declares attention to MegaBake through the
    transformers attention interface instead of leaving it to pattern matching."""
    from transformers import AutoModelForCausalLM

    if attention == ATTENTION_IMPLEMENTATION:
        register_transformers_attention()
    # Checkpoints that ship their own modeling code run it only on request:
    # MEGABAKE_TRUST_REMOTE_CODE=1, with a pinned --revision of code you read.
    trust = os.environ.get("MEGABAKE_TRUST_REMOTE_CODE") == "1"
    if trust and revision is None:
        raise ValueError("MEGABAKE_TRUST_REMOTE_CODE=1 needs a pinned --revision")
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        attn_implementation=attention,
        dtype=torch.bfloat16,
        trust_remote_code=trust,
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
    name: str, build, step, args: argparse.Namespace, traces: Path, pipelined=None, validate=None
) -> dict:
    torch._dynamo.reset()
    function = build()
    started = time.perf_counter()
    first_result = step(function)
    torch.cuda.synchronize()
    first_call = time.perf_counter() - started
    correctness = validate(first_result) if validate is not None else None
    del first_result
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
    if correctness is not None:
        entry["correctness"] = correctness
    summary = {key: value for key, value in entry.items() if not key.endswith("samples_us")}
    print(f"{name}: {json.dumps(summary)}", flush=True)
    return entry


def _measure_or_error(
    name: str, build, step, args: argparse.Namespace, traces: Path, pipelined=None, validate=None
) -> dict:
    try:
        return _measure(name, build, step, args, traces, pipelined, validate)
    except Exception as error:  # noqa: BLE001 - keep benchmarking the other variants
        traceback.print_exc()
        print(f"{name}: FAILED {type(error).__name__}: {error}", flush=True)
        return {"error": f"{type(error).__name__}: {error}"}


def _megabake(mode: str, out: Path):
    return make_backend(
        mode=mode,
        compile_kernel=True,
        cache=False,
        dump_dir=out / "dumps" / mode,
        scheduler="phase",
    )


class GraphedPrefill:
    """Capture one MegaBake prefill with stable input and mask buffers."""

    def __init__(self, function, ids, mask):
        self.graph = graphed(
            lambda tokens, attention_mask: function(tokens, attention_mask=attention_mask),
            ids,
            mask,
        )

    def __call__(self, ids, attention_mask):
        return self.graph(ids, attention_mask)


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
        "megabake_graphed": lambda: GraphedPrefill(
            torch.compile(
                model, backend=_megabake("prefill", args.out), fullgraph=True, dynamic=False
            ),
            ids,
            mask,
        ),
    }
    with torch.no_grad():
        reference = (
            model(ids, attention_mask=mask).logits.detach().float()
            if getattr(args, "check_prefill_correctness", False)
            else None
        )

        def validate(output):
            actual = output.logits.detach().float()
            difference = actual - reference
            return {
                "relative_l2_vs_eager": (difference.norm() / reference.norm()).item(),
                "max_abs_vs_eager": difference.abs().max().item(),
                "top_token_matches": int((actual.argmax(-1) == reference.argmax(-1)).sum().item()),
                "positions": int(reference.shape[-2]),
                "finite": bool(torch.isfinite(actual).all().item()),
            }

        return {
            "prompt_tokens": len(prompt),
            **{
                name: _measure_or_error(
                    f"prefill_{name}",
                    build,
                    lambda function: function(ids, attention_mask=mask),
                    args,
                    args.out / "traces",
                    validate=validate if reference is not None else None,
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
    mixers = [module for module in model.modules() if mixer_of(module) is not None]
    if mixers and len(mixers) == model.config.get_text_config(decoder=True).num_hidden_layers:
        # Every layer's token mixer has a declaration (Mamba2, ComplexKDA and
        # its hybrid attention): ``megabake::linear_attention`` and
        # ``megabake::kv_attention`` over DecodeStep's explicit state.
        step = DecodeStep(model, capacity).eval()
        return step, step.state_shapes()
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
        self.token = token
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
        if token is not self.token:
            self.token.copy_(token)
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
        # The baseline: Hugging Face's own model and static cache (HFStaticDecode).
        "inductor_hf": None,
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
                build = lambda state=state: GraphedStep(
                    torch.compile(stateful, fullgraph=True, dynamic=False), token, state
                )
            if name == "inductor_autotune_graphed":
                build = lambda state=state: GraphedStep(  # noqa: E731
                    torch.compile(stateful, mode=AUTOTUNE_MODE, fullgraph=True, dynamic=False), token, state
                )
            if name == "inductor_hf":
                build = lambda: HFStaticDecode(  # noqa: E731
                    args.model_name, args.model_revision, args.capacity, args.batch, args.hf_attention
                )
            if name == "megabake_graphed":
                build = lambda state=state: graphed(
                    torch.compile(
                        stateful,
                        backend=_megabake("decode", args.out),
                        fullgraph=True,
                        dynamic=False,
                    ),
                    token,
                    torch.zeros((1, 1), dtype=torch.long, device="cuda"),
                    *state,
                )

            hf = name == "inductor_hf"

            def step(function, state=state, position=position, hf=hf):
                position_ids = torch.tensor([[position[0]]], device="cuda")
                position[0] += 1
                return function(token, position_ids) if hf else function(token, position_ids, *state)

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
                if hf:
                    return [lambda row=row: function(token, row) for row in ahead]
                return [lambda row=row: function(token, row, *state) for row in ahead]

            results[name] = _measure_or_error(
                f"decode_{name}", build, step, args, args.out / "traces", pipelined_steps
            )
            results[name]["final_position"] = position[0]
            torch.cuda.empty_cache()
    return results


class HFStaticDecode:
    """Hugging Face decode the way its users compile it: the unmodified
    model with its default attention (``sdpa``) and a ``StaticCache``, one
    ``torch.compile(mode="reduce-overhead")`` over the whole forward, so
    Inductor sees every op (attention included) and replays CUDA graphs.

    Nothing of MegaBake is in this graph. The cache is ``capacity`` keys
    long; Hugging Face attends over the whole static cache with a mask, so
    callers size it to the context they time. Call it as
    ``step(token, position)``; ``position`` is a ``[1, 1]`` device tensor,
    and the cache position is the same value.
    """

    def __init__(
        self, name: str, revision: str | None, capacity: int, batch: int = 1, attention: str = "sdpa"
    ):
        from transformers import AutoModelForCausalLM, StaticCache

        # cuDNN's attention fails to load its sublibrary on the test machine;
        # SDPA then takes flash or memory-efficient attention.
        torch.backends.cuda.enable_cudnn_sdp(False)
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(name, revision=revision)
        window = getattr(config, "sliding_window", None)
        if window and capacity <= window and getattr(config, "layer_types", None):
            # A window at least as long as the cache never drops a key, so
            # sliding layers compute full attention. Built as full-attention
            # layers: Hugging Face's sliding static cache keeps a Python
            # length that recompiles the graph on every call.
            config.layer_types = ["full_attention"] * len(config.layer_types)
            # sliding_window stays set: Gemma 3 builds its sliding mask
            # whatever the layer types, and raises without a window.
            if hasattr(config, "use_sliding_window"):
                config.use_sliding_window = False
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, config=config, attn_implementation=attention, dtype=torch.bfloat16
        ).to("cuda").eval()
        self.cache = StaticCache(config=self.model.config, max_cache_len=capacity)
        # StaticCache allocates its tensors on first use; one eager call does
        # that outside the CUDA graphs, so they stay the graphs' static inputs.
        with torch.no_grad():
            start = torch.zeros((batch, 1), dtype=torch.long, device="cuda")
            self.model(
                input_ids=torch.full((batch, 1), 504, device="cuda"),
                position_ids=start,
                cache_position=start[0],
                past_key_values=self.cache,
                use_cache=True,
            )
        self.forward = torch.compile(
            self.model, mode="reduce-overhead", fullgraph=True, dynamic=False
        )

    def __call__(self, token, position):
        return self.forward(
            input_ids=token,
            position_ids=position,
            cache_position=position.view(-1),
            past_key_values=self.cache,
            use_cache=True,
        ).logits


def _time_contexts(function, token, contexts, args, call=None, lengths=()):
    """{context: complete and pipelined call µs} of ``function`` at each
    context. ``call(function, token, position)`` makes one decode call."""
    call = call or (lambda function, token, position: function(token, position))
    rows = {}
    for context in contexts:
        # Separate device tensors made before timing: a host copy inside the
        # loop would synchronize every call.
        positions = [
            torch.tensor([[context + offset]], device="cuda")
            for offset in range(2 * args.samples + 4)
        ]
        for length in lengths:
            length.fill_(context)
        for position in positions[:4]:
            call(function, token, position)
        torch.cuda.synchronize()
        complete = []
        for position in positions[4 : 4 + args.samples]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            call(function, token, position)
            torch.cuda.synchronize()
            complete.append((time.perf_counter() - started) * 1e6)
        pipelined = _pipelined_calls(
            [
                lambda position=position: call(function, token, position)
                for position in positions[4 + args.samples :]
            ]
        )
        rows[str(context)] = {
            "complete_call_us": statistics.median(complete),
            "pipelined_call_us": pipelined,
        }
    return rows


def bench_hf_contexts(name: str, revision: str | None, args) -> dict:
    """``inductor_hf`` at every context, each with a static cache just long
    enough for its timed calls: the strongest Inductor configuration, since
    Hugging Face attends over the whole static cache."""
    rows = {}
    token = torch.full((args.batch, 1), 504, device="cuda")
    with torch.no_grad():
        for context in sorted(args.decode_contexts):
            torch._dynamo.reset()
            step = None
            try:
                step = HFStaticDecode(
                    name, revision, context + 2 * args.samples + 8, args.batch, args.hf_attention
                )
                step(token, torch.tensor([[0]], device="cuda"))
                rows.update(_time_contexts(step, token, [context], args))
            except Exception as error:  # noqa: BLE001 - one context's failure keeps the others
                print(f"inductor_hf @{context}: FAILED {type(error).__name__}: {error}", flush=True)
                rows[str(context)] = {"error": f"{type(error).__name__}: {error}"}
            del step
            torch.cuda.empty_cache()
    summary = {
        context: round(row["pipelined_call_us"]) if "pipelined_call_us" in row else "FAILED"
        for context, row in rows.items()
    }
    print(f"decode_contexts_inductor_hf pipelined_call_us: {json.dumps(summary)}", flush=True)
    return rows


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
            if name not in args.variants or name == "inductor_hf":
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
                            stateful,
                            backend=_megabake("decode", args.out),
                            fullgraph=True,
                            dynamic=False,
                        ),
                        token,
                        torch.tensor([[0]], device="cuda"),
                        *state,
                    )
                else:
                    function = build()
                function(token, torch.tensor([[0]], device="cuda"), *state)
                rows = _time_contexts(
                    function,
                    token,
                    contexts,
                    args,
                    call=lambda function, token, position: function(token, position, *state),
                    lengths=lengths,
                )
                results[name] = rows
                summary = {
                    context: round(row["pipelined_call_us"]) for context, row in rows.items()
                }
                print(
                    f"decode_contexts_{name} pipelined_call_us: {json.dumps(summary)}", flush=True
                )
            except Exception as error:  # noqa: BLE001 - keep benchmarking the other variants
                traceback.print_exc()
                print(f"decode_contexts_{name}: FAILED {type(error).__name__}: {error}", flush=True)
                results[name] = {"error": f"{type(error).__name__}: {error}"}
            torch.cuda.empty_cache()
    return results


def bench_decode_audit(model, name: str, revision: str | None, args) -> dict:
    """Compare decode graphs and prompt continuations with native eager."""
    warmups, samples, capacity = args.warmup, args.samples, args.capacity
    generated_tokens = args.audit_generated_tokens
    if warmups < 0 or samples < 1 or generated_tokens < 1 or capacity < warmups + samples:
        raise ValueError("capacity must hold warmup and sample positions; counts must be positive")
    if args.attention != "eager":
        raise ValueError("decode-audit requires --attention eager")
    if not 0 <= args.audit_teacher_steps <= capacity:
        raise ValueError("teacher-forced steps must fit in the cache")
    if args.audit_fp32 and not args.audit_teacher_steps:
        raise ValueError("--audit-fp32 requires --audit-teacher-steps")
    from transformers import AutoTokenizer

    prompts = [
        "The capital of France is",
        "Explain why the sky appears blue.",
        "Write a Python function to add two numbers.",
        "2 + 2 =",
        "Once upon a time in a distant village,",
    ]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    prompt_ids = [tokenizer(p, add_special_tokens=True)["input_ids"] for p in prompts]
    if any(not ids or len(ids) + generated_tokens >= capacity for ids in prompt_ids):
        raise ValueError("a prompt is empty or does not fit the audit cache")
    token = torch.tensor([[504]], device="cuda", dtype=torch.long)
    positions = [
        torch.tensor([[i]], device="cuda", dtype=torch.long) for i in range(warmups + samples)
    ]

    def timed(label, step):
        for index in range(warmups):
            step(index)
        torch.cuda.synchronize()
        wall, event = [], []
        for index in range(warmups, warmups + samples):
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter_ns()
            start.record()
            step(index)
            end.record()
            end.synchronize()
            wall.append((time.perf_counter_ns() - t0) / 1_000_000)
            event.append(start.elapsed_time(end))
        return {
            "name": label,
            "samples": samples,
            "warmups": warmups,
            "wall_median_ms": statistics.median(wall),
            "event_span_median_ms": statistics.median(event),
            "wall_samples_ms": wall,
            "event_samples_ms": event,
        }

    def native_reference(ids):
        cache = None
        prompt_logits = []
        for token_id in ids:
            out = model(
                input_ids=torch.tensor([[token_id]], device="cuda"),
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            prompt_logits.append(out.logits[0, -1].detach().float().cpu())
        next_token = int(prompt_logits[-1].argmax().item())
        sequence = []
        for index in range(generated_tokens):
            sequence.append(next_token)
            if index + 1 < generated_tokens:
                out = model(
                    input_ids=torch.tensor([[next_token]], device="cuda"),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                next_token = int(out.logits[0, -1].argmax().item())
        return prompt_logits, sequence

    report = {
        "model": name,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "dtype": str(next(model.parameters()).dtype),
        "capacity": capacity,
        "warmups": warmups,
        "samples": samples,
        "generated_tokens": generated_tokens,
        "timing_notes": "Synchronized host wall time, with device positions prepared before timing. "
        f"Native eager uses HF's growing cache; other variants use capacity-{capacity} "
        "explicit state. CUDA event spans include device idle gaps from host work.",
        "timings": {},
        "prompts": [],
    }

    with torch.no_grad():
        holder = {"cache": None}

        def native_step(_index, holder=holder):
            out = model(input_ids=token, past_key_values=holder["cache"], use_cache=True)
            holder["cache"] = out.past_key_values
            return out.logits

        report["timings"]["native_hf_eager"] = timed("native_hf_eager", native_step)
        del native_step, holder
        native = [native_reference(ids) for ids in prompt_ids]
        print("Native eager timing and prompt references complete", flush=True)

        eager, shapes = decode_module(model, capacity)

        def make_state():
            return [
                torch.zeros(shape, dtype=dtype, device="cuda")
                for layer in shapes
                for shape, dtype in layer
            ]

        eager_state, torch_state, megabake_state = make_state(), make_state(), make_state()
        report["timings"]["wrapper_eager"] = timed(
            "wrapper_eager", lambda i: eager(token, positions[i], *eager_state)
        )
        for value in eager_state:
            value.zero_()

        torch_graph = GraphedStep(
            torch.compile(eager, fullgraph=True, dynamic=False), token.clone(), torch_state
        )
        report["timings"]["torch_graph"] = timed(
            "torch_graph", lambda i: torch_graph(token, positions[i], *torch_state)
        )
        print("PyTorch graph timing complete", flush=True)

        backend = make_backend(mode="decode", compile_kernel=True, cache=False, scheduler="phase")
        compiled = torch.compile(eager, backend=backend, fullgraph=True, dynamic=False)
        megabake_graph = graphed(compiled, token.clone(), positions[0].clone(), *megabake_state)
        report["timings"]["megabake_graph"] = timed(
            "megabake_graph", lambda i: megabake_graph(token, positions[i], *megabake_state)
        )
        print("MegaBake graph timing complete", flush=True)

        variants = (
            ("wrapper_eager", eager, eager_state),
            ("torch_graph", torch_graph, torch_state),
            ("megabake_graph", megabake_graph, megabake_state),
        )
        for prompt, ids, (native_logits, native_sequence) in zip(
            prompts, prompt_ids, native, strict=True
        ):
            case = {
                "prompt": prompt,
                "tokens": len(ids),
                "native_greedy_ids": native_sequence,
                "variants": {},
            }
            for variant, function, state in variants:
                for value in state:
                    value.zero_()
                rels, matches, finite, out = [], 0, True, None
                for position, (token_id, ref) in enumerate(zip(ids, native_logits, strict=True)):
                    out = function(
                        torch.tensor([[token_id]], device="cuda"),
                        torch.tensor([[position]], device="cuda"),
                        *state,
                    )
                    actual = out[0, -1].detach().float().cpu()
                    finite &= bool(torch.isfinite(actual).all())
                    rels.append(float((actual - ref).norm() / ref.norm()))
                    matches += int(actual.argmax().item() == ref.argmax().item())
                next_token = int(actual.argmax().item())
                sequence = []
                for index in range(generated_tokens):
                    sequence.append(next_token)
                    if index + 1 < generated_tokens:
                        output = function(
                            torch.tensor([[next_token]], device="cuda"),
                            torch.tensor([[len(ids) + index]], device="cuda"),
                            *state,
                        )
                        finite &= bool(torch.isfinite(output).all().item())
                        next_token = int(output.argmax().item())
                mismatch = next(
                    (
                        i + 1
                        for i, (a, b) in enumerate(zip(native_sequence, sequence, strict=True))
                        if a != b
                    ),
                    None,
                )
                case["variants"][variant] = {
                    "prompt_top_matches_native": matches,
                    "prompt_positions": len(ids),
                    "prompt_relative_l2_median_vs_native": statistics.median(rels),
                    "prompt_relative_l2_max_vs_native": max(rels),
                    "all_logits_finite": finite,
                    "greedy_first_difference_vs_native": mismatch,
                    "greedy_ids": sequence,
                }
            report["prompts"].append(case)
            print(f"Prompt check complete: {prompt}", flush=True)
        if args.audit_teacher_steps:
            normal_state = make_state()
            for values in (eager_state, megabake_state):
                for value in values:
                    value.zero_()
            relatives, mismatches = [], []
            max_graph_normal = 0.0
            finite = {"eager": True, "megabake": True, "megabake_graph": True}
            for index in range(args.audit_teacher_steps):
                current_token = torch.tensor([[PROMPT[index % len(PROMPT)]]], device="cuda")
                position = torch.tensor([[index]], device="cuda")
                reference = eager(current_token, position, *eager_state).float()
                normal = compiled(current_token, position, *normal_state).float()
                actual = megabake_graph(current_token, position, *megabake_state).float()
                for label, value in (
                    ("eager", reference),
                    ("megabake", normal),
                    ("megabake_graph", actual),
                ):
                    finite[label] &= bool(torch.isfinite(value).all().item())
                relatives.append(float((actual - reference).norm() / reference.norm()))
                if actual.argmax(-1).item() != reference.argmax(-1).item():
                    mismatches.append(index)
                max_graph_normal = max(max_graph_normal, (actual - normal).abs().max().item())
            report["teacher_forced"] = {
                "steps": args.audit_teacher_steps,
                "relative_l2_median_vs_wrapper_eager": statistics.median(relatives),
                "relative_l2_max_vs_wrapper_eager": max(relatives),
                "top_mismatch_positions": mismatches,
                "finite": finite,
                "graph_vs_non_graph_max_abs": max_graph_normal,
                "graph_vs_non_graph_state_equal": all(
                    torch.equal(a, b) for a, b in zip(megabake_state, normal_state, strict=True)
                ),
            }
            if args.audit_fp32:
                import copy

                fp32_model = copy.deepcopy(model).float().eval()
                fp32, fp32_shapes = decode_module(fp32_model, capacity)
                fp32_state = [
                    torch.zeros(
                        shape,
                        device="cuda",
                        dtype=torch.float32 if dtype == torch.bfloat16 else dtype,
                    )
                    for layer in fp32_shapes
                    for shape, dtype in layer
                ]
                comparisons = (
                    ("wrapper_eager", eager, eager_state),
                    ("torch_graph", torch_graph, torch_state),
                    ("megabake", compiled, normal_state),
                    ("megabake_graph", megabake_graph, megabake_state),
                )
                for _, _, state in comparisons:
                    for value in state:
                        value.zero_()
                scores = {
                    name: {"relative_l2": [], "mismatch_positions": []}
                    for name, _, _ in comparisons
                }
                old_tf32 = torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    for index in range(args.audit_teacher_steps):
                        current_token = torch.tensor([[PROMPT[index % len(PROMPT)]]], device="cuda")
                        position = torch.tensor([[index]], device="cuda")
                        reference = fp32(current_token, position, *fp32_state).float()
                        for label, function, state in comparisons:
                            actual = function(current_token, position, *state).float()
                            scores[label]["relative_l2"].append(
                                float((actual - reference).norm() / reference.norm())
                            )
                            if actual.argmax(-1).item() != reference.argmax(-1).item():
                                scores[label]["mismatch_positions"].append(index)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = old_tf32
                for score in scores.values():
                    values = score["relative_l2"]
                    score["median_relative_l2"] = statistics.median(values)
                    score["max_relative_l2"] = max(values)
                    score["top_matches"] = args.audit_teacher_steps - len(
                        score["mismatch_positions"]
                    )
                report["fp32_teacher_forced"] = {
                    "reference": "FP32 arithmetic on BF16-rounded weights with TF32 disabled",
                    "steps": args.audit_teacher_steps,
                    "results": scores,
                }
    return report


def _bench_model(name: str, revision: str | None, args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    args.model_name, args.model_revision = name, revision
    print(f"== {name}@{revision or 'main'} -> {args.out}", flush=True)
    model = _load(name, revision, args.attention)
    report: dict[str, object] = {
        "model": f"{name}@{revision or 'main'}",
        "attention": args.attention,
        "layer_types": sorted(set(layer_types(model))),
        "gpu": torch.cuda.get_device_name(),
        "samples": args.samples,
    }
    if "decode-audit" in args.modes:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        packages = {}
        for package in ("torch", "transformers", "nvidia-cutlass-dsl"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        settings = {
            key: value if value.isdecimal() or value in {"auto", "cta"} else "<set>"
            for key, value in os.environ.items()
            if key.startswith("MEGABAKE_")
        }
        report["environment"] = {
            "commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "tracked_files_changed": bool(changes.stdout) if changes.returncode == 0 else None,
            "packages": packages,
            "cuda_runtime": torch.version.cuda,
            "settings": settings,
        }
    for mode in args.modes:
        if mode == "decode-audit":
            report[mode] = bench_decode_audit(model, name, revision, args)
        else:
            report[mode] = (bench_prefill if mode == "prefill" else bench_decode)(model, args)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.decode_contexts:
        report["decode_contexts"] = bench_decode_contexts(model, args)
        if "inductor_hf" in args.variants:
            try:
                report["decode_contexts"]["inductor_hf"] = bench_hf_contexts(name, revision, args)
            except Exception as error:  # noqa: BLE001 - keep the other variants' results
                traceback.print_exc()
                print(f"decode_contexts_inductor_hf: FAILED {type(error).__name__}: {error}", flush=True)
                report["decode_contexts"]["inductor_hf"] = {"error": f"{type(error).__name__}: {error}"}
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
        "--revision",
        help=f"checkpoint revision (default: pinned for {SMOLLM_MODEL_NAME}, else main)",
    )
    parser.add_argument(
        "--attention",
        choices=("eager", ATTENTION_IMPLEMENTATION),
        default="eager",
        help="prefill attention: eager (pattern-matched by MegaBake) or the declared megabake op",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=("prefill", "decode", "decode-audit"),
        default=["prefill", "decode"],
    )
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
    parser.add_argument(
        "--check-prefill-correctness",
        action="store_true",
        help="compare each prefill variant's logits with BF16 eager",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--audit-generated-tokens", type=int, default=16)
    parser.add_argument(
        "--audit-teacher-steps",
        type=int,
        default=0,
        help="also compare this many teacher-forced decode steps and graph state",
    )
    parser.add_argument(
        "--audit-fp32", action="store_true", help="also compare teacher-forced steps with FP32"
    )
    parser.add_argument("--capacity", type=int, default=128, help="decode KV cache capacity")
    parser.add_argument("--batch", type=int, default=1, help="decode batch size (sequences in lockstep)")
    parser.add_argument(
        "--decode-contexts",
        type=int,
        nargs="*",
        default=[],
        help="also time decode with this many tokens already cached (e.g. 512 2048)",
    )
    parser.add_argument(
        "--hf-attention",
        default="sdpa",
        help="attention implementation of the inductor_hf baseline (sdpa, eager, ...)",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/full_model"))
    args = parser.parse_args()
    root = args.out
    if len(args.model) > 1 and "decode-audit" in args.modes:
        failures = []
        for name in args.model:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model",
                name,
                "--modes",
                *args.modes,
                "--variants",
                *args.variants,
                "--attention",
                args.attention,
                "--samples",
                str(args.samples),
                "--prompt-tokens",
                str(args.prompt_tokens),
                "--warmup",
                str(args.warmup),
                "--capacity",
                str(args.capacity),
                "--audit-generated-tokens",
                str(args.audit_generated_tokens),
                "--audit-teacher-steps",
                str(args.audit_teacher_steps),
                "--out",
                str(root / name.replace("/", "--")),
            ]
            if args.revision:
                command.extend(("--revision", args.revision))
            if args.check_prefill_correctness:
                command.append("--check-prefill-correctness")
            if args.audit_fp32:
                command.append("--audit-fp32")
            if args.decode_contexts:
                command.extend(("--decode-contexts", *map(str, args.decode_contexts)))
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                failures.append(name)
        if failures:
            raise SystemExit(f"Benchmark failed for: {', '.join(failures)}")
        return
    for name in args.model:
        revision = args.revision or (SMOLLM_REVISION if name == SMOLLM_MODEL_NAME else None)
        args.out = root / name.replace("/", "--") if len(args.model) > 1 else root
        _bench_model(name, revision, args)


if __name__ == "__main__":
    main()
