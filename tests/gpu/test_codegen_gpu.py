from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace

import cutlass
import pytest
import torch
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from megabake import (
    assign_storage,
    derive_dependencies,
    emit_cute,
    make_phase_schedule,
    select_bodies,
)
from megabake.bodies import SelectionPolicy, TargetFacts
from megabake.bodies.cuda.sm90.gemm import (
    compile_g0_range_gemm,
    cute_gemm_tensors,
    g0_range_body,
)
from megabake.frontend import prepare_fx_graph
from megabake.graph import import_fx
from megabake.lowering import canonicalize_semantics
from megabake.plan import ConversionBody, CuTeLayout
from megabake.regions import form_regions
from megabake.runtime import make_ready

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0),
    reason="generated CuTe modules require an SM90 GPU",
)


def _target() -> TargetFacts:
    properties = torch.cuda.get_device_properties(0)
    return TargetFacts(
        architecture="sm90a",
        threads_per_warp=32,
        max_threads_per_cta=1024,
        max_shared_memory_per_cta=232_448,
        max_registers_per_thread=255,
        multiprocessor_count=properties.multi_processor_count,
        supports_tma=True,
        supports_wgmma=True,
    )


def _plan(model, *inputs, allow_tuned=True):
    prepared = prepare_fx_graph(torch.export.export(model, inputs).graph_module, inputs)
    graph = form_regions(canonicalize_semantics(import_fx(prepared.graph, prepared.context)))
    selection = select_bodies(graph, _target(), SelectionPolicy(allow_tuned=allow_tuned))
    return assign_storage(make_phase_schedule(selection, derive_dependencies(selection)))


def _load(generated):
    spec = importlib.util.spec_from_file_location(generated.module_name, generated.source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[generated.module_name] = module
    spec.loader.exec_module(module)
    return module


def _view(tensor):
    result = from_dlpack(tensor, assumed_align=16)
    leading_dimension = next(
        (index for index, stride in enumerate(tensor.stride()) if stride == 1),
        None,
    )
    if leading_dimension is None:
        return result
    return result.mark_layout_dynamic(leading_dim=leading_dimension)


def _run_generic(generated, source):
    module = _load(generated)
    tensors = []
    for argument in generated.launch.tensors:
        if argument.kind == "input":
            tensors.append(source)
        else:
            tensors.append(
                torch.empty_strided(
                    argument.layout.shape,
                    argument.layout.strides,
                    dtype=getattr(torch, argument.layout.dtype.removeprefix("torch.")),
                    device="cuda",
                )
            )
    ready = make_ready(generated.launch.grid_size, "cuda")
    work_iterators = [
        torch.arange(iterator.length, device="cuda", dtype=torch.int64) * iterator.step
        for iterator in generated.launch.work_iterators
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    iterator_views = tuple(_view(iterator) for iterator in work_iterators)
    assert generated.launch.launch_policy == "megabake_cooperative_table"
    # Slot addresses, then each phase column's resolved slot address.
    slot_addresses = [tensor.data_ptr() for tensor in tensors]
    slot_addresses += [slot_addresses[slot] for slot in generated.launch.resolved_slots]
    addresses = torch.tensor(slot_addresses, dtype=torch.int64, device="cuda")
    phase_table = torch.tensor(generated.launch.phase_table, dtype=torch.int32, device="cuda")
    # Ring entry 0, stamped with call 0, holds this call's boundary addresses.
    dynamic_addresses = torch.tensor(
        [0, *(slot_addresses[slot] for slot in generated.launch.dynamic_slots)],
        dtype=torch.int64,
    ).pin_memory()
    dynamic_slots = torch.tensor(
        list(generated.launch.dynamic_slots) or [0], dtype=torch.int32, device="cuda"
    )
    phase_floats = torch.empty(
        max(generated.launch.phase_float_elements, 4), dtype=torch.float32, device="cuda"
    )
    phase_counters = torch.zeros(
        max(generated.launch.phase_counter_count, 4), dtype=torch.int32, device="cuda"
    )
    views = [
        _view(ready),
        cutlass.Int32(generated.launch.grid_size),
        _view(addresses),
        _view(phase_table),
        iterator_views,
        # Per-call boundary addresses: a host-mapped buffer and its slots.
        cutlass.Int64(dynamic_addresses.data_ptr()),
        _view(dynamic_slots),
        # Phase scratch: FP32 elements and zeroed Int32 counters.
        _view(phase_floats),
        _view(phase_counters),
    ]
    compiled = cute.compile(module.launch, *views, stream)
    compiled(*views, stream)
    torch.cuda.synchronize()
    return tensors[-1]


def test_generated_cooperative_module_compiles_and_matches_reference(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, value):
            return torch.relu(value) + torch.relu(value)

    source = torch.randn(5, 70)
    plan = _plan(Model(), source, allow_tuned=False)
    generated = emit_cute(plan, tmp_path)
    output = _run_generic(generated, source.cuda())

    torch.testing.assert_close(output.cpu(), torch.relu(source) + torch.relu(source))


def test_generated_layout_conversion_compiles_and_matches_reference(tmp_path, monkeypatch):
    # relu -> sum would fuse as a reduction prologue, leaving no conversion.
    monkeypatch.setenv("MEGABAKE_REDUCTION_PROLOGUE", "0")

    class Model(torch.nn.Module):
        def forward(self, value):
            return value.relu().sum(dim=-1)

    source = torch.randn(2, 3)
    plan = _plan(Model(), source, allow_tuned=False)
    producer = next(
        selected
        for selected in plan.selection.selections
        if selected.program.family == "elementwise"
    )
    consumer = next(
        selected for selected in plan.selection.selections if selected.program.family == "reduction"
    )
    value_id = producer.outputs[0]
    source_layout = CuTeLayout.from_contract(producer.body.layouts.outputs[value_id])
    target_layout = replace(source_layout, strides=(1, 3))
    load = consumer.program.loads[0]
    consumer_program = replace(
        consumer.program,
        loads=(replace(load, strides=target_layout.strides),),
    )
    consumer_body = replace(
        consumer.body,
        layouts=replace(
            consumer.body.layouts,
            inputs={
                value_id: replace(
                    consumer.body.layouts.inputs[value_id],
                    strides=target_layout.strides,
                )
            },
        ),
    )
    conversion = ConversionBody(
        value_id,
        producer.region_id,
        consumer.region_id,
        source_layout,
        target_layout,
        24,
        (producer.region_id,),
    )
    converted_selection = replace(
        plan.selection,
        selections=(
            producer,
            replace(consumer, program=consumer_program, body=consumer_body),
        ),
        conversions=(conversion,),
    )
    converted_plan = assign_storage(
        make_phase_schedule(converted_selection, derive_dependencies(converted_selection))
    )

    generated = emit_cute(converted_plan, tmp_path)
    output = _run_generic(generated, source.cuda())

    torch.testing.assert_close(output.cpu(), source.relu().sum(dim=-1))


def test_generated_g0_kernel_matches_existing_g0_executable(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    source = torch.randn(4, 576, dtype=torch.bfloat16)
    weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    plan = _plan(Model(), source, weight)
    generated = emit_cute(plan, tmp_path)
    assert generated.launch.phases[0].provider == "g0_range"
    assert generated.launch.cooperative
    assert generated.source.count("@cute.kernel") == 1
    assert ".prepare(" in generated.source
    assert ".run(" in generated.source

    module = _load(generated)
    gpu_source = source.cuda()
    gpu_weight = weight.cuda()
    output = torch.empty(4, 1536, device="cuda", dtype=torch.bfloat16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    tensor_a, tensor_b, tensor_c = cute_gemm_tensors(gpu_source, gpu_weight, output)
    ready = make_ready(generated.launch.grid_size, "cuda")
    tensor_views = (
        *(_view(tensor) for tensor in (gpu_source, gpu_weight, output, gpu_weight.t())),
        tensor_a,
        tensor_b,
        tensor_c,
    )
    views = [
        _view(ready),
        cutlass.Int32(generated.launch.grid_size),
        cutlass.Int32(0),
        (),
        tensor_views,
        ((tensor_a, tensor_b, tensor_c),),
    ]
    compiled = cute.compile(module.launch, *views, max_clusters, stream)
    compiled(*views, stream)
    torch.cuda.synchronize()
    generated_output = output.clone()

    existing = compile_g0_range_gemm(
        g0_range_body(plan.phases[0].body),
        gpu_source,
        gpu_weight,
        output,
        max_clusters,
        stream,
    )
    existing(gpu_source, gpu_weight, output, stream)
    torch.cuda.synchronize()

    torch.testing.assert_close(generated_output, output, rtol=0, atol=0)


def _run_mixed(generated, source, weight, output_shape):
    module = _load(generated)
    gpu_source = source.cuda()
    gpu_weight = weight.cuda()
    workspace = torch.empty((4, 1536), device="cuda", dtype=torch.bfloat16)
    output = torch.empty(output_shape, device="cuda", dtype=torch.bfloat16)
    ready = make_ready(generated.launch.grid_size, "cuda")
    work_iterator = (
        torch.arange(
            generated.launch.work_iterators[0].length,
            device="cuda",
            dtype=torch.int64,
        )
        * generated.launch.work_iterators[0].step
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    tensor_a, tensor_b, tensor_c = cute_gemm_tensors(gpu_source, gpu_weight, workspace)
    views = [
        _view(ready),
        cutlass.Int32(generated.launch.grid_size),
        cutlass.Int32(0),
        (_view(work_iterator),),
        (
            *(
                _view(tensor)
                for tensor in (
                    workspace,
                    gpu_source,
                    gpu_weight,
                    output,
                    gpu_weight.t(),
                )
            ),
            tensor_a,
            tensor_b,
            tensor_c,
        ),
        ((tensor_a, tensor_b, tensor_c),),
    ]
    compiled = cute.compile(module.launch, *views, max_clusters, stream)
    compiled(*views, stream)
    torch.cuda.synchronize()
    return output


def test_generated_mixed_g0_and_generic_kernel_matches_reference(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return torch.relu(source @ weight.T)

    source = torch.randn(4, 576, dtype=torch.bfloat16)
    weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    plan = _plan(Model(), source, weight)
    generated = emit_cute(plan, tmp_path)

    assert [phase.provider for phase in generated.launch.phases] == [
        "g0_range",
        "generic",
    ]
    assert generated.launch.threads_per_block == 256
    assert generated.source.count("@cute.kernel") == 1
    assert "grid_barrier(ready, workers, epoch + 0)" in generated.source
    assert generated.launch.phases[0].barrier is not None

    output = _run_mixed(generated, source, weight, (4, 1536))

    torch.testing.assert_close(output.cpu(), torch.relu(source @ weight.T))


def test_generated_mixed_g0_and_reduction_kernel_matches_reference(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, source, weight):
            return (source @ weight.T).sum(dim=-1)

    source = torch.randn(4, 576, dtype=torch.bfloat16)
    weight = torch.randn(1536, 576, dtype=torch.bfloat16)
    plan = _plan(Model(), source, weight)
    generated = emit_cute(plan, tmp_path)

    assert [phase.provider for phase in generated.launch.phases] == [
        "g0_range",
        "generic",
    ]
    assert generated.launch.threads_per_block == 256
    assert generated.launch.work_iterators[0].step == 1

    output = _run_mixed(generated, source, weight, (4,))
    torch.testing.assert_close(
        output.cpu(),
        (source @ weight.T).sum(dim=-1),
        rtol=0.03,
        atol=8,
    )
