import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megabake.bodies import SelectionPolicy, catalog
from megabake.compiler import UnsupportedCompiler, _search_scheduler, _target_facts
from megabake.plan import PlanSearchResult, SearchBudget
from megabake.planner import _contraction_body_order


def test_sm100_table_dispatch_keeps_hopper_on_its_existing_path():
    from megabake.bodies.cuda.sm100 import table

    assert catalog.is_table_gemm("g0_range")
    assert catalog.table_gemm("g0_range") is None
    assert catalog.is_table_gemm("sm100_tma")
    assert catalog.table_gemm("sm100_tma") is table
    assert not catalog.is_table_gemm("generic")
    assert catalog.table_gemm("generic") is None


def test_sm100_split_k_does_not_replace_hopper_helper():
    from megabake.bodies.cuda.sm90.gemm import split_k as hopper_split_k
    from megabake.bodies.cuda.sm100.gemm import split_k as sm100_split_k

    hopper = SimpleNamespace(name="g0_range_swapab_16x64_k4_bias")
    sm100 = SimpleNamespace(name="sm100_tma_swapab_8x128_persistent1_fork_k8_epilogue")
    assert hopper_split_k(hopper) == 4
    assert hopper_split_k(sm100) == 1
    assert sm100_split_k(sm100) == 8


def test_sm100_table_codegen_configuration():
    from megabake.bodies.cuda.sm100 import table
    from megabake.ir import Accumulator

    body = SimpleNamespace(
        name="sm100_tma_swapab_8x128_persistent1_fork_k8_epilogue",
        tiles=SimpleNamespace(cta_shape=(8, 128, 64)),
    )
    assert table.codegen_config(body, SimpleNamespace(post=None)) == (
        (128, 8),
        8,
        False,
        True,
        False,
        True,
    )
    with pytest.raises(ValueError, match="paired contractions"):
        table.codegen_config(body, SimpleNamespace(post=Accumulator("float32", index=1)))


def test_sm100_table_owns_layout_and_launch_decisions():
    from megabake.bodies.cuda.sm100 import table

    table.validate_launch(None, True)
    for schedule, enabled in ((None, False), (object(), True)):
        with pytest.raises(ValueError, match="phase-table launch"):
            table.validate_launch(schedule, enabled)
    assert table.descriptor_splits((1, 4, 8)) == (1, 1, 1)

    selection = object()
    assert table.widen_selection(selection, 256) is selection
    activation = torch.randn(1, 4)
    weight = torch.randn(8, 4)
    output = torch.randn(1, 8)
    a, b, c = table.arrange_operands(activation, (weight,), output)
    assert a is weight and b is activation
    assert c.shape == (8, 1) and c.data_ptr() == output.data_ptr()


def test_sm100_descriptors_fit_the_shared_table_allocation():
    # The shared runtime sizes the descriptor table with Hopper's count.
    from megabake.bodies.cuda.sm90.gemm import TABLE_DESCRIPTORS_PER_GEMM
    from megabake.bodies.cuda.sm100 import descriptors

    assert descriptors._DESCRIPTORS_PER_GEMM == TABLE_DESCRIPTORS_PER_GEMM


@pytest.mark.parametrize(
    ("persistent", "width", "fork", "filename"),
    [
        ("0", "256", "0", "tcgen_gemm.py"),
        ("1", "128", "0", "tcgen_gemm.py"),
        ("1", "256", "0", "persistent_gemm.py"),
        ("1", "256", "1", "cutlass_gemm.py"),
    ],
)
def test_sm100_source_map_uses_the_selected_body(monkeypatch, persistent, width, fork, filename):
    from megabake.bodies.cuda.sm100 import table

    monkeypatch.setenv("MEGABAKE_SM100_PERSISTENT", persistent)
    monkeypatch.setenv("MEGABAKE_SM100_PERSISTENT_WIDTH", width)
    monkeypatch.setenv("MEGABAKE_SM100_CUTLASS_FORK", fork)
    monkeypatch.setenv("MEGABAKE_SM100_PERSISTENT_STAGES", "1")
    assert Path(inspect.getsourcefile(table.source_object())).name == filename


def test_sm100_descriptors_require_full_operands():
    from megabake.bodies.cuda.sm100.gemm import cute_gemm_tensors

    with pytest.raises(ValueError, match="require full operands"):
        cute_gemm_tensors(None, None, None, splits=2)


def test_planner_accepts_provider_selection_priority():
    body = SimpleNamespace(
        provider="new_provider",
        selection_priority=-3,
        participants=SimpleNamespace(participating_threads=128, registers_per_thread=32),
        scratch=SimpleNamespace(size_bytes=0),
        evidence=SimpleNamespace(measurements={}),
    )
    assert _contraction_body_order(body, None, SelectionPolicy())[0] == -3


def test_core_dispatch_does_not_hardcode_table_gemm_providers():
    body_names = {"sm100_tma"}
    core = Path(__file__).resolve().parents[2] / "src" / "megabake"
    hardware_modules = ("megabake.bodies.cuda.sm100", "bodies.cuda.sm100")
    for name in ("codegen.py", "compiler.py", "plan.py", "planner.py"):
        tree = ast.parse((core / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in body_names, (name, node.lineno, node.value)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(hardware_modules)
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(hardware_modules) for alias in node.names)


@pytest.mark.parametrize(
    ("major", "minor", "architecture", "wgmma"),
    [(9, 0, "sm90a", True), (10, 0, "sm100", False), (12, 1, "sm121", False)],
)
def test_cuda_target_facts_preserve_existing_architecture_mapping(
    monkeypatch, major, minor, architecture, wgmma
):
    properties = SimpleNamespace(
        major=major,
        minor=minor,
        warp_size=32,
        max_threads_per_block=1024,
        shared_memory_per_block_optin=200_000,
        multi_processor_count=100,
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    target = _target_facts(torch.device("cuda:0"))
    assert target.architecture == architecture
    assert target.supports_wgmma is wgmma


def test_cuda_target_rejects_unregistered_device(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(major=8, minor=0)
    )
    with pytest.raises(UnsupportedCompiler, match="found sm80"):
        _target_facts(torch.device("cuda:0"))


def test_event_scheduler_preserves_launch_options(monkeypatch):
    options = object()
    seen = []

    def search(graph, target, policy, budget):
        seen.append(policy)
        return PlanSearchResult(None, None, (), (), False, None)

    monkeypatch.setattr("megabake.compiler.search_plans", search)
    monkeypatch.setattr("megabake.compiler._make_plan_evaluator", lambda *args: object())
    policy = SelectionPolicy(table_launch=True, target_options=options)
    _search_scheduler(None, None, policy, SearchBudget(), None, None, None, "region-events")
    assert seen[0].event_schedule
    assert seen[0].table_launch
    assert seen[0].target_options is options
