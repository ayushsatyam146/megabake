"""SM100 GEMM launch and code-generation helpers."""

import inspect

from . import gemm
from .config import persistent_gemm_settings

PROVIDER_NAME = "sm100_tma"


def validate_launch(event_schedule, table):
    if event_schedule is not None or not table:
        raise ValueError("selected GEMM body requires a cooperative phase-table launch")


def widen_selection(selection, threads):
    return selection


def descriptor_splits(splits):
    return tuple(1 for _ in splits)


def arrange_operands(activation, weights, output):
    return (*weights, activation, output.t())


def codegen_config(body, program):
    if gemm.paired_contraction(program):
        raise ValueError("SM100 table GEMM does not support paired contractions")
    rows, columns = body.tiles.cta_shape[:2]
    return (
        (columns, rows),
        gemm.split_k(body),
        False,
        body.name.endswith("_epilogue"),
        False,
        True,
    )


def table_import():
    imports = (
        "from megabake.bodies.cuda.sm100.tcgen_gemm import (\n"
        "    Sm100TmaGemm,\n"
        "    build_sm100_table_descriptors as build_table_descriptors,\n"
        ")\n"
        "from megabake.bodies.cuda.sm100.persistent_gemm import Sm100PersistentGemm\n"
        "from megabake.bodies.cuda.sm100.descriptors import table_descriptors\n"
    )
    if persistent_gemm_settings().cutlass_fork:
        imports += "from megabake.bodies.cuda.sm100.cutlass_gemm import Sm100CutlassForkGemm\n"
    return imports


def table_definition(spec, tile, epilogues):
    settings = persistent_gemm_settings()
    persistent, width, stages, cutlass_fork = (
        settings.enabled,
        settings.width,
        settings.stages,
        settings.cutlass_fork,
    )
    symbol = (
        "Sm100TmaGemm"
        if persistent and width == 128
        else "Sm100CutlassForkGemm"
        if cutlass_fork
        else "Sm100PersistentGemm"
        if persistent
        else "Sm100TmaGemm"
    )
    return (
        f"{spec['symbol']} = {symbol}({tile[0]}, {spec['splits']}, {tile[1]}"
        + (f", acc_stages={stages}" if persistent and width == 256 else "")
        + (", warp_specialized=True" if persistent and width == 128 else "")
        + (f", epilogues=({epilogues})" if epilogues else "")
        + ")"
    )


def source_object():
    settings = persistent_gemm_settings()
    if settings.enabled:
        if settings.cutlass_fork:
            from .cutlass_gemm import PersistentDenseGemmKernel

            return inspect.unwrap(PersistentDenseGemmKernel.table_kernel)
        if settings.width == 128:
            from .tcgen_gemm import _run_tcgen_tile

            return inspect.unwrap(_run_tcgen_tile)
        from .persistent_gemm import _persistent_run

        return inspect.unwrap(_persistent_run)
    from .tcgen_gemm import _run_tcgen_tile

    return inspect.unwrap(_run_tcgen_tile)
