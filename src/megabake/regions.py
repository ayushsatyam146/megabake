"""Region formation: one region per lowered op, each carrying the fusion
candidates that end at it. The ``ops`` families recognize the candidates;
the planner chooses among them."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .graph import ComputeGraph, Region, RegionCandidate, verify_compute_graph
from .ops import attention, elementwise, gemm, linear_attention, norm, prologue, rope
from .ops.matching import external_inputs_of, live_outputs_of


def form_regions(graph: ComputeGraph) -> ComputeGraph:
    verify_compute_graph(graph)
    candidates_by_tail = _fusion_candidates(graph)
    regions = tuple(
        replace(
            Region(
                id=f"region:{operation.id.removeprefix('op:')}",
                source_op_ids=(operation.id,),
                external_inputs=external_inputs_of((operation,), graph),
                live_outputs=live_outputs_of((operation,), graph),
                semantics=gemm.view_semantics(operation, graph),
                tensor_programs=(operation.tensor_program,)
                if operation.tensor_program is not None
                else (),
            ),
            fusion_candidates=candidates_by_tail.get(operation.id, ()),
        )
        for operation in graph.ops.values()
    )
    result = replace(graph, regions=regions)
    verify_compute_graph(result)
    _verify_regions(result)
    return result


def dump_region_graph(graph: ComputeGraph, path: str | Path) -> Path:
    operation_regions = {
        operation_id: region
        for region in graph.regions
        for operation_id in region.source_op_ids
    }
    lines = [
        "# RegionGraph",
        f"values={len(graph.values)} operations={len(graph.ops)} "
        f"regions={len(graph.regions)} inputs={len(graph.inputs)} "
        f"outputs={len(graph.outputs)}",
        "",
        "## Regions",
    ]
    for region in graph.regions:
        lines.append(
            f"{region.id} source_ops={list(region.source_op_ids)} "
            f"external_inputs={list(region.external_inputs)} "
            f"live_outputs={list(region.live_outputs)} "
            f"semantics={region.semantics!r}"
        )
        for candidate in region.fusion_candidates:
            lines.append(
                f"  candidate={candidate.id} source_ops={list(candidate.source_op_ids)} "
                f"executable={candidate.executable} rejections={list(candidate.rejections)}"
            )

    lines.extend(["", "## Dependencies"])
    for consumer in graph.regions:
        for input_id in consumer.external_inputs:
            producer_id = graph.values[input_id].producer
            if producer_id is None:
                continue
            producer = operation_regions[producer_id]
            lines.append(f"{producer.id} -> {consumer.id} value={input_id}")

    lines.extend(["", "## Graph Outputs"])
    for output_id in graph.outputs:
        producer_id = graph.values[output_id].producer
        region_id = operation_regions[producer_id].id if producer_id else "boundary"
        lines.append(f"{region_id} -> output value={output_id}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _fusion_candidates(graph: ComputeGraph) -> dict[str, tuple[RegionCandidate, ...]]:
    """Each op's fusion candidates, in the order the planner tries them."""
    rope_candidates = rope.candidates(graph)
    norm_candidates = norm.candidates(graph)
    reserved = frozenset(
        operation_id
        for candidate in (*rope_candidates, *norm_candidates)
        for operation_id in candidate.source_op_ids
    )
    chain_candidates = elementwise.candidates(graph, reserved)
    prologue_candidates = prologue.candidates(graph)
    declared = (
        candidate
        for operation in graph.ops.values()
        for candidate in (
            attention.stateful_candidate(operation, graph),
            attention.declared_candidate(operation, graph),
            linear_attention.candidate(operation, graph),
        )
        if candidate is not None
    )
    candidates: dict[str, list[RegionCandidate]] = {}
    for candidate in (
        *declared,
        *chain_candidates,
        *prologue_candidates,
        *gemm.epilogue_candidates(graph),
        *norm_candidates,
        *rope_candidates,
        *attention.prologue_candidates(
            graph, (*rope_candidates, *norm_candidates, *chain_candidates)
        ),
        *attention.pattern_candidates(graph),
    ):
        candidates.setdefault(candidate.source_op_ids[-1], []).append(candidate)
    return {tail: tuple(values) for tail, values in candidates.items()}


def _verify_regions(graph: ComputeGraph) -> None:
    source_ids = [operation.id for operation in graph.ops.values()]
    region_ids = [region.id for region in graph.regions]
    covered_ids = [
        operation_id for region in graph.regions for operation_id in region.source_op_ids
    ]
    if len(set(region_ids)) != len(region_ids) or covered_ids != source_ids:
        raise ValueError("Regions do not cover every source operation exactly once")

    operation_regions = {
        operation_id: region
        for region in graph.regions
        for operation_id in region.source_op_ids
    }
    candidate_ids = [
        candidate.id
        for region in graph.regions
        for candidate in region.fusion_candidates
    ]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Fusion candidate IDs are not unique")
    if set(candidate_ids).intersection(region_ids):
        raise ValueError("Fusion candidate IDs collide with region IDs")
    for region in graph.regions:
        for candidate in region.fusion_candidates:
            if not set(candidate.source_op_ids).issubset(graph.ops):
                raise ValueError(f"Candidate {candidate.id} references an unknown operation")
            if any(
                operation_id not in region.source_op_ids
                for operation_id in candidate.source_op_ids
            ):
                if candidate.source_op_ids[-1] not in region.source_op_ids:
                    raise ValueError(f"Candidate {candidate.id} is not attached to its tail region")
            for value_id in (*candidate.external_inputs, *candidate.live_outputs):
                if value_id not in graph.values:
                    raise ValueError(f"Candidate {candidate.id} references an unknown value")
            if any(
                graph.values[value_id].producer not in candidate.source_op_ids
                for value_id in candidate.live_outputs
            ):
                raise ValueError(f"Candidate {candidate.id} outputs a foreign value")
            if any(
                graph.values[value_id].producer in candidate.source_op_ids
                for value_id in candidate.external_inputs
            ):
                raise ValueError(f"Candidate {candidate.id} marks an internal value external")
            if candidate.tensor_program is not None and candidate.tensor_program.store.value_id not in candidate.live_outputs:
                raise ValueError(f"Candidate {candidate.id} does not output its program result")
    for region in graph.regions:
        operations = tuple(graph.ops[operation_id] for operation_id in region.source_op_ids)
        if region.external_inputs != external_inputs_of(operations, graph):
            raise ValueError(f"Region {region.id} has incorrect external inputs")
        if region.live_outputs != live_outputs_of(operations, graph):
            raise ValueError(f"Region {region.id} has incorrect live outputs")
        for result_id in region.live_outputs:
            producer = graph.values[result_id].producer
            if producer not in region.source_op_ids:
                raise ValueError(f"Region {region.id} outputs a foreign value")
        for operand_id in region.external_inputs:
            producer = graph.values[operand_id].producer
            if producer is not None and operation_regions[producer] is region:
                raise ValueError(f"Region {region.id} marks an internal value external")
