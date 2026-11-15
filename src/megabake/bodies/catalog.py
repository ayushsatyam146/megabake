"""The body providers a target runs: the generic ones on every target, plus
those of the GPU generation its architecture belongs to."""

from __future__ import annotations

from typing import Any

from ..graph import ComputeGraph, Region
from .cuda import sm90, sm100
from .cuda.generic import GenericBodyProvider
from .interface import (
    BodyConfig,
    BodyProvider,
    CandidateSet,
    Generation,
    InvalidBodyConfig,
    ProviderRejection,
    SelectionPolicy,
    TargetFacts,
    validate_body_config,
)

GENERIC: tuple[BodyProvider, ...] = (GenericBodyProvider(),)
GENERATIONS: tuple[Generation, ...] = (sm90.GENERATION, sm100.GENERATION)


def generation(target: TargetFacts) -> Generation | None:
    return next(
        (
            candidate
            for candidate in GENERATIONS
            if target.architecture in candidate.architectures
        ),
        None,
    )


def providers(target: TargetFacts) -> tuple[BodyProvider, ...]:
    found = generation(target)
    return GENERIC + (found.providers if found is not None else ())


def provider(name: str) -> BodyProvider:
    """The provider that emits bodies configured with ``BodyConfig.provider``."""
    for candidate in GENERIC + tuple(
        item for found in GENERATIONS for item in found.providers
    ):
        if candidate.provider_name == name:
            return candidate
    raise KeyError(f"no body provider named {name!r}")


def launch_options(
    graph: ComputeGraph, target: TargetFacts, table_launch: bool, device: Any = None
) -> Any:
    """The ``SelectionPolicy.target_options`` of a launch of ``graph``;
    ``device`` is the launch's ``DeviceTree`` when it is planned with one."""
    found = generation(target)
    if found is None or found.launch_options is None:
        return None
    return found.launch_options(graph, table_launch, device)


def get_candidates(
    region: Region, target: TargetFacts, policy: SelectionPolicy
) -> CandidateSet:
    configurations: list[BodyConfig] = []
    rejections: list[ProviderRejection] = []
    for candidate in providers(target):
        result = candidate.get_candidates(region, target, policy)
        for config in result.configurations:
            try:
                validate_body_config(config, target, policy)
            except InvalidBodyConfig as error:
                rejections.append(ProviderRejection(config.provider, str(error)))
            else:
                configurations.append(config)
        rejections.extend(result.rejections)
    return CandidateSet(tuple(configurations), tuple(rejections))
