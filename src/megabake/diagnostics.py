from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompilerDiagnostic:
    stage: str
    message: str
    source_op_ids: tuple[str, ...] = ()
    region_ids: tuple[str, ...] = ()
    requirement: str | None = None


class CompilerError(ValueError):
    def __init__(self, diagnostic: CompilerDiagnostic, *, safe_for_fallback: bool = False):
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic
        self.safe_for_fallback = safe_for_fallback

    def __str__(self) -> str:
        diagnostic = self.diagnostic
        parts = [f"{diagnostic.stage}: {diagnostic.message}"]
        if diagnostic.source_op_ids:
            parts.append("source operations: " + ", ".join(diagnostic.source_op_ids))
        if diagnostic.region_ids:
            parts.append("regions: " + ", ".join(diagnostic.region_ids))
        if diagnostic.requirement:
            parts.append(f"requirement: {diagnostic.requirement}")
        return "; ".join(parts)
