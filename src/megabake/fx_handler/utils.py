from __future__ import annotations

import re
from pathlib import Path
from typing import Any

PHASE_FILES = {
    "input": "phase_0_input_fx.txt",
    "normalized": "phase_1_normalized.txt",
    "pre_grad": "phase_2_pre_grad.txt",
    "aot": "phase_3_aot_inference.txt",
    "prepared": "phase_4_prepared.txt",
    "cache": "phase_5_cache_check.txt",
    "post_grad": "phase_6_post_grad.txt",
}


def dump_graph(
    phase: str,
    graph: Any,
    output_dir: str | Path | None,
    details: str = "",
) -> Path | None:
    if output_dir is None:
        return None

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_phase = re.sub(r"[^a-z0-9_]+", "_", phase.lower()).strip("_")
    filename = PHASE_FILES.get(phase, f"phase_{safe_phase}.txt")
    graph_text = graph.print_readable(print_output=False, include_stride=True, include_device=True)
    contents = f"# {phase}\n"
    if details:
        contents += f"{details}\n"
    contents += f"\n{graph_text}\n"
    path = directory / filename
    path.write_text(contents, encoding="utf-8")
    return path
