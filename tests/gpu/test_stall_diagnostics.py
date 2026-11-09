"""X0.3: a wait that times out names its counter, then the kernel traps.

The trap poisons the CUDA context, so each check runs in a child process.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)),
    reason="stall diagnostics require an SM90 CUDA device",
)

_DECODE = """
import sys
sys.path.insert(0, {root!r})
import torch
import megabake.compiler
from megabake import make_backend
from tests.fixtures import StatefulSmolLMArchitecture

# Real models take the table launch, which carries the checks; make this
# 2-layer model take it too.
megabake.compiler._TABLE_LAUNCH_OPS = 0

torch.manual_seed(3)
layers = 2
model = StatefulSmolLMArchitecture(layers=layers, device="cuda", dtype=torch.bfloat16)
keys = tuple(torch.zeros(1, 4, 3, 64, device="cuda", dtype=torch.bfloat16) for _ in range(layers))
values = tuple(torch.zeros(1, 4, 3, 64, device="cuda", dtype=torch.bfloat16) for _ in range(layers))
lengths = tuple(torch.zeros(1, device="cuda", dtype=torch.int32) for _ in range(layers))
backend = make_backend(mode="decode", compile_kernel=True, cache=False, scheduler="phase")
optimized = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
with torch.no_grad():
    for position in range(2):
        token = torch.tensor([[position + 1]], device="cuda")
        optimized(token, torch.tensor([[position]], device="cuda"), *keys, *values, *lengths)
    torch.cuda.synchronize()
assert backend.megabake_last_compiled.generated.launch.launch_policy == "megabake_cooperative_table"
print("decode finished")
"""


def _run(**environment):
    return subprocess.run(
        [sys.executable, "-c", _DECODE.format(root=str(ROOT))],
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, **environment},
        cwd=ROOT,
    )


def test_injected_missing_arrival_is_reported_and_traps():
    completed = _run(MEGABAKE_STALL_INJECT="1", MEGABAKE_STALL_TIMEOUT_S="0.3")
    assert completed.returncode != 0
    assert "decode finished" not in completed.stdout
    report = completed.stderr[completed.stderr.index("MegaBake stall report"):]
    header = re.match(r"MegaBake stall report: (\d+) of (\d+) CTAs timed out waiting", report)
    assert header and header.group(1) == header.group(2), report[:400]
    rows = [line.split() for line in report.splitlines()[2:] if "grid barrier" in line]
    # Every CTA waits at the same barrier, one arrival short of its target.
    assert len(rows) == 1, report[:800]
    call, _, _, phase, ordinal, target, observed = rows[0][:7]
    workers = int(header.group(2))
    assert int(phase) >= 0 and "closes after" in report
    assert int(target) == (int(ordinal) + 1) * workers
    assert int(observed) == int(target) - 1


def test_producer_ring_stall_names_the_unreleased_page():
    # X1.1: the MMA warps never release page 0, so the producer warp waits
    # for it on its next lap and reports it.
    completed = _run(
        MEGABAKE_PRODUCER_WARP="1", MEGABAKE_STALL_INJECT_RING="1", MEGABAKE_STALL_TIMEOUT_S="0.3"
    )
    assert completed.returncode != 0
    assert "decode finished" not in completed.stdout
    report = completed.stderr[completed.stderr.index("MegaBake stall report"):]
    rows = [line.split() for line in report.splitlines()[2:] if "ring page empty" in line]
    assert rows, report[:800]
    # call kind(3 words) phase counter(page) target(parity) observed
    assert all(row[5] == "0" for row in rows), report[:800]


def test_producer_ring_decode_runs_with_checked_waits():
    completed = _run(MEGABAKE_PRODUCER_WARP="1", MEGABAKE_STALL_TIMEOUT_S="0.3")
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "decode finished" in completed.stdout
    assert "MegaBake stall report" not in completed.stderr


def test_checked_waits_leave_normal_decode_untouched():
    completed = _run(MEGABAKE_STALL_TIMEOUT_S="0.3")
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "decode finished" in completed.stdout
    assert "MegaBake stall report" not in completed.stderr
