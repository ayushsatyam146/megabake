#!/usr/bin/env bash
# One-command setup on a fresh rented GPU box (H100 or any sm90 GPU), then a
# correctness check and the Llama-3.2-1B decode benchmark.
#
#     git clone -b experimentation-profiling https://github.com/ayushsatyam146/megabake.git
#     cd megabake && bash tools/setup_gpu_box.sh
#
# Environment knobs:
#     SKIP_BENCH=1      stop after setup and the smoke test
#     SKIP_CALIBRATE=1  skip measuring this GPU (tools/calibrate.py)
#     MODEL=...         model for the check and benchmark (default Llama-3.2-1B)
#     HF_TOKEN=...      only for gated checkpoints; the defaults are ungated
#
# No CUDA toolkit or nvcc is needed: CuTe DSL compiles kernels itself. What is
# needed is an NVIDIA driver new enough for PyTorch's CUDA 13.0 wheels (>= 580).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-unsloth/Llama-3.2-1B-Instruct}"
OUT="artifacts/$(date +%Y-%m-%d)_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"
step() { printf '\n==> [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

step "Check the GPU and driver (fail fast before spending time on installs)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver on this box"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
[ "${driver%%.*}" -ge 580 ] || die "driver $driver is too old for CUDA 13.0 wheels (need >= 580). Pick another image/host, or ask the provider for a CUDA 13 image."
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
[ "$cap" = "9.0" ] || echo "WARNING: compute capability $cap; MegaBake's tuned bodies target sm90 (H100/H200)."
others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$others" -eq 0 ] || echo "WARNING: $others other process(es) already on the GPU; timings will be inflated."

step "Install uv (fetches Python 3.13 itself, no system Python needed)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

step "Create .venv and install pinned requirements (torch 2.14 cu130, CuTe DSL 4.8)"
[ -x .venv/bin/python ] || uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt

step "Download checkpoints in the background while the smoke test runs"
.venv/bin/python - "$MODEL" HuggingFaceTB/SmolLM-135M > "/tmp/megabake_download.log" 2>&1 <<'EOF' &
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    print(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]), flush=True)
EOF
download_pid=$!

step "Smoke test: torch sees the GPU, CuTe DSL imports"
.venv/bin/python - <<'EOF'
import torch, cutlass, importlib.metadata as md
assert torch.cuda.is_available(), "torch cannot see the GPU"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}); {p.name}, sm{p.major}{p.minor}, {p.multi_processor_count} SMs; cutlass-dsl {md.version('nvidia-cutlass-dsl')}")
EOF

wait "$download_pid" || { cat /tmp/megabake_download.log; die "checkpoint download failed (set HF_TOKEN for gated models)"; }

if [ "${SKIP_CALIBRATE:-0}" != 1 ]; then
  step "Calibrate this GPU (cached under ~/.cache/megabake/machine; the planner uses it)"
  mkdir -p "$OUT"
  .venv/bin/python tools/calibrate.py --out "$OUT/calibration.json"
fi

[ "${SKIP_BENCH:-0}" = 1 ] && { step "Setup done (SKIP_BENCH=1)"; exit 0; }

step "Correctness: 64 greedy tokens against Hugging Face generate"
mkdir -p "$OUT"
.venv/bin/python tools/greedy_check.py --model "$MODEL" --tokens 64 --capacity 256 | tee "$OUT/greedy.out"

step "Benchmark: decode, KV capacity 256"
.venv/bin/python benchmarks/full_model.py --model "$MODEL" --modes decode --capacity 256 --out "$OUT/bench"

step "Done. Results in $OUT"
