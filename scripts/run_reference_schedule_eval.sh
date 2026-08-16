#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

output_root=$1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
encoder_python=${ENCODER_PYTHON:-/home/gear/Projects/gr00t/.venv/bin/python}
sim_python=${SIM_PYTHON:-/home/gear/Projects/SpeedTuning-official-20260813/.venv/bin/python}
encoder_device=${ENCODER_DEVICE:-cpu}
socket_path="$output_root/rn18.sock"
ready_path="$output_root/rn18-ready.json"
mkdir -p "$output_root"

cleanup() {
  if [[ -n "${encoder_pid:-}" ]] && kill -0 "$encoder_pid" 2>/dev/null; then
    kill "$encoder_pid" 2>/dev/null || true
    wait "$encoder_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

OMP_NUM_THREADS=${ENCODER_THREADS:-4} MKL_NUM_THREADS=${ENCODER_THREADS:-4} \
"$encoder_python" "$repo_root/scripts/serve_rn18_embeddings.py" \
  --socket "$socket_path" \
  --ready "$ready_path" \
  --device "$encoder_device" \
  >"$output_root/encoder.log" 2>"$output_root/encoder.err" &
encoder_pid=$!

for _ in $(seq 1 120); do
  [[ -s "$ready_path" && -S "$socket_path" ]] && break
  kill -0 "$encoder_pid"
  sleep 0.25
done
[[ -s "$ready_path" && -S "$socket_path" ]]

run_task() {
  local task=$1 controller=$2 runtime=$3 refs=$4 tests=$5 margin=$6
  MUJOCO_GL=${MUJOCO_GL:-egl} "$sim_python" "$repo_root/scripts/evaluate_reference_aligned_schedule.py" \
    --task "$task" \
    --controller "$controller" \
    --runtime-root "$runtime" \
    --encoder-socket "$socket_path" \
    --output "$output_root/$task" \
    --reference-seeds "$refs" \
    --test-seeds "$tests" \
    --p90-margin "$margin" \
    --confidence-threshold 0.55 \
    >"$output_root/$task.log" 2>"$output_root/$task.err"
}

run_task \
  pick_and_place \
  /dev/shm/speedtuning-r3-pick-causal-4S4OOtFL/pick/configs/selected.json \
  /dev/shm/speedtuning-r3-pick-causal-4S4OOtFL/pick/runtime \
  3201000-3201004 3201010-3201029 0.0372
run_task \
  tea_bag_randomized \
  /dev/shm/speedtuning-r3-tea-successor-UVCXUaQM/tea/configs/selected.json \
  /dev/shm/speedtuning-r3-tea-successor-UVCXUaQM/tea/runtime \
  3202000-3202004 3202010-3202029 0.0688
run_task \
  insertion \
  /dev/shm/speedtuning-r3-tea-insertion-causal-azlgnAya/insertion/configs/repair-two-segment-downstream1_0.json \
  /dev/shm/speedtuning-r3-tea-insertion-causal-azlgnAya/insertion/runtime \
  3203000-3203004 3203010-3203029 0.0446

"$sim_python" - "$output_root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = {
    task: json.loads((root / task / "summary.json").read_text())
    for task in ("pick_and_place", "tea_bag_randomized", "insertion")
}
path = root / "summary.json"
path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "COMPLETE").write_text(f"{digest}  summary.json\n")
print(json.dumps(results, indent=2, sort_keys=True))
PY
