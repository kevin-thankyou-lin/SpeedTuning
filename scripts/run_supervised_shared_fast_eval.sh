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
model_root=${MODEL_ROOT:-/dev/shm/r3-supervised-phase-intent-full-20260816}
dataset_root=${DATASET_ROOT:-/dev/shm/speedtuning-xirl-bank-three-task-20260816-5zrlVY/datasets}
socket_path="$output_root/rn18.sock"
ready_path="$output_root/rn18-ready.json"

if [[ -e "$output_root" ]]; then
  echo "output already exists: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root/portable"

cleanup() {
  if [[ -n "${encoder_pid:-}" ]] && kill -0 "$encoder_pid" 2>/dev/null; then
    kill "$encoder_pid" 2>/dev/null || true
    wait "$encoder_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cat >"$output_root/CONTRACT" <<EOF
source_commit=$(git -C "$repo_root" rev-parse HEAD)
controller=fast:2.0,segment_0:1.0,segment_1:1.0
decision_cadence_physics_steps=5
matched_arms=native_1x,learned_shared_fast
task_order=pick_and_place,tea_bag_randomized,insertion
seeds=4101000-4101004,4102000-4102004,4103000-4103004
runtime_privileged_speed_inputs=false
EOF

export_model() {
  local task_dir=$1 method=$2 output_name=$3
  "$encoder_python" "$repo_root/scripts/export_supervised_phase_model.py" \
    --model "$model_root/$task_dir/$method.pkl" \
    --output "$output_root/portable/$output_name.npz" \
    >"$output_root/portable/$output_name.log" \
    2>"$output_root/portable/$output_name.err"
}

export_model pick action pick-action
export_model tea action tea-action
export_model insertion fused insertion-fused

OMP_NUM_THREADS=${ENCODER_THREADS:-4} MKL_NUM_THREADS=${ENCODER_THREADS:-4} \
"$encoder_python" "$repo_root/scripts/serve_rn18_embeddings.py" \
  --socket "$socket_path" \
  --ready "$ready_path" \
  --device "${ENCODER_DEVICE:-cuda}" \
  >"$output_root/encoder.log" 2>"$output_root/encoder.err" &
encoder_pid=$!
for _ in $(seq 1 120); do
  [[ -s "$ready_path" && -S "$socket_path" ]] && break
  kill -0 "$encoder_pid"
  sleep 0.25
done
[[ -s "$ready_path" && -S "$socket_path" ]]

run_task() {
  local task=$1 task_dir=$2 dataset=$3 method=$4 seeds=$5 portable=$6
  shift 6
  MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} \
  OMP_NUM_THREADS=${SIM_THREADS:-4} MKL_NUM_THREADS=${SIM_THREADS:-4} \
  OPENBLAS_NUM_THREADS=${SIM_THREADS:-4} \
  "$sim_python" "$repo_root/scripts/evaluate_supervised_shared_fast.py" \
    --task "$task" \
    --model-dir "$model_root/$task_dir" \
    --portable-model "$output_root/portable/$portable.npz" \
    --portable-receipt "$output_root/portable/$portable.receipt.json" \
    --dataset-manifest "$dataset_root/$dataset/manifest.json" \
    --action-receipt "$model_root/$task_dir-actions.receipt.json" \
    --method "$method" \
    --seeds $seeds \
    --fast-speed 2 \
    --protected-speed 1 \
    --cadence 5 \
    --output "$output_root/$task" \
    "$@" \
    >"$output_root/$task.log" 2>"$output_root/$task.err"
}

run_task pick_and_place pick pick_and_place action \
  "4101000 4101001 4101002 4101003 4101004" pick-action
run_task tea_bag_randomized tea tea_bag_randomized action \
  "4102000 4102001 4102002 4102003 4102004" tea-action
run_task insertion insertion insertion fused \
  "4103000 4103001 4103002 4103003 4103004" insertion-fused \
  --encoder-socket "$socket_path"

"$sim_python" - "$output_root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tasks = ("pick_and_place", "tea_bag_randomized", "insertion")
results = {task: json.loads((root / task / "results.json").read_text()) for task in tasks}
summary = {
    "schema": "speedtuning-supervised-shared-fast-sequential-v1",
    "tasks": {task: results[task]["summary"] for task in tasks},
    "all_native_success": all(
        result["success"] for task in tasks for result in results[task]["native_1x"]
    ),
    "all_candidate_success": all(
        result["success"] for task in tasks for result in results[task]["candidate"]
    ),
    "new_rollouts": sum(results[task]["summary"]["new_rollouts"] for task in tasks),
}
path = root / "summary.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "COMPLETE").write_text(f"{digest}  summary.json\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
