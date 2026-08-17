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
mkdir -p "$output_root/portable" "$output_root/sentinel"

cleanup() {
  if [[ -n "${encoder_pid:-}" ]] && kill -0 "$encoder_pid" 2>/dev/null; then
    kill "$encoder_pid" 2>/dev/null || true
    wait "$encoder_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cat >"$output_root/CONTRACT" <<EOF
source_commit=$(git -C "$repo_root" rev-parse HEAD)
frozen_models=true
retraining=false
controller=fast:3.0,segment_0:task_specific,segment_1:1.0
pick_speeds=fast:3.0,segment_0:1.0,segment_1:1.0
tea_speeds=fast:3.0,segment_0:1.5,segment_1:1.0
insertion_speeds=fast:3.0,segment_0:1.5,segment_1:1.0
validation_decoder=risk_threshold:0.45,exit_threshold:0.60,exit_stability:1
candidate_decoder=risk_threshold:0.35,exit_threshold:0.60,exit_stability:2
counterexample_first=true
sentinel_seeds=tea:4202000,4202004;insertion:4203000,4203001
sentinel_arms=learned_candidate_only;native_1x_reused_from_prior_matched_receipt
fresh_task_order=pick_and_place,tea_bag_randomized,insertion
fresh_seeds=4401000-4401004,4402000-4402004,4403000-4403004
fresh_arms=native_1x,learned_shared_fast
decision_cadence_physics_steps=5
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
  --device "${ENCODER_DEVICE:-cpu}" \
  >"$output_root/encoder.log" 2>"$output_root/encoder.err" &
encoder_pid=$!
for _ in $(seq 1 120); do
  [[ -s "$ready_path" && -S "$socket_path" ]] && break
  kill -0 "$encoder_pid"
  sleep 0.25
done
[[ -s "$ready_path" && -S "$socket_path" ]]

run_task() {
  local output=$1 task=$2 task_dir=$3 dataset=$4 method=$5 seeds=$6 portable=$7
  shift 7
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
    --fast-speed 3 \
    --protected-speed 1 \
    --decoder-risk-threshold 0.35 \
    --decoder-exit-threshold 0.60 \
    --decoder-exit-stability 2 \
    --cadence 5 \
    --output "$output" \
    "$@" \
    >"$output.log" 2>"$output.err"
}

run_task "$output_root/sentinel/tea_bag_randomized" \
  tea_bag_randomized tea tea_bag_randomized action "4202000 4202004" tea-action \
  --candidate-only --protected-speed-override segment_0=1.5
run_task "$output_root/sentinel/insertion" \
  insertion insertion insertion fused "4203000 4203001" insertion-fused \
  --candidate-only --protected-speed-override segment_0=1.5 \
  --encoder-socket "$socket_path"

"$sim_python" "$repo_root/scripts/summarize_supervised_shared_fast.py" \
  --result "$output_root/sentinel/tea_bag_randomized/results.json" \
  --result "$output_root/sentinel/insertion/results.json" \
  --output "$output_root/sentinel/summary.json" \
  >"$output_root/sentinel/summary.log" 2>"$output_root/sentinel/summary.err"

if ! jq -e '.all_candidate_success == true' "$output_root/sentinel/summary.json" >/dev/null; then
  touch "$output_root/SENTINEL_FAILED"
  cp "$output_root/sentinel/summary.json" "$output_root/summary.json"
else
  run_task "$output_root/pick_and_place" \
    pick_and_place pick pick_and_place action \
    "4401000 4401001 4401002 4401003 4401004" pick-action
  run_task "$output_root/tea_bag_randomized" \
    tea_bag_randomized tea tea_bag_randomized action \
    "4402000 4402001 4402002 4402003 4402004" tea-action \
    --protected-speed-override segment_0=1.5
  run_task "$output_root/insertion" \
    insertion insertion insertion fused \
    "4403000 4403001 4403002 4403003 4403004" insertion-fused \
    --protected-speed-override segment_0=1.5 \
    --encoder-socket "$socket_path"
  "$sim_python" "$repo_root/scripts/summarize_supervised_shared_fast.py" \
    --result "$output_root/pick_and_place/results.json" \
    --result "$output_root/tea_bag_randomized/results.json" \
    --result "$output_root/insertion/results.json" \
    --output "$output_root/summary.json" \
    >"$output_root/summary.log" 2>"$output_root/summary.err"
fi

cleanup
encoder_pid=
summary_hash=$(sha256sum "$output_root/summary.json" | awk '{print $1}')
printf '%s  summary.json\n' "$summary_hash" >"$output_root/COMPLETE"
(
  cd "$output_root"
  find . -type f \
    \( -name '*.json' -o -name '*.npz' -o -name COMPLETE -o -name CONTRACT -o -name SENTINEL_FAILED \) \
    ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum >SHA256SUMS
  sha256sum -c --quiet SHA256SUMS
)
