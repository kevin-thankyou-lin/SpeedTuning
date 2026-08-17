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

python3 - "$output_root/CONTRACT" "$(git -C "$repo_root" rev-parse HEAD)" <<'PY'
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    "\n".join(
        [
            f"source_commit={sys.argv[2]}",
            "task=insertion",
            "frozen_models=true",
            "retraining=false",
            "only_change=exit_stability:2_to_3",
            "controller=fast:3.0,segment_0:1.5,segment_1:1.0",
            "decoder=risk_threshold:0.35,exit_threshold:0.60,exit_stability:3",
            "decision_cadence_physics_steps=5",
            "counterexample_first=true",
            "sentinel_seeds=4403000,4403001,4403002",
            "sentinel_arms=learned_candidate_only",
            "sentinel_native_reference=/dev/shm/r3-supervised-conservative-margin-insertion-discovery5-20260816/insertion/results.json",
            "excluded_prior_success_seeds=4403003,4403004",
            "fresh_seeds=4503000,4503001,4503002,4503003,4503004",
            "fresh_arms=native_1x,learned_shared_fast",
            "runtime_privileged_speed_inputs=false",
            "",
        ]
    )
)
PY

"$encoder_python" "$repo_root/scripts/export_supervised_phase_model.py" \
  --model "$model_root/insertion/fused.pkl" \
  --output "$output_root/portable/insertion-fused.npz" \
  >"$output_root/portable/insertion-fused.log" \
  2>"$output_root/portable/insertion-fused.err"

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

run_eval() {
  local output=$1 seeds=$2
  shift 2
  MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} \
  OMP_NUM_THREADS=${SIM_THREADS:-4} MKL_NUM_THREADS=${SIM_THREADS:-4} \
  OPENBLAS_NUM_THREADS=${SIM_THREADS:-4} \
  "$sim_python" "$repo_root/scripts/evaluate_supervised_shared_fast.py" \
    --task insertion \
    --model-dir "$model_root/insertion" \
    --portable-model "$output_root/portable/insertion-fused.npz" \
    --portable-receipt "$output_root/portable/insertion-fused.receipt.json" \
    --dataset-manifest "$dataset_root/insertion/manifest.json" \
    --action-receipt "$model_root/insertion-actions.receipt.json" \
    --method fused \
    --seeds $seeds \
    --fast-speed 3 \
    --protected-speed 1 \
    --protected-speed-override segment_0=1.5 \
    --decoder-risk-threshold 0.35 \
    --decoder-exit-threshold 0.60 \
    --decoder-exit-stability 3 \
    --cadence 5 \
    --encoder-socket "$socket_path" \
    --output "$output" \
    "$@" \
    >"$output.log" 2>"$output.err"
}

run_eval "$output_root/sentinel/insertion" "4403000 4403001 4403002" \
  --candidate-only
"$sim_python" "$repo_root/scripts/summarize_supervised_shared_fast.py" \
  --result "$output_root/sentinel/insertion/results.json" \
  --output "$output_root/sentinel/summary.json" \
  >"$output_root/sentinel/summary.log" 2>"$output_root/sentinel/summary.err"

if ! jq -e '.all_candidate_success == true' "$output_root/sentinel/summary.json" >/dev/null; then
  touch "$output_root/SENTINEL_FAILED"
  cp "$output_root/sentinel/summary.json" "$output_root/summary.json"
else
  run_eval "$output_root/insertion" \
    "4503000 4503001 4503002 4503003 4503004"
  "$sim_python" "$repo_root/scripts/summarize_supervised_shared_fast.py" \
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
