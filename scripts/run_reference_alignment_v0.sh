#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

output_root=$1
source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sim_python=${SIM_PYTHON:-python}
probe_python=${PROBE_PYTHON:-python}
probe_device=${PROBE_DEVICE:-cpu}
root_floor_bytes=$((20 * 1024 * 1024 * 1024))

if [[ -e "$output_root" ]]; then
  echo "refusing to overwrite existing output root: $output_root" >&2
  exit 1
fi
mkdir -p "$output_root"/{datasets,logs,results}
terminal=$output_root/TERMINAL
trap 'printf "FAILED\n" > "$terminal"' ERR

check_disk() {
  local root_free shm_free
  root_free=$(df -PB1 / | awk 'NR == 2 {print $4}')
  shm_free=$(df -PB1 /dev/shm | awk 'NR == 2 {print $4}')
  if (( root_free < root_floor_bytes || shm_free < root_floor_bytes )); then
    echo "disk guard failed: root=$root_free shm=$shm_free floor=$root_floor_bytes" >&2
    return 1
  fi
}

capture_task() {
  local task=$1
  local seed_start=$2
  local dataset=$output_root/datasets/$task
  local -a seeds=()
  local seed
  for ((seed=seed_start; seed<seed_start+6; seed++)); do
    seeds+=("$seed")
  done
  check_disk
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SPEEDTUNING_RUNTIME_ROOT="$source_root" \
    "$sim_python" "$source_root/scripts/capture_reference_alignment_dataset.py" \
    --task "$task" \
    --seeds "${seeds[@]}" \
    --output "$dataset" \
    --camera angle \
    --frame-stride 5 \
    --landmarks-per-query 12 \
    >"$output_root/logs/capture-$task.log" 2>&1
  jq -e '
    .semantic_segment_labels_present == false and
    .evaluation_truth_used_by_model == false and
    (.trajectories | length) == 6
  ' "$dataset/manifest.json" >/dev/null
  echo "CAPTURE_COMPLETE task=$task"
}

benchmark_task() {
  local task=$1
  check_disk
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    "$probe_python" "$source_root/scripts/benchmark_reference_alignment.py" \
    --dataset "$output_root/datasets/$task" \
    --output "$output_root/results/$task" \
    --encoders rn18_temporal_pool r3d18 \
    --windows 0.25 0.5 1.0 \
    --video-windows 0.5 \
    --device "$probe_device" \
    --batch-size 64 \
    --video-batch-size 16 \
    >"$output_root/logs/benchmark-$task.log" 2>&1
  jq -e '
    .semantic_segment_labels_used == false and
    .semantic_phase_classifier_trained == false and
    .task_specific_training == false and
    .future_query_frames_used == false and
    (.methods | length) == 4
  ' "$output_root/results/$task/results.json" >/dev/null
  echo "BENCHMARK_COMPLETE task=$task"
}

capture_task pick_and_place 3111000
capture_task tea_bag_randomized 3112000
capture_task insertion 3113000

benchmark_task pick_and_place
benchmark_task tea_bag_randomized
benchmark_task insertion

sha256sum \
  "$source_root/reference_alignment.py" \
  "$source_root/scripts/capture_reference_alignment_dataset.py" \
  "$source_root/scripts/benchmark_reference_alignment.py" \
  "$output_root"/datasets/*/manifest.json \
  "$output_root"/results/*/results.json \
  >"$output_root/SHA256SUMS"
printf "COMPLETE\n" > "$terminal"
echo "ALL_COMPLETE"
