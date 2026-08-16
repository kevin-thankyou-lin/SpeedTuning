#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

probe_root=$1
capture_python=/home/gear/Projects/SpeedTuning-official-20260813/.venv/bin/python
probe_python=/home/gear/Projects/gr00t/.venv/bin/python
source_root=/dev/shm/speedtuning-xirl-bank-source
vip_root=/dev/shm/speedtuning-phase-correspondence-20260816/vendor/vip

pick_runtime=/dev/shm/speedtuning-r3-pick-causal-4S4OOtFL/pick/runtime
pick_controller=/dev/shm/speedtuning-r3-pick-causal-4S4OOtFL/pick/configs/selected.json
tea_runtime=/dev/shm/speedtuning-r3-tea-successor-UVCXUaQM/tea/runtime
tea_controller=/dev/shm/speedtuning-r3-tea-successor-UVCXUaQM/tea/configs/selected.json
insertion_runtime=/dev/shm/speedtuning-r3-tea-insertion-causal-azlgnAya/insertion/runtime
insertion_controller=/dev/shm/speedtuning-r3-tea-insertion-causal-azlgnAya/insertion/configs/repair-two-segment-downstream1_0.json

mkdir -p "$probe_root"/{datasets,logs,results}
terminal=$probe_root/logs/terminal.txt
trap 'echo FAILED > "$terminal"' ERR

check_disk() {
  local root_free shm_free floor
  floor=$((20 * 1024 * 1024 * 1024))
  root_free=$(df -PB1 / | awk 'NR==2 {print $4}')
  shm_free=$(df -PB1 /dev/shm | awk 'NR==2 {print $4}')
  if (( root_free < floor || shm_free < floor )); then
    echo "disk guard failed: root=$root_free shm=$shm_free floor=$floor" >&2
    return 1
  fi
}

capture_task() {
  local task=$1
  local runtime=$2
  local controller=$3
  local seed_start=$4
  local dataset=$probe_root/datasets/$task
  local -a seeds=()
  local seed
  for ((seed=seed_start; seed<seed_start+36; seed++)); do
    seeds+=("$seed")
  done
  check_disk
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SPEEDTUNING_RUNTIME_ROOT=$runtime \
    "$capture_python" \
    "$source_root/capture_phase_dataset.py" \
    --task "$task" \
    --controller "$controller" \
    --seeds "${seeds[@]}" \
    --output "$dataset" \
    --camera angle \
    --stride 5 \
    --preentry-margin 1 \
    >"$probe_root/logs/capture-$task.log" 2>&1
  local successes
  successes=$(jq '[.episodes[] | select(.success)] | length' "$dataset/manifest.json")
  if (( successes < 28 )); then
    echo "$task produced only $successes successful rollouts" >&2
    return 1
  fi
  echo "CAPTURE_COMPLETE task=$task successes=$successes attempts=36"
}

probe_task() {
  local task=$1
  check_disk
  env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    taskset -c 0-7 "$probe_python" \
    "$source_root/probe_xirl_bank.py" \
    --dataset "$probe_root/datasets/$task" \
    --vip-root "$vip_root" \
    --output "$probe_root/results/$task.json" \
    --device cpu \
    --epochs 200 \
    --pairs-per-epoch 12 \
    --train-videos 12 \
    --validation-videos 5 \
    --final-videos 10 \
    >"$probe_root/logs/probe-$task.log" 2>&1
  jq -e '
    .schema == "speedtuning-offline-xirl-bank-v1" and
    .runtime_privileged_signals == false and
    .tcc_phase_labels_used == false and
    .split_counts == {
      "labelled_reference": 1,
      "additional_unlabelled_tcc_train": 12,
      "validation": 5,
      "final": 10
    }
  ' "$probe_root/results/$task.json" >/dev/null
  echo "PROBE_COMPLETE task=$task"
}

capture_task pick_and_place "$pick_runtime" "$pick_controller" 2801000
capture_task tea_bag_randomized "$tea_runtime" "$tea_controller" 2802000
capture_task insertion "$insertion_runtime" "$insertion_controller" 2803000

probe_task pick_and_place
probe_task tea_bag_randomized
probe_task insertion

sha256sum \
  "$source_root/capture_phase_dataset.py" \
  "$source_root/probe_phase_correspondence.py" \
  "$source_root/probe_xirl_tcc.py" \
  "$source_root/probe_xirl_bank.py" \
  "$probe_root"/datasets/*/manifest.json \
  "$probe_root"/results/*.json \
  >"$probe_root/SHA256SUMS"
echo COMPLETE >"$terminal"
echo ALL_COMPLETE
