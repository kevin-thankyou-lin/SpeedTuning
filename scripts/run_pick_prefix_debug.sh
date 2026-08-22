#!/usr/bin/env bash
set -euo pipefail

: "${VARIANT:?}"
: "${NORMALIZATION:?}"
: "${STEPS:?}"
: "${CHECKPOINT_EVERY:?}"
: "${LEARNING_RATE:?}"
: "${INITIALIZE_FROM_BASE:?}"
: "${SOURCE_COMMIT:?}"
: "${PYTHON:?}"
: "${PREFIX_GENERATION:?}"
: "${SUPERVISED_HORIZON:?}"
ACT_DETERMINISTIC=${ACT_DETERMINISTIC:-0}
CHUNK_SIZE=${CHUNK_SIZE:-48}

BASE=/mnt/amlfs-04/home/linke/speedtuning-relative-imitation/speedtuning-conditioned-relative-joint-20260821-v2/pick
BASE_CHECKPOINT=$BASE/checkpoints/slow_150/act/best.pt
BASE_CHECKPOINT_SHA=cd984a8812ce5679ee3083cb5c0fb17ff92d29dc62e47b9e11b9d638a5ddd6ba
DATASET=$BASE/datasets/slow_150/pick_and_place
ROOT=/mnt/amlfs-04/home/linke/speedtuning-relative-imitation/$PREFIX_GENERATION/$VARIANT
OUTPUT=/osmo/run/outputs/$VARIANT
START_PROBABILITY=0.25

test -d "$DATASET"
test -s "$BASE_CHECKPOINT"
echo "$BASE_CHECKPOINT_SHA  $BASE_CHECKPOINT" | sha256sum -c -
test ! -e "$ROOT" || { echo "refusing to overwrite $ROOT" >&2; exit 30; }
mkdir -p "$ROOT" "$OUTPUT"

DATASET="$DATASET" "$PYTHON" - <<'PY'
import glob
import h5py
import os

paths = sorted(glob.glob(os.environ["DATASET"] + "/episode_*.hdf5"))
assert len(paths) == 150, len(paths)
seeds = []
for path in paths[:10]:
    with h5py.File(path, "r") as root:
        seeds.append(int(root.attrs["seed"]))
assert seeds == list(range(4100001, 4100011)), seeds
print("training-pose-bank", seeds)
PY

INITIAL_ARGS=()
if [ "$INITIALIZE_FROM_BASE" = 1 ]; then
  INITIAL_ARGS=(--initial-checkpoint "$BASE_CHECKPOINT")
fi
DETERMINISTIC_ARGS=()
if [ "$ACT_DETERMINISTIC" = 1 ]; then
  DETERMINISTIC_ARGS=(--act-deterministic)
fi
(
  set +e
  "$PYTHON" -m scripts.train_relative_imitation \
    --kind act --dataset-dir "$DATASET" --output-dir "$ROOT/checkpoint" \
    --steps "$STEPS" --chunk-size "$CHUNK_SIZE" --batch-size 16 \
    --lr "$LEARNING_RATE" --seed 0 \
    --episode-start-probability "$START_PROBABILITY" \
    --normalization "$NORMALIZATION" \
    --supervised-horizon "$SUPERVISED_HORIZON" \
    "${DETERMINISTIC_ARGS[@]}" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    "${INITIAL_ARGS[@]}" \
    2>&1 | tee "$ROOT/train.log"
  echo "${PIPESTATUS[0]}" > "$ROOT/train.rc"
) &
TRAIN_JOB=$!

evaluate_checkpoint() {
  local checkpoint=$1
  local label marker
  label=$(basename "$checkpoint" .pt)
  marker=$ROOT/.evaluated-$label
  test ! -e "$marker" || return 0
  "$PYTHON" -m scripts.evaluate_relative_imitation \
    --task pick_and_place --checkpoint "$checkpoint" \
    --output "$ROOT/eval-$label-training.json" --episodes 10 \
    --seed-base 4100001 --replan-interval 8 --speed-condition 0 \
    2>&1 | tee "$ROOT/eval-$label-training.log"
  "$PYTHON" -m scripts.evaluate_relative_imitation \
    --task pick_and_place --checkpoint "$checkpoint" \
    --output "$ROOT/eval-$label-fresh.json" --episodes 10 \
    --seed-base 6100000 --replan-interval 8 --speed-condition 0 \
    2>&1 | tee "$ROOT/eval-$label-fresh.log"
  touch "$marker"
}

while kill -0 "$TRAIN_JOB" 2>/dev/null; do
  for checkpoint in "$ROOT"/checkpoint/step-*.pt; do
    test -e "$checkpoint" || continue
    evaluate_checkpoint "$checkpoint"
  done
  sleep 5
done
wait "$TRAIN_JOB"
test "$(cat "$ROOT/train.rc")" = 0
for checkpoint in "$ROOT"/checkpoint/step-*.pt; do
  test -e "$checkpoint" || continue
  evaluate_checkpoint "$checkpoint"
done
evaluate_checkpoint "$ROOT/checkpoint/best.pt"

export ROOT VARIANT START_PROBABILITY NORMALIZATION STEPS SOURCE_COMMIT
export BASE_CHECKPOINT_SHA INITIALIZE_FROM_BASE SUPERVISED_HORIZON
export ACT_DETERMINISTIC
export CHUNK_SIZE
"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
evaluations = {}
for path in sorted(root.glob("eval-*.json")):
    value = json.loads(path.read_text())
    evaluations[path.stem] = {
        key: value[key]
        for key in (
            "checkpoint",
            "episodes",
            "successes",
            "success_rate",
            "successful_mean_steps",
            "clipping",
        )
    }
training = {
    key: value for key, value in evaluations.items() if key.endswith("-training")
}
best_key, best_value = max(
    training.items(), key=lambda item: (item[1]["successes"], item[0])
)
summary = {
    "schema": "pick-executed-prefix-debug-v1",
    "variant": os.environ["VARIANT"],
    "normalization": os.environ["NORMALIZATION"],
    "episode_start_probability": float(os.environ["START_PROBABILITY"]),
    "supervised_horizon": int(os.environ["SUPERVISED_HORIZON"]),
    "chunk_size": int(os.environ["CHUNK_SIZE"]),
    "training_steps": int(os.environ["STEPS"]),
    "initialized_from_base": os.environ["INITIALIZE_FROM_BASE"] == "1",
    "act_deterministic": os.environ["ACT_DETERMINISTIC"] == "1",
    "source_commit": os.environ["SOURCE_COMMIT"],
    "base_checkpoint_sha256": os.environ["BASE_CHECKPOINT_SHA"],
    "training_pose_seeds": list(range(4100001, 4100011)),
    "fresh_monitor_seeds": list(range(6100000, 6100010)),
    "best_training_evaluation": best_key,
    "best_training_successes": best_value["successes"],
    "training_gate_passed": best_value["successes"] >= 9,
    "evaluations": evaluations,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (root / "SHA256SUMS").open("w") as stream:
    for path in sorted(root.glob("checkpoint/*.pt")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stream.write(f"{digest}  {path.relative_to(root)}\n")
print(json.dumps(summary, indent=2))
PY
cp "$ROOT/summary.json" "$ROOT/SHA256SUMS" "$OUTPUT"/
echo "PICK_EXECUTED_PREFIX_DEBUG_COMPLETE variant=$VARIANT root=$ROOT"
