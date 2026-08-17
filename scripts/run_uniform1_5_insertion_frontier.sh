#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

output_root=$1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sim_python=${SIM_PYTHON:-/home/gear/Projects/SpeedTuning-official-20260813/.venv/bin/python}
controller="$repo_root/configs/insertion_uniform_1_5.json"
runtime_root=/dev/shm/speedtuning-r3-tea-insertion-causal-azlgnAya/insertion/runtime
cached_native=/dev/shm/r3-supervised-conservative-margin-insertion-discovery5-20260816/insertion/results.json

if [[ -e "$output_root" ]]; then
  echo "output already exists: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root/sentinel"

python3 - "$output_root/CONTRACT" "$(git -C "$repo_root" rev-parse HEAD)" \
  "$(sha256sum "$controller" | awk '{print $1}')" \
  "$(sha256sum "$cached_native" | awk '{print $1}')" <<'PY'
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    "\n".join(
        [
            f"source_commit={sys.argv[2]}",
            "task=insertion",
            "only_change=uniform_speed:2.0_to_1.5",
            "controller=uniform:1.5",
            "controller_sha256=" + sys.argv[3],
            "counterexample_first=true",
            "sentinel_seeds=4403000,4403002",
            "sentinel_native_reference=cached_same_seed_1x",
            "cached_native_sha256=" + sys.argv[4],
            "sentinel_new_native_rollouts=0",
            "sentinel_new_candidate_rollouts=2",
            "fresh_seeds=4603000,4603001,4603002,4603003,4603004",
            "fresh_arms=native_1x,uniform_1_5x",
            "fresh_launch_requires_sentinel_2_of_2=true",
            "base_policy_failure_disposition=stop_promotion",
            "runtime_boundary_inputs_privileged=false",
            "",
        ]
    )
)
PY

run_eval() {
  local output=$1 seeds=$2
  shift 2
  MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} \
  OMP_NUM_THREADS=${SIM_THREADS:-4} MKL_NUM_THREADS=${SIM_THREADS:-4} \
  OPENBLAS_NUM_THREADS=${SIM_THREADS:-4} \
  "$sim_python" "$repo_root/scripts/evaluate_privileged_segment_schedule.py" \
    --task insertion \
    --controller "$controller" \
    --runtime-root "$runtime_root" \
    --seeds $seeds \
    --arm-name uniform_1_5x \
    --output "$output" \
    "$@" \
    >"$output.log" 2>"$output.err"
}

run_eval "$output_root/sentinel/insertion" "4403000 4403002" \
  --cached-native-results "$cached_native"

if ! jq -e '.summary.candidate_success_rate == 1' \
  "$output_root/sentinel/insertion/results.json" >/dev/null; then
  touch "$output_root/SENTINEL_FAILED"
else
  run_eval "$output_root/insertion" \
    "4603000 4603001 4603002 4603003 4603004"
fi

python3 - "$output_root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sentinel = json.loads((root / "sentinel/insertion/results.json").read_text())
fresh_path = root / "insertion/results.json"
fresh = json.loads(fresh_path.read_text()) if fresh_path.exists() else None

def compact(result):
    return {
        "seeds": result["seeds"],
        "summary": result["summary"],
        "per_seed": [
            {
                "seed": item["seed"],
                "success": item["success"],
                "physics_steps": item["physics_steps"],
                "mean_speed": item["mean_speed"],
            }
            for item in result["candidate"]
        ],
        "native_per_seed": [
            {
                "seed": item["seed"],
                "success": item["success"],
                "physics_steps": item["physics_steps"],
            }
            for item in result["native_1x"]
        ],
    }

summary = {
    "schema": "speedtuning-uniform1_5-frontier-audit-v1",
    "task": "insertion",
    "controller_sha256": sentinel["controller_sha256"],
    "sentinel": compact(sentinel),
    "sentinel_passed": all(item["success"] for item in sentinel["candidate"]),
    "fresh": None if fresh is None else compact(fresh),
}
path = root / "summary.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "COMPLETE").write_text(f"{digest}  summary.json\n")
PY

(
  cd "$output_root"
  find . -type f \
    \( -name '*.json' -o -name COMPLETE -o -name CONTRACT -o -name SENTINEL_FAILED \) \
    ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum >SHA256SUMS
  sha256sum -c --quiet SHA256SUMS
)
