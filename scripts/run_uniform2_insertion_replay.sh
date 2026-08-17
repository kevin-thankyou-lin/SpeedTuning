#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

output_root=$1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sim_python=${SIM_PYTHON:-/home/gear/Projects/SpeedTuning-official-20260813/.venv/bin/python}
controller=/dev/shm/speedtuning-r3-tea-insertion-causal-azlgnAya/insertion/configs/selected.json
runtime_root=/dev/shm/speedtuning-r3-tea-insertion-causal-azlgnAya/insertion/runtime
cached_native=/dev/shm/r3-supervised-conservative-margin-insertion-discovery5-20260816/insertion/results.json

if [[ -e "$output_root" ]]; then
  echo "output already exists: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root"

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
            "arm=uniform_2x",
            "controller=uniform:2.0",
            "controller_sha256=" + sys.argv[3],
            "seeds=4403000,4403001,4403002,4403003,4403004",
            "native_reference=cached_same_seed_1x",
            "cached_native_sha256=" + sys.argv[4],
            "new_native_rollouts=0",
            "new_candidate_rollouts=5",
            "runtime_boundary_inputs_privileged=false",
            "stage=same_seed_baseline_characterization",
            "promotion_authorized=false",
            "",
        ]
    )
)
PY

MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} \
OMP_NUM_THREADS=${SIM_THREADS:-4} MKL_NUM_THREADS=${SIM_THREADS:-4} \
OPENBLAS_NUM_THREADS=${SIM_THREADS:-4} \
"$sim_python" "$repo_root/scripts/evaluate_privileged_segment_schedule.py" \
  --task insertion \
  --controller "$controller" \
  --runtime-root "$runtime_root" \
  --cached-native-results "$cached_native" \
  --seeds 4403000 4403001 4403002 4403003 4403004 \
  --arm-name uniform_2x \
  --output "$output_root/insertion" \
  >"$output_root/insertion.log" 2>"$output_root/insertion.err"

python3 - "$output_root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
result = json.loads((root / "insertion/results.json").read_text())
summary = {
    "schema": "speedtuning-uniform2-same-seed-replay-audit-v1",
    "task": result["task"],
    "seeds": result["seeds"],
    "controller_sha256": result["controller_sha256"],
    "summary": result["summary"],
    "per_seed": [
        {
            "seed": item["seed"],
            "success": item["success"],
            "physics_steps": item["physics_steps"],
            "mean_speed": item["mean_speed"],
            "speed_counts": item["speed_counts"],
        }
        for item in result["candidate"]
    ],
}
path = root / "summary.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "COMPLETE").write_text(f"{digest}  summary.json\n")
PY

(
  cd "$output_root"
  find . -type f \
    \( -name '*.json' -o -name COMPLETE -o -name CONTRACT \) \
    ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum >SHA256SUMS
  sha256sum -c --quiet SHA256SUMS
)
