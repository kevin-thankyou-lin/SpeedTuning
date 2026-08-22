#!/usr/bin/env python3
"""Resume only untouched final evaluations for a three-scene round."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
WORKER_ENV = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "SPEEDTUNING_TORCH_THREADS": "2",
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recovery-note")
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "COMPLETE").exists():
        raise RuntimeError("round is already complete")
    contract_path = root / "CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    for task in contract["tasks"].values():
        if not Path(task["vlm_selection"]).exists():
            raise RuntimeError("missing frozen VLM selection")
        if not Path(task["tabular_checkpoint"]).exists():
            raise RuntimeError("missing frozen tabular checkpoint")
    if args.recovery_note:
        write_json(
            root / "RECOVERY.json",
            {
                "note": args.recovery_note,
                "final_evaluation_only": True,
                "discovery_repeated": False,
                "tabular_training_repeated": False,
                "agents_relaunched": False,
            },
        )
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", **WORKER_ENV)
    processes = []
    for lane in contract["tasks"]:
        for method in ("vlm", "tabular"):
            output = root / "results" / lane / f"{method}.json"
            log = root / "logs" / f"{lane}-{method}-evaluation.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("w")
            command = [
                str(PYTHON), str(REPO_ROOT / "scripts" / "evaluate_one_reset_generalization.py"),
                "--contract", str(contract_path), "--task", lane,
                "--method", method, "--output", str(output),
            ]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, text=True)
            processes.append((lane, method, output, stream, process))
    for lane, method, _, stream, process in processes:
        code = process.wait()
        stream.close()
        if code != 0:
            raise RuntimeError(f"{lane} {method} evaluation failed with {code}")
    results = {
        lane: {
            method: json.loads(output.read_text())
            for current_lane, method, output, _, _ in processes
            if current_lane == lane
        }
        for lane in contract["tasks"]
    }
    write_json(root / "RESULTS.json", results)
    write_json(root / "FINAL_STATUS.json", {"terminal": True, "state": "complete"})
    (root / "COMPLETE").write_text("complete\n")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] three-scene final evaluation complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
