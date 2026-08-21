#!/usr/bin/env python3
"""Replay and relearn one-reset schedules with the sealed learned detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
EXPECTED = {
    "checkpoint_sha256": "c25c3f530da42eb7c60e5f70405b3a99c56ab72c1e53dfd27055dc3d99c3512d",
    "inference_sha256": "1398e1d1b5b4e682f009c6501598e651a516341f6d60822f40fc575a40061815",
    "model_source_sha256": "8a47f110f19f4e52a39b7e0e4f2273c2895690f6332ab17a4b71c8eb5ce4ae37",
}
WORKER_ENV = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "SPEEDTUNING_TORCH_THREADS": "2",
}
TASK_NAMES = {"pick": "pick_and_place", "tea": "tea_bag", "insertion": "insertion"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run_evaluations(contract_path: Path, output_root: Path) -> dict:
    processes = []
    logs = output_root.parent / "logs"
    env = dict(
        os.environ,
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        **WORKER_ENV,
    )
    for lane in TASK_NAMES:
        for method in ("vlm", "tabular"):
            output = output_root / lane / f"{method}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            log_path = logs / f"{lane}-{method}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("w")
            command = [
                str(PYTHON), str(REPO_ROOT / "scripts" / "evaluate_one_reset_generalization.py"),
                "--contract", str(contract_path), "--task", lane,
                "--method", method, "--output", str(output),
            ]
            processes.append((lane, method, output, stream, subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, env=env, text=True
            )))
    for lane, method, _, stream, process in processes:
        code = process.wait()
        stream.close()
        if code != 0:
            raise RuntimeError(f"{lane} {method} learned-detector evaluation failed with {code}")
    return {
        lane: {
            method: json.loads(output.read_text())
            for current_lane, method, output, _, _ in processes
            if current_lane == lane
        }
        for lane in TASK_NAMES
    }


def learned_task_contract(task: str) -> str:
    return f"""# One-reset learned-phase generalization study

Task: `{task}`
Learning evidence: exactly one frozen initial scene, reused for every schedule.
Learning budget: 50 total episodes, including native `1x`.
Runtime phases: `pre_grasp`, `grasp_lift`, `transport`, `interaction` from the sealed RGB/proprio detector, raw argmax with no oracle fallback.
Allowed speeds: `1, 1.5, 2, 2.5, 3, 3.5, 4`.
Client: `python3 /workspace/one_reset.py --api /workspace/runner_api`.

Start with `info` and inspect `media/native.mp4`. Test schedules with `test A,B,C,D`; each unique schedule costs one episode and returns its learned-phase entry trace, phase workload estimate, and MP4. Before choosing among plausible challengers, call `score ANCHOR_HASH A,B,C,D --safe-success-probability P` for each, where `P` is your conservative MP4- and evidence-based probability of safe success. Prioritize the largest `expected_absolute_steps_saved = P * predicted_absolute_steps_saved`, not the largest multiplier or relative speedup. The score is only an acquisition estimate; protect weak phases and retain a safely successful anchor. You may stop early. Before terminating, call `select HASH` for one tested schedule that succeeded without a workspace violation. Optimize for likely randomized-pose generalization first and absolute expected time reduction second; one-scene success is not a reliability claim. Do not access other experiments, seeds, private state, oracle phases, or prior schedules.

Write only `analysis/STATUS.json` and `analysis/FINAL_REPORT.md`. `STATUS.json` must set `terminal: true` and record the selected schedule hash, schedule, budget used, detector type, and the limitation that learning used one state.
"""


def prepare_retrain_contract(source_root: Path, retrain_root: Path, detector: dict) -> Path:
    source = json.loads((source_root / "CONTRACT.json").read_text())
    contract = deepcopy(source)
    contract["schema"] = "one-reset-learned-phase-generalization-v1"
    contract["phase_labels"] = "sealed learned RGB/proprio four-phase raw argmax"
    contract["phase_detector"] = detector
    for lane, task in contract["tasks"].items():
        task["vlm_runner_root"] = str(retrain_root / "vlm_runners" / lane)
        task["vlm_agent_root"] = str(retrain_root / "vlm_agents" / lane)
        task["vlm_selection"] = str(retrain_root / "vlm_runners" / lane / "public" / "SELECTION.json")
        task["tabular_checkpoint"] = str(retrain_root / "tabular" / lane / "checkpoint.json")
        task["tabular_report"] = str(retrain_root / "tabular" / lane / "training.json")
        source_agent = source_root / "vlm_agents" / lane
        target_agent = retrain_root / "vlm_agents" / lane
        target_agent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_agent / "AGENTS.md", target_agent / "AGENTS.md")
        shutil.copy2(REPO_ROOT / "scripts" / "one_reset_client.py", target_agent / "one_reset.py")
        (target_agent / "TASK_CONTRACT.md").write_text(
            learned_task_contract(TASK_NAMES[lane])
        )
    path = retrain_root / "CONTRACT.json"
    write_json(path, contract)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    root = args.root.resolve()
    if args.resume:
        if not root.is_dir():
            raise RuntimeError("resume root does not exist")
    else:
        root.mkdir(parents=True, exist_ok=False)

    detector_source = args.detector_source.resolve()
    detector_checkpoint = args.detector_checkpoint.resolve()
    actual = {
        "checkpoint_sha256": sha256(detector_checkpoint),
        "inference_sha256": sha256(detector_source / "phase_detector" / "rgb_inference.py"),
        "model_source_sha256": sha256(detector_source / "phase_detector" / "rgb_proprio.py"),
    }
    if actual != EXPECTED:
        raise RuntimeError(f"sealed learned detector hash mismatch: {actual}")
    detector = {
        "checkpoint_path": str(detector_checkpoint),
        "source_root": str(detector_source),
        **actual,
        "device": "cuda",
        "history_stride": 5,
        "cpu_threads_per_worker": 2,
        "render_camera_names": ["angle"],
    }
    detector_path = root / "DETECTOR.json"
    if args.resume and detector_path.exists():
        if json.loads(detector_path.read_text()) != detector:
            raise RuntimeError("resume detector identity mismatch")
    write_json(detector_path, detector)

    replay_contract = json.loads((source_root / "CONTRACT.json").read_text())
    replay_contract["phase_detector"] = detector
    replay_path = root / "replay" / "CONTRACT.json"
    write_json(replay_path, replay_contract)
    log("starting frozen-schedule replay with learned detector")
    replay = run_evaluations(replay_path, root / "replay" / "results")
    write_json(root / "replay" / "RESULTS.json", replay)
    log("frozen-schedule learned-detector replay complete")

    retrain_root = root / "retrain"
    prepare_retrain_contract(source_root, retrain_root, detector)
    log("starting one-reset VLM and tabular relearning with learned detector")
    env = dict(
        os.environ,
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        **WORKER_ENV,
    )
    code = subprocess.call(
        [str(PYTHON), str(REPO_ROOT / "scripts" / "manage_one_reset_round.py"),
         "--root", str(retrain_root)],
        env=env,
    )
    if code != 0:
        raise RuntimeError(f"learned-detector relearning manager failed with {code}")
    retrained = json.loads((retrain_root / "RESULTS.json").read_text())
    write_json(root / "RESULTS.json", {"replay": replay, "retrained": retrained})
    write_json(root / "FINAL_STATUS.json", {"terminal": True, "state": "complete"})
    (root / "COMPLETE").write_text("complete\n")
    log("learned-detector one-reset follow-up complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
