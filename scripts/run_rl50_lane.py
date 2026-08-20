#!/usr/bin/env python3
"""Run one 50-episode SpeedTuning lane across the three scripted tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


TASKS = (
    ("pick", "pick_and_place", "scripted-pick-and-place", 41000),
    ("tea", "tea_bag", "scripted-tea-bag-randomized", 42000),
    ("insertion", "insertion", "scripted-insertion", 43000),
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, stdout_path: Path, stderr_path: Path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        subprocess.run(command, check=True, stdout=stdout, stderr=stderr, env=os.environ)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("phase", "full"), required=True)
    parser.add_argument("--round-root", type=Path, required=True)
    args = parser.parse_args()
    lane = args.round_root / args.method
    lane.mkdir(parents=True, exist_ok=True)
    results = {}
    try:
        for label, task, config, train_seed in TASKS:
            task_dir = lane / label
            checkpoint = task_dir / "policy.pt"
            training_report = task_dir / "training.json"
            status = {
                "state": "training",
                "method": args.method,
                "task": label,
                "completed_tasks": list(results),
            }
            write_json(lane / "STATUS.json", status)
            observation_args = []
            if args.method == "phase":
                observation_args = [
                    "--speed-observation", "external",
                    "--observation-encoder-loader",
                    "oracle_phase_observation:create_oracle_phase_encoder",
                ]
            train_command = [
                sys.executable,
                "scripts/train_speed_policy.py",
                "--config", config,
                "--task", task,
                "--training-episodes", "50",
                "--decisions", "100000",
                "--seed", str(train_seed),
                "--device", "cpu",
                "--terminate-on-success",
                "--output", str(checkpoint),
                "--report", str(training_report),
                "--quiet",
                *observation_args,
            ]
            run(train_command, task_dir / "train.stdout", task_dir / "train.stderr")
            training = json.loads(training_report.read_text())
            if training["summary"]["episodes"] != 50:
                raise RuntimeError(f"{label} completed {training['summary']['episodes']} episodes")

            eval_seed = train_seed + 100000
            seeds = ",".join(str(value) for value in range(eval_seed, eval_seed + 100))
            evaluation_path = task_dir / "evaluation.json"
            eval_command = [
                sys.executable,
                "scripts/eval_speed_policy.py",
                "--config", config,
                "--task", task,
                "--speed-policy", "rainbow",
                "--speed-checkpoint", str(checkpoint),
                "--seeds", seeds,
                "--device", "cpu",
                "--terminate-on-success",
                *observation_args,
            ]
            run(eval_command, evaluation_path, task_dir / "eval.stderr")
            evaluation = json.loads(evaluation_path.read_text())
            results[label] = {
                "training": training["summary"],
                "evaluation": evaluation,
                "checkpoint_sha256": sha256(checkpoint),
            }
            write_json(lane / "RESULTS.json", results)

        write_json(lane / "STATUS.json", {
            "state": "complete", "method": args.method, "completed_tasks": list(results)
        })
        (lane / "COMPLETE").touch()
        return 0
    except Exception as exc:
        write_json(lane / "STATUS.json", {
            "state": "failed", "method": args.method, "error": str(exc),
            "traceback": traceback.format_exc(), "completed_tasks": list(results),
        })
        (lane / "FAILED").touch()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
