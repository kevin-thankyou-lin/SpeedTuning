#!/usr/bin/env python3
"""Wait for both RL lanes, cache native evaluations, and compute canonical metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


TASKS = (
    ("pick", "pick_and_place", "scripted-pick-and-place", 141000),
    ("tea", "tea_bag", "scripted-tea-bag-randomized", 142000),
    ("insertion", "insertion", "scripted-insertion", 143000),
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def successful_mean_steps(report):
    values = [item["physics_steps"] for item in report["rollouts"] if item["success"]]
    return None if not values else sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.round_root
    while True:
        if any((root / method / "FAILED").exists() for method in ("phase", "full")):
            write_json(root / "FINAL_STATUS.json", {"state": "failed", "reason": "lane failed"})
            return 1
        if all((root / method / "COMPLETE").exists() for method in ("phase", "full")):
            break
        time.sleep(5)

    native_results = {}
    native_root = root / "native"
    for label, task, config, eval_seed in TASKS:
        task_dir = native_root / label
        task_dir.mkdir(parents=True, exist_ok=True)
        seeds = ",".join(str(value) for value in range(eval_seed, eval_seed + 100))
        output = task_dir / "evaluation.json"
        command = [
            sys.executable, "scripts/eval_speed_policy.py",
            "--config", config, "--task", task,
            "--speed-policy", "fixed", "--speed", "1",
            "--seeds", seeds,
        ]
        with output.open("w") as stdout, (task_dir / "eval.stderr").open("w") as stderr:
            subprocess.run(command, check=True, stdout=stdout, stderr=stderr)
        native_results[label] = json.loads(output.read_text())
    write_json(native_root / "RESULTS.json", native_results)

    comparison = {}
    for method in ("phase", "full"):
        method_results = json.loads((root / method / "RESULTS.json").read_text())
        comparison[method] = {}
        for label, _, _, _ in TASKS:
            candidate = method_results[label]["evaluation"]
            native = native_results[label]
            native_steps = successful_mean_steps(native)
            candidate_steps = successful_mean_steps(candidate)
            comparison[method][label] = {
                "training_episodes": method_results[label]["training"]["episodes"],
                "training_decisions": method_results[label]["training"]["decisions"],
                "candidate_successes": candidate["successes"],
                "native_successes": native["successes"],
                "candidate_successful_mean_steps": candidate_steps,
                "native_successful_mean_steps": native_steps,
                "success_only_speedup": (
                    None if candidate_steps is None or native_steps is None
                    else native_steps / candidate_steps
                ),
            }
    write_json(root / "COMPARISON.json", comparison)
    write_json(root / "FINAL_STATUS.json", {"state": "complete"})
    (root / "COMPLETE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
