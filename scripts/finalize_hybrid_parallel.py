#!/usr/bin/env python3
"""Merge three parallel hybrid task workers against the cached native bank."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


TASKS = ("pick", "tea", "insertion")


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
    parser.add_argument("--native-results", type=Path, required=True)
    args = parser.parse_args()
    while True:
        failed = [
            task for task in TASKS
            if (args.round_root / "workers" / task / "hybrid" / "FAILED").exists()
        ]
        if failed:
            write_json(args.round_root / "FINAL_STATUS.json", {
                "state": "failed", "failed_tasks": failed
            })
            return 1
        if all(
            (args.round_root / "workers" / task / "hybrid" / "COMPLETE").exists()
            for task in TASKS
        ):
            break
        time.sleep(5)

    native = json.loads(args.native_results.read_text())
    result = {}
    for task in TASKS:
        candidate = json.loads(
            (args.round_root / "workers" / task / "hybrid" / "RESULTS.json").read_text()
        )[task]
        candidate_eval = candidate["evaluation"]
        native_eval = native[task]
        candidate_steps = successful_mean_steps(candidate_eval)
        native_steps = successful_mean_steps(native_eval)
        result[task] = {
            "training_episodes": candidate["training"]["episodes"],
            "training_decisions": candidate["training"]["decisions"],
            "candidate_successes": candidate_eval["successes"],
            "native_successes": native_eval["successes"],
            "candidate_successful_mean_steps": candidate_steps,
            "native_successful_mean_steps": native_steps,
            "success_only_speedup": (
                None if candidate_steps is None or native_steps is None
                else native_steps / candidate_steps
            ),
        }
    write_json(args.round_root / "COMPARISON.json", {"hybrid": result})
    write_json(args.round_root / "FINAL_STATUS.json", {"state": "complete"})
    (args.round_root / "COMPLETE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
