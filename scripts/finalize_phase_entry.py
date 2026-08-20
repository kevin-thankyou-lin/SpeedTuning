#!/usr/bin/env python3
"""Finalize one phase-entry lane against an existing matched native bank."""

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
    lane = args.round_root / "phase_entry"
    while not (lane / "COMPLETE").exists():
        if (lane / "FAILED").exists():
            write_json(args.round_root / "FINAL_STATUS.json", {
                "state": "failed", "reason": "phase-entry lane failed"
            })
            return 1
        time.sleep(5)

    candidates = json.loads((lane / "RESULTS.json").read_text())
    native = json.loads(args.native_results.read_text())
    result = {}
    for task in TASKS:
        candidate_eval = candidates[task]["evaluation"]
        native_eval = native[task]
        candidate_steps = successful_mean_steps(candidate_eval)
        native_steps = successful_mean_steps(native_eval)
        result[task] = {
            "training_episodes": candidates[task]["training"]["episodes"],
            "training_decisions": candidates[task]["training"]["decisions"],
            "candidate_successes": candidate_eval["successes"],
            "native_successes": native_eval["successes"],
            "candidate_successful_mean_steps": candidate_steps,
            "native_successful_mean_steps": native_steps,
            "success_only_speedup": (
                None if candidate_steps is None or native_steps is None
                else native_steps / candidate_steps
            ),
        }
    write_json(args.round_root / "COMPARISON.json", {"phase_entry": result})
    write_json(args.round_root / "FINAL_STATUS.json", {"state": "complete"})
    (args.round_root / "COMPLETE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
