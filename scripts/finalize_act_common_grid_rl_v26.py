#!/usr/bin/env python3
"""Seal aggregate receipts for the common-grid RL v26 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from act_speed_benchmark import SPEED_VALUES, sha256
from scripts.run_act_speed_benchmark_cell import immutable_json

TASKS = ("pick", "tea", "insertion")
METHODS = ("learned_phase_tabular_rl", "learned_phase_rainbow_rl")


def checked(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed receipt: {path}")
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    manifest = checked(args.run_manifest)
    if manifest.get("schema") != "act-common-grid-rl-run-manifest-v26":
        raise RuntimeError("unexpected manifest")

    cells = {}
    invalid = safety = 0
    for task in TASKS:
        cells[task] = {}
        for method in METHODS:
            cell = args.root / "cells" / task / method
            search_complete = checked(cell / "search" / "COMPLETE.json")
            selected = checked(cell / "search" / "selected.json")
            final_complete = checked(cell / "final" / "COMPLETE.json")
            result = checked(cell / "final" / "result.json")
            if search_complete.get("episodes") != 25 or final_complete.get("episodes") != 50:
                raise RuntimeError(f"budget mismatch: {task}/{method}")
            if search_complete.get("selected_sha256") != sha256(cell / "search" / "selected.json"):
                raise RuntimeError(f"selection hash mismatch: {task}/{method}")
            if final_complete.get("result_sha256") != sha256(cell / "final" / "result.json"):
                raise RuntimeError(f"result hash mismatch: {task}/{method}")
            policy = selected["selected_policy"]
            grid = policy.get("speed_values")
            if grid is None and method == "learned_phase_rainbow_rl":
                grid = list(SPEED_VALUES)
            if tuple(grid or ()) != SPEED_VALUES:
                raise RuntimeError(f"controller grid mismatch: {task}/{method}")
            invalid += int(result.get("physics_errors", 0))
            safety += int(result.get("safety_violations", 0))
            cells[task][method] = {
                "search_complete_sha256": sha256(cell / "search" / "COMPLETE.json"),
                "selected_sha256": sha256(cell / "search" / "selected.json"),
                "final_complete_sha256": sha256(cell / "final" / "COMPLETE.json"),
                "result_sha256": sha256(cell / "final" / "result.json"),
                "selected_policy": policy,
                "result": result,
            }
    aggregate = {
        "schema": "act-common-grid-rl-result-v26",
        "run_manifest_sha256": sha256(args.run_manifest),
        "action_grid": list(SPEED_VALUES),
        "cells": cells,
        "accounting": {
            "new_training_rollouts": 150,
            "new_final_rollouts": 300,
            "prior_rollouts_reexecuted": 0,
            "simulator_invalid_attempts": invalid,
            "safety_incidents": safety,
        },
    }
    result_path = args.root / "RESULT.json"
    immutable_json(result_path, aggregate)
    immutable_json(
        args.root / "COMPLETE.json",
        {
            "schema": "act-common-grid-rl-completion-v26",
            "run_manifest_sha256": sha256(args.run_manifest),
            "result_sha256": sha256(result_path),
            **aggregate["accounting"],
        },
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
