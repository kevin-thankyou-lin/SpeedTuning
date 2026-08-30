#!/usr/bin/env python3
"""Seal aggregate STRIDER v27 results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import sha256
from scripts.run_act_speed_benchmark_cell import immutable_json

TASKS = ("pick", "tea", "insertion")


def checked(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing receipt: {path}")
    return json.loads(path.read_text())


def immutable_or_equal(path: Path, value: dict) -> None:
    if path.exists():
        if checked(path) != value:
            raise RuntimeError(f"sealed aggregate differs: {path}")
    else:
        immutable_json(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--study-version", choices=("v27", "v28"), default="v27")
    args = parser.parse_args()
    search_per_task = 32 if args.study_version == "v27" else 25
    results = {}
    physics = safety = new_final = final_cache_hits = 0
    for task in TASKS:
        task_root = args.root / task
        complete = checked(task_root / "COMPLETE.json")
        result = checked(task_root / "RESULT.json")
        selection = checked(task_root / "SELECTION.json")
        if complete.get("result_sha256") != sha256(task_root / "RESULT.json"):
            raise RuntimeError(f"result hash mismatch: {task}")
        if complete.get("selection_sha256") != sha256(task_root / "SELECTION.json"):
            raise RuntimeError(f"selection hash mismatch: {task}")
        if complete.get("search_scientific_rollouts") != search_per_task or complete.get("final_scientific_rollouts") != 50:
            raise RuntimeError(f"budget mismatch: {task}")
        physics += int(complete["physics_errors"])
        safety += int(complete["safety_violations"])
        if args.study_version == "v28":
            if int(complete.get("new_final_rollouts", -1)) + int(complete.get("final_cache_hits", -1)) != 50:
                raise RuntimeError(f"final cache accounting mismatch: {task}")
            new_final += int(complete["new_final_rollouts"])
            final_cache_hits += int(complete["final_cache_hits"])
        results[task] = {
            "selected_name": selection["selected_name"],
            "selected_schedule": selection["selected_schedule"],
            "selection_sha256": sha256(task_root / "SELECTION.json"),
            "result_sha256": sha256(task_root / "RESULT.json"),
            "summary": result["summary"],
        }
    aggregate = {
        "schema": f"act-common-grid-strider-aggregate-{args.study_version}",
        "tasks": results,
        "accounting": {
            "search_scientific_rollouts": 3 * search_per_task,
            "final_scientific_rollouts": 150,
            "new_final_rollouts": new_final if args.study_version == "v28" else 150,
            "final_cache_hits": final_cache_hits,
            "v20_v26_rollouts_reexecuted": 0,
            "v27_rollouts_reexecuted": 0,
            "physics_errors": physics,
            "safety_violations": safety,
        },
    }
    result_path = args.root / "RESULT.json"
    immutable_or_equal(result_path, aggregate)
    immutable_or_equal(args.root / "COMPLETE.json", {
        "schema": f"act-common-grid-strider-aggregate-completion-{args.study_version}",
        "result_sha256": sha256(result_path),
        **aggregate["accounting"],
    })
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
