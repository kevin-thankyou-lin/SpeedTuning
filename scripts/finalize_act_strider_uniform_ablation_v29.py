#!/usr/bin/env python3
"""Seal the paired STRIDER-versus-uniform v29 ablation."""

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


def checked(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed receipt: {path}")
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
    parser.add_argument("--v28-root", type=Path, required=True)
    args = parser.parse_args()

    v28_result_path = args.v28_root / "RESULT.json"
    v28_complete = checked(args.v28_root / "COMPLETE.json")
    v28_result = checked(v28_result_path)
    if v28_complete.get("result_sha256") != sha256(v28_result_path):
        raise RuntimeError("v28 aggregate hash mismatch")

    tasks = {}
    physics = safety = new_rollouts = 0
    for task in ("pick", "tea"):
        task_root = args.root / task
        result_path = task_root / "RESULT.json"
        complete = checked(task_root / "COMPLETE.json")
        result = checked(result_path)
        if complete.get("result_sha256") != sha256(result_path):
            raise RuntimeError(f"v29 result hash mismatch: {task}")
        if complete.get("new_rollouts") != 50:
            raise RuntimeError(f"v29 rollout accounting mismatch: {task}")
        physics += int(complete["physics_errors"])
        safety += int(complete["safety_violations"])
        new_rollouts += int(complete["new_rollouts"])
        tasks[task] = result

    insertion = v28_result["tasks"]["insertion"]
    if insertion["selected_schedule"] != [1.5, 1.5, 1.5, 1.5]:
        raise RuntimeError("Insertion STRIDER is not the registered uniform identity case")
    tasks["insertion"] = {
        "schema": "act-strider-uniform-identity-result-v29",
        "task_label": "insertion",
        "strider_schedule": insertion["selected_schedule"],
        "uniform_schedule": insertion["selected_schedule"],
        "strider_summary": insertion["summary"],
        "uniform_summary": insertion["summary"],
        "paired": {
            "episodes": 50,
            "success_contingency": {
                "both_success": insertion["summary"]["successes"],
                "strider_only": 0,
                "uniform_only": 0,
                "both_fail": 50 - insertion["summary"]["successes"],
            },
            "success_delta_strider_minus_uniform": 0,
            "identity_comparison": True,
        },
        "episodes_per_controller": 50,
        "new_rollouts": 0,
        "v28_result_sha256": sha256(v28_result_path),
    }
    aggregate = {
        "schema": "act-strider-uniform-ablation-aggregate-v29",
        "study_type": "posthoc_paired_adaptivity_ablation",
        "tasks": tasks,
        "accounting": {
            "new_rollouts": new_rollouts,
            "insertion_identity_reuse_rollouts": 50,
            "v20_v26_v27_v28_rollouts_reexecuted": 0,
            "physics_errors": physics,
            "safety_violations": safety,
        },
        "interpretation": "paired posthoc ablation on an already-opened final bank",
    }
    result_path = args.root / "RESULT.json"
    immutable_or_equal(result_path, aggregate)
    immutable_or_equal(args.root / "COMPLETE.json", {
        "schema": "act-strider-uniform-ablation-aggregate-completion-v29",
        "result_sha256": sha256(result_path),
        **aggregate["accounting"],
    })
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
