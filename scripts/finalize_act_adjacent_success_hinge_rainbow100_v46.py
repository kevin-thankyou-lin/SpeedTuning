#!/usr/bin/env python3
"""Seal the V46 Adjacent-success-hinge Rainbow-100 aggregate receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import run_act_adjacent_success_hinge_rainbow100_v46 as v46
from scripts.run_act_strider_frontier_v4 import file_sha256


def read_final(root: Path, task: str, method: str, seeds: list[int]):
    method_root = root / "final" / task / "methods" / method
    result = v46.checked_json(method_root / "RESULT.json")
    complete = v46.checked_json(method_root / "COMPLETE.json")
    identity = v46.checked_json(method_root / "IDENTITY.json")
    if int(result["episodes"]) != v46.FINAL_EPISODES:
        raise RuntimeError(f"v46 final count mismatch: {task}/{method}")
    if complete["result_sha256"] != file_sha256(method_root / "RESULT.json"):
        raise RuntimeError(f"v46 final completion mismatch: {task}/{method}")
    records = [
        v46.checked_json(method_root / "states" / f"{seed}.json") for seed in seeds
    ]
    if [record["seed"] for record in records] != seeds:
        raise RuntimeError(f"v46 final seed order mismatch: {task}/{method}")
    if any(record["identity_sha256"] != identity["identity_sha256"] for record in records):
        raise RuntimeError(f"v46 final state identity mismatch: {task}/{method}")
    return result, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    banks = v46.checked_json(args.banks)
    v46.validate_banks(banks)
    v46.require_all_search(root)

    tasks = {}
    prefinal_physics = 0
    prefinal_safety = 0
    final_physics = 0
    final_safety = 0
    for task in v46.TASKS:
        task_seed_banks = v46.task_banks(banks["tasks"][task])
        seeds = task_seed_banks["final"]
        finals = {}
        for method in v46.FINAL_METHODS:
            finals[method], _ = read_final(root, task, method, seeds)
            final_physics += int(finals[method]["summary"]["physics_errors"])
            final_safety += int(finals[method]["summary"]["safety_violations"])
        search = v46.checked_json(root / "search" / task / "SEARCH_RESULT.json")
        prefinal_physics += int(search["incident_totals"]["physics_errors"])
        prefinal_safety += int(search["incident_totals"]["safety_violations"])
        tasks[task] = {
            "search": search,
            "final": finals,
            "paired_baseline_receipts": (
                "read from sealed V43 and V45 by downstream comparison"
            ),
        }

    result = {
        "schema": "act-adjacent-success-hinge-rainbow100-heldout-result-v46",
        "label": (
            "Adjacent-success zero-margin L1 hinge Rainbow from scratch on "
            "V43/V45 paired randomized resets"
        ),
        "tasks": tasks,
        "accounting": {
            "training_rollouts": len(v46.TASKS) * v46.TRAINING_EPISODES,
            "checkpoint_probe_rollouts": (
                len(v46.TASKS) * len(v46.CHECKPOINT_EPISODES) * v46.PROBE_EPISODES
            ),
            "prefinal_scientific_rollouts": len(v46.TASKS) * v46.PREFINAL_ROLLOUTS,
            "heldout_final_rollouts": (
                len(v46.TASKS) * len(v46.FINAL_METHODS) * v46.FINAL_EPISODES
            ),
            "historical_rollouts_reexecuted": 0,
            "v43_baseline_rollouts_reexecuted": 0,
            "v45_baseline_rollouts_reexecuted": 0,
            "checkpoint_probe_outcomes_used_for_training_or_selection": False,
            "prefinal_physics_errors": prefinal_physics,
            "prefinal_safety_violations": prefinal_safety,
            "heldout_physics_errors": final_physics,
            "heldout_safety_violations": final_safety,
            "physics_errors": prefinal_physics + final_physics,
            "safety_violations": prefinal_safety + final_safety,
        },
    }
    v46.immutable_or_verify(root / "RESULT.json", result)
    v46.immutable_or_verify(root / "COMPLETE.json", {
        "schema": "act-adjacent-success-hinge-rainbow100-heldout-completion-v46",
        **result["accounting"],
        "banks_sha256": file_sha256(args.banks),
        "result_sha256": file_sha256(root / "RESULT.json"),
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
