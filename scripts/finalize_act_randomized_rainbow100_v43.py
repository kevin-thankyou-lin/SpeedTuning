#!/usr/bin/env python3
"""Seal the V43 randomized-reset Rainbow-100 aggregate receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import run_act_champion_challenger_v37 as v37
from scripts import run_act_randomized_rainbow100_v43 as v43
from scripts.run_act_strider_frontier_v4 import file_sha256


def read_final(root: Path, task: str, method: str, seeds: list[int]):
    method_root = root / "final" / task / "methods" / method
    result = v43.checked_json(method_root / "RESULT.json")
    complete = v43.checked_json(method_root / "COMPLETE.json")
    identity = v43.checked_json(method_root / "IDENTITY.json")
    if int(result["episodes"]) != v43.FINAL_EPISODES:
        raise RuntimeError(f"v43 final count mismatch: {task}/{method}")
    if complete["result_sha256"] != file_sha256(method_root / "RESULT.json"):
        raise RuntimeError(f"v43 final completion mismatch: {task}/{method}")
    records = [
        v43.checked_json(method_root / "states" / f"{seed}.json") for seed in seeds
    ]
    if [record["seed"] for record in records] != seeds:
        raise RuntimeError(f"v43 final seed order mismatch: {task}/{method}")
    if any(record["identity_sha256"] != identity["identity_sha256"] for record in records):
        raise RuntimeError(f"v43 final state identity mismatch: {task}/{method}")
    return result, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    banks = v43.checked_json(args.banks)
    v43.validate_banks(banks)
    v43.require_all_search(root)

    tasks = {}
    prefinal_physics = 0
    prefinal_safety = 0
    final_physics = 0
    final_safety = 0
    for task in v43.TASKS:
        task_seed_banks = v43.task_banks(banks["tasks"][task])
        seeds = task_seed_banks["final"]
        finals, states = {}, {}
        for method in v43.FINAL_METHODS:
            finals[method], states[method] = read_final(root, task, method, seeds)
            final_physics += int(finals[method]["summary"]["physics_errors"])
            final_safety += int(finals[method]["summary"]["safety_violations"])
        search = v43.checked_json(root / "search" / task / "SEARCH_RESULT.json")
        prefinal_physics += int(search["incident_totals"]["physics_errors"])
        prefinal_safety += int(search["incident_totals"]["safety_violations"])
        tasks[task] = {
            "search": search,
            "final": finals,
            "paired": {
                "rainbow_vs_native_1x": v37.paired_receipt(
                    states["rainbow"], states["native_1x"]
                ),
                "rainbow_vs_uniform_2x": v37.paired_receipt(
                    states["rainbow"], states["uniform_2x"]
                ),
            },
        }

    result = {
        "schema": "act-randomized-rainbow100-heldout-result-v43",
        "label": "Rainbow from scratch on 100 distinct randomized training resets",
        "tasks": tasks,
        "accounting": {
            "training_rollouts": len(v43.TASKS) * v43.TRAINING_EPISODES,
            "shared_native_probe_rollouts": len(v43.TASKS) * v43.PROBE_EPISODES,
            "checkpoint_probe_rollouts": (
                len(v43.TASKS) * len(v43.CHECKPOINT_EPISODES) * v43.PROBE_EPISODES
            ),
            "prefinal_scientific_rollouts": len(v43.TASKS) * v43.PREFINAL_ROLLOUTS,
            "heldout_final_rollouts": (
                len(v43.TASKS) * len(v43.FINAL_METHODS) * v43.FINAL_EPISODES
            ),
            "historical_rollouts_reexecuted": 0,
            "checkpoint_probe_outcomes_used_for_training_or_selection": False,
            "prefinal_physics_errors": prefinal_physics,
            "prefinal_safety_violations": prefinal_safety,
            "heldout_physics_errors": final_physics,
            "heldout_safety_violations": final_safety,
            "physics_errors": prefinal_physics + final_physics,
            "safety_violations": prefinal_safety + final_safety,
        },
    }
    v43.immutable_or_verify(root / "RESULT.json", result)
    v43.immutable_or_verify(root / "COMPLETE.json", {
        "schema": "act-randomized-rainbow100-heldout-completion-v43",
        **result["accounting"],
        "banks_sha256": file_sha256(args.banks),
        "result_sha256": file_sha256(root / "RESULT.json"),
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
