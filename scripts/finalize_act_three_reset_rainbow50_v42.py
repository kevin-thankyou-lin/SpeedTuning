#!/usr/bin/env python3
"""Seal the V42 three-reset Rainbow 50-episode continuation benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_act_champion_challenger_v37 as v37  # noqa: E402
from scripts import run_act_three_reset_rainbow50_v42 as v42  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256  # noqa: E402


def checked_states(root: Path, task: str, method: str, seeds: list[int]):
    alias = v42.checked_json(root / "final" / task / "methods" / method / "RESULT.json")
    controller_root = root / "final" / task / "controllers" / alias["controller_sha256"]
    result = v42.checked_json(controller_root / "RESULT.json")
    complete = v42.checked_json(controller_root / "COMPLETE.json")
    identity = v42.checked_json(controller_root / "IDENTITY.json")
    if alias["controller_result_sha256"] != file_sha256(controller_root / "RESULT.json"):
        raise RuntimeError(f"v42 method alias mismatch: {task}/{method}")
    if int(complete["episodes"]) != len(seeds):
        raise RuntimeError(f"v42 controller count mismatch: {task}/{method}")
    if complete["result_sha256"] != file_sha256(controller_root / "RESULT.json"):
        raise RuntimeError(f"v42 controller completion mismatch: {task}/{method}")
    states = [
        v42.checked_json(controller_root / "states" / f"{seed}.json") for seed in seeds
    ]
    if any(item.get("identity_sha256") != identity["identity_sha256"] for item in states):
        raise RuntimeError(f"v42 controller state mismatch: {task}/{method}")
    return result, states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    banks = v42.checked_json(args.banks)
    v42.validate_banks(banks)
    v42.require_all_search(root)
    tasks = {}
    heldout_physics_errors = 0
    heldout_safety_violations = 0
    prefinal_physics_errors = 0
    prefinal_safety_violations = 0
    executed_controllers = 0
    for task in v42.TASKS:
        seeds = list(map(int, banks["tasks"][task]["final"]))
        finals, states, hashes = {}, {}, set()
        for method in v42.FINAL_METHODS:
            finals[method], states[method] = checked_states(root, task, method, seeds)
            hashes.add(finals[method]["controller"]["controller_sha256"])
        executed_controllers += len(hashes)
        for digest in hashes:
            result = v42.checked_json(
                root / "final" / task / "controllers" / digest / "RESULT.json"
            )
            heldout_physics_errors += int(result["summary"]["physics_errors"])
            heldout_safety_violations += int(result["summary"]["safety_violations"])
        selection = v42.checked_json(root / "search" / task / "SELECTION.json")
        prefinal_physics_errors += int(selection["incident_totals"]["physics_errors"])
        prefinal_safety_violations += int(selection["incident_totals"]["safety_violations"])
        tasks[task] = {
            "search_selection": selection,
            "fixed_pose_matched_probe": selection["fixed_probe"],
            "final": finals,
            "paired": {
                "rainbow_vs_uniform_2x": v37.paired_receipt(
                    states["rainbow"], states["uniform_2x"]
                ),
                "selected_vs_uniform_2x": v37.paired_receipt(
                    states["selected"], states["uniform_2x"]
                ),
                "selected_vs_native_1x": v37.paired_receipt(
                    states["selected"], states["native_1x"]
                ),
            },
        }
    result = {
        "schema": "act-three-reset-rainbow50-heldout-result-v42",
        "label": "V41 Rainbow continued from 18 to 50 episodes on the same three resets",
        "tasks": tasks,
        "accounting": {
            "parent_v41_training_rollouts_inherited": len(v42.TASKS) * v42.PARENT_TRAINING_EPISODES,
            "new_extension_training_rollouts": len(v42.TASKS) * v42.EXTENSION_EPISODES,
            "new_fixed_probe_rollouts": len(v42.TASKS) * 2 * v42.FIXED_PROBE_EPISODES,
            "new_fresh_screen_rollouts": len(v42.TASKS) * v42.SCREEN_EPISODES,
            "new_prefinal_scientific_rollouts": len(v42.TASKS) * v42.NEW_PREFINAL_ROLLOUTS,
            "heldout_unique_controllers": executed_controllers,
            "heldout_final_rollouts_executed": executed_controllers * v42.FINAL_EPISODES,
            "heldout_method_cells": len(v42.TASKS) * len(v42.FINAL_METHODS),
            "heldout_cache_saved_rollouts": (
                len(v42.TASKS) * len(v42.FINAL_METHODS) - executed_controllers
            ) * v42.FINAL_EPISODES,
            "parent_v41_training_state_inherited": True,
            "parent_v41_screen_or_heldout_outcomes_used": False,
            "parent_rollouts_reexecuted": 0,
            "prefinal_physics_errors": prefinal_physics_errors,
            "prefinal_safety_violations": prefinal_safety_violations,
            "heldout_physics_errors": heldout_physics_errors,
            "heldout_safety_violations": heldout_safety_violations,
            "physics_errors": prefinal_physics_errors + heldout_physics_errors,
            "safety_violations": prefinal_safety_violations + heldout_safety_violations,
        },
    }
    v42.immutable_or_verify(root / "RESULT.json", result)
    v42.immutable_or_verify(root / "COMPLETE.json", {
        "schema": "act-three-reset-rainbow50-heldout-completion-v42",
        **result["accounting"], "banks_sha256": file_sha256(args.banks),
        "result_sha256": file_sha256(root / "RESULT.json"),
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
