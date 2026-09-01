#!/usr/bin/env python3
"""Seal the v41 three-reset Rainbow held-out comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_act_champion_challenger_v37 as v37  # noqa: E402
from scripts import run_act_three_reset_rainbow25_v41 as v41  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256  # noqa: E402


def checked_states(root: Path, task: str, method: str, seeds: list[int]):
    alias = v41.checked_json(root / "final" / task / "methods" / method / "RESULT.json")
    controller_root = root / "final" / task / "controllers" / alias["controller_sha256"]
    result = v41.checked_json(controller_root / "RESULT.json")
    complete = v41.checked_json(controller_root / "COMPLETE.json")
    identity = v41.checked_json(controller_root / "IDENTITY.json")
    if alias["controller_result_sha256"] != file_sha256(controller_root / "RESULT.json"):
        raise RuntimeError(f"v41 method alias mismatch: {task}/{method}")
    if int(complete["episodes"]) != len(seeds):
        raise RuntimeError(f"v41 controller count mismatch: {task}/{method}")
    if complete["result_sha256"] != file_sha256(controller_root / "RESULT.json"):
        raise RuntimeError(f"v41 controller completion mismatch: {task}/{method}")
    states = [
        v41.checked_json(controller_root / "states" / f"{seed}.json") for seed in seeds
    ]
    if any(item.get("identity_sha256") != identity["identity_sha256"] for item in states):
        raise RuntimeError(f"v41 controller state mismatch: {task}/{method}")
    return result, states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    banks = v41.checked_json(args.banks)
    v41.validate_banks(banks)
    v41.require_all_search(root)
    tasks = {}
    physics_errors = 0
    safety_violations = 0
    executed_controllers = 0
    for task in v41.TASKS:
        seeds = list(map(int, banks["tasks"][task]["final"]))
        finals, states, hashes = {}, {}, set()
        for method in v41.FINAL_METHODS:
            finals[method], states[method] = checked_states(root, task, method, seeds)
            hashes.add(finals[method]["controller"]["controller_sha256"])
        executed_controllers += len(hashes)
        for digest in hashes:
            result = v41.checked_json(
                root / "final" / task / "controllers" / digest / "RESULT.json"
            )
            physics_errors += int(result["summary"]["physics_errors"])
            safety_violations += int(result["summary"]["safety_violations"])
        tasks[task] = {
            "search_selection": v41.checked_json(root / "search" / task / "SELECTION.json"),
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
        "schema": "act-three-reset-rainbow25-heldout-result-v41",
        "label": "fresh phase-conditioned Rainbow trained on three repeated resets",
        "tasks": tasks,
        "accounting": {
            "online_search_rollouts": len(v41.TASKS) * v41.SEARCH_BUDGET,
            "online_search_rollouts_per_task": v41.SEARCH_BUDGET,
            "heldout_unique_controllers": executed_controllers,
            "heldout_final_rollouts_executed": executed_controllers * v41.FINAL_EPISODES,
            "heldout_method_cells": len(v41.TASKS) * len(v41.FINAL_METHODS),
            "heldout_cache_saved_rollouts": (
                len(v41.TASKS) * len(v41.FINAL_METHODS) - executed_controllers
            ) * v41.FINAL_EPISODES,
            "historical_speed_outcomes_used_for_initialization": False,
            "historical_v20_v40_rollouts_reexecuted": 0,
            "physics_errors": physics_errors,
            "safety_violations": safety_violations,
        },
    }
    v41.immutable_or_verify(root / "RESULT.json", result)
    v41.immutable_or_verify(
        root / "COMPLETE.json",
        {"schema": "act-three-reset-rainbow25-heldout-completion-v41",
         **result["accounting"], "banks_sha256": file_sha256(args.banks),
         "result_sha256": file_sha256(root / "RESULT.json")},
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
