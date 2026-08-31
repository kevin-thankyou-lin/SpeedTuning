#!/usr/bin/env python3
"""Seal the v37 accelerated champion-challenger comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_act_champion_challenger_v37 as v37
from scripts.run_act_strider_frontier_v4 import file_sha256


def checked_states(
    root: Path, task: str, method: str, seeds: list[int]
) -> tuple[dict, list[dict]]:
    alias = v37.v33.checked_json(
        root / "final" / task / "methods" / method / "RESULT.json"
    )
    controller = root / "final" / task / "controllers" / alias["controller_sha256"]
    result = v37.v33.checked_json(controller / "RESULT.json")
    complete = v37.v33.checked_json(controller / "COMPLETE.json")
    identity = v37.v33.checked_json(controller / "IDENTITY.json")
    if alias["controller_result_sha256"] != file_sha256(controller / "RESULT.json"):
        raise RuntimeError(f"v37 method alias mismatch: {task}/{method}")
    if int(complete["episodes"]) != len(seeds):
        raise RuntimeError(f"v37 controller count mismatch: {task}/{method}")
    if complete["result_sha256"] != file_sha256(controller / "RESULT.json"):
        raise RuntimeError(f"v37 controller completion mismatch: {task}/{method}")
    states = [
        v37.v33.checked_json(controller / "states" / f"{seed}.json") for seed in seeds
    ]
    if any(
        item.get("identity_sha256") != identity["identity_sha256"] for item in states
    ):
        raise RuntimeError(f"v37 controller state mismatch: {task}/{method}")
    return result, states


def paired(left: list[dict], right: list[dict]) -> dict:
    if [item["seed"] for item in left] != [item["seed"] for item in right]:
        raise RuntimeError("v37 final paired seed order differs")
    pairs = list(zip(left, right))
    common = [(a, b) for a, b in pairs if v37.successful(a) and v37.successful(b)]
    return {
        "pairs": len(pairs),
        "both_success": len(common),
        "left_only_success": sum(
            v37.successful(a) and not v37.successful(b) for a, b in pairs
        ),
        "right_only_success": sum(
            not v37.successful(a) and v37.successful(b) for a, b in pairs
        ),
        "left_speedup_vs_right_on_common_success": (
            None
            if not common
            else (sum(v37.metric_steps(b) for _, b in common) / len(common))
            / (sum(v37.metric_steps(a) for a, _ in common) / len(common))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--champions", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    banks = v37.v33.checked_json(args.banks)
    v37.validate_banks(banks)
    v37.require_all_search(root)
    tasks = {}
    physics_errors = 0
    safety_violations = 0
    executed_controllers = 0
    for task in v37.TASKS:
        seeds = list(map(int, banks["tasks"][task]["final"]))
        finals = {}
        states = {}
        hashes = set()
        for method in v37.FINAL_METHODS:
            finals[method], states[method] = checked_states(root, task, method, seeds)
            hashes.add(finals[method]["schedule_sha256"])
        executed_controllers += len(hashes)
        for controller_hash in hashes:
            summary = v37.v33.checked_json(
                root / "final" / task / "controllers" / controller_hash / "RESULT.json"
            )["summary"]
            physics_errors += int(summary["physics_errors"])
            safety_violations += int(summary["safety_violations"])
        tasks[task] = {
            "search_selection": v37.v33.checked_json(
                root / "search" / task / "SELECTION.json"
            ),
            "final": finals,
            "paired": {
                "selected_vs_champion": paired(states["selected"], states["champion"]),
                "selected_vs_native_1x": paired(
                    states["selected"], states["native_1x"]
                ),
                "champion_vs_native_1x": paired(
                    states["champion"], states["native_1x"]
                ),
            },
        }
    result = {
        "schema": "act-champion-challenger-heldout-result-v37",
        "label": "exact-25 accelerated incumbent versus adjacent challenger",
        "tasks": tasks,
        "accounting": {
            "online_search_rollouts": len(v37.TASKS) * v37.SEARCH_BUDGET,
            "online_search_rollouts_per_task": v37.SEARCH_BUDGET,
            "heldout_unique_controllers": executed_controllers,
            "heldout_final_rollouts_executed": executed_controllers * 50,
            "heldout_method_cells": len(v37.TASKS) * len(v37.FINAL_METHODS),
            "heldout_cache_saved_rollouts": (
                len(v37.TASKS) * len(v37.FINAL_METHODS) - executed_controllers
            )
            * 50,
            "historical_v20_v36_rollouts_reexecuted": 0,
            "physics_errors": physics_errors,
            "safety_violations": safety_violations,
        },
    }
    v37.v33.immutable_or_verify(root / "RESULT.json", result)
    v37.v33.immutable_or_verify(
        root / "COMPLETE.json",
        {
            "schema": "act-champion-challenger-heldout-completion-v37",
            **result["accounting"],
            "banks_sha256": file_sha256(args.banks),
            "champions_sha256": file_sha256(args.champions),
            "result_sha256": file_sha256(root / "RESULT.json"),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
