#!/usr/bin/env python3
"""Seal the v38 transport-first terminal-boundary comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_act_transport3_boundary_v38 as v38
from scripts.run_act_strider_frontier_v4 import file_sha256


def checked_states(root: Path, task: str, method: str, seeds: list[int]):
    alias = v38.v33.checked_json(
        root / "final" / task / "methods" / method / "RESULT.json"
    )
    controller = root / "final" / task / "controllers" / alias["controller_sha256"]
    result = v38.v33.checked_json(controller / "RESULT.json")
    complete = v38.v33.checked_json(controller / "COMPLETE.json")
    identity = v38.v33.checked_json(controller / "IDENTITY.json")
    if alias["controller_result_sha256"] != file_sha256(controller / "RESULT.json"):
        raise RuntimeError(f"v38 method alias mismatch: {task}/{method}")
    if int(complete["episodes"]) != len(seeds):
        raise RuntimeError(f"v38 controller count mismatch: {task}/{method}")
    if complete["result_sha256"] != file_sha256(controller / "RESULT.json"):
        raise RuntimeError(f"v38 controller completion mismatch: {task}/{method}")
    states = [
        v38.v33.checked_json(controller / "states" / f"{seed}.json") for seed in seeds
    ]
    if any(item.get("identity_sha256") != identity["identity_sha256"] for item in states):
        raise RuntimeError(f"v38 controller state mismatch: {task}/{method}")
    return result, states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--controllers", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    banks = v38.v33.checked_json(args.banks)
    v38.validate_banks(banks)
    v38.require_all_search(root)
    tasks = {}
    physics_errors = 0
    safety_violations = 0
    executed_controllers = 0
    gate_events = 0
    for task in v38.TASKS:
        seeds = list(map(int, banks["tasks"][task]["final"]))
        finals = {}
        states = {}
        hashes = set()
        for method in v38.FINAL_METHODS:
            finals[method], states[method] = checked_states(root, task, method, seeds)
            hashes.add(finals[method]["controller"]["controller_sha256"])
        executed_controllers += len(hashes)
        for digest in hashes:
            result = v38.v33.checked_json(
                root / "final" / task / "controllers" / digest / "RESULT.json"
            )
            physics_errors += int(result["summary"]["physics_errors"])
            safety_violations += int(result["summary"]["safety_violations"])
            gate_events += int(result["terminal_approach_events"])
        tasks[task] = {
            "search_selection": v38.v33.checked_json(
                root / "search" / task / "SELECTION.json"
            ),
            "final": finals,
            "paired": {
                "selected_vs_champion": v38.v37.paired_receipt(
                    states["selected"], states["champion"]
                ),
                "selected_vs_native_1x": v38.v37.paired_receipt(
                    states["selected"], states["native_1x"]
                ),
                "champion_vs_native_1x": v38.v37.paired_receipt(
                    states["champion"], states["native_1x"]
                ),
            },
        }
    result = {
        "schema": "act-transport3-boundary-heldout-result-v38",
        "label": "3x free-space transport with current-observation terminal downshift",
        "tasks": tasks,
        "accounting": {
            "online_search_rollouts": len(v38.TASKS) * v38.SEARCH_BUDGET,
            "online_search_rollouts_per_task": v38.SEARCH_BUDGET,
            "heldout_unique_controllers": executed_controllers,
            "heldout_final_rollouts_executed": executed_controllers * 50,
            "heldout_method_cells": len(v38.TASKS) * len(v38.FINAL_METHODS),
            "heldout_cache_saved_rollouts": (
                len(v38.TASKS) * len(v38.FINAL_METHODS) - executed_controllers
            )
            * 50,
            "historical_v20_v37_rollouts_reexecuted": 0,
            "terminal_approach_events": gate_events,
            "physics_errors": physics_errors,
            "safety_violations": safety_violations,
        },
    }
    v38.v33.immutable_or_verify(root / "RESULT.json", result)
    v38.v33.immutable_or_verify(
        root / "COMPLETE.json",
        {
            "schema": "act-transport3-boundary-heldout-completion-v38",
            **result["accounting"],
            "banks_sha256": file_sha256(args.banks),
            "controllers_sha256": file_sha256(args.controllers),
            "result_sha256": file_sha256(root / "RESULT.json"),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
