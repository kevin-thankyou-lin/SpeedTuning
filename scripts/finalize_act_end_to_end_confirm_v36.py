#!/usr/bin/env python3
"""Seal the v36 end-to-end-confirmed schedule comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_act_end_to_end_confirm_v36 as v36  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256  # noqa: E402


def checked_states(root: Path, task: str, method: str, seeds: list[int]) -> tuple[dict, list[dict]]:
    alias = v36.v33.checked_json(root / "final" / task / "methods" / method / "RESULT.json")
    controller = root / "final" / task / "controllers" / alias["controller_sha256"]
    result = v36.v33.checked_json(controller / "RESULT.json")
    complete = v36.v33.checked_json(controller / "COMPLETE.json")
    identity = v36.v33.checked_json(controller / "IDENTITY.json")
    if alias["controller_result_sha256"] != file_sha256(controller / "RESULT.json"):
        raise RuntimeError(f"v36 method alias mismatch: {task}/{method}")
    if int(complete["episodes"]) != len(seeds) or complete["result_sha256"] != file_sha256(controller / "RESULT.json"):
        raise RuntimeError(f"v36 controller completion mismatch: {task}/{method}")
    states = [v36.v33.checked_json(controller / "states" / f"{seed}.json") for seed in seeds]
    if any(item.get("identity_sha256") != identity["identity_sha256"] for item in states):
        raise RuntimeError(f"v36 controller state mismatch: {task}/{method}")
    return result, states


def metric_steps(record: dict) -> int:
    first = record.get("first_success_step")
    return int(record["physics_steps"] if first is None else first)


def paired(left: list[dict], right: list[dict]) -> dict:
    if [item["seed"] for item in left] != [item["seed"] for item in right]:
        raise RuntimeError("v36 paired seed order differs")
    common = [(a, b) for a, b in zip(left, right) if v36.successful(a) and v36.successful(b)]
    return {
        "pairs": len(left),
        "both_success": len(common),
        "left_only_success": sum(v36.successful(a) and not v36.successful(b) for a, b in zip(left, right)),
        "right_only_success": sum(not v36.successful(a) and v36.successful(b) for a, b in zip(left, right)),
        "left_speedup_vs_right_on_common_success": (
            None
            if not common
            else (sum(metric_steps(b) for _, b in common) / len(common))
            / (sum(metric_steps(a) for a, _ in common) / len(common))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--offline-priors", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    banks = v36.v33.checked_json(args.banks)
    v36.require_all_search(root)
    tasks = {}
    physics_errors = 0
    safety_violations = 0
    executed_controllers = 0
    for task in v36.TASKS:
        seeds = list(map(int, banks["tasks"][task]["final"]))
        finals = {}
        states = {}
        hashes = set()
        for method in v36.FINAL_METHODS:
            finals[method], states[method] = checked_states(root, task, method, seeds)
            hashes.add(finals[method]["schedule_sha256"])
        executed_controllers += len(hashes)
        for controller_hash in hashes:
            summary = v36.v33.checked_json(root / "final" / task / "controllers" / controller_hash / "RESULT.json")["summary"]
            physics_errors += int(summary["physics_errors"])
            safety_violations += int(summary["safety_violations"])
        tasks[task] = {
            "search_selection": v36.v33.checked_json(root / "search" / task / "SELECTION.json"),
            "final": finals,
            "paired": {
                "confirmed_vs_native_1x": paired(states["confirmed_phase_schedule"], states["native_1x"]),
            },
        }
    result = {
        "schema": "act-end-to-end-confirm-heldout-result-v36",
        "label": "exact-25 complete-schedule discovery and finalist confirmation",
        "tasks": tasks,
        "accounting": {
            "offline_prior_training_rollouts_reused": 60,
            "offline_prior_rollouts_reexecuted": 0,
            "online_search_rollouts": 75,
            "online_search_rollouts_per_task": 25,
            "heldout_unique_controllers": executed_controllers,
            "heldout_final_rollouts_executed": executed_controllers * 50,
            "heldout_method_cells": len(v36.TASKS) * len(v36.FINAL_METHODS),
            "within_v36_duplicate_final_rollouts": 0,
            "historical_v20_v35_rollouts_reexecuted": 0,
            "physics_errors": physics_errors,
            "safety_violations": safety_violations,
        },
    }
    v36.v33.immutable_or_verify(root / "RESULT.json", result)
    v36.v33.immutable_or_verify(
        root / "COMPLETE.json",
        {
            "schema": "act-end-to-end-confirm-heldout-completion-v36",
            **result["accounting"],
            "banks_sha256": file_sha256(args.banks),
            "offline_priors_sha256": file_sha256(args.offline_priors),
            "result_sha256": file_sha256(root / "RESULT.json"),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
