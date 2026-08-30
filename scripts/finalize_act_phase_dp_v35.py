#!/usr/bin/env python3
"""Seal the v35 exact-25 phase-DP search and held-out comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_act_phase_dp_v35 as v35  # noqa: E402
from scripts.run_act_speed_benchmark_cell import immutable_json  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256  # noqa: E402


def checked_states(root: Path, task: str, method: str, seeds: list[int]) -> tuple[dict, list[dict]]:
    result = v35.v33.checked_json(root / "final" / task / "methods" / method / "RESULT.json")
    controller = root / "final" / task / "controllers" / result["controller_sha256"]
    controller_result = v35.v33.checked_json(controller / "RESULT.json")
    complete = v35.v33.checked_json(controller / "COMPLETE.json")
    identity = v35.v33.checked_json(controller / "IDENTITY.json")
    if result["controller_result_sha256"] != file_sha256(controller / "RESULT.json"):
        raise RuntimeError(f"v35 method alias mismatch: {task}/{method}")
    if complete["episodes"] != len(seeds) or complete["result_sha256"] != file_sha256(controller / "RESULT.json"):
        raise RuntimeError(f"v35 controller completion mismatch: {task}/{method}")
    states = [v35.v33.checked_json(controller / "states" / f"{seed}.json") for seed in seeds]
    if any(item.get("identity_sha256") != identity["identity_sha256"] for item in states):
        raise RuntimeError(f"v35 controller state mismatch: {task}/{method}")
    return controller_result, states


def paired(left: list[dict], right: list[dict]) -> dict:
    if [x["seed"] for x in left] != [x["seed"] for x in right]:
        raise RuntimeError("v35 paired seed order differs")
    common = [(a, b) for a, b in zip(left, right) if v35.successful(a) and v35.successful(b)]
    left_steps = [v35.metric_steps(a) for a, _ in common]
    right_steps = [v35.metric_steps(b) for _, b in common]
    return {
        "pairs": len(left),
        "both_success": len(common),
        "left_only_success": sum(v35.successful(a) and not v35.successful(b) for a, b in zip(left, right)),
        "right_only_success": sum(not v35.successful(a) and v35.successful(b) for a, b in zip(left, right)),
        "left_speedup_vs_right_on_common_success": (
            None
            if not common
            else (sum(right_steps) / len(right_steps)) / (sum(left_steps) / len(left_steps))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--frozen-v34", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    banks = v35.v33.checked_json(args.banks)
    v35.require_all_search(root)
    tasks = {}
    physics_errors = 0
    safety_violations = 0
    unique_controllers = 0
    for task in v35.TASKS:
        seeds = list(map(int, banks["tasks"][task]["final"]))
        finals = {}
        states = {}
        hashes = set()
        for method in v35.FINAL_METHODS:
            finals[method], states[method] = checked_states(root, task, method, seeds)
            hashes.add(finals[method]["schedule_sha256"])
        unique_controllers += len(hashes)
        for controller_hash in hashes:
            summary = v35.v33.checked_json(root / "final" / task / "controllers" / controller_hash / "RESULT.json")["summary"]
            physics_errors += int(summary["physics_errors"])
            safety_violations += int(summary["safety_violations"])
        tasks[task] = {
            "search_selection": v35.v33.checked_json(root / "search" / task / "SELECTION.json"),
            "final": finals,
            "paired": {
                "phase_dp_vs_native_1x": paired(states["phase_dp"], states["native_1x"]),
                "phase_dp_vs_v34_phase_only": paired(states["phase_dp"], states["v34_phase_only"]),
            },
        }
    result = {
        "schema": "act-phase-dp-heldout-result-v35",
        "label": "exact-25 orthogonal phase experiment plus finite-horizon backward induction",
        "tasks": tasks,
        "accounting": {
            "online_search_rollouts": 75,
            "online_search_rollouts_per_task": 25,
            "heldout_unique_controllers": unique_controllers,
            "heldout_final_rollouts_executed": unique_controllers * 50,
            "heldout_method_cells": len(v35.TASKS) * len(v35.FINAL_METHODS),
            "within_v35_duplicate_final_rollouts": 0,
            "historical_v20_v34_rollouts_reexecuted": 0,
            "physics_errors": physics_errors,
            "safety_violations": safety_violations,
        },
    }
    v35.v33.immutable_or_verify(root / "RESULT.json", result)
    v35.v33.immutable_or_verify(
        root / "COMPLETE.json",
        {
            "schema": "act-phase-dp-heldout-completion-v35",
            **result["accounting"],
            "banks_sha256": file_sha256(args.banks),
            "design_sha256": file_sha256(args.design),
            "frozen_v34_sha256": file_sha256(args.frozen_v34),
            "result_sha256": file_sha256(root / "RESULT.json"),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
