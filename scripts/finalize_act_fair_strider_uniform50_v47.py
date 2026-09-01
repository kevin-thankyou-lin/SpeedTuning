#!/usr/bin/env python3
"""Seal the V47 fair STRIDER-versus-uniform aggregate receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_act_fair_strider_uniform50_v47 as v47
from scripts.run_act_strider_frontier_v4 import file_sha256


def checked_method(root: Path, task: str, method: str, seeds: list[int]):
    method_root = root / "final" / task / "methods" / method
    alias = v47.v39.v38.v33.checked_json(method_root / "RESULT.json")
    controller_root = (
        root / "final" / task / "controllers" / alias["controller_sha256"]
    )
    result = v47.v39.v38.v33.checked_json(controller_root / "RESULT.json")
    complete = v47.v39.v38.v33.checked_json(controller_root / "COMPLETE.json")
    if int(result["episodes"]) != v47.FINAL_EPISODES:
        raise RuntimeError(f"v47 final count mismatch: {task}/{method}")
    if complete["result_sha256"] != file_sha256(controller_root / "RESULT.json"):
        raise RuntimeError(f"v47 final hash mismatch: {task}/{method}")
    records = [
        v47.v39.v38.v33.checked_json(controller_root / "states" / f"{seed}.json")
        for seed in seeds
    ]
    if [int(record["seed"]) for record in records] != seeds:
        raise RuntimeError(f"v47 final seed order mismatch: {task}/{method}")
    return alias, result, records


def method_comparison(uniform_records: list[dict], strider_records: list[dict]):
    return v47.paired_receipt(strider_records, uniform_records)


def native_comparison(native: dict, candidate: dict) -> dict:
    native_summary = native["summary"]
    candidate_summary = candidate["summary"]
    native_steps = native_summary["successful_mean_first_success_steps"]
    candidate_steps = candidate_summary["successful_mean_first_success_steps"]
    native_throughput = native_summary["achieved_throughput_per_step"]
    candidate_throughput = candidate_summary["achieved_throughput_per_step"]
    return {
        "candidate_successes": candidate_summary["successes"],
        "native_successes": native_summary["successes"],
        "successful_rollout_speedup_vs_native": (
            None
            if native_steps is None or candidate_steps in (None, 0)
            else native_steps / candidate_steps
        ),
        "failure_aware_throughput_ratio_vs_native": (
            None
            if native_throughput <= 0
            else candidate_throughput / native_throughput
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    banks = v47.v39.v38.v33.checked_json(args.banks)
    v47.validate_banks(banks)
    v47.require_all_search(root)

    tasks = {}
    search_physics = search_safety = 0
    final_physics = final_safety = 0
    unique_final_controllers = set()
    for task in v47.TASKS:
        spec = v47.expand_task_banks(banks["tasks"][task])
        searches = {
            arm: v47.v39.v38.v33.checked_json(
                root / "search" / task / arm / "SELECTION.json"
            )
            for arm in v47.SEARCH_ARMS
        }
        for search in searches.values():
            search_physics += int(search["incident_totals"]["physics_errors"])
            search_safety += int(search["incident_totals"]["safety_violations"])
        aliases = {}
        finals = {}
        records = {}
        for method in v47.FINAL_METHODS:
            aliases[method], finals[method], records[method] = checked_method(
                root, task, method, spec["final"]
            )
            unique_final_controllers.add((task, aliases[method]["controller_sha256"]))
        for digest in {aliases[m]["controller_sha256"] for m in v47.FINAL_METHODS}:
            result = v47.v39.v38.v33.checked_json(
                root / "final" / task / "controllers" / digest / "RESULT.json"
            )
            final_physics += int(result["summary"]["physics_errors"])
            final_safety += int(result["summary"]["safety_violations"])
        tasks[task] = {
            "search": searches,
            "final": aliases,
            "paired_strider_vs_uniform": method_comparison(
                records["uniform_selected"], records["strider_selected"]
            ),
            "uniform_vs_native": native_comparison(
                finals["native_1x"], finals["uniform_selected"]
            ),
            "strider_vs_native": native_comparison(
                finals["native_1x"], finals["strider_selected"]
            ),
        }

    result = {
        "schema": "act-fair-strider-uniform50-heldout-result-v47",
        "label": (
            "Fresh symmetric 50-rollout uniform-search versus STRIDER-search "
            "with paired 100-reset final evaluation"
        ),
        "tasks": tasks,
        "accounting": {
            "search_rollouts_per_arm_per_task": v47.SEARCH_BUDGET,
            "uniform_search_rollouts": len(v47.TASKS) * v47.SEARCH_BUDGET,
            "strider_search_rollouts": len(v47.TASKS) * v47.SEARCH_BUDGET,
            "total_search_rollouts": (
                len(v47.TASKS) * len(v47.SEARCH_ARMS) * v47.SEARCH_BUDGET
            ),
            "final_method_evaluations": (
                len(v47.TASKS) * len(v47.FINAL_METHODS) * v47.FINAL_EPISODES
            ),
            "unique_final_controller_rollouts": (
                len(unique_final_controllers) * v47.FINAL_EPISODES
            ),
            "historical_speed_outcomes_used_for_initialization": False,
            "historical_speed_schedules_used_for_initialization": False,
            "historical_rollouts_reexecuted": 0,
            "search_physics_errors": search_physics,
            "search_safety_violations": search_safety,
            "final_physics_errors": final_physics,
            "final_safety_violations": final_safety,
            "physics_errors": search_physics + final_physics,
            "safety_violations": search_safety + final_safety,
        },
    }
    v47.v39.v38.v33.immutable_or_verify(root / "RESULT.json", result)
    v47.v39.v38.v33.immutable_or_verify(
        root / "COMPLETE.json",
        {
            "schema": "act-fair-strider-uniform50-heldout-completion-v47",
            **result["accounting"],
            "banks_sha256": file_sha256(args.banks),
            "result_sha256": file_sha256(root / "RESULT.json"),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
