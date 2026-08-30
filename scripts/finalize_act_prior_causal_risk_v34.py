#!/usr/bin/env python3
"""Seal the v34 exact-budget search and fresh held-out comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import canonical_sha256  # noqa: E402
from scripts import run_act_prior_causal_risk_v34 as v34  # noqa: E402
from scripts.run_act_speed_benchmark_cell import immutable_json  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256  # noqa: E402


def paired(left_states: list[dict], right_states: list[dict]) -> dict:
    if [x["seed"] for x in left_states] != [x["seed"] for x in right_states]:
        raise RuntimeError("v34 paired seed order differs")
    both = [
        (left, right)
        for left, right in zip(left_states, right_states)
        if left["success"] and right["success"]
    ]
    left_steps = [int(x[0]["first_success_step"]) for x in both]
    right_steps = [int(x[1]["first_success_step"]) for x in both]
    return {
        "pairs": len(left_states),
        "both_success": len(both),
        "left_only_success": sum(bool(a["success"]) and not bool(b["success"]) for a, b in zip(left_states, right_states)),
        "right_only_success": sum(not bool(a["success"]) and bool(b["success"]) for a, b in zip(left_states, right_states)),
        "left_speedup_vs_right_on_common_success": (
            (sum(right_steps) / len(right_steps)) / (sum(left_steps) / len(left_steps))
            if both else None
        ),
    }


def load_final(root: Path, task: str, method: str, seeds: list[int]) -> tuple[dict, list[dict]]:
    directory = root / "final" / task / method
    result = v34.v33.checked_json(directory / "RESULT.json")
    complete = v34.v33.checked_json(directory / "COMPLETE.json")
    identity = v34.v33.checked_json(directory / "IDENTITY.json")
    if complete["episodes"] != len(seeds) or complete["result_sha256"] != file_sha256(directory / "RESULT.json"):
        raise RuntimeError(f"invalid v34 completion: {task}/{method}")
    if identity["seed_bank"] != {"seeds": seeds, "sha256": canonical_sha256(seeds)}:
        raise RuntimeError(f"v34 final bank mismatch: {task}/{method}")
    states = [v34.v33.checked_json(directory / "states" / f"{seed}.json") for seed in seeds]
    if any(item["identity_sha256"] != identity["identity_sha256"] for item in states):
        raise RuntimeError(f"v34 final state identity mismatch: {task}/{method}")
    return result, states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--offline-priors", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    banks = v34.v33.checked_json(args.banks)
    v34.all_search_complete(root)
    tasks = {}
    physics_errors = 0
    safety_violations = 0
    for task in v34.TASKS:
        seeds = list(map(int, banks["tasks"][task]["final"]))
        finals = {}
        states = {}
        for method in v34.FINAL_METHODS:
            finals[method], states[method] = load_final(root, task, method, seeds)
            physics_errors += int(finals[method]["summary"]["physics_errors"])
            safety_violations += int(finals[method]["summary"]["safety_violations"])
        tasks[task] = {
            "final": finals,
            "paired": {
                "risk_gated_vs_native_1x": paired(states["risk_gated"], states["native_1x"]),
                "risk_gated_vs_phase_only": paired(states["risk_gated"], states["phase_only"]),
            },
            "selection": v34.v33.checked_json(root / "search" / task / "SELECTION.json"),
        }
    result = {
        "schema": "act-prior-causal-risk-heldout-result-v34",
        "label": "semantic-plus-SAIL-inspired prior, causal repair, observation risk gate",
        "tasks": tasks,
        "accounting": {
            "offline_prior_training_rollouts_reused": 60,
            "offline_prior_rollouts_reexecuted": 0,
            "online_search_rollouts": 75,
            "online_search_rollouts_per_task": 25,
            "heldout_final_rollouts": 450,
            "historical_v20_v33_rollouts_reexecuted": 0,
            "physics_errors": physics_errors,
            "safety_violations": safety_violations,
        },
    }
    immutable_json(root / "RESULT.json", result)
    complete = {
        "schema": "act-prior-causal-risk-heldout-completion-v34",
        **result["accounting"],
        "banks_sha256": file_sha256(args.banks),
        "offline_priors_sha256": file_sha256(args.offline_priors),
        "result_sha256": file_sha256(root / "RESULT.json"),
    }
    immutable_json(root / "COMPLETE.json", complete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
