#!/usr/bin/env python3
"""Seal exact-budget SAIL-warm-start search and held-out ACT results."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import canonical_sha256  # noqa: E402
from scripts.run_act_sail_warmstart_v33 import (  # noqa: E402
    FINAL_METHODS,
    SEARCH_METHODS,
    TASKS,
    checked_json,
    immutable_or_verify,
)
from scripts.run_act_strider_frontier_v4 import file_sha256  # noqa: E402


def load_final(root: Path, seeds: list[int]):
    identity = checked_json(root / "IDENTITY.json")
    result = checked_json(root / "RESULT.json")
    complete = checked_json(root / "COMPLETE.json")
    if complete["episodes"] != 50 or complete["result_sha256"] != file_sha256(root / "RESULT.json"):
        raise RuntimeError(f"invalid v33 final completion receipt: {root}")
    if identity["seed_bank"] != {"seeds": seeds, "sha256": canonical_sha256(seeds)}:
        raise RuntimeError(f"v33 final bank mismatch: {root}")
    records = [checked_json(root / "states" / f"{seed}.json") for seed in seeds]
    if any(item["identity_sha256"] != identity["identity_sha256"] for item in records):
        raise RuntimeError(f"v33 final state identity mismatch: {root}")
    return result, records


def step(record: dict) -> int:
    return int(record["physics_steps"] if record.get("first_success_step") is None else record["first_success_step"])


def paired(left: list[dict], right: list[dict]) -> dict:
    pairs = list(zip(left, right, strict=True))
    common = [(a, b) for a, b in pairs if a["success"] and b["success"]]
    left_steps = [step(a) for a, _ in common]
    right_steps = [step(b) for _, b in common]
    return {
        "pairs": len(pairs),
        "both_success": len(common),
        "left_only_success": sum(a["success"] and not b["success"] for a, b in pairs),
        "right_only_success": sum(not a["success"] and b["success"] for a, b in pairs),
        "both_failure": sum(not a["success"] and not b["success"] for a, b in pairs),
        "common_success_left_mean_steps": None if not common else statistics.fmean(left_steps),
        "common_success_right_mean_steps": None if not common else statistics.fmean(right_steps),
        "left_speedup_vs_right": None if not common else statistics.fmean(right_steps) / statistics.fmean(left_steps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v32-banks", type=Path, required=True)
    args = parser.parse_args()
    banks = checked_json(args.v32_banks)
    search_rollouts = final_rollouts = physics = safety = 0
    tasks = {}
    for task in TASKS:
        search = {}
        for method in SEARCH_METHODS:
            cell = args.root / "search" / task / method
            completion = checked_json(cell / "SEARCH_COMPLETE.json")
            selection = checked_json(cell / "SELECTION.json")
            if completion["search_scientific_rollouts"] != 25:
                raise RuntimeError(f"v33 search budget mismatch: {task}/{method}")
            if completion["selection_sha256"] != file_sha256(cell / "SELECTION.json"):
                raise RuntimeError(f"v33 search selection hash mismatch: {task}/{method}")
            search[method] = {
                "selected_schedule": selection["selected_schedule"],
                "selection_sha256": completion["selection_sha256"],
                "offline_prior": selection["offline_prior"],
            }
            search_rollouts += 25
            physics += completion["physics_errors"]
            safety += completion["safety_violations"]
        spec = banks["tasks"][task]["final"]
        seeds = list(range(spec["start"], spec["start"] + spec["count"]))
        finals = {}
        records = {}
        for method in FINAL_METHODS:
            result, method_records = load_final(args.root / "final" / task / method, seeds)
            finals[method] = result
            records[method] = method_records
            final_rollouts += 50
            physics += result["summary"]["physics_errors"]
            safety += result["summary"]["safety_violations"]
        comparisons = {}
        for method in FINAL_METHODS[1:]:
            comparisons[f"{method}_vs_native_1x"] = paired(records[method], records["native_1x"])
        comparisons["sail_causal_vs_strider_v32"] = paired(records["sail_causal"], records["strider_v32"])
        comparisons["sail_tabular_vs_strider_v32"] = paired(records["sail_tabular"], records["strider_v32"])
        comparisons["agent_causal_vs_strider_v32"] = paired(records["agent_causal"], records["strider_v32"])
        comparisons["sail_causal_vs_sail_tabular"] = paired(records["sail_causal"], records["sail_tabular"])
        comparisons["agent_causal_vs_sail_causal"] = paired(records["agent_causal"], records["sail_causal"])
        tasks[task] = {"search": search, "final": finals, "paired": comparisons}
    result = {
        "schema": "act-sail-warmstart-heldout-result-v33",
        "label": "SAIL-inspired prior, not paper-faithful SAIL",
        "tasks": tasks,
        "accounting": {
            "online_search_rollouts": search_rollouts,
            "online_search_rollouts_per_task_per_method": 25,
            "offline_prior_rollouts": 0,
            "heldout_final_rollouts": final_rollouts,
            "v20_v32_rollouts_reexecuted": 0,
            "physics_errors": physics,
            "safety_violations": safety,
        },
    }
    immutable_or_verify(args.root / "RESULT.json", result)
    complete = {
        "schema": "act-sail-warmstart-heldout-completion-v33",
        **result["accounting"],
        "result_sha256": file_sha256(args.root / "RESULT.json"),
        "v32_banks_sha256": file_sha256(args.v32_banks),
    }
    immutable_or_verify(args.root / "COMPLETE.json", complete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
