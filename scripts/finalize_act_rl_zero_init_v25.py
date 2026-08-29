#!/usr/bin/env python3
"""Seal v25 zero-training RL results and paired comparisons."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import sha256
from scripts.run_act_speed_benchmark_cell import immutable_json

TASKS = ("pick", "tea", "insertion")
METHODS = ("learned_phase_tabular_rl", "learned_phase_rainbow_rl")


def checked(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed receipt: {path}")
    return json.loads(path.read_text())


def load_cell(root: Path, seeds: list[int]) -> tuple[dict, list[dict]]:
    identity = checked(root / "identity.json")
    result = checked(root / "result.json")
    complete = checked(root / "COMPLETE.json")
    if complete.get("episodes") != 50 or complete.get("result_sha256") != sha256(root / "result.json"):
        raise RuntimeError(f"bad completion: {root}")
    if identity.get("seed_bank", {}).get("seeds") != seeds:
        raise RuntimeError(f"seed mismatch: {root}")
    records = [checked(root / "states" / f"{seed}.json") for seed in seeds]
    if any(record.get("seed") != seed or record.get("identity_sha256") != identity["identity_sha256"] for seed, record in zip(seeds, records, strict=True)):
        raise RuntimeError(f"state identity mismatch: {root}")
    return result, records


def metric_step(record: dict) -> int:
    return int(record["physics_steps"] if record.get("first_success_step") is None else record["first_success_step"])


def paired(a: list[dict], b: list[dict]) -> dict:
    pairs = list(zip(a, b, strict=True))
    common = [(x, y) for x, y in pairs if x["success"] and y["success"]]
    ax = [metric_step(x) for x, _ in common]
    bx = [metric_step(y) for _, y in common]
    return {
        "pairs": len(pairs),
        "both_success": len(common),
        "zero_only_success": sum(x["success"] and not y["success"] for x, y in pairs),
        "comparator_only_success": sum(not x["success"] and y["success"] for x, y in pairs),
        "zero_mean_steps": None if not ax else statistics.fmean(ax),
        "comparator_mean_steps": None if not bx else statistics.fmean(bx),
        "zero_speedup_vs_comparator": None if not ax else statistics.fmean(bx) / statistics.fmean(ax),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v22-root", type=Path, required=True)
    parser.add_argument("--v23-root", type=Path, required=True)
    parser.add_argument("--v24-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = checked(args.run_manifest)
    for source in (args.v22_root, args.v23_root, args.v24_root):
        checked(source / "COMPLETE.json")
    tasks = {}
    invalid = incidents = 0
    for task in TASKS:
        seeds = manifest["tasks"][task]["final_bank"]["seeds"]
        strider_result, strider_records = load_cell(args.v24_root / "final" / task / "strider_v17", seeds)
        tab20_result, tab20_records = load_cell(args.v23_root / "final" / task / "learned_phase_tabular_rl", seeds)
        methods = {}
        for method in METHODS:
            result, records = load_cell(args.root / "final" / task / method, seeds)
            invalid += int(result["physics_errors"])
            incidents += int(result["safety_violations"])
            methods[method] = {
                "result": result,
                "paired_vs_strider": paired(records, strider_records),
                "paired_vs_tabular20": paired(records, tab20_records),
            }
        tasks[task] = {"methods": methods, "strider_v24": strider_result, "tabular20_v23": tab20_result}
    result = {
        "schema": "act-rl-zero-init-paired-result-v25",
        "new_training_rollouts": 0,
        "new_final_rollouts": 300,
        "cached_v20_v22_v23_v24_rollouts_reexecuted": 0,
        "tasks": tasks,
        "simulator_invalid_attempts": invalid,
        "safety_incidents": incidents,
        "interpretation": "zero-training initialization control, not learned RL",
    }
    result_path = args.root / "RESULT.json"
    if result_path.exists() and checked(result_path) != result:
        raise RuntimeError("existing result differs")
    if not result_path.exists():
        immutable_json(result_path, result)
    complete = {
        "schema": "act-rl-zero-init-eval-completion-v25",
        "controllers": 6,
        "new_training_rollouts": 0,
        "new_final_rollouts": 300,
        "cached_v20_v22_v23_v24_rollouts_reexecuted": 0,
        "simulator_invalid_attempts": invalid,
        "safety_incidents": incidents,
        "result_sha256": sha256(result_path),
        "run_manifest_sha256": sha256(args.run_manifest),
    }
    complete_path = args.root / "COMPLETE.json"
    if complete_path.exists() and checked(complete_path) != complete:
        raise RuntimeError("existing completion differs")
    if not complete_path.exists():
        immutable_json(complete_path, complete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
