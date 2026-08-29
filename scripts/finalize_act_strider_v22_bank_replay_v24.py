#!/usr/bin/env python3
"""Seal paired STRIDER versus low-budget RL results on the v22 bank."""

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
COMPARATORS = {
    "tabular_episode20_v23": ("v23", "learned_phase_tabular_rl"),
    "tabular_episode25_v22": ("v22", "learned_phase_tabular_rl"),
    "rainbow_episode25_v22": ("v22", "learned_phase_rainbow_rl"),
}


def checked_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed receipt: {path}")
    return json.loads(path.read_text())


def load_cell(root: Path, seeds: list[int]) -> tuple[dict, list[dict]]:
    identity_path = root / "identity.json"
    result_path = root / "result.json"
    complete_path = root / "COMPLETE.json"
    identity = checked_json(identity_path)
    result = checked_json(result_path)
    complete = checked_json(complete_path)
    if complete.get("episodes") != 50 or complete.get("result_sha256") != sha256(result_path):
        raise RuntimeError(f"invalid completion receipt: {root}")
    observed = identity.get("seed_bank", {}).get("seeds")
    if observed != seeds:
        raise RuntimeError(f"seed mismatch: {root}")
    records = []
    for seed in seeds:
        record = checked_json(root / "states" / f"{seed}.json")
        if record.get("seed") != seed or record.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError(f"state identity mismatch: {root}/{seed}")
        records.append(record)
    return result, records


def metric_step(record: dict) -> int:
    value = record.get("first_success_step")
    return int(record["physics_steps"] if value is None else value)


def paired(strider: list[dict], comparator: list[dict]) -> dict:
    pairs = list(zip(strider, comparator, strict=True))
    common = [(a, b) for a, b in pairs if a["success"] and b["success"]]
    strider_steps = [metric_step(a) for a, _ in common]
    comparator_steps = [metric_step(b) for _, b in common]
    return {
        "pairs": len(pairs),
        "both_success": len(common),
        "both_failure": sum(not a["success"] and not b["success"] for a, b in pairs),
        "strider_only_success": sum(a["success"] and not b["success"] for a, b in pairs),
        "comparator_only_success": sum(not a["success"] and b["success"] for a, b in pairs),
        "common_success_time_to_success": {
            "pairs": len(common),
            "strider_mean_steps": None if not common else statistics.fmean(strider_steps),
            "comparator_mean_steps": None if not common else statistics.fmean(comparator_steps),
            "strider_speedup_vs_comparator": (
                None
                if not common
                else statistics.fmean(comparator_steps) / statistics.fmean(strider_steps)
            ),
            "strider_fewer_steps": sum(a < b for a, b in zip(strider_steps, comparator_steps)),
            "comparator_fewer_steps": sum(b < a for a, b in zip(strider_steps, comparator_steps)),
            "equal_steps": sum(a == b for a, b in zip(strider_steps, comparator_steps)),
        },
    }


def immutable_or_verify(path: Path, value: dict) -> None:
    if path.exists():
        if checked_json(path) != value:
            raise RuntimeError(f"existing sealed receipt differs: {path}")
    else:
        immutable_json(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v22-root", type=Path, required=True)
    parser.add_argument("--v23-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = checked_json(args.run_manifest)
    if not (args.v22_root / "COMPLETE.json").is_file() or not (args.v23_root / "COMPLETE.json").is_file():
        raise RuntimeError("v22 and v23 must be sealed before v24 finalization")

    tasks = {}
    invalid = 0
    incidents = 0
    for task in TASKS:
        seeds = manifest["tasks"][task]["final_bank"]["seeds"]
        strider_result, strider_records = load_cell(
            args.root / "final" / task / "strider_v17", seeds
        )
        invalid += int(strider_result["summary"]["physics_errors"])
        incidents += int(strider_result["summary"]["safety_violations"])
        comparisons = {}
        for name, (source, method) in COMPARATORS.items():
            source_root = args.v22_root if source == "v22" else args.v23_root
            comparator_result, comparator_records = load_cell(
                source_root / "final" / task / method, seeds
            )
            comparisons[name] = {
                "comparator_result": comparator_result,
                "paired": paired(strider_records, comparator_records),
            }
        tasks[task] = {
            "schedule": manifest["tasks"][task]["strider"]["schedule"],
            "strider": strider_result,
            "comparisons": comparisons,
        }

    result = {
        "schema": "act-strider-v22-bank-paired-result-v24",
        "new_search_rollouts": 0,
        "new_training_rollouts": 0,
        "new_final_rollouts": 150,
        "cached_v20_v22_v23_rollouts_reexecuted": 0,
        "tasks": tasks,
        "simulator_invalid_attempts": invalid,
        "safety_incidents": incidents,
        "interpretation": "post_hoc_same_seed_frozen_controller_replay",
    }
    result_path = args.root / "RESULT.json"
    immutable_or_verify(result_path, result)
    complete = {
        "schema": "act-strider-v22-bank-replay-completion-v24",
        "controllers": 3,
        "matched_pairs_per_comparator": 150,
        "new_final_rollouts": 150,
        "cached_v20_v22_v23_rollouts_reexecuted": 0,
        "simulator_invalid_attempts": invalid,
        "safety_incidents": incidents,
        "result_sha256": sha256(result_path),
        "run_manifest_sha256": sha256(args.run_manifest),
        "v22_completion_sha256": sha256(args.v22_root / "COMPLETE.json"),
        "v23_completion_sha256": sha256(args.v23_root / "COMPLETE.json"),
    }
    immutable_or_verify(args.root / "COMPLETE.json", complete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
