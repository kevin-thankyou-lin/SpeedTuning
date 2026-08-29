#!/usr/bin/env python3
"""Seal the matched v22 episode-25 Tabular/Rainbow comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import canonical_sha256, sha256
from scripts.run_act_speed_benchmark_cell import immutable_json


TASKS = ("pick", "tea", "insertion")
METHODS = ("learned_phase_tabular_rl", "learned_phase_rainbow_rl")


def immutable_or_verify(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"existing sealed receipt differs: {path}")
    else:
        immutable_json(path, value)


def load_cell(root: Path, seeds: list[int]) -> tuple[dict, list[dict]]:
    complete_path = root / "COMPLETE.json"
    result_path = root / "result.json"
    identity_path = root / "identity.json"
    for path in (complete_path, result_path, identity_path):
        if not path.is_file():
            raise RuntimeError(f"missing final receipt: {path}")
    complete = json.loads(complete_path.read_text())
    result = json.loads(result_path.read_text())
    identity = json.loads(identity_path.read_text())
    if complete.get("episodes") != 50 or not result.get("exact_budget_complete"):
        raise RuntimeError(f"incomplete final cell: {root}")
    if complete.get("result_sha256") != sha256(result_path):
        raise RuntimeError(f"final result hash mismatch: {root}")
    if identity.get("seed_bank", {}).get("seeds") != seeds:
        raise RuntimeError(f"final seed mismatch: {root}")
    records = []
    for seed in seeds:
        path = root / "states" / f"{seed}.json"
        record = json.loads(path.read_text())
        if record.get("seed") != seed or record.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError(f"final state identity mismatch: {path}")
        records.append(record)
    return result, records


def paired(tabular: list[dict], rainbow: list[dict]) -> dict:
    pairs = list(zip(tabular, rainbow, strict=True))
    return {
        "pairs": len(pairs),
        "both_success": sum(a["success"] and b["success"] for a, b in pairs),
        "both_failure": sum(not a["success"] and not b["success"] for a, b in pairs),
        "tabular_only_success": sum(a["success"] and not b["success"] for a, b in pairs),
        "rainbow_only_success": sum(not a["success"] and b["success"] for a, b in pairs),
        "tabular_fewer_physics_steps": sum(a["physics_steps"] < b["physics_steps"] for a, b in pairs),
        "rainbow_fewer_physics_steps": sum(b["physics_steps"] < a["physics_steps"] for a, b in pairs),
        "equal_physics_steps": sum(a["physics_steps"] == b["physics_steps"] for a, b in pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.run_manifest.read_text())
    if not (args.root / "frozen" / "FROZEN_CONTROLLERS_COMPLETE.json").is_file():
        raise RuntimeError("controllers were not all frozen before final evaluation")
    tasks = {}
    total_invalid = 0
    total_incidents = 0
    for task in TASKS:
        seeds = manifest["tasks"][task]["final_bank"]["seeds"]
        values = {}
        records = {}
        for method in METHODS:
            cell = args.root / "final" / task / method
            values[method], records[method] = load_cell(cell, seeds)
            total_invalid += int(values[method]["physics_errors"])
            total_incidents += int(values[method]["safety_violations"])
        tasks[task] = {
            "final_bank_sha256": canonical_sha256(seeds),
            "methods": values,
            "paired": paired(records[METHODS[0]], records[METHODS[1]]),
        }
    result = {
        "schema": "act-controller-budget25-matched-result-v22",
        "training_budget_per_controller": 25,
        "new_training_rollouts": 0,
        "new_final_rollouts": 300,
        "tasks": tasks,
        "simulator_invalid_attempts": total_invalid,
        "safety_incidents": total_incidents,
        "interpretation": "retrospective_reproduction_on_fresh_unopened_final_banks",
    }
    result_path = args.root / "RESULT.json"
    immutable_or_verify(result_path, result)
    complete = {
        "schema": "act-controller-budget25-completion-v22",
        "controllers": 6,
        "matched_final_pairs": 300,
        "new_final_rollouts": 300,
        "simulator_invalid_attempts": total_invalid,
        "safety_incidents": total_incidents,
        "result_sha256": sha256(result_path),
        "run_manifest_sha256": sha256(args.run_manifest),
    }
    immutable_or_verify(args.root / "COMPLETE.json", complete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
