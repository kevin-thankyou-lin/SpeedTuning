#!/usr/bin/env python3
"""Seal v23 and compare episode-20 Tabular with episode-25 on matched seeds."""

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
METHOD = "learned_phase_tabular_rl"


def immutable_or_verify(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"existing sealed receipt differs: {path}")
    else:
        immutable_json(path, value)


def load_cell(root: Path, seeds: list[int]) -> tuple[dict, list[dict]]:
    result_path = root / "result.json"
    complete_path = root / "COMPLETE.json"
    identity_path = root / "identity.json"
    for path in (result_path, complete_path, identity_path):
        if not path.is_file():
            raise RuntimeError(f"missing sealed cell receipt: {path}")
    result = json.loads(result_path.read_text())
    complete = json.loads(complete_path.read_text())
    identity = json.loads(identity_path.read_text())
    if complete.get("episodes") != 50 or complete.get("result_sha256") != sha256(result_path):
        raise RuntimeError(f"invalid cell completion: {root}")
    if identity.get("seed_bank", {}).get("seeds") != seeds:
        raise RuntimeError(f"seed mismatch: {root}")
    records = []
    for seed in seeds:
        record = json.loads((root / "states" / f"{seed}.json").read_text())
        if record.get("seed") != seed or record.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError(f"state identity mismatch: {root}/{seed}")
        records.append(record)
    return result, records


def paired(first: list[dict], second: list[dict]) -> dict:
    pairs = list(zip(first, second, strict=True))
    return {
        "pairs": 50,
        "both_success": sum(a["success"] and b["success"] for a, b in pairs),
        "both_failure": sum(not a["success"] and not b["success"] for a, b in pairs),
        "episode20_only_success": sum(a["success"] and not b["success"] for a, b in pairs),
        "episode25_only_success": sum(not a["success"] and b["success"] for a, b in pairs),
        "episode20_fewer_physics_steps": sum(a["physics_steps"] < b["physics_steps"] for a, b in pairs),
        "episode25_fewer_physics_steps": sum(b["physics_steps"] < a["physics_steps"] for a, b in pairs),
        "equal_physics_steps": sum(a["physics_steps"] == b["physics_steps"] for a, b in pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v22-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.run_manifest.read_text())
    if not (args.root / "frozen/FROZEN_CONTROLLERS_COMPLETE.json").is_file():
        raise RuntimeError("episode-20 controllers are not all frozen")
    if not (args.v22_root / "COMPLETE.json").is_file():
        raise RuntimeError("v22 comparison is not sealed")
    tasks = {}
    invalid = 0
    incidents = 0
    for task in TASKS:
        seeds = manifest["tasks"][task]["final_bank"]["seeds"]
        result20, records20 = load_cell(args.root / "final" / task / METHOD, seeds)
        result25, records25 = load_cell(args.v22_root / "final" / task / METHOD, seeds)
        invalid += int(result20["physics_errors"])
        incidents += int(result20["safety_violations"])
        tasks[task] = {
            "final_bank_sha256": canonical_sha256(seeds),
            "episode20": result20,
            "episode25": result25,
            "paired": paired(records20, records25),
            "success_difference_episode20_minus_25": result20["successes"] - result25["successes"],
            "mean_acceleration_difference_episode20_minus_25": result20["mean_acceleration"] - result25["mean_acceleration"],
        }
    result = {
        "schema": "act-tabular-budget20-paired-result-v23",
        "training_budget": 20,
        "reference_training_budget": 25,
        "new_training_rollouts": 0,
        "new_final_rollouts": 150,
        "tasks": tasks,
        "simulator_invalid_attempts": invalid,
        "safety_incidents": incidents,
        "interpretation": "post_hoc_paired_training_budget_curve",
    }
    result_path = args.root / "RESULT.json"
    immutable_or_verify(result_path, result)
    complete = {
        "schema": "act-tabular-budget20-completion-v23",
        "controllers": 3,
        "matched_final_pairs_against_v22": 150,
        "new_final_rollouts": 150,
        "simulator_invalid_attempts": invalid,
        "safety_incidents": incidents,
        "result_sha256": sha256(result_path),
        "run_manifest_sha256": sha256(args.run_manifest),
        "v22_completion_sha256": sha256(args.v22_root / "COMPLETE.json"),
    }
    immutable_or_verify(args.root / "COMPLETE.json", complete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
