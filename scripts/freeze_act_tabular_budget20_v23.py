#!/usr/bin/env python3
"""Freeze exact Tabular policies after the first 20 v20 training episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import canonical_sha256, sha256
from scripts.freeze_act_controller_budget25_v22 import checked_prefix, freeze_tabular
from scripts.run_act_speed_benchmark_cell import immutable_json


TASKS = ("pick", "tea", "insertion")
METHOD = "learned_phase_tabular_rl"
EPISODES = 20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--v20-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.run_manifest.read_text())
    completions = []
    for task in TASKS:
        source = args.v20_run / "cells" / task / METHOD / "search"
        destination = args.output_root / task / METHOD
        if (destination / "COMPLETE.json").exists():
            completions.append(json.loads((destination / "COMPLETE.json").read_text()))
            continue
        seeds = manifest["tasks"][task]["search_bank"]["seeds"]
        records, source_identity = checked_prefix(source, seeds, METHOD, task, episodes=EPISODES)
        receipt_hashes = [sha256(source / "states" / f"{seed}.json") for seed in seeds[:EPISODES]]
        identity = {
            "schema": "act-tabular-budget20-frozen-identity-v23",
            "task_label": task,
            "method": METHOD,
            "training_episodes": EPISODES,
            "training_rollouts_reexecuted": 0,
            "source_identity_sha256": source_identity["identity_sha256"],
            "source_prefix_seeds": seeds[:EPISODES],
            "source_prefix_receipt_sha256": receipt_hashes,
            "source_prefix_sha256": canonical_sha256(receipt_hashes),
        }
        identity["identity_sha256"] = canonical_sha256(identity)
        selected_policy, evidence = freeze_tabular(records)
        selected = {
            "schema": "act-speed-selected-method-v1",
            "method": METHOD,
            "task_label": task,
            "identity_sha256": identity["identity_sha256"],
            "terminal_artifact_only": True,
            "selected_policy": selected_policy,
            "episode_20_evidence": evidence,
        }
        destination.mkdir(parents=True, exist_ok=True)
        immutable_json(destination / "identity.json", identity)
        immutable_json(destination / "selected.json", selected)
        complete = {
            "schema": "act-tabular-budget20-frozen-completion-v23",
            "task_label": task,
            "method": METHOD,
            "episodes": EPISODES,
            "training_rollouts_reexecuted": 0,
            "identity_sha256": sha256(destination / "identity.json"),
            "selected_sha256": sha256(destination / "selected.json"),
            "simulator_invalid_attempts_in_prefix": sum("physics_error" in item for item in records),
            "safety_incidents_in_prefix": sum(item.get("safety_violation") is not None for item in records),
        }
        immutable_json(destination / "COMPLETE.json", complete)
        completions.append(complete)
    marker = {
        "schema": "act-tabular-budget20-all-frozen-v23",
        "controllers": 3,
        "training_rollouts_reexecuted": 0,
        "all_frozen_before_final_bank": True,
        "completion_sha256": [sha256(args.output_root / item["task_label"] / METHOD / "COMPLETE.json") for item in completions],
    }
    marker_path = args.output_root / "FROZEN_CONTROLLERS_COMPLETE.json"
    if marker_path.exists():
        if json.loads(marker_path.read_text()) != marker:
            raise RuntimeError("existing freeze marker differs")
    else:
        immutable_json(marker_path, marker)
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
