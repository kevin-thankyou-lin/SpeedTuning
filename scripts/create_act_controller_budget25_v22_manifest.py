#!/usr/bin/env python3
"""Create the v22 manifest from sealed v20 search receipts and fresh banks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_act_speed_benchmark_cell import immutable_json, sha256
from act_speed_benchmark import canonical_sha256


V20_COMMIT = "9aa797796ba5dfee7db2a7a4dab73183c04585bd"
TASKS = ("pick", "tea", "insertion")
METHODS = ("learned_phase_tabular_rl", "learned_phase_rainbow_rl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--v20-manifest", type=Path, required=True)
    parser.add_argument("--v20-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository = REPO_ROOT
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    if head != args.source_commit:
        raise RuntimeError("manifest source commit mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip():
        raise RuntimeError("manifest requires a clean source worktree")
    contract = json.loads(args.contract.read_text())
    source = json.loads(args.v20_manifest.read_text())
    if source.get("source", {}).get("commit") != V20_COMMIT:
        raise RuntimeError("unexpected v20 source commit")
    if not source.get("parity_gate", {}).get("passed"):
        raise RuntimeError("v20 parity gate was not passed")

    tasks = {}
    source_cells = {}
    for task in TASKS:
        old_task = source["tasks"][task]
        expected_search = list(range(contract["tasks"][task]["search_seed_base"], contract["tasks"][task]["search_seed_base"] + 50))
        if old_task["search_bank"]["seeds"] != expected_search:
            raise RuntimeError(f"v20 search bank mismatch for {task}")
        final = list(range(contract["tasks"][task]["final_seed_base"], contract["tasks"][task]["final_seed_base"] + 50))
        for name, expected in old_task["artifacts"].items():
            path = Path(contract["tasks"][task]["root"]) / "checkpoints" / name
            if sha256(path) != expected:
                raise RuntimeError(f"ACT artifact mismatch: {path}")
        tasks[task] = {
            "task": contract["tasks"][task]["task"],
            "policy_root": contract["tasks"][task]["root"],
            "artifacts": old_task["artifacts"],
            "search_bank": {"seeds": expected_search, "sha256": canonical_sha256(expected_search)},
            "final_bank": {"seeds": final, "sha256": canonical_sha256(final)},
        }
        source_cells[task] = {}
        for method in METHODS:
            root = args.v20_run / "cells" / task / method / "search"
            complete = root / "COMPLETE.json"
            if not complete.is_file():
                raise RuntimeError(f"missing completed v20 search cell: {root}")
            source_cells[task][method] = {
                "root": str(root),
                "complete_sha256": sha256(complete),
                "identity_sha256": sha256(root / "identity.json"),
                "preregistration_sha256": sha256(root / "preregistration.json"),
            }

    tracked = subprocess.check_output(["git", "ls-files"], cwd=repository, text=True).splitlines()
    manifest = {
        "schema": "act-controller-budget25-run-manifest-v22",
        "source": {
            "repository": "https://github.com/kevin-thankyou-lin/SpeedTuning.git",
            "commit": args.source_commit,
            "tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repository, text=True).strip(),
            "tracked_file_sha256": {name: sha256(repository / name) for name in tracked if (repository / name).is_file()},
        },
        "contract": {"path": str(args.contract), "sha256": sha256(args.contract), "payload": contract},
        "learned_phase_detector": source["learned_phase_detector"],
        "tasks": tasks,
        "parity_gate": source["parity_gate"],
        "v20_source": {
            "commit": V20_COMMIT,
            "manifest_path": str(args.v20_manifest),
            "manifest_sha256": sha256(args.v20_manifest),
            "cells": source_cells,
        },
        "provenance": {
            "training_rollouts_reexecuted": 0,
            "training_prefix_episodes": 25,
            "controller_status": "episode_25_retrospective_reproduction",
            "v20_outcomes_previously_observed": True,
            "v22_final_outcomes_available_at_freeze": False,
        },
    }
    if args.output.exists():
        if json.loads(args.output.read_text()) != manifest:
            raise RuntimeError("existing v22 manifest differs")
    else:
        immutable_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
