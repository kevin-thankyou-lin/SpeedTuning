#!/usr/bin/env python3
"""Create the hash-bound manifest for the v23 Tabular-20 extension."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import canonical_sha256
from scripts.run_act_speed_benchmark_cell import immutable_json, sha256


V20_COMMIT = "9aa797796ba5dfee7db2a7a4dab73183c04585bd"
TASKS = ("pick", "tea", "insertion")
METHOD = "learned_phase_tabular_rl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--v20-manifest", type=Path, required=True)
    parser.add_argument("--v20-run", type=Path, required=True)
    parser.add_argument("--v22-complete", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if head != args.source_commit:
        raise RuntimeError("source commit mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip():
        raise RuntimeError("manifest requires a clean worktree")
    if not args.v22_complete.is_file():
        raise RuntimeError("v22 must be sealed before v23 is created")
    v22_complete = json.loads(args.v22_complete.read_text())
    if v22_complete.get("controllers") != 6 or v22_complete.get("new_final_rollouts") != 300:
        raise RuntimeError("unexpected v22 completion receipt")
    contract = json.loads(args.contract.read_text())
    v20 = json.loads(args.v20_manifest.read_text())
    if v20.get("source", {}).get("commit") != V20_COMMIT or not v20.get("parity_gate", {}).get("passed"):
        raise RuntimeError("invalid v20 source manifest")

    tasks = {}
    source_cells = {}
    for task in TASKS:
        old = v20["tasks"][task]
        search = list(range(contract["tasks"][task]["search_seed_base"], contract["tasks"][task]["search_seed_base"] + 50))
        final = list(range(contract["tasks"][task]["final_seed_base"], contract["tasks"][task]["final_seed_base"] + 50))
        if old["search_bank"]["seeds"] != search:
            raise RuntimeError(f"v20 search bank mismatch for {task}")
        for name, expected in old["artifacts"].items():
            path = Path(contract["tasks"][task]["root"]) / "checkpoints" / name
            if sha256(path) != expected:
                raise RuntimeError(f"ACT artifact mismatch: {path}")
        source_root = args.v20_run / "cells" / task / METHOD / "search"
        if not (source_root / "COMPLETE.json").is_file():
            raise RuntimeError(f"missing sealed v20 search: {source_root}")
        source_cells[task] = {
            "root": str(source_root),
            "complete_sha256": sha256(source_root / "COMPLETE.json"),
            "identity_sha256": sha256(source_root / "identity.json"),
        }
        tasks[task] = {
            "task": contract["tasks"][task]["task"],
            "policy_root": contract["tasks"][task]["root"],
            "artifacts": old["artifacts"],
            "search_bank": {"seeds": search, "sha256": canonical_sha256(search)},
            "final_bank": {"seeds": final, "sha256": canonical_sha256(final)},
        }
    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True).splitlines()
    manifest = {
        "schema": "act-tabular-budget20-run-manifest-v23",
        "source": {
            "repository": "https://github.com/kevin-thankyou-lin/SpeedTuning.git",
            "commit": args.source_commit,
            "tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip(),
            "tracked_file_sha256": {name: sha256(REPO_ROOT / name) for name in tracked if (REPO_ROOT / name).is_file()},
        },
        "contract": {"path": str(args.contract), "sha256": sha256(args.contract), "payload": contract},
        "learned_phase_detector": v20["learned_phase_detector"],
        "tasks": tasks,
        "parity_gate": v20["parity_gate"],
        "v20_source": {"commit": V20_COMMIT, "manifest_sha256": sha256(args.v20_manifest), "cells": source_cells},
        "v22_reference": {"completion_path": str(args.v22_complete), "completion_sha256": sha256(args.v22_complete)},
        "provenance": {
            "training_prefix_episodes": 20,
            "training_rollouts_reexecuted": 0,
            "v22_partial_outcomes_available_before_v23_request": True,
            "interpretation": "post_hoc_paired_training_budget_curve",
        },
    }
    if args.output.exists():
        if json.loads(args.output.read_text()) != manifest:
            raise RuntimeError("existing v23 manifest differs")
    else:
        immutable_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
