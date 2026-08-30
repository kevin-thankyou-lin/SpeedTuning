#!/usr/bin/env python3
"""Create the immutable prospective common-grid RL v26 manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import SPEED_VALUES, canonical_sha256, sha256
from scripts.create_act_controller_retrain_v20_manifest import (
    DETECTOR_HASHES,
    EXPECTED_ARTIFACTS,
)
from scripts.run_act_speed_benchmark_cell import immutable_json

TASKS = ("pick", "tea", "insertion")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = REPO_ROOT
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != args.source_commit:
        raise RuntimeError("manifest source commit mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise RuntimeError("manifest requires a clean source worktree")
    contract = json.loads(args.contract.read_text())
    if contract.get("study_schema") != "act-common-grid-rl-contract-v26":
        raise RuntimeError("unexpected v26 contract")
    if tuple(contract.get("action_grid", ())) != SPEED_VALUES:
        raise RuntimeError("contract and executable action grids differ")

    observed_detector = {
        "checkpoint": sha256(args.detector_checkpoint),
        "inference": sha256(args.detector_source / "phase_detector/rgb_inference.py"),
        "model_source": sha256(args.detector_source / "phase_detector/rgb_proprio.py"),
    }
    if observed_detector != DETECTOR_HASHES:
        raise RuntimeError(f"detector hash mismatch: {observed_detector}")

    tasks = {}
    all_banks = []
    for task in TASKS:
        config = contract["tasks"][task]
        checkpoint_root = Path(config["root"]) / "checkpoints"
        artifacts = {name: sha256(checkpoint_root / name) for name in EXPECTED_ARTIFACTS[task]}
        if artifacts != EXPECTED_ARTIFACTS[task]:
            raise RuntimeError(f"ACT artifact mismatch for {task}")
        search = list(range(config["search_seed_base"], config["search_seed_base"] + 25))
        final = list(range(config["final_seed_base"], config["final_seed_base"] + 50))
        all_banks.extend((set(search), set(final)))
        tasks[task] = {
            "task": config["task"],
            "policy_root": config["root"],
            "artifacts": artifacts,
            "search_bank": {"seeds": search, "sha256": canonical_sha256(search)},
            "final_bank": {"seeds": final, "sha256": canonical_sha256(final)},
        }
    for index, bank in enumerate(all_banks):
        if any(bank & other for other in all_banks[index + 1 :]):
            raise RuntimeError("registered banks overlap")

    tracked = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines()
    manifest = {
        "schema": "act-common-grid-rl-run-manifest-v26",
        "source": {
            "repository": "https://github.com/kevin-thankyou-lin/SpeedTuning.git",
            "commit": args.source_commit,
            "tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True).strip(),
            "tracked_file_sha256": {name: sha256(repo / name) for name in tracked if (repo / name).is_file()},
        },
        "contract": {"path": str(args.contract), "sha256": sha256(args.contract), "payload": contract},
        "learned_phase_detector": {"sha256": observed_detector},
        "tasks": tasks,
        "parity_gate": {
            "passed": True,
            "basis": "hash-identical ACT and learned-phase artifacts from sealed v20 staging",
        },
        "provenance": {
            "controller_status": "fresh_25_episode_training_on_common_grid",
            "prior_rollouts_reexecuted": 0,
            "final_bank_opened_at_manifest_creation": False,
        },
    }
    if args.output.exists():
        if json.loads(args.output.read_text()) != manifest:
            raise RuntimeError("existing manifest differs")
    else:
        immutable_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
