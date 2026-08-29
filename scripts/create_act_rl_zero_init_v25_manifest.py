#!/usr/bin/env python3
"""Create the immutable v25 zero-training RL control manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import canonical_sha256, sha256
from scripts.run_act_speed_benchmark_cell import immutable_json

TASKS = ("pick", "tea", "insertion")


def checked(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed input: {path}")
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--v22-root", type=Path, required=True)
    parser.add_argument("--v23-root", type=Path, required=True)
    parser.add_argument("--v24-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if head != args.source_commit:
        raise RuntimeError("source commit mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip():
        raise RuntimeError("manifest requires a clean worktree")
    contract = checked(args.contract)
    if contract.get("study_schema") != "act-rl-zero-init-eval-contract-v25":
        raise RuntimeError("unexpected v25 contract")
    v22_manifest = checked(args.v22_root / "RUN_MANIFEST.json")
    for root in (args.v22_root, args.v23_root, args.v24_root):
        checked(root / "COMPLETE.json")
    if not v22_manifest.get("parity_gate", {}).get("passed"):
        raise RuntimeError("v22 parity gate missing")

    tasks = {}
    for task in TASKS:
        source = v22_manifest["tasks"][task]
        seeds = list(source["final_bank"]["seeds"])
        expected = list(range(contract["tasks"][task]["final_seed_base"], contract["tasks"][task]["final_seed_base"] + 50))
        if seeds != expected or source["final_bank"]["sha256"] != canonical_sha256(seeds):
            raise RuntimeError(f"bank mismatch: {task}")
        tasks[task] = {
            "task": contract["tasks"][task]["task"],
            "policy_root": source["policy_root"],
            "artifacts": source["artifacts"],
            "search_bank": source["search_bank"],
            "final_bank": source["final_bank"],
        }
    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True).splitlines()
    manifest = {
        "schema": "act-rl-zero-init-eval-manifest-v25",
        "source": {
            "repository": "https://github.com/kevin-thankyou-lin/SpeedTuning.git",
            "commit": args.source_commit,
            "tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True).strip(),
            "tracked_file_sha256": {name: sha256(REPO_ROOT / name) for name in tracked if (REPO_ROOT / name).is_file()},
        },
        "contract": {"path": str(args.contract), "sha256": sha256(args.contract), "payload": contract},
        "tasks": tasks,
        "parity_gate": v22_manifest["parity_gate"],
        "learned_phase_detector": v22_manifest["learned_phase_detector"],
        "references": {
            "v22_completion_sha256": sha256(args.v22_root / "COMPLETE.json"),
            "v23_completion_sha256": sha256(args.v23_root / "COMPLETE.json"),
            "v24_completion_sha256": sha256(args.v24_root / "COMPLETE.json"),
        },
        "provenance": {
            "training_rollouts": 0,
            "all_controllers_frozen_before_final_bank": True,
            "cached_v20_v22_v23_v24_rollouts_reexecuted": 0,
            "rainbow_is_new_seed_fixed_initialization": True,
            "rainbow_is_historical_episode0_reconstruction": False,
        },
    }
    if args.output.exists():
        if checked(args.output) != manifest:
            raise RuntimeError("existing manifest differs")
    else:
        immutable_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
