#!/usr/bin/env python3
"""Create the hash-bound manifest for the v24 STRIDER replay."""

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
from scripts.run_act_strider_frontier_v4 import schedule_sha256, validate_schedule


TASKS = ("pick", "tea", "insertion")


def checked_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed receipt: {path}")
    return json.loads(path.read_text())


def checked_v18_schedule(root: Path, task: str, expected: list[float]) -> dict:
    task_root = root / task
    identity_path = task_root / "IDENTITY.json"
    result_path = task_root / "RESULT.json"
    complete_path = task_root / "COMPLETE.json"
    identity = checked_json(identity_path)
    complete = checked_json(complete_path)
    checked_json(result_path)
    if identity.get("task_label") != task:
        raise RuntimeError(f"v18 task mismatch: {task}")
    if complete.get("identity_sha256") != sha256(identity_path):
        raise RuntimeError(f"v18 identity hash mismatch: {task}")
    if complete.get("result_sha256") != sha256(result_path):
        raise RuntimeError(f"v18 result hash mismatch: {task}")
    if int(complete.get("simulator_invalid_pairs", -1)) != 0:
        raise RuntimeError(f"v18 receipt contains invalid pairs: {task}")
    schedule = list(validate_schedule(identity["frozen_selected_schedule"]))
    if schedule != list(map(float, expected)):
        raise RuntimeError(f"unexpected frozen STRIDER schedule: {task}: {schedule}")
    if schedule_sha256(schedule) != identity["frozen_selected_schedule_sha256"]:
        raise RuntimeError(f"v18 schedule hash mismatch: {task}")
    return {
        "schedule": schedule,
        "schedule_sha256": schedule_sha256(schedule),
        "v17_selection_sha256": identity["v17_selection_sha256"],
        "v18_identity_sha256": sha256(identity_path),
        "v18_result_sha256": sha256(result_path),
        "v18_completion_sha256": sha256(complete_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--v18-receipts", type=Path, required=True)
    parser.add_argument("--v22-manifest", type=Path, required=True)
    parser.add_argument("--v22-complete", type=Path, required=True)
    parser.add_argument("--v23-complete", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if head != args.source_commit:
        raise RuntimeError("source commit mismatch")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ).strip():
        raise RuntimeError("manifest requires a clean worktree")

    contract = checked_json(args.contract)
    if contract.get("schema") != "act-strider-v22-bank-replay-contract-v24":
        raise RuntimeError("unexpected v24 contract schema")
    v22_manifest = checked_json(args.v22_manifest)
    v22_complete = checked_json(args.v22_complete)
    v23_complete = checked_json(args.v23_complete)
    if v22_complete.get("controllers") != 6 or v22_complete.get("new_final_rollouts") != 300:
        raise RuntimeError("v22 is not the sealed six-controller evaluation")
    if v23_complete.get("controllers") != 3 or v23_complete.get("new_final_rollouts") != 150:
        raise RuntimeError("v23 is not the sealed Tabular-20 evaluation")
    if not v22_manifest.get("parity_gate", {}).get("passed"):
        raise RuntimeError("v22 manifest lacks the ACT parity gate")

    tasks = {}
    for task in TASKS:
        task_contract = contract["tasks"][task]
        source = v22_manifest["tasks"][task]
        seeds = list(source["final_bank"]["seeds"])
        expected = list(
            range(task_contract["final_seed_base"], task_contract["final_seed_base"] + 50)
        )
        if seeds != expected or source["final_bank"]["sha256"] != canonical_sha256(seeds):
            raise RuntimeError(f"v22 final-bank mismatch: {task}")
        tasks[task] = {
            "task": task_contract["task"],
            "policy_root": source["policy_root"],
            "artifacts": source["artifacts"],
            "final_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
            "strider": checked_v18_schedule(
                args.v18_receipts, task, task_contract["expected_schedule"]
            ),
        }

    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=REPO_ROOT, text=True
    ).splitlines()
    manifest = {
        "schema": "act-strider-v22-bank-replay-manifest-v24",
        "source": {
            "repository": "https://github.com/kevin-thankyou-lin/SpeedTuning.git",
            "commit": args.source_commit,
            "tree": subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True
            ).strip(),
            "tracked_file_sha256": {
                name: sha256(REPO_ROOT / name)
                for name in tracked
                if (REPO_ROOT / name).is_file()
            },
        },
        "contract": {
            "path": str(args.contract),
            "sha256": sha256(args.contract),
            "payload": contract,
        },
        "tasks": tasks,
        "parity_gate": v22_manifest["parity_gate"],
        "learned_phase_detector": v22_manifest["learned_phase_detector"],
        "references": {
            "v22_manifest_sha256": sha256(args.v22_manifest),
            "v22_completion_sha256": sha256(args.v22_complete),
            "v23_completion_sha256": sha256(args.v23_complete),
        },
        "provenance": {
            "all_strider_schedules_frozen_before_evaluation": True,
            "search_or_tuning_permitted": False,
            "cached_v20_v22_v23_rollouts_reexecuted": 0,
            "interpretation": "post_hoc_same_seed_frozen_controller_replay",
        },
    }
    if args.output.exists():
        if checked_json(args.output) != manifest:
            raise RuntimeError("existing v24 manifest differs")
    else:
        immutable_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
