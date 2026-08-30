#!/usr/bin/env python3
"""Seal aggregate receipts for the three-task v30 causal search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_act_strider_frontier_v4 import file_sha256, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    tasks = {}
    native = candidate = physics = safety = 0
    for task in ("pick", "tea", "insertion"):
        root = args.root / task
        identity = root / "IDENTITY.json"
        selection = root / "SELECTION.json"
        completion = json.loads((root / "SEARCH_COMPLETE.json").read_text())
        if completion["identity_sha256"] != file_sha256(identity):
            raise RuntimeError(f"{task} identity hash mismatch")
        if completion["selection_sha256"] != file_sha256(selection):
            raise RuntimeError(f"{task} selection hash mismatch")
        if completion["final_bank_opened"] is not False:
            raise RuntimeError(f"{task} final bank was opened")
        value = json.loads(selection.read_text())
        tasks[task] = {
            "selected_schedule": value["selected_schedule"],
            "selected_role": value["selected_role"],
            "chronology": value["chronology"],
            "attribution_receipts": value["attribution_receipts"],
            "native_reference_rollouts": value["native_reference_rollouts"],
            "candidate_rollouts": value["candidate_rollouts"],
            "selection_sha256": completion["selection_sha256"],
        }
        native += int(value["native_reference_rollouts"])
        candidate += int(value["candidate_rollouts"])
        physics += int(completion["physics_errors"])
        safety += int(completion["safety_violations"])
    result = {
        "schema": "act-common-grid-strider-causal-search-aggregate-v30",
        "tasks": tasks,
        "accounting": {
            "native_reference_rollouts": native,
            "candidate_rollouts": candidate,
            "candidate_budget_maximum": 300,
            "physics_errors": physics,
            "safety_violations": safety,
            "prior_rollouts_reexecuted": 0,
            "all_final_banks_opened": False,
        },
    }
    result_path = args.root / "RESULT.json"
    write_json(result_path, result)
    write_json(
        args.root / "COMPLETE.json",
        {
            "schema": "act-common-grid-strider-causal-search-completion-v30",
            "result_sha256": file_sha256(result_path),
            **result["accounting"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
