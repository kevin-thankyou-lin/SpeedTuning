#!/usr/bin/env python3
"""Seal the aggregate exact-25 v32 causal STRIDER search receipt."""

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
    total = physics = safety = 0
    for task in ("pick", "tea", "insertion"):
        root = args.root / task
        identity = root / "IDENTITY.json"
        selection = root / "SELECTION.json"
        completion = json.loads((root / "SEARCH_COMPLETE.json").read_text())
        if completion["identity_sha256"] != file_sha256(identity):
            raise RuntimeError(f"{task} identity hash mismatch")
        if completion["selection_sha256"] != file_sha256(selection):
            raise RuntimeError(f"{task} selection hash mismatch")
        if completion["search_scientific_rollouts"] != 25:
            raise RuntimeError(f"{task} did not use exactly 25 search episodes")
        value = json.loads(selection.read_text())
        tasks[task] = {
            "selection_status": value["selection_status"],
            "selected_schedule": value["selected_schedule"],
            "update_receipts": value["update_receipts"],
            "finalists": value["finalists"],
            "selection_sha256": completion["selection_sha256"],
        }
        total += completion["search_scientific_rollouts"]
        physics += completion["physics_errors"]
        safety += completion["safety_violations"]
    result = {
        "schema": "act-common-grid-strider-causal25-search-aggregate-v32",
        "tasks": tasks,
        "accounting": {
            "search_scientific_rollouts": total,
            "search_rollouts_per_task": 25,
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
            "schema": "act-common-grid-strider-causal25-search-completion-v32",
            "result_sha256": file_sha256(result_path),
            **result["accounting"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
