#!/usr/bin/env python3
"""Evaluate frozen v17 STRIDER selections on a fresh matched final-50 bank."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import run_act_strider_codex_v14_six_schedule as v14
from scripts import run_act_strider_frontier_v4 as v4
from scripts import run_act_strider_vlm_v10 as base


FINAL_VALID_TARGET = 50
EXPECTED_CONTROLLERS = ("native_1x", "uniform_1p5", "strider_v17")


def checked_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed receipt: {path}")
    return json.loads(path.read_text())


def checked_v17_selection(root: Path, task_label: str) -> tuple[dict, dict, dict]:
    identity_path = root / "IDENTITY.json"
    selection_path = root / "SELECTION.json"
    complete_path = root / "COMPLETE.json"
    identity = checked_json(identity_path)
    selection = checked_json(selection_path)
    complete = checked_json(complete_path)
    if identity.get("task_label") != task_label:
        raise RuntimeError("v17 task mismatch")
    if complete.get("identity_sha256") != v4.file_sha256(identity_path):
        raise RuntimeError("v17 identity hash mismatch")
    if complete.get("selection_sha256") != v4.file_sha256(selection_path):
        raise RuntimeError("v17 selection hash mismatch")
    if complete.get("opens_final_bank") is not False:
        raise RuntimeError("v17 unexpectedly opened a final bank")
    if selection.get("opens_final_bank") is not False:
        raise RuntimeError("v17 selection does not preserve the unopened-bank contract")
    schedule = list(v4.validate_schedule(selection["selected_schedule"]))
    if v4.schedule_sha256(schedule) != selection.get("selected_schedule_sha256"):
        raise RuntimeError("v17 selected schedule hash mismatch")
    return identity, selection, complete


def checked_final_pool(banks: dict, task_label: str) -> list[int]:
    if banks.get("schema") != "act-strider-representative-final-banks-v18":
        raise RuntimeError("v18 bank schema mismatch")
    if banks.get("controllers") != list(EXPECTED_CONTROLLERS):
        raise RuntimeError("v18 controller registry mismatch")
    task = banks["tasks"][task_label]
    primary = base._range(task["final_primary"])
    reserve = base._range(task["final_reserve"])
    if len(primary) != FINAL_VALID_TARGET or len(reserve) != 20:
        raise RuntimeError("v18 requires 50 primary and 20 reserve seeds")
    pool = primary + reserve
    if len(pool) != len(set(pool)):
        raise RuntimeError("v18 task bank overlaps itself")
    return pool


def evaluate_final(runtime, root: Path, schedule: list[float], final_pool: list[int]) -> dict:
    named = {
        "native_1x": [1.0] * 4,
        "uniform_1p5": [1.5] * 4,
        "strider_v17": schedule,
    }
    if len({tuple(value) for value in named.values()}) != len(EXPECTED_CONTROLLERS):
        raise RuntimeError("v18 requires three unique controllers")
    ledger = base.ValidVideoLedger(runtime, root, [], final_pool)
    result = ledger.evaluate_final_paired(named)
    if result.get("scientific_rollouts") != FINAL_VALID_TARGET * len(named):
        raise RuntimeError("v18 final scientific-rollout count mismatch")
    if result.get("unique_controllers_evaluated") != len(named):
        raise RuntimeError("v18 did not evaluate three unique controllers")
    if len(result.get("valid_pair_seeds", ())) != FINAL_VALID_TARGET:
        raise RuntimeError("v18 final bank lacks 50 valid matched poses")
    return result


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v17-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--success-criterion", type=Path)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    banks = checked_json(args.banks)
    final_pool = checked_final_pool(banks, args.task_label)
    v17_root = args.v17_root.resolve()
    v17_identity, v17_selection, _ = checked_v17_selection(v17_root, args.task_label)
    runtime, criterion_receipt = v14.build_runtime(args)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": "act-strider-representative-final-identity-v18",
        "method": "strider_v17_untouched_matched_final50",
        "source_commit": args.source_commit,
        "contract_sha256": v4.file_sha256(args.contract),
        "banks_sha256": v4.file_sha256(args.banks),
        "task_final_seed_pool": final_pool,
        "v17_root": str(v17_root),
        "v17_identity_sha256": v4.file_sha256(v17_root / "IDENTITY.json"),
        "v17_selection_sha256": v4.file_sha256(v17_root / "SELECTION.json"),
        "v17_complete_sha256": v4.file_sha256(v17_root / "COMPLETE.json"),
        "v17_source_commit": v17_identity["source_commit"],
        "frozen_selected_schedule": v17_selection["selected_schedule"],
        "frozen_selected_schedule_sha256": v17_selection["selected_schedule_sha256"],
        "controllers": list(EXPECTED_CONTROLLERS),
        "selection_frozen_before_final_bank": True,
        "post_v16_design_provenance_retained": True,
        "tea_success_criterion": criterion_receipt,
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and checked_json(identity_path) != identity:
        raise RuntimeError("v18 identity mismatch")
    v4.write_json(identity_path, identity)
    identity_hash = v4.file_sha256(identity_path)

    final = evaluate_final(
        runtime,
        root,
        list(v4.validate_schedule(v17_selection["selected_schedule"])),
        final_pool,
    )
    if v4.file_sha256(identity_path) != identity_hash:
        raise RuntimeError("v18 identity changed after final bank opened")
    result = {
        "schema": "act-strider-representative-final-result-v18",
        "task_label": args.task_label,
        "identity_sha256": identity_hash,
        "v17_selection_sha256": identity["v17_selection_sha256"],
        "final": final,
        "accounting": {
            "final_scientific_rollouts": final["scientific_rollouts"],
            "final_physical_attempts": final["new_physical_attempts"],
            "simulator_invalid_pairs": len(final["simulator_invalid_pairs"]),
            "selection_frozen_before_final_bank": True,
        },
    }
    result_path = root / "RESULT.json"
    v4.write_json(result_path, result)
    complete = {
        "schema": "act-strider-representative-final-completion-v18",
        "identity_sha256": identity_hash,
        "result_sha256": v4.file_sha256(result_path),
        **result["accounting"],
    }
    v4.write_json(root / "COMPLETE.json", complete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

