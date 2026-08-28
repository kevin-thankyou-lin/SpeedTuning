#!/usr/bin/env python3
"""Evaluate frozen STRIDER v13/v14 selections on fresh matched 50-pose banks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import run_act_strider_codex_v13_diverse_four_reset as v13
from scripts import run_act_strider_codex_v14_six_schedule as v14
from scripts import run_act_strider_frontier_v4 as v4
from scripts import run_act_strider_vlm_v10 as base


FINAL_VALID_TARGET = 50


def checked_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed receipt: {path}")
    return json.loads(path.read_text())


def checked_parents(
    *,
    task_label: str,
    v13_root: Path,
    v14_root: Path,
    v13_banks: Path,
) -> tuple[dict, dict, dict]:
    v13_selection, _, _ = v14.checked_parent(v13_root, v13_banks)
    if v13_selection.get("task_label") != task_label:
        raise RuntimeError("v13 parent task mismatch")

    v14_identity_path = v14_root / "IDENTITY.json"
    v14_selection_path = v14_root / "SELECTION.json"
    v14_completion_path = v14_root / "SEARCH_COMPLETE.json"
    v14_identity = checked_json(v14_identity_path)
    v14_selection = checked_json(v14_selection_path)
    v14_completion = checked_json(v14_completion_path)
    if v14_completion.get("selection_sha256") != v4.file_sha256(v14_selection_path):
        raise RuntimeError("v14 selection hash mismatch")
    if v14_completion.get("identity_sha256") != v4.file_sha256(v14_identity_path):
        raise RuntimeError("v14 identity hash mismatch")
    if v14_completion.get("final_bank_opened") is not False:
        raise RuntimeError("v14 parent was not search-only")
    if v14_identity.get("parent_selection_sha256") != v4.file_sha256(
        v13_root / "SELECTION.json"
    ):
        raise RuntimeError("v14 does not descend from the supplied v13 parent")
    if v14_identity.get("banks_sha256") != v4.file_sha256(v13_banks):
        raise RuntimeError("v14 parent bank hash mismatch")
    if v14_selection.get("task_label") != task_label:
        raise RuntimeError("v14 parent task mismatch")
    if int(v14_selection.get("total_unique_schedules", -1)) != 6:
        raise RuntimeError("v14 parent is not the sealed six-schedule search")
    return v13_selection, v14_selection, v14_identity


def named_schedules(v13_selection: dict, v14_selection: dict) -> dict[str, list[float]]:
    incumbent = v13_selection.get("uniform_incumbent")
    uniform = [1.0] * 4 if incumbent is None else incumbent["schedule"]
    schedules = {
        "native_1x": [1.0] * 4,
        "uniform_incumbent": uniform,
        "strider_v13": v13_selection["selected_schedule"],
        "strider_v14": v14_selection["selected_schedule"],
    }
    return {name: list(v4.validate_schedule(value)) for name, value in schedules.items()}


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v13-root", type=Path, required=True)
    parser.add_argument("--v14-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--v13-banks", type=Path, required=True)
    parser.add_argument("--success-criterion", type=Path)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    v13.configure()
    root = args.root.resolve()
    v13_root = args.v13_root.resolve()
    v14_root = args.v14_root.resolve()
    v13_selection, v14_selection, v14_identity = checked_parents(
        task_label=args.task_label,
        v13_root=v13_root,
        v14_root=v14_root,
        v13_banks=args.v13_banks,
    )
    schedules = named_schedules(v13_selection, v14_selection)

    banks = checked_json(args.banks)
    task_banks = banks["tasks"][args.task_label]
    final_pool = base._range(task_banks["final_primary"]) + base._range(
        task_banks["final_reserve"]
    )
    if len(final_pool) < FINAL_VALID_TARGET or len(final_pool) != len(set(final_pool)):
        raise RuntimeError("invalid final seed pool")

    runtime, criterion_receipt = v14.build_runtime(args)
    root.mkdir(parents=True, exist_ok=True)
    schedule_registry = {
        name: {
            "schedule": schedule,
            "schedule_sha256": v4.schedule_sha256(schedule),
        }
        for name, schedule in schedules.items()
    }
    identity = {
        **runtime.identity(),
        "schema": "act-strider-codex-final50-identity-v15",
        "method": "frozen_strider_v13_v14_matched_final50",
        "task_label": args.task_label,
        "source_commit": args.source_commit,
        "contract_sha256": v4.file_sha256(args.contract),
        "banks_sha256": v4.file_sha256(args.banks),
        "v13_banks_sha256": v4.file_sha256(args.v13_banks),
        "v13_root": str(v13_root),
        "v13_selection_sha256": v4.file_sha256(v13_root / "SELECTION.json"),
        "v14_root": str(v14_root),
        "v14_identity_sha256": v4.file_sha256(v14_root / "IDENTITY.json"),
        "v14_selection_sha256": v4.file_sha256(v14_root / "SELECTION.json"),
        "v14_parent_selection_sha256": v14_identity["parent_selection_sha256"],
        "selection_frozen_before_final": True,
        "final_valid_target": FINAL_VALID_TARGET,
        "final_seed_pool": final_pool,
        "named_schedule_registry": schedule_registry,
        "unique_controller_count": len(
            {item["schedule_sha256"] for item in schedule_registry.values()}
        ),
        "tea_success_criterion": criterion_receipt,
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and checked_json(identity_path) != identity:
        raise RuntimeError("final identity mismatch")
    v4.write_json(identity_path, identity)

    parent_hashes_before = {
        "v13_selection_sha256": v4.file_sha256(v13_root / "SELECTION.json"),
        "v14_selection_sha256": v4.file_sha256(v14_root / "SELECTION.json"),
    }
    ledger = base.ValidVideoLedger(runtime, root, [], final_pool)
    final = ledger.evaluate_final_paired(schedules)
    if len(final["valid_pair_seeds"]) != FINAL_VALID_TARGET:
        raise RuntimeError("final result did not seal 50 valid matched pairs")
    parent_hashes_after = {
        "v13_selection_sha256": v4.file_sha256(v13_root / "SELECTION.json"),
        "v14_selection_sha256": v4.file_sha256(v14_root / "SELECTION.json"),
    }
    if parent_hashes_before != parent_hashes_after:
        raise RuntimeError("sealed parent selection changed during final evaluation")

    result = {
        "schema": "act-strider-codex-final50-result-v15",
        "task_label": args.task_label,
        "identity_sha256": v4.file_sha256(identity_path),
        "parent_hashes": parent_hashes_after,
        "named_schedule_registry": schedule_registry,
        "final": final,
        "accounting": {
            "valid_matched_poses": FINAL_VALID_TARGET,
            "unique_controllers": final["unique_controllers_evaluated"],
            "scientific_rollouts": final["scientific_rollouts"],
            "physical_attempts": final["new_physical_attempts"],
            "simulator_invalid_pairs": len(final["simulator_invalid_pairs"]),
            "selection_rollouts_rerun": 0,
        },
    }
    result_path = root / "RESULT.json"
    v4.write_json(result_path, result)
    complete = {
        "schema": "act-strider-codex-final50-completion-v15",
        "identity_sha256": v4.file_sha256(identity_path),
        "result_sha256": v4.file_sha256(result_path),
        "valid_matched_poses": FINAL_VALID_TARGET,
        "scientific_rollouts": final["scientific_rollouts"],
        "parent_hashes_unchanged": True,
    }
    v4.write_json(root / "COMPLETE.json", complete)
    print(json.dumps({"result": result, "completion": complete}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
