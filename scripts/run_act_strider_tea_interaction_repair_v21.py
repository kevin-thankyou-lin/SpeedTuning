#!/usr/bin/env python3
"""Test one causal Tea interaction-speed repair on fresh representative poses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import build_representative_reset_panel as panel_base
from scripts import run_act_strider_codex_v14_six_schedule as v14
from scripts import run_act_strider_frontier_v4 as v4
from scripts import run_act_strider_representative_confirmation_v17 as v17


INCUMBENT = [2.0, 2.5, 1.5, 2.0]
REPAIR = [2.0, 2.5, 1.5, 1.5]
STAGE_PREFIXES = (4, 8, 16)


def checked_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed receipt: {path}")
    return json.loads(path.read_text())


def checked_panel(path: Path) -> dict:
    panel = checked_json(path)
    if panel.get("schema") != "act-tea-representative-repair-panel-v21":
        raise RuntimeError("v21 representative panel schema mismatch")
    if panel.get("task_label") != "tea":
        raise RuntimeError("v21 representative panel task mismatch")
    if panel.get("selection_uses_policy_outcomes") is not False:
        raise RuntimeError("v21 representative panel is not outcome blind")
    if panel.get("stage_prefix_sizes") != list(STAGE_PREFIXES):
        raise RuntimeError("v21 representative panel stage prefixes mismatch")
    ids = list(map(int, panel.get("panel_ids", ())))
    poses = panel.get("object_pose_vectors", ())
    if len(ids) != 16 or len(set(ids)) != 16 or len(poses) != 16:
        raise RuntimeError("v21 representative panel must contain sixteen unique poses")
    if panel.get("object_pose_vectors_sha256") != panel_base.canonical_sha256(poses):
        raise RuntimeError("v21 representative pose hash mismatch")
    return panel


def incidents(summary: dict) -> int:
    return int(summary["safety_violations"]) + int(summary["physics_errors"])


def gate_decision(stage: int, summaries: dict[str, dict]) -> str:
    incumbent = summaries["incumbent"]
    repair = summaries["interaction_repair"]
    if incidents(repair):
        return "reject_repair_incident"
    if stage == 4:
        return "reject_repair_futility" if repair["successes"] <= 2 else "continue"
    if stage == 8:
        return "reject_repair_absolute_reliability" if repair["successes"] <= 6 else "continue"
    if stage != 16:
        raise ValueError(f"unsupported v21 stage: {stage}")
    if repair["successes"] < 15:
        return "reject_repair_absolute_reliability"
    if incidents(incumbent):
        return "select_repair_incumbent_incident"
    if repair["successes"] > incumbent["successes"]:
        return "select_repair_reliability_gain"
    if repair["successes"] < incumbent["successes"]:
        return "retain_incumbent_repair_regression"
    return "retain_incumbent_no_demonstrated_reliability_gain"


def run_study(runtime, root: Path, panel: dict) -> dict:
    schedules = {"incumbent": INCUMBENT, "interaction_repair": REPAIR}
    ledger = v17.RepresentativePoseLedger(runtime, root / "paired", panel)
    records = {name: [] for name in schedules}
    stages = []
    physical_attempts = 0
    decision = None
    for target in STAGE_PREFIXES:
        start = len(records["incumbent"])
        for pose_id in ledger.ids[start:target]:
            for name, schedule in schedules.items():
                record, ran = ledger.run_or_load(name, schedule, pose_id)
                physical_attempts += int(ran)
                records[name].append(record)
        summaries = {name: v4.summarize(value) for name, value in records.items()}
        decision = gate_decision(target, summaries)
        gate = {
            "target_representative_poses_per_controller": target,
            "panel_ids": ledger.ids[:target],
            "summaries": summaries,
            "decision": decision,
        }
        stages.append(gate)
        v4.write_json(root / f"GATE_{target}.json", gate)
        if decision != "continue":
            break
    if decision is None or decision == "continue":
        raise RuntimeError("v21 study lacked a terminal gate decision")
    selected = "interaction_repair" if decision.startswith("select_repair") else "incumbent"
    return {
        "schema": "act-strider-tea-interaction-repair-selection-v21",
        "schedules": schedules,
        "stages": stages,
        "decision": decision,
        "selected_name": selected,
        "selected_schedule": schedules[selected],
        "selected_schedule_sha256": v4.schedule_sha256(schedules[selected]),
        "new_scientific_rollouts": 2 * len(records["incumbent"]),
        "new_physical_attempts": physical_attempts,
        "opens_final_bank": False,
        "post_v18_causal_diagnostic": True,
    }


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--success-criterion", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.task_label = "tea"
    panel = checked_panel(args.panel)
    runtime, criterion = v14.build_runtime(args)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": "act-strider-tea-interaction-repair-identity-v21",
        "method": "one_variable_interaction_speed_repair",
        "source_commit": args.source_commit,
        "contract_sha256": v4.file_sha256(args.contract),
        "panel_sha256": v4.file_sha256(args.panel),
        "tea_success_criterion": criterion,
        "frozen_schedules": {"incumbent": INCUMBENT, "interaction_repair": REPAIR},
        "v18_outcomes_used_only_to_formulate_repair": True,
        "v18_final_bank_read_by_selector": False,
        "opens_final_bank": False,
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and checked_json(identity_path) != identity:
        raise RuntimeError("v21 identity mismatch")
    v4.write_json(identity_path, identity)
    selection = run_study(runtime, root, panel)
    selection_path = root / "SELECTION.json"
    v4.write_json(selection_path, selection)
    completion = {
        "schema": "act-strider-tea-interaction-repair-completion-v21",
        "identity_sha256": v4.file_sha256(identity_path),
        "selection_sha256": v4.file_sha256(selection_path),
        "selected_name": selection["selected_name"],
        "new_scientific_rollouts": selection["new_scientific_rollouts"],
        "new_physical_attempts": selection["new_physical_attempts"],
        "opens_final_bank": False,
    }
    v4.write_json(root / "COMPLETE.json", completion)
    print(json.dumps({"identity": identity, "selection": selection, "complete": completion}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
