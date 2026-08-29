#!/usr/bin/env python3
"""Reselect frozen v16 finalists on mathematical representative reset poses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import build_representative_reset_panel as panel_builder
from scripts import run_act_strider_codex_v14_six_schedule as v14
from scripts import run_act_strider_frontier_v4 as v4
from scripts import run_act_strider_vlm_v10 as base


DISCOVERY_CACHE_ROLLOUTS = 24
STAGE_PREFIXES = (4, 8)
MIN_EIGHT_SUCCESS = 7
MIN_THROUGHPUT_GAIN = 0.03


def checked_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed receipt: {path}")
    return json.loads(path.read_text())


def checked_v16_selection(root: Path, task_label: str) -> tuple[dict, dict]:
    identity_path = root / "IDENTITY.json"
    selection_path = root / "SELECTION.json"
    result_path = root / "RESULT.json"
    complete_path = root / "COMPLETE.json"
    identity = checked_json(identity_path)
    selection = checked_json(selection_path)
    result = checked_json(result_path)
    complete = checked_json(complete_path)
    if identity.get("task_label") != task_label or result.get("task_label") != task_label:
        raise RuntimeError("v16 task label mismatch")
    if complete.get("identity_sha256") != v4.file_sha256(identity_path):
        raise RuntimeError("v16 identity hash mismatch")
    if complete.get("selection_sha256") != v4.file_sha256(selection_path):
        raise RuntimeError("v16 selection hash mismatch")
    if complete.get("result_sha256") != v4.file_sha256(result_path):
        raise RuntimeError("v16 result hash mismatch")
    finalists = selection.get("finalists")
    if set(finalists or {}) != {"uniform", "adaptive"}:
        raise RuntimeError("v16 selection does not contain exactly two finalists")
    return identity, selection


def checked_panel(path: Path, task_label: str) -> dict:
    panel = checked_json(path)
    if panel.get("schema") != "act-representative-reset-panel-v1":
        raise RuntimeError("representative panel schema mismatch")
    if panel.get("task_label") != task_label:
        raise RuntimeError("representative panel task mismatch")
    if panel.get("selection_uses_policy_outcomes") is not False:
        raise RuntimeError("representative panel is not outcome blind")
    if panel.get("stage_prefix_sizes") != [4, 8]:
        raise RuntimeError("representative panel stage prefixes mismatch")
    ids = list(map(int, panel.get("panel_ids", ())))
    poses = panel.get("object_pose_vectors", ())
    if len(ids) != 8 or len(set(ids)) != 8 or len(poses) != 8:
        raise RuntimeError("representative panel must contain eight unique poses")
    if panel.get("object_pose_vectors_sha256") != panel_builder.canonical_sha256(poses):
        raise RuntimeError("representative pose hash mismatch")
    return panel


def incident_count(summary: dict) -> int:
    return int(summary["safety_violations"]) + int(summary["physics_errors"])


def gate_decision(stage: int, summaries: dict[str, dict]) -> str:
    uniform = summaries["uniform"]
    adaptive = summaries["adaptive"]
    if incident_count(adaptive):
        return "reject_adaptive_incident"
    if stage == 4:
        return "reject_adaptive_futility" if adaptive["successes"] <= 2 else "continue"
    if stage != 8:
        raise ValueError(f"unsupported representative-panel stage: {stage}")

    adaptive_eligible = adaptive["successes"] >= MIN_EIGHT_SUCCESS
    uniform_eligible = (
        uniform["successes"] >= MIN_EIGHT_SUCCESS and incident_count(uniform) == 0
    )
    if not adaptive_eligible:
        return "reject_adaptive_absolute_reliability"
    if not uniform_eligible:
        return "select_adaptive_uniform_ineligible"
    if adaptive["successes"] < uniform["successes"]:
        return "reject_adaptive_paired_reliability_regression"
    if adaptive["achieved_throughput_per_step"] < (
        (1.0 + MIN_THROUGHPUT_GAIN) * uniform["achieved_throughput_per_step"]
    ):
        return "reject_adaptive_throughput"
    return "select_adaptive"


def selected_name(
    decision: str,
    summaries: dict[str, dict],
    *,
    stage: int,
    native_deployment_fallback: bool,
) -> str:
    if decision.startswith("select_adaptive"):
        return "adaptive"
    if stage == 4:
        return "native" if native_deployment_fallback else "uniform"
    uniform = summaries["uniform"]
    if uniform["successes"] >= MIN_EIGHT_SUCCESS and incident_count(uniform) == 0:
        return "uniform"
    return "native"


def v16_native_deployment_fallback(selection: dict) -> bool:
    """Read the fallback flag across pre- and post-e55def2 v16 receipts."""

    if "native_deployment_fallback" in selection:
        return bool(selection["native_deployment_fallback"])
    selected = selection.get("selected_name")
    if selected not in {"native", "uniform", "adaptive"}:
        raise RuntimeError("legacy v16 selection lacks a recognized selected controller")
    return selected == "native"


class RepresentativePoseLedger:
    def __init__(self, runtime, root: Path, panel: dict):
        self.runtime = runtime
        self.root = root
        self.ids = list(map(int, panel["panel_ids"]))
        self.poses = {
            pose_id: list(map(float, pose))
            for pose_id, pose in zip(self.ids, panel["object_pose_vectors"])
        }

    def run_or_load(
        self, name: str, schedule: list[float], pose_id: int
    ) -> tuple[dict, bool]:
        schedule_hash = v4.schedule_sha256(schedule)
        base_root = self.root / "controllers" / schedule_hash
        state_path = base_root / "states" / f"{pose_id}.json"
        video_path = base_root / "videos" / f"{pose_id}.mp4"
        pose = self.poses[pose_id]
        pose_hash = panel_builder.canonical_sha256(pose)
        if state_path.exists():
            record = base.checked_video_record(state_path, video_path, schedule, pose_id)
            if record.get("representative_pose_sha256") != pose_hash:
                raise RuntimeError(f"cached representative pose mismatch: {state_path}")
            return record, False
        if video_path.exists():
            raise RuntimeError(f"unreceipted representative video: {video_path}")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        record = self.runtime.rollout(
            schedule,
            pose_id,
            object_pose=pose,
            video_path=video_path,
            record_attribution_telemetry=True,
        )
        if list(map(float, record.get("schedule", ()))) != schedule:
            raise RuntimeError("runtime returned a different schedule")
        record = {
            **record,
            "representative_panel_name": name,
            "representative_pose": pose,
            "representative_pose_sha256": pose_hash,
        }
        if base.simulator_valid(record):
            if not video_path.is_file() or video_path.stat().st_size <= 0:
                raise RuntimeError("simulator-valid representative rollout lacks video")
            record["video_sha256"] = v4.file_sha256(video_path)
            record["video_bytes"] = video_path.stat().st_size
        else:
            record["simulator_invalid"] = True
        v4.write_json(state_path, record)
        return record, True


def run_confirmation(runtime, root: Path, selection: dict, panel: dict) -> dict:
    schedules = {
        name: list(v4.validate_schedule(selection["finalists"][name]["schedule"]))
        for name in ("uniform", "adaptive")
    }
    ledger = RepresentativePoseLedger(runtime, root / "confirmation", panel)
    records = {name: [] for name in schedules}
    stages = []
    physical_attempts = 0
    decision = None
    for target in STAGE_PREFIXES:
        for pose_id in ledger.ids[len(records["uniform"]):target]:
            for name, schedule in schedules.items():
                record, ran = ledger.run_or_load(name, schedule, pose_id)
                physical_attempts += int(ran)
                records[name].append(record)
        summaries = {name: v4.summarize(value) for name, value in records.items()}
        decision = gate_decision(target, summaries)
        stage = {
            "target_representative_poses_per_finalist": target,
            "panel_ids": ledger.ids[:target],
            "summaries": summaries,
            "decision": decision,
        }
        stages.append(stage)
        v4.write_json(root / "confirmation" / f"GATE_{target}.json", stage)
        if decision != "continue":
            break
    if decision is None or decision == "continue":
        raise RuntimeError("representative confirmation lacked a terminal decision")
    final_summaries = stages[-1]["summaries"]
    chosen = selected_name(
        decision,
        final_summaries,
        stage=int(stages[-1]["target_representative_poses_per_finalist"]),
        native_deployment_fallback=v16_native_deployment_fallback(selection),
    )
    selected_schedule = [1.0] * 4 if chosen == "native" else schedules[chosen]
    valid_per_finalist = len(records["uniform"])
    return {
        "schema": "act-strider-representative-confirmation-selection-v17",
        "v16_finalists": schedules,
        "stages": stages,
        "decision": decision,
        "selected_name": chosen,
        "selected_schedule": selected_schedule,
        "selected_schedule_sha256": v4.schedule_sha256(selected_schedule),
        "cached_discovery_rollouts": DISCOVERY_CACHE_ROLLOUTS,
        "new_confirmation_scientific_rollouts": 2 * valid_per_finalist,
        "new_confirmation_physical_attempts": physical_attempts,
        "effective_search_rollouts": DISCOVERY_CACHE_ROLLOUTS + 2 * valid_per_finalist,
        "opens_final_bank": False,
    }


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v16-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--success-criterion", type=Path)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    v16_root = args.v16_root.resolve()
    v16_identity, v16_selection = checked_v16_selection(v16_root, args.task_label)
    representative_panel = checked_panel(args.panel, args.task_label)
    runtime, criterion_receipt = v14.build_runtime(args)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": "act-strider-representative-confirmation-identity-v17",
        "method": "strider_v16_finalist_reselection_on_mathematical_representative_panel",
        "source_commit": args.source_commit,
        "contract_sha256": v4.file_sha256(args.contract),
        "panel_sha256": v4.file_sha256(args.panel),
        "v16_root": str(v16_root),
        "v16_identity_sha256": v4.file_sha256(v16_root / "IDENTITY.json"),
        "v16_selection_sha256": v4.file_sha256(v16_root / "SELECTION.json"),
        "v16_source_commit": v16_identity["source_commit"],
        "historical_final_outcomes_available_but_not_read_by_selector": True,
        "diagnostic_not_independent_replication": True,
        "tea_success_criterion": criterion_receipt,
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and checked_json(identity_path) != identity:
        raise RuntimeError("v17 identity mismatch")
    v4.write_json(identity_path, identity)
    selection = run_confirmation(runtime, root, v16_selection, representative_panel)
    selection_path = root / "SELECTION.json"
    v4.write_json(selection_path, selection)
    complete = {
        "schema": "act-strider-representative-confirmation-completion-v17",
        "identity_sha256": v4.file_sha256(identity_path),
        "selection_sha256": v4.file_sha256(selection_path),
        "selected_name": selection["selected_name"],
        "effective_search_rollouts": selection["effective_search_rollouts"],
        "new_confirmation_scientific_rollouts": selection[
            "new_confirmation_scientific_rollouts"
        ],
        "new_confirmation_physical_attempts": selection[
            "new_confirmation_physical_attempts"
        ],
        "opens_final_bank": False,
    }
    v4.write_json(root / "COMPLETE.json", complete)
    print(json.dumps({"identity": identity, "selection": selection, "complete": complete}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
