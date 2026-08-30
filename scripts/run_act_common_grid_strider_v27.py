#!/usr/bin/env python3
"""Search and evaluate capped-grid STRIDER on the paired v26 bank."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import COMMON_GRID_SPEED_VALUES, canonical_sha256, sha256
from scripts.act_vlm_frontier_server import ACTFrontierRuntime
from scripts.build_common_grid_strider_panel_v27 import base as panel_base
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json
from scripts.run_act_strider_frontier_v4 import schedule_sha256, summarize, validate_schedule

PHASES = ("approach", "interaction", "transport", "release")
BASELINE = [1.5] * 4
DISCOVERY = [
    BASELINE,
    [2.0] * 4,
    [2.5, 2.0, 2.0, 2.0],
    [2.0, 2.5, 2.0, 2.0],
    [2.0, 2.0, 2.5, 2.0],
    [2.0, 2.0, 2.0, 2.5],
]


def checked(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed input: {path}")
    return json.loads(path.read_text())


def immutable_or_equal(path: Path, value: dict) -> None:
    if path.exists():
        if checked(path) != value:
            raise RuntimeError(f"sealed receipt differs: {path}")
    else:
        immutable_json(path, value)


def incidents(summary: dict) -> int:
    return int(summary["physics_errors"]) + int(summary["safety_violations"])


class Ledger:
    def __init__(self, runtime, root: Path, panel: dict, identity: str):
        self.runtime = runtime
        self.root = root
        self.ids = list(map(int, panel["panel_ids"]))
        self.poses = dict(zip(self.ids, panel["object_pose_vectors"]))
        self.identity = identity

    def one(self, schedule: list[float], pose_id: int) -> dict:
        digest = schedule_sha256(schedule)
        cell = self.root / digest
        state = cell / "states" / f"{pose_id}.json"
        video = cell / "videos" / f"{pose_id}.mp4"
        if state.exists():
            value = checked(state)
            if value.get("identity_sha256") != self.identity:
                raise RuntimeError("cached search identity mismatch")
            return value
        video.parent.mkdir(parents=True, exist_ok=True)
        value = self.runtime.rollout(
            schedule, pose_id, object_pose=self.poses[pose_id], video_path=video,
            record_attribution_telemetry=False,
        )
        value["identity_sha256"] = self.identity
        value["representative_pose_sha256"] = panel_base.canonical_sha256(self.poses[pose_id])
        if value.get("physics_error") is None:
            value["video_sha256"] = sha256(video)
        immutable_json(state, value)
        return value


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--v26-root", type=Path, required=True)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if tuple(COMMON_GRID_SPEED_VALUES) != (1.0, 1.5, 2.0, 2.5, 3.0):
        raise RuntimeError("common grid definition changed")
    v26_manifest_path = args.v26_root / "RUN_MANIFEST.json"
    v26_complete_path = args.v26_root / "COMPLETE.json"
    v26_manifest = checked(v26_manifest_path)
    v26_complete = checked(v26_complete_path)
    if v26_complete.get("new_final_rollouts") != 300 or v26_complete.get("simulator_invalid_attempts") != 0:
        raise RuntimeError("v26 aggregate is not cleanly sealed")
    panel = checked(args.panel)
    if panel.get("task_label") != args.task_label or panel.get("selection_uses_policy_outcomes") is not False:
        raise RuntimeError("panel contract mismatch")
    if len(panel.get("panel_ids", ())) != 8:
        raise RuntimeError("panel must contain eight poses")

    runtime = ACTFrontierRuntime(
        source_commit=args.source_commit,
        run_manifest=v26_manifest_path,
        task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
    )
    root = args.root.resolve() / args.task_label
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".lane.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another process owns this STRIDER task") from exc

    final_seeds = list(v26_manifest["tasks"][args.task_label]["final_bank"]["seeds"])
    identity = {
        **runtime.identity(),
        "schema": "act-common-grid-strider-identity-v27",
        "method": "strider_task_independent_one_phase_common_grid_search",
        "task_label": args.task_label,
        "action_grid": list(COMMON_GRID_SPEED_VALUES),
        "candidate_schedules": DISCOVERY,
        "panel_sha256": sha256(args.panel),
        "v26_manifest_sha256": sha256(v26_manifest_path),
        "v26_completion_sha256": sha256(v26_complete_path),
        "final_bank": {"seeds": final_seeds, "sha256": canonical_sha256(final_seeds)},
        "selection_reads_v26_final_outcomes": False,
        "post_v26_paired_extension": True,
        "search_budget": 32,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    identity_path = root / "IDENTITY.json"
    immutable_or_equal(identity_path, identity)
    complete_path = root / "COMPLETE.json"
    if complete_path.exists():
        complete = checked(complete_path)
        if complete.get("identity_sha256") != sha256(identity_path):
            raise RuntimeError("completed STRIDER identity hash mismatch")
        if complete.get("result_sha256") != sha256(root / "RESULT.json"):
            raise RuntimeError("completed STRIDER result hash mismatch")
        print(json.dumps(checked(root / "RESULT.json"), sort_keys=True))
        return 0

    ledger = Ledger(runtime, root / "search", panel, identity["identity_sha256"])
    discovery = {}
    for schedule in DISCOVERY:
        records = [ledger.one(schedule, pose_id) for pose_id in ledger.ids[:4]]
        discovery[schedule_sha256(schedule)] = {"schedule": schedule, "summary": summarize(records)}
    baseline = discovery[schedule_sha256(BASELINE)]
    adaptive = max(
        (value for key, value in discovery.items() if key != schedule_sha256(BASELINE)),
        key=lambda value: (value["summary"]["successes"], value["summary"]["achieved_throughput_per_step"]),
    )
    combined = {}
    for name, candidate in (("uniform_1_5", baseline), ("adaptive", adaptive)):
        schedule = candidate["schedule"]
        records = [ledger.one(schedule, pose_id) for pose_id in ledger.ids]
        combined[name] = {"schedule": schedule, "summary": summarize(records)}
    uniform_summary = combined["uniform_1_5"]["summary"]
    adaptive_summary = combined["adaptive"]["summary"]
    eligible = (
        adaptive_summary["successes"] >= 7
        and adaptive_summary["successes"] >= uniform_summary["successes"]
        and incidents(adaptive_summary) == 0
        and adaptive_summary["achieved_throughput_per_step"]
        >= 1.03 * uniform_summary["achieved_throughput_per_step"]
    )
    selected_name = "adaptive" if eligible else "uniform_1_5"
    selected_schedule = combined[selected_name]["schedule"]
    selection = {
        "schema": "act-common-grid-strider-selection-v27",
        "discovery": discovery,
        "confirmation": combined,
        "selected_name": selected_name,
        "selected_schedule": selected_schedule,
        "selected_schedule_sha256": schedule_sha256(selected_schedule),
        "search_scientific_rollouts": 32,
        "selection_reads_v26_final_outcomes": False,
    }
    selection_path = root / "SELECTION.json"
    immutable_or_equal(selection_path, selection)

    final_root = root / "final"
    states = final_root / "states"
    records = []
    missing = False
    for seed in final_seeds:
        path = states / f"{seed}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError("non-contiguous STRIDER final states")
        value = checked(path)
        if value.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError("STRIDER final identity mismatch")
        records.append(value)
    for seed in final_seeds[len(records):]:
        value = runtime.rollout(selected_schedule, seed, record_attribution_telemetry=False)
        value["identity_sha256"] = identity["identity_sha256"]
        immutable_json(states / f"{seed}.json", value)
        records.append(value)
        atomic_json(final_root / "progress.json", {
            "completed": len(records), "successes": sum(bool(item["success"]) for item in records),
            "physics_errors": sum(item.get("physics_error") is not None for item in records),
            "safety_violations": sum(item.get("safety_violation") is not None for item in records),
        })
        print(json.dumps({"task": args.task_label, "completed": len(records), "successes": sum(bool(item["success"]) for item in records)}), flush=True)
    result = {
        "schema": "act-common-grid-strider-result-v27",
        "task_label": args.task_label,
        "schedule": selected_schedule,
        "selection_sha256": sha256(selection_path),
        "summary": summarize(records),
        "episodes": 50,
    }
    result_path = root / "RESULT.json"
    immutable_or_equal(result_path, result)
    immutable_or_equal(complete_path, {
        "schema": "act-common-grid-strider-completion-v27",
        "identity_sha256": sha256(identity_path),
        "selection_sha256": sha256(selection_path),
        "result_sha256": sha256(result_path),
        "search_scientific_rollouts": 32,
        "final_scientific_rollouts": 50,
        "physics_errors": result["summary"]["physics_errors"],
        "safety_violations": result["summary"]["safety_violations"],
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
