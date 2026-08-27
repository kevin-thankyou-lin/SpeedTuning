#!/usr/bin/env python3
"""Run the pre-outcome delayed-release schedule on v9's matched final bank."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from scripts import run_act_strider_frontier_v4 as v4
from scripts import run_act_strider_tea_release_v9 as v9
from scripts import run_act_strider_tea_volume_v5 as tea


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--success-criterion", type=Path, required=True)
    parser.add_argument("--final-seeds", type=v4.comma_ints, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    proposal = json.loads(args.proposal.read_text())
    if proposal.get("schema") != "act-strider-tea-delayed-release-final-proposal-v1":
        raise ValueError("unexpected delayed-release proposal schema")
    if proposal.get("parent_final_outcomes_visible") is not False:
        raise ValueError("proposal must be frozen before parent final outcomes")
    expected_seeds = list(
        range(
            int(proposal["final_seed_start"]),
            int(proposal["final_seed_start"]) + int(proposal["final_seed_count"]),
        )
    )
    if args.final_seeds != expected_seeds or len(set(args.final_seeds)) != 50:
        raise ValueError("final seeds do not match the frozen proposal")
    schedule = list(v4.validate_schedule(proposal["schedule"]))
    if schedule != v9.DELAYED_RELEASE:
        raise ValueError("proposal is not the registered delayed-release schedule")

    parent_selection_path = args.parent_root / "SELECTION.json"
    parent_result_path = args.parent_root / "RESULT.json"
    if v4.file_sha256(parent_selection_path) != proposal["parent_selection_sha256"]:
        raise RuntimeError("parent selection hash mismatch")
    parent_selection = json.loads(parent_selection_path.read_text())
    if parent_selection["selected_schedule"] != v9.UNIFORM:
        raise RuntimeError("parent did not select the registered uniform incumbent")
    if not parent_result_path.is_file():
        raise RuntimeError("parent final bank is not sealed")

    tea.SUCCESS_CRITERION_SCHEMA = "tea-cup-center-success-v1"
    criterion = tea.checked_success_criterion(args.success_criterion)
    from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head

    if git_head() != args.source_commit:
        raise RuntimeError("checked-out source does not match requested commit")
    runtime = ACTFrontierRuntime(
        source_commit=args.source_commit,
        run_manifest=args.run_manifest,
        task_label="tea",
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
        critical_source_overrides={
            "sim_tasks.py": criterion["files"]["sim_tasks.py"]["sha256"]
        },
    )
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": "act-strider-tea-delayed-release-final-identity-v1",
        "method": "pre_outcome_user_informed_matched_final_comparator",
        "proposal_sha256": v4.file_sha256(args.proposal),
        "parent_source_commit": proposal["parent_source_commit"],
        "parent_selection_sha256": proposal["parent_selection_sha256"],
        "parent_final_outcomes_visible_at_proposal": False,
        "schedule": schedule,
        "final_seeds": args.final_seeds,
        "actual_episode_videos": True,
        "same_gpu_controller_concurrency": False,
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("delayed-release final root identity mismatch")
    v4.write_json(identity_path, identity)

    ledger = v9.VideoFinalLedger(
        runtime,
        root,
        [],
        args.final_seeds,
        record_search_telemetry=False,
    )
    delayed, _ = ledger.evaluate_final(schedule)
    parent = json.loads(parent_result_path.read_text())
    native = parent["final"]["methods"]["native_1x"]["summary"]
    uniform = parent["final"]["methods"]["uniform_1p5x"]["summary"]
    summary = delayed["summary"]
    summary["successful_rollout_speedup"] = (
        native["successful_mean_first_success_steps"]
        / summary["successful_mean_first_success_steps"]
    )
    summary["throughput_delta_percent_vs_native"] = 100.0 * (
        summary["achieved_throughput_per_step"]
        / native["achieved_throughput_per_step"]
        - 1.0
    )
    result = {
        "schema": "act-strider-tea-delayed-release-final-result-v1",
        "identity_sha256": v4.file_sha256(identity_path),
        "parent_result_sha256": v4.file_sha256(parent_result_path),
        "schedule": schedule,
        "summary": summary,
        "uniform_1p5x_parent_summary": uniform,
        "paired_success_delta": summary["successes"] - uniform["successes"],
        "paired_throughput_delta": (
            summary["achieved_throughput_per_step"]
            - uniform["achieved_throughput_per_step"]
        ),
        "new_final_rollouts": 50,
        "actual_episode_videos": delayed["actual_episode_videos"],
        "not_used_for_parent_selection": True,
    }
    result_path = root / "RESULT.json"
    v4.write_json(result_path, result)
    v4.write_json(
        root / "COMPLETE.json",
        {
            "schema": "act-strider-tea-delayed-release-final-completion-v1",
            "identity_sha256": v4.file_sha256(identity_path),
            "result_sha256": v4.file_sha256(result_path),
            "new_final_rollouts": 50,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
