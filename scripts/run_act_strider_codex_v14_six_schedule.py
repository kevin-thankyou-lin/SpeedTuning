#!/usr/bin/env python3
"""Extend diverse-panel STRIDER search to six unique schedules per task."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from one_reset_phase_schedule import ALLOWED_SPEEDS, PHASES, estimate_phase_workload
from scripts import run_act_strider_codex_v13_diverse_four_reset as v13
from scripts import run_act_strider_frontier_v4 as v4
from scripts import run_act_strider_vlm_v10 as base


TARGET_UNIQUE_SCHEDULES = 6


def promotion_candidates(
    selected_schedule: list[float],
    median_workload: dict[str, float],
    frozen_phases: set[str],
    existing_schedules: set[tuple[float, ...]],
    count: int,
) -> tuple[list[dict], list[dict]]:
    ranked = []
    for index, (phase, speed) in enumerate(zip(PHASES, selected_schedule)):
        speed = float(speed)
        if phase in frozen_phases or speed == max(ALLOWED_SPEEDS):
            continue
        next_speed = ALLOWED_SPEEDS[ALLOWED_SPEEDS.index(speed) + 1]
        schedule = list(map(float, selected_schedule))
        schedule[index] = next_speed
        if tuple(schedule) in existing_schedules:
            continue
        saved = float(median_workload[phase]) * (1.0 / speed - 1.0 / next_speed)
        ranked.append(
            {
                "phase": phase,
                "from_speed": speed,
                "to_speed": next_speed,
                "schedule": schedule,
                "predicted_saved_steps": saved,
            }
        )
    ranked.sort(key=lambda item: (-item["predicted_saved_steps"], PHASES.index(item["phase"])))
    if len(ranked) < count:
        raise RuntimeError(
            f"only {len(ranked)} eligible one-phase promotions for {count} slots"
        )
    return ranked[:count], ranked


def checked_parent(parent: Path, banks_path: Path) -> tuple[dict, list[dict], list[dict]]:
    identity_path = parent / "IDENTITY.json"
    selection_path = parent / "SELECTION.json"
    completion_path = parent / "SEARCH_COMPLETE.json"
    for path in (identity_path, selection_path, completion_path):
        if not path.is_file():
            raise RuntimeError(f"missing sealed parent receipt: {path}")
    identity = json.loads(identity_path.read_text())
    selection = json.loads(selection_path.read_text())
    completion = json.loads(completion_path.read_text())
    if completion.get("selection_sha256") != v4.file_sha256(selection_path):
        raise RuntimeError("parent selection hash mismatch")
    if completion.get("final_bank_opened") is not False:
        raise RuntimeError("parent was not a search-only run")
    if identity.get("banks_sha256") != v4.file_sha256(banks_path):
        raise RuntimeError("parent bank hash mismatch")

    reports = []
    import_receipts = []
    for summary_path in sorted((parent / "search" / "candidates").glob("*/SUMMARY.json")):
        report = json.loads(summary_path.read_text())
        schedule = list(v4.validate_schedule(report["schedule"]))
        schedule_hash = v4.schedule_sha256(schedule)
        if summary_path.parent.name != schedule_hash:
            raise RuntimeError(f"parent schedule directory mismatch: {summary_path}")
        if int(report["summary"]["episodes"]) != 4:
            raise RuntimeError("parent candidate is not a four-reset result")
        state_hashes = {}
        for seed in identity["search_seed_pool"][:4]:
            state_path = summary_path.parent / "states" / f"{seed}.json"
            video_path = summary_path.parent / "videos" / f"{seed}.mp4"
            base.checked_video_record(state_path, video_path, schedule, int(seed))
            state_hashes[str(seed)] = v4.file_sha256(state_path)
        reports.append(report)
        import_receipts.append(
            {
                "schedule": schedule,
                "schedule_sha256": schedule_hash,
                "summary_sha256": v4.file_sha256(summary_path),
                "state_sha256": state_hashes,
            }
        )
    if not reports:
        raise RuntimeError("parent has no candidate reports")
    return selection, reports, import_receipts


def selected_records(parent: Path, selection: dict) -> list[dict]:
    root = (
        parent
        / "search"
        / "candidates"
        / selection["selected_schedule_sha256"]
        / "states"
    )
    records = [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]
    successful = [
        record
        for record in records
        if v4.base.successful(record) and record.get("physics_error") is None
    ]
    if len(successful) != 4:
        raise RuntimeError("selected parent schedule is not successful on all four resets")
    return successful


def frozen_phases(selection: dict) -> set[str]:
    selected = list(map(float, selection["selected_schedule"]))
    rejected = selection.get("rejected_uniform")
    if selection["selected_role"] in {"vlm_causal_repair", "telemetry_repair_control"}:
        if rejected is None:
            raise RuntimeError("repair selection lacks its rejected uniform parent")
        return {
            phase
            for phase, selected_speed, rejected_speed in zip(
                PHASES, selected, rejected["schedule"]
            )
            if selected_speed < float(rejected_speed)
        }
    attribution = selection.get("vlm_attribution") or {}
    phase = attribution.get("selected_phase")
    return set() if phase is None else {str(phase)}


def build_runtime(args):
    overrides = {"sim_tasks.py": v4.file_sha256(Path("sim_tasks.py"))}
    criterion_receipt = None
    if args.task_label == "tea":
        if args.success_criterion is None:
            raise ValueError("Tea requires the frozen center-inside success criterion")
        from scripts import run_act_strider_tea_volume_v5 as tea

        tea.SUCCESS_CRITERION_SCHEMA = "tea-cup-center-success-v1"
        criterion_receipt = tea.checked_success_criterion(args.success_criterion)
        if overrides["sim_tasks.py"] != criterion_receipt["files"]["sim_tasks.py"]["sha256"]:
            raise RuntimeError("Tea success criterion does not match sim_tasks.py")

    from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head

    if git_head() != args.source_commit:
        raise RuntimeError("checked-out source does not match requested commit")
    runtime = ACTFrontierRuntime(
        source_commit=args.source_commit,
        run_manifest=args.run_manifest,
        task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
        critical_source_overrides=overrides,
    )
    return runtime, criterion_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
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

    v13.configure()
    banks = json.loads(args.banks.read_text())
    task_banks = banks["tasks"][args.task_label]
    search_pool = base._range(task_banks["search_primary"]) + base._range(
        task_banks["search_reserve"]
    )
    final_pool = base._range(task_banks["final_primary"]) + base._range(
        task_banks["final_reserve"]
    )
    if set(search_pool) & set(final_pool):
        raise RuntimeError("search and final pools overlap")

    parent = args.parent_root.resolve()
    parent_selection, parent_reports, imports = checked_parent(parent, args.banks)
    if parent_selection["task_label"] != args.task_label:
        raise RuntimeError("parent task mismatch")
    existing_schedules = {tuple(map(float, report["schedule"])) for report in parent_reports}
    missing = TARGET_UNIQUE_SCHEDULES - len(existing_schedules)
    if missing <= 0:
        raise RuntimeError("parent already contains six schedules")

    records = selected_records(parent, parent_selection)
    workloads = [estimate_phase_workload(record) for record in records]
    median_workload = {
        phase: statistics.median(item[phase] for item in workloads) for phase in PHASES
    }
    frozen = frozen_phases(parent_selection)
    proposed, full_ranking = promotion_candidates(
        list(map(float, parent_selection["selected_schedule"])),
        median_workload,
        frozen,
        existing_schedules,
        missing,
    )

    runtime, criterion_receipt = build_runtime(args)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": "act-strider-six-schedule-extension-identity-v14",
        "method": "strider_diverse_four_reset_six_schedule_frontier_extension",
        "source_commit": args.source_commit,
        "contract_sha256": v4.file_sha256(args.contract),
        "banks_sha256": v4.file_sha256(args.banks),
        "parent_root": str(parent),
        "parent_selection_sha256": v4.file_sha256(parent / "SELECTION.json"),
        "target_unique_schedules": TARGET_UNIQUE_SCHEDULES,
        "search_seed_pool": search_pool,
        "final_seed_pool": final_pool,
        "new_candidate_gate": [{"valid_rollouts": 4, "minimum_successes": 4}],
        "final_bank_opened": False,
        "tea_success_criterion": criterion_receipt,
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("extension identity mismatch")
    v4.write_json(identity_path, identity)
    v4.write_json(
        root / "PARENT_IMPORT.json",
        {
            "schema": "act-strider-parent-search-import-v1",
            "parent_selection_sha256": identity["parent_selection_sha256"],
            "cache_hits": sum(report["summary"]["episodes"] for report in parent_reports),
            "candidate_receipts": imports,
        },
    )
    promotion_receipt = {
        "schema": "act-strider-one-phase-promotion-receipt-v1",
        "selected_parent_schedule": parent_selection["selected_schedule"],
        "frozen_phases": sorted(frozen, key=PHASES.index),
        "median_native_equivalent_phase_workload": median_workload,
        "ranking_rule": "predicted saved steps, then registered phase order",
        "full_ranking": full_ranking,
        "proposed": proposed,
    }
    v4.write_json(root / "PROMOTION_RECEIPT.json", promotion_receipt)

    ledger = base.ValidVideoLedger(runtime, root, search_pool, final_pool)
    new_reports = []
    for proposal in proposed:
        report, _ = ledger.evaluate_search(proposal["schedule"], "one_phase_promotion")
        new_reports.append({**report, "promoted_phase": proposal["phase"]})

    parent_selected = next(
        report
        for report in parent_reports
        if report["schedule_sha256"] == parent_selection["selected_schedule_sha256"]
    )
    selectable = [parent_selected] + [
        report
        for report in new_reports
        if report["qualified"]
        and report["summary"]["achieved_throughput_per_step"]
        >= parent_selected["summary"]["achieved_throughput_per_step"]
    ]
    selected = max(
        selectable,
        key=lambda report: report["summary"]["achieved_throughput_per_step"],
    )
    selection = {
        "schema": "act-strider-six-schedule-extension-selection-v14",
        "task_label": args.task_label,
        "parent_unique_schedules": len(existing_schedules),
        "new_unique_schedules": len(new_reports),
        "total_unique_schedules": len(existing_schedules) + len(new_reports),
        "parent_selected_schedule": parent_selection["selected_schedule"],
        "new_reports": new_reports,
        "selected_schedule": selected["schedule"],
        "selected_schedule_sha256": selected["schedule_sha256"],
        "selected_from_extension": selected in new_reports,
        "new_scientific_rollouts": ledger.search_valid_rollouts_used(),
        "cache_hits": sum(report["summary"]["episodes"] for report in parent_reports),
        "final_bank_opened": False,
    }
    selection_path = root / "SELECTION.json"
    v4.write_json(selection_path, selection)
    completion = {
        "schema": "act-strider-six-schedule-extension-completion-v14",
        "identity_sha256": v4.file_sha256(identity_path),
        "selection_sha256": v4.file_sha256(selection_path),
        "new_scientific_rollouts": selection["new_scientific_rollouts"],
        "total_unique_schedules": selection["total_unique_schedules"],
        "final_bank_opened": False,
    }
    v4.write_json(root / "SEARCH_COMPLETE.json", completion)
    print(json.dumps({"selection": selection, "completion": completion}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
