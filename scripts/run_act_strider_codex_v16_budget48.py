#!/usr/bin/env python3
"""Confirm fresh six-schedule STRIDER discovery under a 48-rollout cap."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import run_act_strider_codex_v13_diverse_four_reset as v13
from scripts import run_act_strider_codex_v14_six_schedule as v14
from scripts import run_act_strider_frontier_v4 as v4
from scripts import run_act_strider_vlm_v10 as base


DISCOVERY_ROLLOUTS = 24
CONFIRMATION_STAGES = ((8, 7), (16, 15))
MAX_SEARCH_ROLLOUTS = 48
MIN_THROUGHPUT_GAIN = 0.03


def checked_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed receipt: {path}")
    return json.loads(path.read_text())


def checked_discovery(
    *, task_label: str, v13_root: Path, v14_root: Path, banks_path: Path
) -> tuple[dict, dict, list[tuple[dict, Path]], dict]:
    v13_selection, parent_reports, _ = v14.checked_parent(v13_root, banks_path)
    v14_identity_path = v14_root / "IDENTITY.json"
    v14_selection_path = v14_root / "SELECTION.json"
    v14_complete_path = v14_root / "SEARCH_COMPLETE.json"
    identity = checked_json(v14_identity_path)
    selection = checked_json(v14_selection_path)
    complete = checked_json(v14_complete_path)
    if complete.get("identity_sha256") != v4.file_sha256(v14_identity_path):
        raise RuntimeError("v14 identity hash mismatch")
    if complete.get("selection_sha256") != v4.file_sha256(v14_selection_path):
        raise RuntimeError("v14 selection hash mismatch")
    if complete.get("final_bank_opened") is not False:
        raise RuntimeError("discovery final bank was opened")
    if identity.get("parent_selection_sha256") != v4.file_sha256(
        v13_root / "SELECTION.json"
    ):
        raise RuntimeError("v14 does not descend from supplied v13 search")
    if identity.get("banks_sha256") != v4.file_sha256(banks_path):
        raise RuntimeError("discovery bank hash mismatch")
    if selection.get("task_label") != task_label or v13_selection.get("task_label") != task_label:
        raise RuntimeError("discovery task mismatch")
    if int(selection.get("total_unique_schedules", -1)) != 6:
        raise RuntimeError("discovery did not test exactly six schedules")

    reports: list[tuple[dict, Path]] = []
    for report in parent_reports:
        reports.append(
            (
                report,
                v13_root / "search" / "candidates" / report["schedule_sha256"],
            )
        )
    for report in selection["new_reports"]:
        schedule = list(v4.validate_schedule(report["schedule"]))
        schedule_hash = v4.schedule_sha256(schedule)
        if schedule_hash != report["schedule_sha256"]:
            raise RuntimeError("v14 report schedule hash mismatch")
        reports.append((report, v14_root / "search" / "candidates" / schedule_hash))
    if len(reports) != 6:
        raise RuntimeError("discovery report count mismatch")
    if sum(int(report["summary"]["episodes"]) for report, _ in reports) != DISCOVERY_ROLLOUTS:
        raise RuntimeError("discovery did not consume exactly 24 valid rollouts")
    return v13_selection, selection, reports, identity


def choose_finalists(v13_selection: dict, reports: list[tuple[dict, Path]]) -> dict:
    incumbent = v13_selection.get("uniform_incumbent")
    if incumbent is None:
        uniform = {
            "role": "native_fallback",
            "schedule": [1.0] * 4,
            "schedule_sha256": v4.schedule_sha256([1.0] * 4),
            "qualified": True,
            "summary": None,
        }
        uniform_root = None
    else:
        uniform = incumbent
        uniform_root = next(
            root
            for report, root in reports
            if report["schedule_sha256"] == uniform["schedule_sha256"]
        )

    adaptive = [
        (report, root)
        for report, root in reports
        if report.get("qualified")
        and report.get("role")
        in {"vlm_causal_repair", "telemetry_repair_control", "one_phase_promotion"}
    ]
    if not adaptive:
        raise RuntimeError("six-schedule discovery produced no qualified adaptive finalist")
    adaptive_report, adaptive_root = max(
        adaptive,
        key=lambda item: (
            item[0]["summary"]["achieved_throughput_per_step"],
            item[0]["summary"]["successes"],
        ),
    )
    if adaptive_report["schedule_sha256"] == uniform["schedule_sha256"]:
        raise RuntimeError("adaptive and uniform finalists are identical")
    return {
        "uniform": {"report": uniform, "candidate_root": uniform_root},
        "adaptive": {"report": adaptive_report, "candidate_root": adaptive_root},
    }


def initial_records(finalist: dict, primary_seeds: list[int], runtime) -> list[dict]:
    schedule = list(v4.validate_schedule(finalist["report"]["schedule"]))
    root = finalist["candidate_root"]
    if root is None:
        raise RuntimeError("native fallback confirmation requires discovery receipts")
    records = []
    for seed in primary_seeds[:4]:
        state_path = root / "states" / f"{seed}.json"
        video_path = root / "videos" / f"{seed}.mp4"
        record = base.checked_video_record(state_path, video_path, schedule, seed)
        if not base.simulator_valid(record):
            raise RuntimeError("discovery primary pose was simulator-invalid")
        records.append(record)
    if len(records) != 4:
        raise RuntimeError("finalist lacks four discovery records")
    return records


def confirmation_decision(stage: int, summaries: dict[str, dict]) -> str:
    uniform = summaries["uniform"]
    adaptive = summaries["adaptive"]
    if adaptive["safety_violations"] or adaptive["physics_errors"]:
        return "reject_safety_or_physics"
    minimum = dict(CONFIRMATION_STAGES)[stage]
    if adaptive["successes"] < minimum:
        return "reject_absolute_reliability"
    if adaptive["successes"] < uniform["successes"]:
        return "reject_paired_reliability_regression"
    if stage < CONFIRMATION_STAGES[-1][0]:
        return "continue"
    if adaptive["achieved_throughput_per_step"] < (
        (1.0 + MIN_THROUGHPUT_GAIN) * uniform["achieved_throughput_per_step"]
    ):
        return "reject_throughput"
    return "select_adaptive"


def run_confirmation(
    *, runtime, root: Path, finalists: dict, primary_seeds: list[int], reserve_seeds: list[int]
) -> dict:
    schedules = {
        name: list(v4.validate_schedule(item["report"]["schedule"]))
        for name, item in finalists.items()
    }
    records = {
        name: initial_records(item, primary_seeds, runtime)
        for name, item in finalists.items()
    }
    ledger = base.ValidVideoLedger(runtime, root, [], [])
    extra_pool = primary_seeds[4:] + reserve_seeds
    invalid_pairs = []
    physical_attempts = 0
    cursor = 0
    stages = []
    final_decision = None
    for target, _ in CONFIRMATION_STAGES:
        while len(records["uniform"]) < target:
            if cursor >= len(extra_pool):
                raise RuntimeError("confirmation reserve exhausted")
            seed = extra_pool[cursor]
            cursor += 1
            pair = {}
            errors = []
            for name, schedule in schedules.items():
                schedule_hash = v4.schedule_sha256(schedule)
                record, ran = ledger._run_or_load(
                    root / "confirmation" / "controllers" / schedule_hash,
                    schedule,
                    seed,
                    telemetry_enabled=True,
                )
                physical_attempts += int(ran)
                pair[name] = record
                if not base.simulator_valid(record):
                    errors.append({"name": name, "physics_error": record.get("physics_error")})
            if errors:
                invalid_pairs.append({"seed": seed, "details": errors})
                continue
            for name, record in pair.items():
                records[name].append(record)
        summaries = {name: v4.summarize(value) for name, value in records.items()}
        decision = confirmation_decision(target, summaries)
        receipt = {
            "target_total_valid_poses_per_finalist": target,
            "decision": decision,
            "summaries": summaries,
            "valid_seeds": [int(record["seed"]) for record in records["uniform"]],
            "simulator_invalid_pairs": list(invalid_pairs),
        }
        stages.append(receipt)
        v4.write_json(root / "confirmation" / f"GATE_{target}.json", receipt)
        if decision != "continue":
            final_decision = decision
            break
    if final_decision is None:
        raise RuntimeError("confirmation did not reach a terminal decision")
    selected_name = "adaptive" if final_decision == "select_adaptive" else "uniform"
    additional_valid_rollouts = 2 * (len(records["uniform"]) - 4)
    total_search_rollouts = DISCOVERY_ROLLOUTS + additional_valid_rollouts
    if total_search_rollouts > MAX_SEARCH_ROLLOUTS:
        raise RuntimeError("48-rollout search budget exceeded")
    return {
        "schema": "act-strider-budget48-confirmation-v16",
        "finalists": {
            name: {
                "schedule": schedules[name],
                "schedule_sha256": v4.schedule_sha256(schedules[name]),
                "discovery_role": finalists[name]["report"]["role"],
            }
            for name in finalists
        },
        "stages": stages,
        "decision": final_decision,
        "selected_name": selected_name,
        "selected_schedule": schedules[selected_name],
        "selected_schedule_sha256": v4.schedule_sha256(schedules[selected_name]),
        "discovery_valid_rollouts": DISCOVERY_ROLLOUTS,
        "confirmation_valid_rollouts": additional_valid_rollouts,
        "search_valid_rollouts": total_search_rollouts,
        "confirmation_physical_attempts": physical_attempts,
        "simulator_invalid_pairs": invalid_pairs,
    }


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
    parser.add_argument("--success-criterion", type=Path)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    v13.configure()
    banks = checked_json(args.banks)
    task_banks = banks["tasks"][args.task_label]
    primary = base._range(task_banks["search_primary"])
    reserve = base._range(task_banks["search_reserve"])
    final_pool = base._range(task_banks["final_primary"]) + base._range(
        task_banks["final_reserve"]
    )
    if len(primary) != 16 or len(final_pool) < 70:
        raise RuntimeError("v16 requires 16 search-primary and at least 70 final seeds")
    if len(set(primary + reserve + final_pool)) != len(primary + reserve + final_pool):
        raise RuntimeError("v16 banks overlap")

    v13_root = args.v13_root.resolve()
    v14_root = args.v14_root.resolve()
    v13_selection, _, reports, v14_identity = checked_discovery(
        task_label=args.task_label,
        v13_root=v13_root,
        v14_root=v14_root,
        banks_path=args.banks,
    )
    finalists = choose_finalists(v13_selection, reports)
    runtime, criterion_receipt = v14.build_runtime(args)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": "act-strider-budget48-identity-v16",
        "method": "strider_codex_diverse_six_schedule_budget48",
        "source_commit": args.source_commit,
        "contract_sha256": v4.file_sha256(args.contract),
        "banks_sha256": v4.file_sha256(args.banks),
        "v13_root": str(v13_root),
        "v13_selection_sha256": v4.file_sha256(v13_root / "SELECTION.json"),
        "v14_root": str(v14_root),
        "v14_identity_sha256": v4.file_sha256(v14_root / "IDENTITY.json"),
        "v14_selection_sha256": v4.file_sha256(v14_root / "SELECTION.json"),
        "v14_parent_selection_sha256": v14_identity["parent_selection_sha256"],
        "search_primary_seeds": primary,
        "search_reserve_seeds": reserve,
        "final_seed_pool": final_pool,
        "maximum_search_valid_rollouts": MAX_SEARCH_ROLLOUTS,
        "selection_frozen_before_final": True,
        "tea_success_criterion": criterion_receipt,
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and checked_json(identity_path) != identity:
        raise RuntimeError("v16 identity mismatch")
    v4.write_json(identity_path, identity)

    confirmation = run_confirmation(
        runtime=runtime,
        root=root,
        finalists=finalists,
        primary_seeds=primary,
        reserve_seeds=reserve,
    )
    selection_path = root / "SELECTION.json"
    v4.write_json(selection_path, confirmation)
    selection_hash = v4.file_sha256(selection_path)

    named = {
        "native_1x": [1.0] * 4,
        "uniform_finalist": confirmation["finalists"]["uniform"]["schedule"],
        "adaptive_finalist": confirmation["finalists"]["adaptive"]["schedule"],
        "strider_v16": confirmation["selected_schedule"],
    }
    final_ledger = base.ValidVideoLedger(runtime, root, [], final_pool)
    final = final_ledger.evaluate_final_paired(named)
    if v4.file_sha256(selection_path) != selection_hash:
        raise RuntimeError("v16 selection changed after final bank opened")
    result = {
        "schema": "act-strider-budget48-result-v16",
        "task_label": args.task_label,
        "identity_sha256": v4.file_sha256(identity_path),
        "selection_sha256_before_final": selection_hash,
        "selection": confirmation,
        "final": final,
        "accounting": {
            "search_valid_rollouts": confirmation["search_valid_rollouts"],
            "search_maximum_valid_rollouts": MAX_SEARCH_ROLLOUTS,
            "confirmation_physical_attempts": confirmation["confirmation_physical_attempts"],
            "final_scientific_rollouts": final["scientific_rollouts"],
            "final_physical_attempts": final["new_physical_attempts"],
            "final_bank_opened_only_after_selection": True,
        },
    }
    result_path = root / "RESULT.json"
    v4.write_json(result_path, result)
    v4.write_json(
        root / "COMPLETE.json",
        {
            "schema": "act-strider-budget48-completion-v16",
            "identity_sha256": v4.file_sha256(identity_path),
            "selection_sha256": selection_hash,
            "result_sha256": v4.file_sha256(result_path),
            **result["accounting"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
