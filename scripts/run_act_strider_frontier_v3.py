#!/usr/bin/env python3
"""Run telemetry-paired, budget-complete STRIDER search and final ACT evaluation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scripts import run_act_strider_frontier_v2 as base


PHASES = base.PHASES
ALLOWED_SPEEDS = base.ALLOWED_SPEEDS
STAGES = base.STAGES
SEARCH_BUDGET = base.SEARCH_BUDGET
UNIFORM_LADDER = base.UNIFORM_LADDER
OBJECT_DIVERGENCE_METERS = 0.03
REWARD_DIVERGENCE_EPSILON = 1e-6

canonical_sha256 = base.canonical_sha256
file_sha256 = base.file_sha256
write_json = base.write_json
comma_ints = base.comma_ints
validate_schedule = base.validate_schedule
schedule_sha256 = base.schedule_sha256
successful = base.successful
summarize = base.summarize
gate_decision = base.gate_decision
phase_workloads = base.phase_workloads
adjacent_speed = base.adjacent_speed
make_backoff = base.make_backoff
make_promotion = base.make_promotion
choose_uniform_incumbent = base.choose_uniform_incumbent
bend_replaces_uniform = base.bend_replaces_uniform
pareto_names = base.pareto_names
RolloutLedger = base.RolloutLedger
run_final = base.run_final


SEMANTIC_FALLBACK = {
    "pick": "grasp_lift",
    "tea": "grasp_lift",
    "insertion": "interaction",
}


def _phase_exit_snapshots(record: dict) -> dict[str, dict]:
    """Return the last observable state inside each detector-defined phase."""

    telemetry = list(record.get("attribution_telemetry", ()))
    decisions = list(record.get("phase_decisions", ()))
    snapshots = {}
    for index, decision in enumerate(decisions):
        phase = str(decision.get("phase"))
        if phase not in PHASES:
            continue
        start = int(decision.get("physics_step", 0))
        end = (
            int(
                decisions[index + 1].get("physics_step", record.get("physics_steps", 0))
            )
            if index + 1 < len(decisions)
            else int(record.get("physics_steps", 0)) + 1
        )
        segment = [
            item
            for item in telemetry
            if start < int(item.get("physics_step", -1)) <= end
        ]
        if segment:
            snapshots[phase] = {
                "physics_step": int(segment[-1]["physics_step"]),
                "policy_time": float(segment[-1]["policy_time"]),
                "task_reward": max(
                    float(item.get("task_reward", 0.0)) for item in segment
                ),
                "object_positions": segment[-1].get("object_positions", []),
            }
    return snapshots


def _max_object_position_delta(left: dict, right: dict) -> float | None:
    left_positions = left.get("object_positions", ())
    right_positions = right.get("object_positions", ())
    if not left_positions or len(left_positions) != len(right_positions):
        return None
    return max(
        math.dist(tuple(map(float, a)), tuple(map(float, b)))
        for a, b in zip(left_positions, right_positions)
    )


def paired_divergence_for_failure(candidate: dict, reference: dict) -> tuple[str, dict]:
    """Locate the earliest physical mismatch against a same-seed success."""

    if int(candidate.get("seed", -1)) != int(reference.get("seed", -2)):
        raise ValueError("paired attribution requires the same seed")
    if successful(candidate) or not successful(reference):
        raise ValueError(
            "paired attribution requires failed candidate and successful reference"
        )
    candidate_snapshots = _phase_exit_snapshots(candidate)
    reference_snapshots = _phase_exit_snapshots(reference)
    for phase in PHASES:
        reference_snapshot = reference_snapshots.get(phase)
        if reference_snapshot is None:
            continue
        candidate_snapshot = candidate_snapshots.get(phase)
        if candidate_snapshot is None:
            return phase, {
                "seed": int(candidate["seed"]),
                "phase": phase,
                "cause": "candidate_did_not_complete_reference_phase",
            }
        if (
            candidate_snapshot["task_reward"] + REWARD_DIVERGENCE_EPSILON
            < reference_snapshot["task_reward"]
        ):
            return phase, {
                "seed": int(candidate["seed"]),
                "phase": phase,
                "cause": "task_reward_progress_lag",
                "candidate_reward": candidate_snapshot["task_reward"],
                "reference_reward": reference_snapshot["task_reward"],
            }
        position_delta = _max_object_position_delta(
            candidate_snapshot, reference_snapshot
        )
        if position_delta is not None and position_delta > OBJECT_DIVERGENCE_METERS:
            return phase, {
                "seed": int(candidate["seed"]),
                "phase": phase,
                "cause": "object_position_left_matched_reference_envelope",
                "object_position_delta_m": position_delta,
                "threshold_m": OBJECT_DIVERGENCE_METERS,
            }
    reached = [
        str(item.get("phase"))
        for item in candidate.get("phase_decisions", ())
        if str(item.get("phase")) in PHASES
    ]
    phase = reached[-1] if reached else PHASES[0]
    return phase, {
        "seed": int(candidate["seed"]),
        "phase": phase,
        "cause": "no_earlier_observable_divergence_terminal_phase_fallback",
    }


def paired_failure_phase(
    candidate_records: list[dict],
    reference_records: list[dict],
    fallback_phase: str,
) -> tuple[str, dict]:
    """Aggregate same-seed counterexamples without using historical outcomes."""

    references = {
        int(record["seed"]): record
        for record in reference_records
        if successful(record)
    }
    evidence = []
    unmatched = []
    for candidate in candidate_records:
        if successful(candidate):
            continue
        seed = int(candidate["seed"])
        reference = references.get(seed)
        if reference is None:
            unmatched.append(seed)
            continue
        if not candidate.get("attribution_telemetry") or not reference.get(
            "attribution_telemetry"
        ):
            unmatched.append(seed)
            continue
        _, item = paired_divergence_for_failure(candidate, reference)
        evidence.append(item)
    if evidence:
        phase = min((item["phase"] for item in evidence), key=PHASES.index)
        return phase, {
            "method": "same_seed_phase_exit_physical_divergence",
            "selected_phase": phase,
            "paired_counterexamples": evidence,
            "unmatched_failed_seeds": unmatched,
            "object_position_threshold_m": OBJECT_DIVERGENCE_METERS,
        }
    return fallback_phase, {
        "method": "preregistered_semantic_fallback",
        "selected_phase": fallback_phase,
        "reason": "no failed candidate had a successful same-seed telemetry reference",
        "unmatched_failed_seeds": unmatched,
    }


def highest_bang_for_buck_phase(
    records: list[dict], schedule: list[float], frozen_phases: set[str]
) -> tuple[str | None, dict]:
    usable = [record for record in records if successful(record)]
    if not usable:
        return None, {"reason": "no successful incumbent telemetry"}
    means = {
        phase: statistics.fmean(phase_workloads(record)[phase] for record in usable)
        for phase in PHASES
    }
    scores = {}
    for phase, speed in zip(PHASES, schedule):
        if phase in frozen_phases or speed == ALLOWED_SPEEDS[-1]:
            continue
        promoted = adjacent_speed(speed, 1)
        scores[phase] = means[phase] * (1.0 / speed - 1.0 / promoted)
    if not scores:
        return None, {"reason": "no unfrozen phase has a higher adjacent speed"}
    phase = max(
        PHASES, key=lambda item: (scores.get(item, float("-inf")), -PHASES.index(item))
    )
    return phase, {
        "method": "preregistered_bang_for_buck",
        "selected_phase": phase,
        "phase_native_equivalent_work": means,
        "predicted_steps_saved": scores,
    }


def _candidate_better(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return bool(candidate["qualified"])
    return bend_replaces_uniform(candidate, incumbent)


def _remaining_full_gate(ledger: RolloutLedger) -> bool:
    return ledger.search_rollouts_used() + STAGES[-1][0] <= SEARCH_BUDGET


def run_search(ledger: RolloutLedger, task_label: str) -> dict:
    chronology = []
    uniform_reports = []
    adaptive_reports = []
    attribution_receipts = []
    records_by_hash = {}

    anchor, anchor_records = ledger.evaluate_search([2.0] * 4, "uniform_anchor")
    uniform_reports.append(anchor)
    chronology.append(anchor["schedule_sha256"])
    records_by_hash[anchor["schedule_sha256"]] = anchor_records
    rejected = None
    rejected_records = None

    if anchor["qualified"]:
        for speed in (2.5, 3.0):
            if not _remaining_full_gate(ledger):
                break
            report, records = ledger.evaluate_search([speed] * 4, "uniform_ladder")
            uniform_reports.append(report)
            chronology.append(report["schedule_sha256"])
            records_by_hash[report["schedule_sha256"]] = records
            if not report["qualified"]:
                rejected, rejected_records = report, records
                break
    else:
        rejected, rejected_records = anchor, anchor_records
        if _remaining_full_gate(ledger):
            fallback, fallback_records = ledger.evaluate_search(
                [1.5] * 4, "uniform_fallback"
            )
            uniform_reports.append(fallback)
            chronology.append(fallback["schedule_sha256"])
            records_by_hash[fallback["schedule_sha256"]] = fallback_records

    uniform_incumbent = choose_uniform_incumbent(uniform_reports)
    search_incumbent = uniform_incumbent
    search_incumbent_records = (
        []
        if search_incumbent is None
        else records_by_hash[search_incumbent["schedule_sha256"]]
    )
    frozen_phases: set[str] = set()

    while _remaining_full_gate(ledger):
        if rejected is not None:
            reference_records = search_incumbent_records
            phase, evidence = paired_failure_phase(
                rejected_records or [],
                reference_records,
                SEMANTIC_FALLBACK[task_label],
            )
            proposed = make_backoff(rejected["schedule"], phase)
            operation = "one_rung_causal_backoff"
        elif search_incumbent is not None:
            phase, evidence = highest_bang_for_buck_phase(
                search_incumbent_records,
                search_incumbent["schedule"],
                frozen_phases,
            )
            if phase is None:
                break
            proposed = make_promotion(search_incumbent["schedule"], phase)
            operation = "one_rung_bang_for_buck_promotion"
        else:
            break

        proposed_hash = schedule_sha256(proposed)
        attempted = set(chronology)
        if proposed_hash in attempted:
            break
        attribution_receipts.append(
            {
                "operation": operation,
                "phase": phase,
                "source_schedule": (
                    rejected["schedule"]
                    if rejected is not None
                    else search_incumbent["schedule"]
                ),
                "proposed_schedule": proposed,
                "evidence": evidence,
            }
        )
        report, records = ledger.evaluate_search(proposed, "phase_conditioned_frontier")
        adaptive_reports.append(report)
        chronology.append(report["schedule_sha256"])
        records_by_hash[report["schedule_sha256"]] = records

        if operation == "one_rung_causal_backoff":
            frozen_phases.add(phase)
        if report["qualified"]:
            rejected = None
            rejected_records = None
            if _candidate_better(report, search_incumbent):
                search_incumbent = report
                search_incumbent_records = records
        else:
            rejected = report
            rejected_records = records

    if search_incumbent is not None:
        selected = search_incumbent
        selection_reason = "selected fastest qualified schedule that preserved the qualified uniform incumbent"
    else:
        selected = {
            "role": "native_fallback",
            "schedule": [1.0] * 4,
            "schedule_sha256": schedule_sha256([1.0] * 4),
            "qualified": True,
            "summary": None,
        }
        selection_reason = "no accelerated schedule qualified; fail closed to native"

    if uniform_incumbent is not None and selected is not uniform_incumbent:
        if not _candidate_better(selected, uniform_incumbent):
            raise RuntimeError(
                "selected adaptive schedule regressed the uniform incumbent"
            )

    qualified_reports = [
        report for report in uniform_reports + adaptive_reports if report["qualified"]
    ]
    qualified = {
        report["schedule_sha256"]: report["summary"] for report in qualified_reports
    }
    search_frontier = []
    for candidate_hash, candidate in qualified.items():
        dominated = any(
            other_hash != candidate_hash
            and other["success_rate"] >= candidate["success_rate"]
            and other["achieved_throughput_per_step"]
            >= candidate["achieved_throughput_per_step"]
            and (
                other["success_rate"] > candidate["success_rate"]
                or other["achieved_throughput_per_step"]
                > candidate["achieved_throughput_per_step"]
            )
            for other_hash, other in qualified.items()
        )
        if not dominated:
            search_frontier.append(candidate_hash)

    return {
        "schema": "act-strider-frontier-selection-v3",
        "task_label": task_label,
        "selected_schedule": selected["schedule"],
        "selected_schedule_sha256": selected["schedule_sha256"],
        "selected_role": selected["role"],
        "selection_reason": selection_reason,
        "uniform_incumbent_sha256": (
            None if uniform_incumbent is None else uniform_incumbent["schedule_sha256"]
        ),
        "uniform_reports": uniform_reports,
        "adaptive_reports": adaptive_reports,
        "attribution_receipts": attribution_receipts,
        "chronology": chronology,
        "search_frontier_sha256": sorted(search_frontier),
        "search_rollouts": ledger.search_rollouts_used(),
        "search_budget": SEARCH_BUDGET,
        "unused_budget": SEARCH_BUDGET - ledger.search_rollouts_used(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument(
        "--task-label", choices=("pick", "tea", "insertion"), required=True
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--search-seeds", type=comma_ints, required=True)
    parser.add_argument("--final-seeds", type=comma_ints, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if len(args.search_seeds) != 15 or len(set(args.search_seeds)) != 15:
        raise ValueError("STRIDER v3 requires fifteen unique search seeds")
    if len(args.final_seeds) != 50 or len(set(args.final_seeds)) != 50:
        raise ValueError("STRIDER v3 requires fifty unique final seeds")
    if set(args.search_seeds) & set(args.final_seeds):
        raise ValueError("search and final banks must be disjoint")
    banks = json.loads(args.banks.read_text())
    task_banks = banks["tasks"][args.task_label]
    expected_search = list(
        range(
            int(task_banks["search"]["start"]),
            int(task_banks["search"]["start"])
            + int(task_banks["search"]["count"]),
        )
    )
    expected_final = list(
        range(
            int(task_banks["final"]["start"]),
            int(task_banks["final"]["start"])
            + int(task_banks["final"]["count"]),
        )
    )
    if args.search_seeds != expected_search or args.final_seeds != expected_final:
        raise ValueError("runtime seed arguments do not match frozen STRIDER v3 banks")

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
    )
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": "act-strider-frontier-identity-v3",
        "method": "strider_paired_causal_frontier",
        "contract_sha256": file_sha256(args.contract),
        "banks_sha256": file_sha256(args.banks),
        "search_seeds": args.search_seeds,
        "final_seeds": args.final_seeds,
        "search_budget": SEARCH_BUDGET,
        "stages": [
            {"episodes": count, "minimum_successes": minimum}
            for count, minimum in STAGES
        ],
        "attribution": {
            "same_seed_reference": True,
            "phase_exit_object_position_threshold_m": OBJECT_DIVERGENCE_METERS,
            "historical_speed_results_visible": False,
        },
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("STRIDER v3 root identity mismatch")
    write_json(identity_path, identity)

    ledger = RolloutLedger(
        runtime,
        root,
        args.search_seeds,
        args.final_seeds,
        record_search_telemetry=True,
    )
    selection = run_search(ledger, args.task_label)
    selection_path = root / "SELECTION.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("sealed STRIDER v3 selection changed during resume")
    write_json(selection_path, selection)
    selection_hash = file_sha256(selection_path)

    final = run_final(ledger, selection)
    result = {
        "schema": "act-strider-frontier-result-v3",
        "task_label": args.task_label,
        "identity_sha256": file_sha256(identity_path),
        "selection_sha256_before_final": selection_hash,
        "selection": selection,
        "final": final,
        "accounting": {
            "search_rollouts": ledger.search_rollouts_used(),
            "search_budget": SEARCH_BUDGET,
            "new_final_rollouts": final["new_final_rollouts"],
            "total_new_rollouts": ledger.search_rollouts_used()
            + final["new_final_rollouts"],
            "final_bank_opened_only_after_selection": True,
        },
    }
    result_path = root / "RESULT.json"
    write_json(result_path, result)
    write_json(
        root / "COMPLETE.json",
        {
            "schema": "act-strider-frontier-completion-v3",
            "identity_sha256": file_sha256(identity_path),
            "selection_sha256": selection_hash,
            "result_sha256": file_sha256(result_path),
            **result["accounting"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
