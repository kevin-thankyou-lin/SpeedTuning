#!/usr/bin/env python3
"""Run conservative paired STRIDER search and a fresh ACT final benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import run_act_strider_frontier_v2 as base
from scripts import run_act_strider_frontier_v3 as v3


PHASES = base.PHASES
ALLOWED_SPEEDS = base.ALLOWED_SPEEDS
STAGES = ((5, 4), (10, 9), (20, 19))
SEARCH_BUDGET = 60
MIN_THROUGHPUT_GAIN = 0.03

canonical_sha256 = base.canonical_sha256
file_sha256 = base.file_sha256
write_json = base.write_json
comma_ints = base.comma_ints
validate_schedule = base.validate_schedule
schedule_sha256 = base.schedule_sha256
summarize = base.summarize
make_backoff = base.make_backoff
make_promotion = base.make_promotion
choose_uniform_incumbent = base.choose_uniform_incumbent
pareto_names = base.pareto_names
run_final = base.run_final


class RolloutLedger(base.RolloutLedger):
    """v2 ledger with the v4 5->10->20 gate and unchanged receipt format."""

    def gate_decision(self, summary: dict) -> str:
        if summary["physics_errors"]:
            return "halt_physics_error"
        if summary["safety_violations"]:
            return "reject_safety"
        thresholds = dict(STAGES)
        episodes = summary["episodes"]
        if episodes not in thresholds:
            raise ValueError(f"gate decision requires a staged count, got {episodes}")
        if summary["successes"] < thresholds[episodes]:
            return "reject_reliability"
        return "qualified" if episodes == STAGES[-1][0] else "continue"

    def evaluate_search(self, schedule, role: str) -> tuple[dict, list[dict]]:
        schedule = list(validate_schedule(schedule))
        candidate_hash = schedule_sha256(schedule)
        candidate_root = self.root / "search" / "candidates" / candidate_hash
        schedule_path = candidate_root / "SCHEDULE.json"
        schedule_receipt = {
            "schedule": schedule,
            "schedule_sha256": candidate_hash,
        }
        if (
            schedule_path.exists()
            and json.loads(schedule_path.read_text()) != schedule_receipt
        ):
            raise RuntimeError(f"candidate schedule identity mismatch: {schedule_path}")
        write_json(schedule_path, schedule_receipt)

        records: list[dict] = []
        stage_receipts = []
        final_decision = None
        for target, _ in STAGES:
            for seed in self.search_seeds[len(records) : target]:
                path = candidate_root / "states" / f"{seed}.json"
                if path.exists():
                    record = self._checked_record(path, schedule, seed)
                else:
                    if self.search_rollouts_used() >= SEARCH_BUDGET:
                        raise RuntimeError("STRIDER v4 search rollout budget exhausted")
                    record = self.runtime.rollout(
                        schedule,
                        seed,
                        record_attribution_telemetry=self.record_search_telemetry,
                    )
                    if list(map(float, record.get("schedule", ()))) != schedule:
                        raise RuntimeError("runtime returned a different schedule")
                    write_json(path, record)
                records.append(record)
            stage_summary = summarize(records)
            decision = self.gate_decision(stage_summary)
            stage_receipts.append(
                {"target": target, "decision": decision, "summary": stage_summary}
            )
            write_json(candidate_root / f"GATE_{target}.json", stage_receipts[-1])
            if decision != "continue":
                final_decision = decision
                break
        if final_decision is None:
            raise RuntimeError("candidate did not reach a terminal search decision")
        report = {
            "role": role,
            "schedule": schedule,
            "schedule_sha256": candidate_hash,
            "decision": final_decision,
            "qualified": final_decision == "qualified",
            "summary": summarize(records),
            "stages": stage_receipts,
            "receipt_paths": [
                str(candidate_root / "states" / f"{record['seed']}.json")
                for record in records
            ],
        }
        write_json(candidate_root / "SUMMARY.json", report)
        if final_decision == "halt_physics_error":
            raise RuntimeError(f"physics error in candidate {schedule}")
        return report, records


def adaptive_replaces_uniform(candidate: dict, incumbent: dict | None) -> bool:
    """Require a qualified uniform lower bound and a material paired gain."""

    if incumbent is None or not candidate["qualified"]:
        return False
    candidate_summary = candidate["summary"]
    incumbent_summary = incumbent["summary"]
    if candidate_summary["safety_violations"] or candidate_summary["physics_errors"]:
        return False
    has_slower_phase = any(
        candidate_speed < incumbent_speed
        for candidate_speed, incumbent_speed in zip(
            candidate["schedule"], incumbent["schedule"]
        )
    )
    minimum_successes = incumbent_summary["successes"] + int(has_slower_phase)
    return (
        candidate_summary["successes"] >= minimum_successes
        and candidate_summary["achieved_throughput_per_step"]
        >= (1.0 + MIN_THROUGHPUT_GAIN)
        * incumbent_summary["achieved_throughput_per_step"]
    )


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
            phase, evidence = v3.paired_failure_phase(
                rejected_records or [],
                search_incumbent_records,
                v3.SEMANTIC_FALLBACK[task_label],
            )
            phase_index = PHASES.index(phase)
            if float(rejected["schedule"][phase_index]) == min(ALLOWED_SPEEDS):
                attribution_receipts.append(
                    {
                        "operation": "causal_backoff_exhausted",
                        "phase": phase,
                        "source_schedule": rejected["schedule"],
                        "proposed_schedule": None,
                        "evidence": evidence,
                        "reason": "implicated phase is already at the minimum registered speed",
                    }
                )
                break
            proposed = make_backoff(rejected["schedule"], phase)
            operation = "one_rung_causal_backoff"
        elif search_incumbent is not None:
            phase, evidence = v3.highest_bang_for_buck_phase(
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
        if proposed_hash in set(chronology):
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
        report, records = ledger.evaluate_search(
            proposed, "phase_conditioned_frontier"
        )
        adaptive_reports.append(report)
        chronology.append(report["schedule_sha256"])
        records_by_hash[report["schedule_sha256"]] = records

        if operation == "one_rung_causal_backoff":
            frozen_phases.add(phase)
        if report["qualified"]:
            rejected = None
            rejected_records = None
            if adaptive_replaces_uniform(report, uniform_incumbent):
                search_incumbent = report
                search_incumbent_records = records
        else:
            rejected = report
            rejected_records = records

    if search_incumbent is not None:
        selected = search_incumbent
        selection_reason = (
            "selected a strict-gate schedule that materially improves the qualified "
            "uniform incumbent"
            if selected is not uniform_incumbent
            else "retained the strict-gate uniform incumbent"
        )
    else:
        selected = {
            "role": "native_fallback",
            "schedule": [1.0] * 4,
            "schedule_sha256": schedule_sha256([1.0] * 4),
            "qualified": True,
            "summary": None,
        }
        selection_reason = (
            "no uniform controller reached 19/20; fail closed to native"
        )

    if uniform_incumbent is not None and selected is not uniform_incumbent:
        if not adaptive_replaces_uniform(selected, uniform_incumbent):
            raise RuntimeError("selected adaptive schedule regressed its uniform lower bound")
    if uniform_incumbent is None and selected["role"] != "native_fallback":
        raise RuntimeError("adaptive controller selected without a qualified uniform lower bound")

    qualified_reports = [
        report
        for report in uniform_reports + adaptive_reports
        if report["qualified"]
    ]
    qualified = {
        report["schedule_sha256"]: report["summary"]
        for report in qualified_reports
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
        "schema": "act-strider-frontier-selection-v4",
        "task_label": task_label,
        "selected_schedule": selected["schedule"],
        "selected_schedule_sha256": selected["schedule_sha256"],
        "selected_role": selected["role"],
        "selection_reason": selection_reason,
        "uniform_incumbent_sha256": (
            None
            if uniform_incumbent is None
            else uniform_incumbent["schedule_sha256"]
        ),
        "uniform_reports": uniform_reports,
        "adaptive_reports": adaptive_reports,
        "attribution_receipts": attribution_receipts,
        "chronology": chronology,
        "search_frontier_sha256": sorted(search_frontier),
        "search_rollouts": ledger.search_rollouts_used(),
        "search_budget": SEARCH_BUDGET,
        "unused_budget": SEARCH_BUDGET - ledger.search_rollouts_used(),
        "adaptive_minimum_throughput_gain": MIN_THROUGHPUT_GAIN,
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

    if len(args.search_seeds) != 20 or len(set(args.search_seeds)) != 20:
        raise ValueError("STRIDER v4 requires twenty unique search seeds")
    if len(args.final_seeds) != 50 or len(set(args.final_seeds)) != 50:
        raise ValueError("STRIDER v4 requires fifty unique final seeds")
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
        raise ValueError("runtime seed arguments do not match frozen STRIDER v4 banks")

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
        "schema": "act-strider-frontier-identity-v4",
        "method": "strider_conservative_uniform_lower_bound",
        "contract_sha256": file_sha256(args.contract),
        "banks_sha256": file_sha256(args.banks),
        "search_seeds": args.search_seeds,
        "final_seeds": args.final_seeds,
        "search_budget": SEARCH_BUDGET,
        "stages": [
            {"episodes": count, "minimum_successes": minimum}
            for count, minimum in STAGES
        ],
        "adaptive_minimum_throughput_gain": MIN_THROUGHPUT_GAIN,
        "attribution": {
            "same_seed_reference": True,
            "phase_exit_object_position_threshold_m": v3.OBJECT_DIVERGENCE_METERS,
            "historical_speed_results_visible": False,
        },
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("STRIDER v4 root identity mismatch")
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
        raise RuntimeError("sealed STRIDER v4 selection changed during resume")
    write_json(selection_path, selection)
    selection_hash = file_sha256(selection_path)

    final = run_final(ledger, selection)
    result = {
        "schema": "act-strider-frontier-result-v4",
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
            "schema": "act-strider-frontier-completion-v4",
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
