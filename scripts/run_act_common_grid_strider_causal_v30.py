#!/usr/bin/env python3
"""Run a fresh common-grid anchor-repair-promote STRIDER search."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import COMMON_GRID_SPEED_VALUES
from scripts import run_act_strider_frontier_v3 as telemetry
from scripts import run_act_strider_frontier_v4 as prior

PHASES = prior.PHASES
GRID = tuple(COMMON_GRID_SPEED_VALUES)
STAGES = ((5, 3), (10, 9), (20, 18))
CANDIDATE_BUDGET = 100
NATIVE_REFERENCES = 20
MIN_THROUGHPUT_GAIN = 0.03
OBSERVABLE_DIVERGENCE_CAUSES = {
    "candidate_did_not_complete_reference_phase",
    "task_reward_progress_lag",
    "object_position_left_matched_reference_envelope",
}

canonical_sha256 = prior.canonical_sha256
file_sha256 = prior.file_sha256
write_json = prior.write_json
successful = prior.base.successful
summarize = prior.summarize
phase_workloads = prior.base.phase_workloads


def validate_schedule(schedule) -> tuple[float, ...]:
    values = tuple(float(value) for value in schedule)
    if len(values) != len(PHASES):
        raise ValueError(f"schedule must have {len(PHASES)} phase speeds")
    if any(value not in GRID for value in values):
        raise ValueError(f"schedule values must be in the common grid {GRID}")
    return values


def schedule_sha256(schedule) -> str:
    return canonical_sha256(list(validate_schedule(schedule)))


def adjacent_speed(speed: float, direction: int) -> float:
    index = GRID.index(float(speed)) + int(direction)
    if not 0 <= index < len(GRID):
        raise ValueError(f"no adjacent common-grid speed from {speed}")
    return GRID[index]


def make_backoff(schedule, phase: str) -> list[float]:
    result = list(validate_schedule(schedule))
    index = PHASES.index(phase)
    result[index] = adjacent_speed(result[index], -1)
    return result


def make_promotion(schedule, phase: str) -> list[float]:
    result = list(validate_schedule(schedule))
    index = PHASES.index(phase)
    result[index] = adjacent_speed(result[index], 1)
    return result


def gate_decision(summary: dict) -> str:
    if summary["physics_errors"]:
        return "halt_physics_error"
    if summary["safety_violations"]:
        return "reject_safety"
    threshold = dict(STAGES).get(int(summary["episodes"]))
    if threshold is None:
        raise ValueError("gate decision requires a registered stage")
    if int(summary["successes"]) < threshold:
        return "reject_reliability"
    return "qualified" if int(summary["episodes"]) == STAGES[-1][0] else "continue"


class Ledger(prior.RolloutLedger):
    def gate_decision(self, summary: dict) -> str:
        return gate_decision(summary)

    def evaluate_search(self, schedule, role: str) -> tuple[dict, list[dict]]:
        schedule = list(validate_schedule(schedule))
        candidate_hash = schedule_sha256(schedule)
        candidate_root = self.root / "search" / "candidates" / candidate_hash
        schedule_receipt = {"schedule": schedule, "schedule_sha256": candidate_hash}
        schedule_path = candidate_root / "SCHEDULE.json"
        if schedule_path.exists() and json.loads(schedule_path.read_text()) != schedule_receipt:
            raise RuntimeError("candidate schedule identity mismatch")
        write_json(schedule_path, schedule_receipt)
        records = []
        stage_receipts = []
        final_decision = None
        for target, _ in STAGES:
            for seed in self.search_seeds[len(records) : target]:
                path = candidate_root / "states" / f"{seed}.json"
                if path.exists():
                    record = self._checked_record(path, schedule, seed)
                else:
                    if self.search_rollouts_used() >= CANDIDATE_BUDGET:
                        raise RuntimeError("causal STRIDER candidate budget exhausted")
                    record = self.runtime.rollout(
                        schedule, seed, record_attribution_telemetry=True
                    )
                    if list(map(float, record.get("schedule", ()))) != schedule:
                        raise RuntimeError("runtime returned a different schedule")
                    write_json(path, record)
                    print(
                        json.dumps(
                            {
                                "kind": "candidate",
                                "role": role,
                                "schedule": schedule,
                                "completed_for_schedule": len(records) + 1,
                                "candidate_rollouts_used": self.search_rollouts_used(),
                                "success": bool(record.get("success")),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                records.append(record)
            summary = summarize(records)
            decision = self.gate_decision(summary)
            receipt = {"target": target, "decision": decision, "summary": summary}
            stage_receipts.append(receipt)
            write_json(candidate_root / f"GATE_{target}.json", receipt)
            if decision != "continue":
                final_decision = decision
                break
        if final_decision is None:
            raise RuntimeError("candidate did not reach a terminal decision")
        report = {
            "role": role,
            "schedule": schedule,
            "schedule_sha256": candidate_hash,
            "decision": final_decision,
            "qualified": final_decision == "qualified",
            "summary": summarize(records),
            "stages": stage_receipts,
        }
        write_json(candidate_root / "SUMMARY.json", report)
        if final_decision == "halt_physics_error":
            raise RuntimeError(f"physics error in candidate {schedule}")
        return report, records

    def native_references(self) -> tuple[dict, list[dict]]:
        schedule = [1.0] * len(PHASES)
        root = self.root / "search" / "native_reference"
        records = []
        for seed in self.search_seeds[:NATIVE_REFERENCES]:
            path = root / "states" / f"{seed}.json"
            if path.exists():
                record = self._checked_record(path, schedule, seed)
            else:
                record = self.runtime.rollout(
                    schedule, seed, record_attribution_telemetry=True
                )
                if list(map(float, record.get("schedule", ()))) != schedule:
                    raise RuntimeError("runtime returned a different native schedule")
                write_json(path, record)
                print(
                    json.dumps(
                        {
                            "kind": "native_reference",
                            "completed": len(records) + 1,
                            "success": bool(record.get("success")),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            records.append(record)
        summary = summarize(records)
        if summary["physics_errors"] or summary["safety_violations"]:
            raise RuntimeError("native reference produced an incident")
        if summary["successes"] < 18:
            raise RuntimeError("native reference is below 18/20")
        result = {
            "schedule": schedule,
            "schedule_sha256": schedule_sha256(schedule),
            "qualified": True,
            "role": "native_reference",
            "summary": summary,
        }
        write_json(root / "SUMMARY.json", result)
        return result, records


def remaining_full_gate(ledger: Ledger) -> bool:
    return ledger.search_rollouts_used() + STAGES[-1][0] <= CANDIDATE_BUDGET


def choose_uniform(reports: list[dict]) -> dict | None:
    qualified = [item for item in reports if item["qualified"]]
    if not qualified:
        return None
    return max(
        qualified,
        key=lambda item: (
            item["summary"]["successes"],
            item["summary"]["achieved_throughput_per_step"],
        ),
    )


def replaces(candidate: dict, lower: dict) -> bool:
    summary = candidate["summary"]
    baseline = lower["summary"]
    return (
        candidate["qualified"]
        and not summary["physics_errors"]
        and not summary["safety_violations"]
        and summary["successes"] >= baseline["successes"]
        and summary["achieved_throughput_per_step"]
        >= (1.0 + MIN_THROUGHPUT_GAIN)
        * baseline["achieved_throughput_per_step"]
    )


def merged_references(
    rejected_records: list[dict], incumbent_records: list[dict], native_records: list[dict]
) -> list[dict]:
    native = {int(item["seed"]): item for item in native_records if successful(item)}
    incumbent = {
        int(item["seed"]): item for item in incumbent_records if successful(item)
    }
    return [
        incumbent.get(int(item["seed"]), native.get(int(item["seed"])))
        for item in rejected_records
        if incumbent.get(int(item["seed"]), native.get(int(item["seed"]))) is not None
    ]


def causal_failure_phase(
    rejected_records: list[dict], references: list[dict], fallback_phase: str
) -> tuple[str | None, dict]:
    phase, evidence = telemetry.paired_failure_phase(
        rejected_records, references, fallback_phase
    )
    if evidence.get("method") != "same_seed_phase_exit_physical_divergence":
        return None, evidence
    causal = [
        item
        for item in evidence.get("paired_counterexamples", ())
        if item.get("cause") in OBSERVABLE_DIVERGENCE_CAUSES
    ]
    if not causal:
        return None, {
            "method": "matched_telemetry_without_observable_divergence",
            "reason": "paired failures had no preregistered physical divergence",
            "paired_counterexamples": evidence.get("paired_counterexamples", []),
            "unmatched_failed_seeds": evidence.get("unmatched_failed_seeds", []),
        }
    selected = min((item["phase"] for item in causal), key=PHASES.index)
    return selected, {
        **evidence,
        "selected_phase": selected,
        "paired_counterexamples": causal,
        "noncausal_paired_counterexamples": [
            item
            for item in evidence.get("paired_counterexamples", ())
            if item.get("cause") not in OBSERVABLE_DIVERGENCE_CAUSES
        ],
    }


def promotion_phase(records: list[dict], schedule: list[float], frozen: set[str]):
    usable = [item for item in records if successful(item)]
    if not usable:
        return None, {"reason": "no successful incumbent telemetry"}
    workloads = {
        phase: statistics.fmean(phase_workloads(item)[phase] for item in usable)
        for phase in PHASES
    }
    scores = {}
    for phase, old in zip(PHASES, schedule):
        if phase in frozen or old == GRID[-1]:
            continue
        new = adjacent_speed(old, 1)
        scores[phase] = workloads[phase] * (1.0 / old - 1.0 / new)
    if not scores:
        return None, {"reason": "no unfrozen phase has a higher common-grid rung"}
    phase = max(scores, key=lambda item: (scores[item], -PHASES.index(item)))
    return phase, {
        "method": "preregistered_bang_for_buck",
        "selected_phase": phase,
        "phase_native_equivalent_work": workloads,
        "predicted_steps_saved": scores,
    }


def run_search(ledger: Ledger, task: str) -> dict:
    native, native_records = ledger.native_references()
    chronology = []
    uniform_reports = []
    adaptive_reports = []
    attribution_receipts = []
    records_by_hash = {}
    rejected = None
    rejected_records = []

    anchor, records = ledger.evaluate_search([2.0] * 4, "uniform_anchor")
    uniform_reports.append(anchor)
    chronology.append(anchor["schedule_sha256"])
    records_by_hash[anchor["schedule_sha256"]] = records
    if anchor["qualified"]:
        for speed in (2.5, 3.0):
            if not remaining_full_gate(ledger):
                break
            report, records = ledger.evaluate_search([speed] * 4, "uniform_ladder")
            uniform_reports.append(report)
            chronology.append(report["schedule_sha256"])
            records_by_hash[report["schedule_sha256"]] = records
            if not report["qualified"]:
                rejected, rejected_records = report, records
                break
    else:
        rejected, rejected_records = anchor, records
        if remaining_full_gate(ledger):
            fallback, records = ledger.evaluate_search([1.5] * 4, "uniform_fallback")
            uniform_reports.append(fallback)
            chronology.append(fallback["schedule_sha256"])
            records_by_hash[fallback["schedule_sha256"]] = records

    uniform = choose_uniform(uniform_reports)
    incumbent = uniform or native
    incumbent_records = (
        native_records
        if uniform is None
        else records_by_hash[uniform["schedule_sha256"]]
    )
    frozen = set()
    while remaining_full_gate(ledger):
        if rejected is not None:
            references = merged_references(
                rejected_records, incumbent_records, native_records
            )
            phase, evidence = causal_failure_phase(
                rejected_records, references, telemetry.SEMANTIC_FALLBACK[task]
            )
            if phase is None:
                attribution_receipts.append(
                    {
                        "operation": "causal_attribution_unavailable",
                        "phase": None,
                        "source_schedule": rejected["schedule"],
                        "proposed_schedule": None,
                        "evidence": evidence,
                    }
                )
                break
            index = PHASES.index(phase)
            if rejected["schedule"][index] == GRID[0]:
                attribution_receipts.append(
                    {
                        "operation": "causal_backoff_exhausted",
                        "phase": phase,
                        "source_schedule": rejected["schedule"],
                        "evidence": evidence,
                    }
                )
                break
            proposed = make_backoff(rejected["schedule"], phase)
            operation = "one_rung_causal_backoff"
        else:
            phase, evidence = promotion_phase(
                incumbent_records, incumbent["schedule"], frozen
            )
            if phase is None:
                break
            proposed = make_promotion(incumbent["schedule"], phase)
            operation = "one_rung_bang_for_buck_promotion"
        proposed_hash = schedule_sha256(proposed)
        if proposed_hash in set(chronology):
            break
        attribution_receipts.append(
            {
                "operation": operation,
                "phase": phase,
                "source_schedule": (
                    rejected["schedule"] if rejected is not None else incumbent["schedule"]
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
            frozen.add(phase)
        if report["qualified"]:
            rejected, rejected_records = None, []
            if replaces(report, incumbent):
                incumbent, incumbent_records = report, records
        else:
            rejected, rejected_records = report, records

    selected = incumbent
    all_reports = uniform_reports + adaptive_reports
    incident_totals = {
        "physics_errors": native["summary"]["physics_errors"]
        + sum(item["summary"]["physics_errors"] for item in all_reports),
        "safety_violations": native["summary"]["safety_violations"]
        + sum(item["summary"]["safety_violations"] for item in all_reports),
    }
    selection = {
        "schema": "act-common-grid-strider-causal-selection-v30",
        "task_label": task,
        "selected_schedule": selected["schedule"],
        "selected_schedule_sha256": selected["schedule_sha256"],
        "selected_role": selected["role"],
        "native_reference": native,
        "uniform_incumbent": uniform,
        "uniform_reports": uniform_reports,
        "adaptive_reports": adaptive_reports,
        "attribution_receipts": attribution_receipts,
        "chronology": chronology,
        "frozen_repaired_phases": sorted(frozen, key=PHASES.index),
        "native_reference_rollouts": len(native_records),
        "candidate_rollouts": ledger.search_rollouts_used(),
        "candidate_budget": CANDIDATE_BUDGET,
        "unused_candidate_budget": CANDIDATE_BUDGET - ledger.search_rollouts_used(),
        "incident_totals": incident_totals,
        "final_bank_opened": False,
    }
    if any(speed not in GRID for speed in selection["selected_schedule"]):
        raise RuntimeError("selection escaped the common grid")
    return selection


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    banks = json.loads(args.banks.read_text())
    specs = banks["tasks"][args.task_label]
    if int(specs["search"]["count"]) != 20 or int(specs["final"]["count"]) != 50:
        raise RuntimeError("v30 requires registered 20-search and 50-final banks")
    search_seeds = list(
        range(int(specs["search"]["start"]), int(specs["search"]["start"]) + 20)
    )
    final_seeds = list(
        range(int(specs["final"]["start"]), int(specs["final"]["start"]) + 50)
    )
    all_seeds = []
    for task in banks["tasks"].values():
        for name in ("search", "final"):
            spec = task[name]
            all_seeds.extend(range(int(spec["start"]), int(spec["start"]) + int(spec["count"])))
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("registered v30 banks overlap")

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
    root = args.root.resolve() / args.task_label
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": "act-common-grid-strider-causal-identity-v30",
        "method": "common_grid_anchor_repair_promote",
        "task_label": args.task_label,
        "contract_sha256": file_sha256(args.contract),
        "banks_sha256": file_sha256(args.banks),
        "action_grid": list(GRID),
        "phases": list(PHASES),
        "search_seeds": search_seeds,
        "final_seeds_registered_unopened": final_seeds,
        "native_reference_rollouts": NATIVE_REFERENCES,
        "candidate_budget": CANDIDATE_BUDGET,
        "stages": [
            {"episodes": count, "minimum_successes": minimum}
            for count, minimum in STAGES
        ],
        "historical_schedule_outcomes_visible": False,
        "attribution": {
            "same_seed_incumbent_then_native_reference": True,
            "phase_exit_object_position_threshold_m": telemetry.OBJECT_DIVERGENCE_METERS,
        },
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("v30 identity differs on resume")
    write_json(identity_path, identity)
    complete_path = root / "SEARCH_COMPLETE.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if complete["identity_sha256"] != file_sha256(identity_path):
            raise RuntimeError("completed v30 identity hash mismatch")
        selection_path = root / "SELECTION.json"
        if complete["selection_sha256"] != file_sha256(selection_path):
            raise RuntimeError("completed v30 selection hash mismatch")
        print(json.dumps(json.loads(selection_path.read_text()), sort_keys=True))
        return 0

    ledger = Ledger(
        runtime,
        root,
        search_seeds,
        final_seeds,
        record_search_telemetry=True,
    )
    selection = run_search(ledger, args.task_label)
    selection_path = root / "SELECTION.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("sealed v30 selection changed on resume")
    write_json(selection_path, selection)
    completion = {
        "schema": "act-common-grid-strider-causal-search-completion-v30",
        "task_label": args.task_label,
        "identity_sha256": file_sha256(identity_path),
        "selection_sha256": file_sha256(selection_path),
        "native_reference_rollouts": selection["native_reference_rollouts"],
        "candidate_rollouts": selection["candidate_rollouts"],
        "physics_errors": selection["incident_totals"]["physics_errors"],
        "safety_violations": selection["incident_totals"]["safety_violations"],
        "final_bank_opened": False,
    }
    write_json(complete_path, completion)
    print(json.dumps({"selection": selection, "completion": completion}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
