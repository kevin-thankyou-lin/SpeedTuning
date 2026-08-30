#!/usr/bin/env python3
"""Run exact-25 common-grid STRIDER with causal one-phase repair."""

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

from scripts import run_act_common_grid_strider_causal_v30 as v30
from scripts import run_act_strider_frontier_v4 as prior

PHASES = v30.PHASES
GRID = v30.GRID
DISCOVERY_SCHEDULES = 5
DISCOVERY_SEEDS = 3
CONFIRMATION_SEEDS = 5
SEARCH_BUDGET = DISCOVERY_SCHEDULES * DISCOVERY_SEEDS + 2 * CONFIRMATION_SEEDS

write_json = prior.write_json
file_sha256 = prior.file_sha256
summarize = prior.summarize
successful = prior.base.successful
schedule_sha256 = v30.schedule_sha256
validate_schedule = v30.validate_schedule


def safe(report: dict) -> bool:
    summary = report["summary"]
    return (
        summary["episodes"] == DISCOVERY_SEEDS
        and summary["successes"] == DISCOVERY_SEEDS
        and summary["physics_errors"] == 0
        and summary["safety_violations"] == 0
    )


def is_uniform(schedule) -> bool:
    return len(set(map(float, schedule))) == 1


class Ledger:
    def __init__(self, runtime, root: Path, discovery: list[int], confirmation: list[int]):
        self.runtime = runtime
        self.root = root
        self.discovery = discovery
        self.confirmation = confirmation

    def used(self) -> int:
        return len(list((self.root / "states").glob("*/*.json")))

    def one(self, schedule, seed: int) -> dict:
        schedule = list(validate_schedule(schedule))
        path = self.root / "states" / schedule_sha256(schedule) / f"{seed}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if int(record.get("seed", -1)) != seed:
                raise RuntimeError("cached v31 seed mismatch")
            if list(map(float, record.get("schedule", ()))) != schedule:
                raise RuntimeError("cached v31 schedule mismatch")
            return record
        if self.used() >= SEARCH_BUDGET:
            raise RuntimeError("v31 exact-25 search budget exhausted")
        record = self.runtime.rollout(
            schedule, seed, record_attribution_telemetry=True
        )
        if list(map(float, record.get("schedule", ()))) != schedule:
            raise RuntimeError("runtime returned a different v31 schedule")
        write_json(path, record)
        print(
            json.dumps(
                {
                    "kind": "search",
                    "schedule": schedule,
                    "seed": seed,
                    "success": successful(record),
                    "search_rollouts_used": self.used(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return record

    def discovery_report(self, schedule, role: str) -> tuple[dict, list[dict]]:
        schedule = list(validate_schedule(schedule))
        records = [self.one(schedule, seed) for seed in self.discovery]
        report = {
            "role": role,
            "schedule": schedule,
            "schedule_sha256": schedule_sha256(schedule),
            "summary": summarize(records),
        }
        report["safe_3_of_3"] = safe(report)
        if report["summary"]["physics_errors"]:
            raise RuntimeError(f"physics error in v31 discovery schedule {schedule}")
        write_json(
            self.root / "reports" / f"{report['schedule_sha256']}.json", report
        )
        return report, records

    def confirmation_report(self, report: dict) -> tuple[dict, list[dict]]:
        records = [self.one(report["schedule"], seed) for seed in self.discovery]
        records.extend(self.one(report["schedule"], seed) for seed in self.confirmation)
        value = {
            "schedule": report["schedule"],
            "schedule_sha256": report["schedule_sha256"],
            "discovery_role": report["role"],
            "summary": summarize(records),
        }
        if value["summary"]["physics_errors"]:
            raise RuntimeError(f"physics error in v31 finalist {report['schedule']}")
        write_json(
            self.root / "confirmation" / f"{report['schedule_sha256']}.json", value
        )
        return value, records

    def incident_totals(self) -> dict:
        records = [json.loads(path.read_text()) for path in (self.root / "states").glob("*/*.json")]
        return {
            "physics_errors": sum(item.get("physics_error") is not None for item in records),
            "safety_violations": sum(item.get("safety_violation") is not None for item in records),
        }


def best_safe(reports: list[dict]) -> dict:
    candidates = [report for report in reports if safe(report)]
    if not candidates:
        raise RuntimeError("v31 has no safe discovery controller")
    return max(
        candidates,
        key=lambda report: (
            report["summary"]["achieved_throughput_per_step"],
            -len(set(report["schedule"])),
        ),
    )


def references_for(failed_records, incumbent_records, native_records):
    return v30.merged_references(failed_records, incumbent_records, native_records)


def causal_backoff(report, records, incumbent, incumbent_records, native_records):
    refs = references_for(records, incumbent_records, native_records)
    phase, evidence = v30.causal_failure_phase(records, refs, PHASES[0])
    if phase is None:
        return None, {
            "operation": "causal_attribution_unavailable",
            "source_schedule": report["schedule"],
            "proposed_schedule": None,
            "phase": None,
            "evidence": evidence,
        }
    index = PHASES.index(phase)
    if report["schedule"][index] == GRID[0]:
        return None, {
            "operation": "causal_backoff_exhausted",
            "source_schedule": report["schedule"],
            "proposed_schedule": None,
            "phase": phase,
            "evidence": evidence,
        }
    proposed = v30.make_backoff(report["schedule"], phase)
    return proposed, {
        "operation": "one_rung_causal_backoff",
        "source_schedule": report["schedule"],
        "proposed_schedule": proposed,
        "phase": phase,
        "evidence": evidence,
    }


def ranked_promotions(records, schedule, frozen: set[str]):
    usable = [record for record in records if successful(record)]
    if not usable:
        return []
    work = {
        phase: statistics.fmean(prior.base.phase_workloads(item)[phase] for item in usable)
        for phase in PHASES
    }
    candidates = []
    for phase, speed in zip(PHASES, schedule):
        if phase in frozen or speed == GRID[-1]:
            continue
        new = v30.adjacent_speed(speed, 1)
        score = work[phase] * (1.0 / speed - 1.0 / new)
        candidates.append((score, -PHASES.index(phase), phase, v30.make_promotion(schedule, phase)))
    return sorted(candidates, reverse=True)


def unused_uniform(tried: set[str]):
    for speed in (2.5, 1.5, 3.0):
        schedule = [speed] * len(PHASES)
        if schedule_sha256(schedule) not in tried:
            return schedule
    return None


def next_safe_proposal(incumbent, incumbent_records, tried, frozen, last):
    if last["schedule"] == [3.0] * len(PHASES):
        bracket = [2.5] * len(PHASES)
        if schedule_sha256(bracket) not in tried:
            return bracket, {
                "operation": "registered_uniform_bracket",
                "source_schedule": last["schedule"],
                "proposed_schedule": bracket,
            }
    for _, _, phase, schedule in ranked_promotions(
        incumbent_records, incumbent["schedule"], frozen
    ):
        if schedule_sha256(schedule) not in tried:
            return schedule, {
                "operation": "one_rung_bang_for_buck_promotion",
                "source_schedule": incumbent["schedule"],
                "proposed_schedule": schedule,
                "phase": phase,
            }
    return unused_uniform(tried), {"operation": "registered_uniform_fallback"}


def run_search(ledger: Ledger, task: str) -> dict:
    reports = []
    records_by_hash = {}
    receipts = []
    frozen: set[str] = set()

    def evaluate(schedule, role):
        report, records = ledger.discovery_report(schedule, role)
        reports.append(report)
        records_by_hash[report["schedule_sha256"]] = records
        return report, records

    native, native_records = evaluate([1.0] * len(PHASES), "native_reference")
    if not safe(native):
        raise RuntimeError("v31 native reference is not safe 3/3")
    anchor, anchor_records = evaluate([2.0] * len(PHASES), "uniform_anchor")
    last, last_records = anchor, anchor_records

    while len(reports) < DISCOVERY_SCHEDULES:
        tried = {report["schedule_sha256"] for report in reports}
        incumbent = best_safe(reports)
        incumbent_records = records_by_hash[incumbent["schedule_sha256"]]
        if len(reports) == 2 and safe(anchor):
            proposed = [3.0] * len(PHASES)
            receipt = {
                "operation": "registered_aggressive_ceiling_diagnostic",
                "source_schedule": anchor["schedule"],
                "proposed_schedule": proposed,
            }
        elif not safe(last):
            proposed, receipt = causal_backoff(
                last,
                last_records,
                incumbent,
                incumbent_records,
                native_records,
            )
            receipts.append(receipt)
            if receipt.get("phase") is not None:
                frozen.add(receipt["phase"])
            if proposed is not None and schedule_sha256(proposed) in tried:
                receipts.append(
                    {
                        "operation": "causal_backoff_to_cached_safe_controller",
                        "schedule": proposed,
                    }
                )
                proposed = None
            if proposed is None:
                if receipt["operation"] == "causal_attribution_unavailable":
                    proposed = [1.5] * len(PHASES)
                    if schedule_sha256(proposed) in tried:
                        proposed = unused_uniform(tried)
                    receipts.append(
                        {
                            "operation": "registered_uniform_fallback_after_unavailable_attribution",
                            "proposed_schedule": proposed,
                        }
                    )
                else:
                    proposed, fallback = next_safe_proposal(
                        incumbent, incumbent_records, tried, frozen, last
                    )
                    receipts.append(fallback)
        else:
            proposed, receipt = next_safe_proposal(
                incumbent, incumbent_records, tried, frozen, last
            )
            receipts.append(receipt)
        if proposed is None or schedule_sha256(proposed) in tried:
            raise RuntimeError("v31 could not fill five unique discovery slots")
        last, last_records = evaluate(proposed, "causal_frontier")

    ranked = sorted(
        reports,
        key=lambda report: (
            safe(report),
            report["summary"]["successes"],
            report["summary"]["achieved_throughput_per_step"],
        ),
        reverse=True,
    )
    finalists = ranked[:2]
    confirmation = []
    for report in finalists:
        value, _ = ledger.confirmation_report(report)
        confirmation.append(value)
    if ledger.used() != SEARCH_BUDGET:
        raise RuntimeError(f"v31 search used {ledger.used()}, expected exactly 25")
    eligible = [
        value
        for value in confirmation
        if value["summary"]["successes"] >= 7
        and value["summary"]["physics_errors"] == 0
        and value["summary"]["safety_violations"] == 0
    ]
    selected = max(
        eligible or confirmation,
        key=lambda value: (
            value["summary"]["successes"],
            value["summary"]["achieved_throughput_per_step"],
            -len(set(value["schedule"])),
        ),
    )
    return {
        "schema": "act-common-grid-strider-causal25-selection-v31",
        "task_label": task,
        "discovery_reports": reports,
        "update_receipts": receipts,
        "frozen_causal_phases": sorted(frozen, key=PHASES.index),
        "finalists": confirmation,
        "selected_schedule": selected["schedule"],
        "selected_schedule_sha256": selected["schedule_sha256"],
        "search_scientific_rollouts": ledger.used(),
        "search_budget": SEARCH_BUDGET,
        "prior_rollouts_reexecuted": 0,
        "incident_totals": ledger.incident_totals(),
        "final_bank_opened": False,
    }


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
    spec = banks["tasks"][args.task_label]
    discovery = list(range(spec["discovery"]["start"], spec["discovery"]["start"] + 3))
    confirmation = list(
        range(spec["confirmation"]["start"], spec["confirmation"]["start"] + 5)
    )
    final = list(range(spec["final"]["start"], spec["final"]["start"] + 50))
    all_seeds = []
    for item in banks["tasks"].values():
        for name in ("discovery", "confirmation", "final"):
            bank = item[name]
            all_seeds.extend(range(bank["start"], bank["start"] + bank["count"]))
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("v31 registered banks overlap")

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
        "schema": "act-common-grid-strider-causal25-identity-v31",
        "task_label": args.task_label,
        "contract_sha256": file_sha256(args.contract),
        "banks_sha256": file_sha256(args.banks),
        "action_grid": list(GRID),
        "phases": list(PHASES),
        "discovery_seeds": discovery,
        "confirmation_seeds": confirmation,
        "final_seeds_registered_unopened": final,
        "search_budget": SEARCH_BUDGET,
        "historical_schedule_outcomes_visible": False,
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("v31 identity differs on resume")
    write_json(identity_path, identity)
    selection_path = root / "SELECTION.json"
    complete_path = root / "SEARCH_COMPLETE.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if complete["identity_sha256"] != file_sha256(identity_path):
            raise RuntimeError("v31 completed identity hash mismatch")
        if complete["selection_sha256"] != file_sha256(selection_path):
            raise RuntimeError("v31 completed selection hash mismatch")
        print(json.dumps(json.loads(selection_path.read_text()), sort_keys=True))
        return 0

    selection = run_search(Ledger(runtime, root / "search", discovery, confirmation), args.task_label)
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("sealed v31 selection changed on resume")
    write_json(selection_path, selection)
    incidents = selection["incident_totals"]
    completion = {
        "schema": "act-common-grid-strider-causal25-search-completion-v31",
        "task_label": args.task_label,
        "identity_sha256": file_sha256(identity_path),
        "selection_sha256": file_sha256(selection_path),
        "search_scientific_rollouts": SEARCH_BUDGET,
        **incidents,
        "prior_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    }
    write_json(complete_path, completion)
    print(json.dumps({"selection": selection, "completion": completion}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
