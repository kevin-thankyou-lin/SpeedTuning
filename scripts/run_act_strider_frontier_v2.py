#!/usr/bin/env python3
"""Run corrected uniform-anchored STRIDER search and fresh final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PHASES = ("pre_grasp", "grasp_lift", "transport", "interaction")
ALLOWED_SPEEDS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
STAGES = ((5, 4), (10, 9), (15, 14))
SEARCH_BUDGET = 60
UNIFORM_LADDER = (1.5, 2.0, 2.5, 3.0)


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def validate_schedule(schedule) -> tuple[float, ...]:
    values = tuple(float(value) for value in schedule)
    if len(values) != len(PHASES):
        raise ValueError(f"schedule must have {len(PHASES)} values")
    if any(value not in ALLOWED_SPEEDS for value in values):
        raise ValueError(f"schedule values must be in {ALLOWED_SPEEDS}")
    return values


def schedule_sha256(schedule) -> str:
    return canonical_sha256(list(validate_schedule(schedule)))


def successful(record: dict) -> bool:
    return bool(record.get("success")) and record.get("safety_violation") is None and record.get("physics_error") is None


def episode_metric_steps(record: dict) -> int:
    if successful(record) and record.get("first_success_step") is not None:
        return int(record["first_success_step"])
    return int(record["physics_steps"])


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("cannot summarize zero records")
    successes = [record for record in records if successful(record)]
    total_steps = sum(episode_metric_steps(record) for record in records)
    return {
        "episodes": len(records),
        "successes": len(successes),
        "success_rate": len(successes) / len(records),
        "successful_mean_first_success_steps": (
            None
            if not successes
            else statistics.fmean(episode_metric_steps(record) for record in successes)
        ),
        "total_episode_metric_steps": total_steps,
        "achieved_throughput_per_step": len(successes) / total_steps,
        "safety_violations": sum(record.get("safety_violation") is not None for record in records),
        "physics_errors": sum(record.get("physics_error") is not None for record in records),
    }


def gate_decision(summary: dict) -> str:
    if summary["physics_errors"]:
        return "halt_physics_error"
    if summary["safety_violations"]:
        return "reject_safety"
    episodes = summary["episodes"]
    successes = summary["successes"]
    thresholds = dict(STAGES)
    if episodes not in thresholds:
        raise ValueError(f"gate decision requires a staged count, got {episodes}")
    if successes < thresholds[episodes]:
        return "reject_reliability"
    return "qualified" if episodes == STAGES[-1][0] else "continue"


def earliest_failed_phase(records: list[dict], fallback: str) -> tuple[str, str]:
    phases = []
    for record in records:
        if successful(record):
            continue
        reached = [str(item.get("phase")) for item in record.get("phase_decisions", ())]
        indices = [PHASES.index(phase) for phase in reached if phase in PHASES]
        phases.append(PHASES[max(indices)] if indices else PHASES[0])
    if phases:
        phase = min(phases, key=PHASES.index)
        return phase, f"earliest implicated phase across {len(phases)} failed rollout(s)"
    return fallback, f"no failed rollout; semantic-risk fallback protects {fallback}"


def phase_workloads(record: dict) -> dict[str, float]:
    workloads = {phase: 0.0 for phase in PHASES}
    decisions = list(record.get("phase_decisions", ()))
    final_step = episode_metric_steps(record)
    for index, decision in enumerate(decisions):
        phase = str(decision.get("phase"))
        if phase not in workloads:
            continue
        start = min(int(decision["physics_step"]), final_step)
        end = min(
            int(decisions[index + 1]["physics_step"])
            if index + 1 < len(decisions)
            else final_step,
            final_step,
        )
        speed = float(decision.get("speed", 1.0))
        workloads[phase] += max(end - start, 0) * speed
    return workloads


def highest_workload_phase(records: list[dict], fallback: str) -> tuple[str, str]:
    usable = [record for record in records if successful(record)]
    if not usable:
        return fallback, f"no successful rollout; semantic-risk fallback protects {fallback}"
    means = {
        phase: statistics.fmean(phase_workloads(record)[phase] for record in usable)
        for phase in PHASES
    }
    phase = max(PHASES, key=lambda item: (means[item], -PHASES.index(item)))
    return phase, f"largest mean native-equivalent workload ({means[phase]:.3f})"


def adjacent_speed(speed: float, direction: int) -> float:
    index = ALLOWED_SPEEDS.index(float(speed)) + direction
    if not 0 <= index < len(ALLOWED_SPEEDS):
        raise ValueError(f"no adjacent speed from {speed} in direction {direction}")
    return ALLOWED_SPEEDS[index]


def make_backoff(rejected_schedule, phase: str) -> list[float]:
    schedule = list(validate_schedule(rejected_schedule))
    schedule[PHASES.index(phase)] = adjacent_speed(schedule[PHASES.index(phase)], -1)
    return schedule


def make_promotion(incumbent_schedule, phase: str) -> list[float]:
    schedule = list(validate_schedule(incumbent_schedule))
    schedule[PHASES.index(phase)] = adjacent_speed(schedule[PHASES.index(phase)], 1)
    return schedule


def choose_uniform_incumbent(reports: list[dict]) -> dict | None:
    qualified = [report for report in reports if report["qualified"]]
    if not qualified:
        return None
    return max(
        qualified,
        key=lambda report: (
            report["summary"]["achieved_throughput_per_step"],
            report["summary"]["successes"],
        ),
    )


def bend_replaces_uniform(bend: dict, incumbent: dict | None) -> bool:
    if not bend["qualified"]:
        return False
    if incumbent is None:
        return True
    bend_summary = bend["summary"]
    incumbent_summary = incumbent["summary"]
    return (
        bend_summary["successes"] >= incumbent_summary["successes"]
        and bend_summary["achieved_throughput_per_step"]
        > incumbent_summary["achieved_throughput_per_step"]
    )


def pareto_names(summaries: dict[str, dict]) -> list[str]:
    frontier = []
    for name, candidate in summaries.items():
        dominated = False
        for other_name, other in summaries.items():
            if other_name == name:
                continue
            no_worse = (
                other["success_rate"] >= candidate["success_rate"]
                and other["throughput_delta_percent_vs_native"]
                >= candidate["throughput_delta_percent_vs_native"]
            )
            strictly_better = (
                other["success_rate"] > candidate["success_rate"]
                or other["throughput_delta_percent_vs_native"]
                > candidate["throughput_delta_percent_vs_native"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return sorted(frontier)


class RolloutLedger:
    def __init__(
        self,
        runtime,
        root: Path,
        search_seeds: list[int],
        final_seeds: list[int],
        *,
        record_search_telemetry: bool = False,
    ):
        self.runtime = runtime
        self.root = root
        self.search_seeds = search_seeds
        self.final_seeds = final_seeds
        self.record_search_telemetry = record_search_telemetry

    def _checked_record(self, path: Path, schedule: list[float], seed: int) -> dict:
        record = json.loads(path.read_text())
        if int(record.get("seed", -1)) != seed:
            raise RuntimeError(f"cached seed mismatch: {path}")
        if list(map(float, record.get("schedule", ()))) != schedule:
            raise RuntimeError(f"cached schedule mismatch: {path}")
        return record

    def search_rollouts_used(self) -> int:
        return len(list((self.root / "search" / "candidates").glob("*/states/*.json")))

    def evaluate_search(self, schedule, role: str) -> tuple[dict, list[dict]]:
        schedule = list(validate_schedule(schedule))
        candidate_hash = schedule_sha256(schedule)
        candidate_root = self.root / "search" / "candidates" / candidate_hash
        schedule_path = candidate_root / "SCHEDULE.json"
        schedule_receipt = {"schedule": schedule, "schedule_sha256": candidate_hash}
        if schedule_path.exists() and json.loads(schedule_path.read_text()) != schedule_receipt:
            raise RuntimeError(f"candidate schedule identity mismatch: {schedule_path}")
        write_json(schedule_path, schedule_receipt)
        records: list[dict] = []
        stage_receipts = []
        final_decision = None
        for target, _ in STAGES:
            for seed in self.search_seeds[len(records):target]:
                path = candidate_root / "states" / f"{seed}.json"
                if path.exists():
                    record = self._checked_record(path, schedule, seed)
                else:
                    if self.search_rollouts_used() >= SEARCH_BUDGET:
                        raise RuntimeError("STRIDER search rollout budget exhausted")
                    if self.record_search_telemetry:
                        record = self.runtime.rollout(
                            schedule,
                            seed,
                            record_attribution_telemetry=True,
                        )
                    else:
                        record = self.runtime.rollout(schedule, seed)
                    if list(map(float, record.get("schedule", ()))) != schedule:
                        raise RuntimeError("runtime returned a different schedule")
                    write_json(path, record)
                records.append(record)
            stage_summary = summarize(records)
            decision = gate_decision(stage_summary)
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
                str(candidate_root / "states" / f"{record['seed']}.json") for record in records
            ],
        }
        write_json(candidate_root / "SUMMARY.json", report)
        if final_decision == "halt_physics_error":
            raise RuntimeError(f"physics error in candidate {schedule}")
        return report, records

    def evaluate_final(self, schedule) -> tuple[dict, list[dict]]:
        schedule = list(validate_schedule(schedule))
        controller_hash = schedule_sha256(schedule)
        controller_root = self.root / "final" / "controllers" / controller_hash
        schedule_receipt = {"schedule": schedule, "schedule_sha256": controller_hash}
        schedule_path = controller_root / "SCHEDULE.json"
        if schedule_path.exists() and json.loads(schedule_path.read_text()) != schedule_receipt:
            raise RuntimeError(f"final controller identity mismatch: {schedule_path}")
        write_json(schedule_path, schedule_receipt)
        records = []
        for seed in self.final_seeds:
            path = controller_root / "states" / f"{seed}.json"
            if path.exists():
                record = self._checked_record(path, schedule, seed)
            else:
                record = self.runtime.rollout(schedule, seed)
                if list(map(float, record.get("schedule", ()))) != schedule:
                    raise RuntimeError("runtime returned a different final schedule")
                write_json(path, record)
            records.append(record)
        result = {
            "schedule": schedule,
            "schedule_sha256": controller_hash,
            "summary": summarize(records),
        }
        write_json(controller_root / "SUMMARY.json", result)
        return result, records


def run_search(ledger: RolloutLedger, task_label: str) -> dict:
    protected_phase = {
        "pick": "grasp_lift",
        "tea": "grasp_lift",
        "insertion": "interaction",
    }[task_label]
    chronology = []
    uniform_reports = []
    report_2x, records_2x = ledger.evaluate_search([2.0] * 4, "uniform_anchor")
    uniform_reports.append(report_2x)
    chronology.append(report_2x["schedule_sha256"])
    rejected_report = None
    rejected_records = None
    if report_2x["qualified"]:
        for speed in (2.5, 3.0):
            report, records = ledger.evaluate_search([speed] * 4, "uniform_ladder")
            uniform_reports.append(report)
            chronology.append(report["schedule_sha256"])
            if not report["qualified"]:
                rejected_report, rejected_records = report, records
                break
    else:
        rejected_report, rejected_records = report_2x, records_2x
        report_15, _ = ledger.evaluate_search([1.5] * 4, "uniform_fallback")
        uniform_reports.append(report_15)
        chronology.append(report_15["schedule_sha256"])

    incumbent = choose_uniform_incumbent(uniform_reports)
    bend_report = None
    attribution = None
    if rejected_report is not None:
        phase, evidence = earliest_failed_phase(rejected_records or [], protected_phase)
        bend_schedule = make_backoff(rejected_report["schedule"], phase)
        attribution = {
            "operation": "one_rung_backoff",
            "phase": phase,
            "evidence": evidence,
            "source_schedule": rejected_report["schedule"],
        }
    elif incumbent is not None:
        incumbent_records = []
        states_root = ledger.root / "search" / "candidates" / incumbent["schedule_sha256"] / "states"
        for seed in ledger.search_seeds[: incumbent["summary"]["episodes"]]:
            incumbent_records.append(json.loads((states_root / f"{seed}.json").read_text()))
        phase, evidence = highest_workload_phase(incumbent_records, protected_phase)
        bend_schedule = make_promotion(incumbent["schedule"], phase)
        attribution = {
            "operation": "one_rung_promotion",
            "phase": phase,
            "evidence": evidence,
            "source_schedule": incumbent["schedule"],
        }
    else:
        phase, evidence = earliest_failed_phase(rejected_records or [], protected_phase)
        bend_schedule = make_backoff(rejected_report["schedule"], phase)
        attribution = {
            "operation": "one_rung_backoff_without_qualified_uniform",
            "phase": phase,
            "evidence": evidence,
            "source_schedule": rejected_report["schedule"],
        }

    if schedule_sha256(bend_schedule) not in {report["schedule_sha256"] for report in uniform_reports}:
        bend_report, _ = ledger.evaluate_search(bend_schedule, "phase_conditioned_bend")
        chronology.append(bend_report["schedule_sha256"])

    if bend_report is not None and bend_replaces_uniform(bend_report, incumbent):
        selected_report = bend_report
        selection_reason = "bend matched incumbent reliability and improved failure-aware throughput"
    elif incumbent is not None:
        selected_report = incumbent
        selection_reason = "retained best qualified uniform incumbent"
    else:
        selected_report = {
            "role": "native_fallback",
            "schedule": [1.0] * 4,
            "schedule_sha256": schedule_sha256([1.0] * 4),
            "qualified": True,
            "summary": None,
        }
        selection_reason = "no accelerated schedule qualified; fail closed to native"

    qualified = {
        report["schedule_sha256"]: report["summary"]
        for report in uniform_reports + ([bend_report] if bend_report else [])
        if report["qualified"]
    }
    search_frontier = []
    for candidate_hash, candidate in qualified.items():
        dominated = any(
            other_hash != candidate_hash
            and other["success_rate"] >= candidate["success_rate"]
            and other["achieved_throughput_per_step"] >= candidate["achieved_throughput_per_step"]
            and (
                other["success_rate"] > candidate["success_rate"]
                or other["achieved_throughput_per_step"] > candidate["achieved_throughput_per_step"]
            )
            for other_hash, other in qualified.items()
        )
        if not dominated:
            search_frontier.append(candidate_hash)

    selection = {
        "schema": "act-strider-frontier-selection-v2",
        "task_label": task_label,
        "selected_schedule": selected_report["schedule"],
        "selected_schedule_sha256": selected_report["schedule_sha256"],
        "selected_role": selected_report["role"],
        "selection_reason": selection_reason,
        "uniform_incumbent_sha256": None if incumbent is None else incumbent["schedule_sha256"],
        "bend_replaced_uniform": bend_report is not None and selected_report is bend_report,
        "attribution": attribution,
        "uniform_reports": uniform_reports,
        "bend_report": bend_report,
        "chronology": chronology,
        "search_frontier_sha256": sorted(search_frontier),
        "search_rollouts": ledger.search_rollouts_used(),
        "search_budget": SEARCH_BUDGET,
    }
    if selection["search_rollouts"] > SEARCH_BUDGET:
        raise RuntimeError("search exceeded the frozen rollout budget")
    if incumbent is not None and selected_report is not incumbent and not bend_replaces_uniform(selected_report, incumbent):
        raise RuntimeError("selected bend does not preserve its uniform incumbent")
    return selection


def run_final(ledger: RolloutLedger, selection: dict) -> dict:
    named_schedules = {
        "native_1x": [1.0] * 4,
        "uniform_1p5x": [1.5] * 4,
        "uniform_2x": [2.0] * 4,
        "uniform_2p5x": [2.5] * 4,
        "uniform_3x": [3.0] * 4,
    }
    selected_schedule = selection["selected_schedule"]
    selected_hash = schedule_sha256(selected_schedule)
    unique_results = {}
    methods = {}
    for name, schedule in named_schedules.items():
        controller_hash = schedule_sha256(schedule)
        if controller_hash not in unique_results:
            unique_results[controller_hash], _ = ledger.evaluate_final(schedule)
        methods[name] = {**unique_results[controller_hash], "selected_by_strider": controller_hash == selected_hash}
    if selected_hash not in unique_results:
        unique_results[selected_hash], _ = ledger.evaluate_final(selected_schedule)
        methods["strider_selected"] = {**unique_results[selected_hash], "selected_by_strider": True}
    else:
        selected_name = next(name for name, value in methods.items() if value["schedule_sha256"] == selected_hash)
        methods["strider_selected"] = {
            **methods[selected_name],
            "alias_of": selected_name,
            "selected_by_strider": True,
        }

    native = methods["native_1x"]["summary"]
    native_throughput = native["achieved_throughput_per_step"]
    native_mean = native["successful_mean_first_success_steps"]
    for method in methods.values():
        candidate = method["summary"]
        candidate_mean = candidate["successful_mean_first_success_steps"]
        candidate["successful_rollout_speedup"] = (
            None if candidate_mean is None or native_mean is None else native_mean / candidate_mean
        )
        candidate["throughput_delta_percent_vs_native"] = 100.0 * (
            candidate["achieved_throughput_per_step"] / native_throughput - 1.0
        )

    unique_methods = {
        name: method["summary"]
        for name, method in methods.items()
        if name != "strider_selected"
    }
    if "strider_selected" in methods and "alias_of" not in methods["strider_selected"]:
        unique_methods["strider_selected"] = methods["strider_selected"]["summary"]
    frontier = pareto_names(unique_methods)
    selected_frontier_name = (
        methods["strider_selected"].get("alias_of", "strider_selected")
    )
    return {
        "methods": methods,
        "empirical_frontier": frontier,
        "selected_on_empirical_frontier": selected_frontier_name in frontier,
        "selected_empirical_frontier_name": selected_frontier_name,
        "unique_controllers_evaluated": len(unique_results),
        "new_final_rollouts": len(unique_results) * len(ledger.final_seeds),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--search-seeds", type=comma_ints, required=True)
    parser.add_argument("--final-seeds", type=comma_ints, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if len(args.search_seeds) != 15 or len(set(args.search_seeds)) != 15:
        raise ValueError("STRIDER v2 requires fifteen unique search seeds")
    if len(args.final_seeds) != 50 or len(set(args.final_seeds)) != 50:
        raise ValueError("STRIDER v2 requires fifty unique final seeds")
    if set(args.search_seeds) & set(args.final_seeds):
        raise ValueError("search and final banks must be disjoint")

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
        "schema": "act-strider-frontier-identity-v2",
        "method": "strider_uniform_anchored_phase_frontier",
        "contract_sha256": file_sha256(args.contract),
        "search_seeds": args.search_seeds,
        "final_seeds": args.final_seeds,
        "search_budget": SEARCH_BUDGET,
        "stages": [{"episodes": count, "minimum_successes": minimum} for count, minimum in STAGES],
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("STRIDER v2 root identity mismatch")
    write_json(identity_path, identity)

    ledger = RolloutLedger(runtime, root, args.search_seeds, args.final_seeds)
    selection = run_search(ledger, args.task_label)
    selection_path = root / "SELECTION.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("sealed STRIDER selection changed during resume")
    write_json(selection_path, selection)
    selection_sha256 = file_sha256(selection_path)

    final = run_final(ledger, selection)
    result = {
        "schema": "act-strider-frontier-result-v2",
        "task_label": args.task_label,
        "identity_sha256": file_sha256(identity_path),
        "selection_sha256_before_final": selection_sha256,
        "selection": selection,
        "final": final,
        "accounting": {
            "search_rollouts": ledger.search_rollouts_used(),
            "search_budget": SEARCH_BUDGET,
            "new_final_rollouts": final["new_final_rollouts"],
            "total_new_rollouts": ledger.search_rollouts_used() + final["new_final_rollouts"],
            "final_bank_opened_only_after_selection": True,
        },
    }
    result_path = root / "RESULT.json"
    write_json(result_path, result)
    completion = {
        "schema": "act-strider-frontier-completion-v2",
        "identity_sha256": file_sha256(identity_path),
        "selection_sha256": selection_sha256,
        "result_sha256": file_sha256(result_path),
        **result["accounting"],
    }
    write_json(root / "COMPLETE.json", completion)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
