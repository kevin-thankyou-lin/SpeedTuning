#!/usr/bin/env python3
"""Run STRIDER with blinded VLM failure attribution on all ACT tasks."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from scripts import run_act_strider_frontier_v3 as telemetry
from scripts import run_act_strider_frontier_v4 as v4
from scripts.codex_agent_failure_attribution import CodexExchangeAttributor
from scripts.qwen_vlm_failure_attribution import (
    QwenVideoAttributor,
    aggregate_attributions,
)


PHASES = v4.PHASES
STAGES = v4.STAGES
SEARCH_BUDGET = 120
SEARCH_VALID_TARGET = 20
FINAL_VALID_TARGET = 50
MAX_ATTRIBUTION_PAIRS = 3
UNIFORM_LADDER = (2.0, 2.5, 3.0, 3.5)
VLM_LOCK = Path("/tmp/strider-qwen-v10.lock")


def simulator_valid(record: dict) -> bool:
    return record.get("physics_error") is None


def checked_video_record(
    path: Path, video_path: Path, schedule: list[float], seed: int
) -> dict:
    record = json.loads(path.read_text())
    if int(record.get("seed", -1)) != seed:
        raise RuntimeError(f"cached seed mismatch: {path}")
    if list(map(float, record.get("schedule", ()))) != schedule:
        raise RuntimeError(f"cached schedule mismatch: {path}")
    if simulator_valid(record):
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError(f"valid cached rollout lacks video: {path}")
        if record.get("video_sha256") != v4.file_sha256(video_path):
            raise RuntimeError(f"cached video hash mismatch: {video_path}")
    return record


class ValidVideoLedger:
    """Receipt ledger with video evidence and registered QACC replacement."""

    def __init__(
        self,
        runtime,
        root: Path,
        search_seed_pool: list[int],
        final_seed_pool: list[int],
    ):
        self.runtime = runtime
        self.root = root
        self.search_seed_pool = search_seed_pool
        self.final_seed_pool = final_seed_pool

    def _run_or_load(
        self,
        base: Path,
        schedule: list[float],
        seed: int,
        *,
        telemetry_enabled: bool,
    ) -> tuple[dict, bool]:
        state_path = base / "states" / f"{seed}.json"
        video_path = base / "videos" / f"{seed}.mp4"
        if state_path.exists():
            return checked_video_record(state_path, video_path, schedule, seed), False
        if video_path.exists():
            raise RuntimeError(f"unreceipted video requires audit: {video_path}")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        record = self.runtime.rollout(
            schedule,
            seed,
            video_path=video_path,
            record_attribution_telemetry=telemetry_enabled,
        )
        if list(map(float, record.get("schedule", ()))) != schedule:
            raise RuntimeError("runtime returned a different schedule")
        if simulator_valid(record):
            if not video_path.is_file() or video_path.stat().st_size <= 0:
                raise RuntimeError("simulator-valid rollout lacks its evidence video")
            record = {
                **record,
                "video_sha256": v4.file_sha256(video_path),
                "video_bytes": video_path.stat().st_size,
            }
        else:
            record = {
                **record,
                "simulator_invalid": True,
                "counted_in_scientific_denominator": False,
                "video_missing_allowed_for_physics_error": not video_path.is_file(),
            }
        v4.write_json(state_path, record)
        return record, True

    def search_valid_rollouts_used(self) -> int:
        total = 0
        for path in (self.root / "search" / "candidates").glob("*/states/*.json"):
            total += int(simulator_valid(json.loads(path.read_text())))
        return total

    def search_attempts_used(self) -> int:
        return len(
            list((self.root / "search" / "candidates").glob("*/states/*.json"))
        )

    def evaluate_search(self, schedule, role: str) -> tuple[dict, list[dict]]:
        schedule = list(v4.validate_schedule(schedule))
        schedule_hash = v4.schedule_sha256(schedule)
        candidate_root = self.root / "search" / "candidates" / schedule_hash
        schedule_path = candidate_root / "SCHEDULE.json"
        receipt = {"schedule": schedule, "schedule_sha256": schedule_hash}
        if schedule_path.exists() and json.loads(schedule_path.read_text()) != receipt:
            raise RuntimeError("candidate schedule identity mismatch")
        v4.write_json(schedule_path, receipt)

        final_decision = None
        valid_records: list[dict] = []
        invalid_records: list[dict] = []
        stages = []
        for target, minimum in STAGES:
            valid_records = []
            invalid_records = []
            for seed in self.search_seed_pool:
                if len(valid_records) >= target:
                    break
                path = candidate_root / "states" / f"{seed}.json"
                if not path.exists() and self.search_valid_rollouts_used() >= SEARCH_BUDGET:
                    raise RuntimeError("STRIDER VLM search valid-rollout budget exhausted")
                record, _ = self._run_or_load(
                    candidate_root,
                    schedule,
                    seed,
                    telemetry_enabled=True,
                )
                if simulator_valid(record):
                    valid_records.append(record)
                else:
                    invalid_records.append(record)
            if len(valid_records) != target:
                raise RuntimeError("search reserve pool exhausted before staged target")
            summary = v4.summarize(valid_records)
            if summary["safety_violations"]:
                decision = "reject_safety"
            elif summary["successes"] < minimum:
                decision = "reject_reliability"
            else:
                decision = "qualified" if target == SEARCH_VALID_TARGET else "continue"
            stage = {
                "target_valid_rollouts": target,
                "minimum_successes": minimum,
                "decision": decision,
                "summary": summary,
                "simulator_invalid_attempts": len(invalid_records),
            }
            stages.append(stage)
            v4.write_json(candidate_root / f"GATE_{target}.json", stage)
            if decision != "continue":
                final_decision = decision
                break
        if final_decision is None:
            raise RuntimeError("candidate did not reach a terminal gate decision")
        report = {
            "role": role,
            "schedule": schedule,
            "schedule_sha256": schedule_hash,
            "decision": final_decision,
            "qualified": final_decision == "qualified",
            "summary": v4.summarize(valid_records),
            "valid_rollouts": len(valid_records),
            "simulator_invalid_attempts": len(invalid_records),
            "stages": stages,
        }
        v4.write_json(candidate_root / "SUMMARY.json", report)
        return report, valid_records

    def evaluate_final_paired(self, named_schedules: dict[str, list[float]]) -> dict:
        unique = {}
        names_by_hash = {}
        for name, schedule in named_schedules.items():
            checked = list(v4.validate_schedule(schedule))
            schedule_hash = v4.schedule_sha256(checked)
            unique.setdefault(schedule_hash, checked)
            names_by_hash.setdefault(schedule_hash, []).append(name)

        valid = {schedule_hash: [] for schedule_hash in unique}
        valid_seeds = []
        invalid_pairs = []
        new_attempts = 0
        for seed in self.final_seed_pool:
            if len(valid_seeds) >= FINAL_VALID_TARGET:
                break
            pair = {}
            errors = []
            for schedule_hash, schedule in unique.items():
                root = self.root / "final" / "controllers" / schedule_hash
                record, ran = self._run_or_load(
                    root, schedule, seed, telemetry_enabled=False
                )
                new_attempts += int(ran)
                pair[schedule_hash] = record
                if not simulator_valid(record):
                    errors.append(
                        {
                            "schedule_sha256": schedule_hash,
                            "physics_error": record.get("physics_error"),
                        }
                    )
            if errors:
                invalid_pairs.append(
                    {
                        "seed": seed,
                        "reason": "physics_error",
                        "details": errors,
                        "counted_in_scientific_denominator": False,
                    }
                )
                continue
            valid_seeds.append(seed)
            for schedule_hash, record in pair.items():
                valid[schedule_hash].append(record)
        if len(valid_seeds) != FINAL_VALID_TARGET:
            raise RuntimeError("final reserve pool exhausted before 50 valid pairs")

        by_hash = {}
        for schedule_hash, records in valid.items():
            by_hash[schedule_hash] = {
                "schedule": unique[schedule_hash],
                "schedule_sha256": schedule_hash,
                "summary": v4.summarize(records),
                "valid_pair_seeds": valid_seeds,
            }
        methods = {}
        for schedule_hash, names in names_by_hash.items():
            for index, name in enumerate(names):
                methods[name] = {
                    **by_hash[schedule_hash],
                    **({"alias_of": names[0]} if index else {}),
                }
        native = methods["native_1x"]["summary"]
        for method in methods.values():
            summary = method["summary"]
            summary["successful_rollout_speedup"] = (
                None
                if summary["successful_mean_first_success_steps"] is None
                or native["successful_mean_first_success_steps"] is None
                else native["successful_mean_first_success_steps"]
                / summary["successful_mean_first_success_steps"]
            )
            summary["throughput_delta_percent_vs_native"] = 100.0 * (
                summary["achieved_throughput_per_step"]
                / native["achieved_throughput_per_step"]
                - 1.0
            )
        return {
            "methods": methods,
            "valid_pair_seeds": valid_seeds,
            "simulator_invalid_pairs": invalid_pairs,
            "unique_controllers_evaluated": len(unique),
            "new_physical_attempts": new_attempts,
            "scientific_rollouts": len(unique) * FINAL_VALID_TARGET,
        }


def _can_add_full_candidate(ledger: ValidVideoLedger) -> bool:
    return ledger.search_valid_rollouts_used() + SEARCH_VALID_TARGET <= SEARCH_BUDGET


def _backoff(schedule: list[float], phase: str) -> list[float] | None:
    index = PHASES.index(phase)
    if schedule[index] == min(v4.ALLOWED_SPEEDS):
        return None
    return v4.make_backoff(schedule, phase)


def _diagnose_vlm(
    *,
    attributor: QwenVideoAttributor,
    task_label: str,
    rejected_records: list[dict],
    reference_records: list[dict],
    output_root: Path,
) -> tuple[dict, list[dict]]:
    references = {
        int(record["seed"]): record
        for record in reference_records
        if v4.base.successful(record)
    }
    receipts = []
    failures = []
    with VLM_LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        failures_to_diagnose = [
            candidate
            for candidate in rejected_records
            if not v4.base.successful(candidate)
        ][:MAX_ATTRIBUTION_PAIRS]
        for candidate in failures_to_diagnose:
            seed = int(candidate["seed"])
            reference = references.get(seed)
            if reference is None:
                failures.append({"seed": seed, "reason": "no_successful_same_seed_reference"})
                continue
            receipt_path = output_root / f"{seed}.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
            else:
                try:
                    receipt = attributor.diagnose(
                        task_label=task_label,
                        reference_record=reference,
                        candidate_record=candidate,
                        reference_video=Path(reference["video_path"]),
                        candidate_video=Path(candidate["video_path"]),
                    )
                    v4.write_json(receipt_path, receipt)
                except Exception as exc:
                    failures.append(
                        {"seed": seed, "reason": f"{type(exc).__name__}: {exc}"}
                    )
                    continue
            receipts.append(receipt)
        attributor.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
    fallback = telemetry.SEMANTIC_FALLBACK[task_label]
    aggregate = aggregate_attributions(
        [receipt["attribution"] for receipt in receipts], fallback
    )
    aggregate["unattributed_failures"] = failures
    return aggregate, receipts


def run_search(
    ledger: ValidVideoLedger,
    task_label: str,
    attributor: QwenVideoAttributor,
) -> dict:
    uniform_reports = []
    records_by_hash = {}
    rejected = rejected_records = None
    for speed in UNIFORM_LADDER:
        if not _can_add_full_candidate(ledger):
            break
        report, records = ledger.evaluate_search(
            [speed] * len(PHASES),
            "uniform_anchor" if speed == 2.0 else "uniform_ladder",
        )
        uniform_reports.append(report)
        records_by_hash[report["schedule_sha256"]] = records
        if not report["qualified"]:
            rejected, rejected_records = report, records
            break
    if uniform_reports and not uniform_reports[0]["qualified"] and _can_add_full_candidate(ledger):
        fallback, records = ledger.evaluate_search([1.5] * 4, "uniform_fallback")
        uniform_reports.append(fallback)
        records_by_hash[fallback["schedule_sha256"]] = records

    uniform_incumbent = v4.choose_uniform_incumbent(uniform_reports)
    adaptive_reports = []
    repair_reports = {}
    telemetry_receipt = None
    vlm_aggregate = None
    vlm_receipts = []
    if rejected is not None and uniform_incumbent is not None:
        reference_records = records_by_hash[uniform_incumbent["schedule_sha256"]]
        telemetry_phase, telemetry_evidence = telemetry.paired_failure_phase(
            rejected_records or [],
            reference_records,
            telemetry.SEMANTIC_FALLBACK[task_label],
        )
        telemetry_receipt = {
            "selected_phase": telemetry_phase,
            "evidence": telemetry_evidence,
        }
        vlm_aggregate, vlm_receipts = _diagnose_vlm(
            attributor=attributor,
            task_label=task_label,
            rejected_records=rejected_records or [],
            reference_records=reference_records,
            output_root=ledger.root / "search" / "vlm_attribution",
        )
        proposed = []
        for method, phase in (
            ("vlm_causal_repair", vlm_aggregate["selected_phase"]),
            ("telemetry_repair_control", telemetry_phase),
        ):
            schedule = _backoff(rejected["schedule"], phase)
            if schedule is None:
                repair_reports[method] = {
                    "not_run": True,
                    "reason": "attributed phase already at minimum speed",
                    "phase": phase,
                }
                continue
            schedule_hash = v4.schedule_sha256(schedule)
            prior = next(
                (
                    report
                    for report in adaptive_reports
                    if report["schedule_sha256"] == schedule_hash
                ),
                None,
            )
            if prior is not None:
                repair_reports[method] = {**prior, "alias_of": prior["role"]}
                continue
            if not _can_add_full_candidate(ledger):
                repair_reports[method] = {
                    "not_run": True,
                    "reason": "registered 120-valid-rollout budget exhausted",
                    "phase": phase,
                    "schedule": schedule,
                }
                continue
            report, records = ledger.evaluate_search(schedule, method)
            adaptive_reports.append(report)
            records_by_hash[schedule_hash] = records
            repair_reports[method] = report

    selectable = []
    if uniform_incumbent is not None:
        selectable.append(uniform_incumbent)
        selectable.extend(
            report
            for report in adaptive_reports
            if v4.adaptive_replaces_uniform(report, uniform_incumbent)
        )
    if selectable:
        selected = max(
            selectable,
            key=lambda report: (
                report["summary"]["achieved_throughput_per_step"],
                report["summary"]["successes"],
            ),
        )
    else:
        selected = {
            "role": "native_fallback",
            "schedule": [1.0] * 4,
            "schedule_sha256": v4.schedule_sha256([1.0] * 4),
            "qualified": True,
            "summary": None,
        }
    comparison = {
        "attribution_was_triggered": rejected is not None,
        "vlm_and_telemetry_agree": (
            None
            if vlm_aggregate is None or telemetry_receipt is None
            else vlm_aggregate["selected_phase"]
            == telemetry_receipt["selected_phase"]
        ),
        "distinct_repairs_tested": len(
            {
                report["schedule_sha256"]
                for report in adaptive_reports
            }
        ),
        "repair_reports": repair_reports,
    }
    return {
        "schema": "act-strider-vlm-selection-v10",
        "task_label": task_label,
        "selected_schedule": selected["schedule"],
        "selected_schedule_sha256": selected["schedule_sha256"],
        "selected_role": selected["role"],
        "uniform_incumbent": uniform_incumbent,
        "rejected_uniform": rejected,
        "uniform_reports": uniform_reports,
        "adaptive_reports": adaptive_reports,
        "telemetry_attribution": telemetry_receipt,
        "vlm_attribution": vlm_aggregate,
        "vlm_pair_receipts": vlm_receipts,
        "attribution_comparison": comparison,
        "search_valid_rollouts": ledger.search_valid_rollouts_used(),
        "search_physical_attempts": ledger.search_attempts_used(),
        "search_budget_valid_rollouts": SEARCH_BUDGET,
    }


def _range(spec: dict) -> list[int]:
    return list(range(int(spec["start"]), int(spec["start"]) + int(spec["count"])))


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--success-criterion", type=Path)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument(
        "--attribution-backend", choices=("qwen", "codex-agent"), default="qwen"
    )
    parser.add_argument("--qwen-model", type=Path)
    parser.add_argument("--codex-exchange-root", type=Path)
    parser.add_argument("--codex-model", default="gpt-5.6-sol")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    banks = json.loads(args.banks.read_text())
    task_banks = banks["tasks"][args.task_label]
    search_pool = _range(task_banks["search_primary"]) + _range(task_banks["search_reserve"])
    final_pool = _range(task_banks["final_primary"]) + _range(task_banks["final_reserve"])
    if len(set(search_pool + final_pool)) != len(search_pool) + len(final_pool):
        raise ValueError("search/final primary and reserve banks must be disjoint")
    if len(search_pool) < 40 or len(final_pool) < 70:
        raise ValueError("insufficient registered reserve seeds")

    # The frozen ACT run manifest predates the task-native success-checker
    # update in this source commit.  Pin the current source file explicitly for
    # every task: Pick and Insertion retain their existing task checks, while
    # Tea additionally verifies the center-inside-cup criterion receipt below.
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
    if args.attribution_backend == "qwen":
        if args.qwen_model is None:
            raise ValueError("Qwen attribution requires --qwen-model")
        attributor = QwenVideoAttributor(args.qwen_model, device=args.device)
        study_version = "v10"
    else:
        if args.codex_exchange_root is None:
            raise ValueError("Codex attribution requires --codex-exchange-root")
        attributor = CodexExchangeAttributor(
            args.codex_exchange_root,
            model=args.codex_model,
        )
        study_version = "v11"
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": f"act-strider-vlm-identity-{study_version}",
        "method": (
            "strider_blinded_codex_agent_causal_failure_attribution"
            if args.attribution_backend == "codex-agent"
            else "strider_blinded_vlm_causal_failure_attribution"
        ),
        "contract_sha256": v4.file_sha256(args.contract),
        "banks_sha256": v4.file_sha256(args.banks),
        "search_seed_pool": search_pool,
        "final_seed_pool": final_pool,
        "search_valid_rollout_budget": SEARCH_BUDGET,
        "search_stages": [
            {"valid_rollouts": count, "minimum_successes": minimum}
            for count, minimum in STAGES
        ],
        "physics_error_policy": "exclude_invalid_pair_and_use_registered_reserve",
        "phase_source": "learned_online_rgb_plus_robot_proprioception",
        "vlm": attributor.identity(),
        "maximum_matched_failure_pairs_per_rejected_schedule": MAX_ATTRIBUTION_PAIRS,
        "tea_success_criterion": criterion_receipt,
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("STRIDER VLM root identity mismatch")
    v4.write_json(identity_path, identity)

    ledger = ValidVideoLedger(runtime, root, search_pool, final_pool)
    selection = run_search(ledger, args.task_label, attributor)
    selection_path = root / "SELECTION.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("sealed selection changed during resume")
    v4.write_json(selection_path, selection)
    selection_hash = v4.file_sha256(selection_path)

    named = {
        "native_1x": [1.0] * 4,
        "uniform_incumbent": (
            [1.0] * 4
            if selection["uniform_incumbent"] is None
            else selection["uniform_incumbent"]["schedule"]
        ),
        "strider_selected": selection["selected_schedule"],
    }
    for name in ("vlm_causal_repair", "telemetry_repair_control"):
        report = selection["attribution_comparison"]["repair_reports"].get(name)
        if report and not report.get("not_run"):
            named[name] = report["schedule"]
    final = ledger.evaluate_final_paired(named)
    if v4.file_sha256(selection_path) != selection_hash:
        raise RuntimeError("selection changed after final bank opened")
    result = {
        "schema": f"act-strider-vlm-result-{study_version}",
        "task_label": args.task_label,
        "identity_sha256": v4.file_sha256(identity_path),
        "selection_sha256_before_final": selection_hash,
        "selection": selection,
        "final": final,
        "accounting": {
            "search_valid_rollouts": ledger.search_valid_rollouts_used(),
            "search_physical_attempts": ledger.search_attempts_used(),
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
            "schema": f"act-strider-vlm-completion-{study_version}",
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
