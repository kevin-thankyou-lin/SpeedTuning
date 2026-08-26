#!/usr/bin/env python3
"""Confirm a delayed Tea release, then run a video-complete final bank."""

from __future__ import annotations

import json

from scripts import run_act_strider_frontier_v4 as v4
from scripts import run_act_strider_tea_center_v7 as v7
from scripts import run_act_strider_tea_volume_v5 as implementation


NATIVE = [1.0] * 4
UNIFORM = [1.5] * 4
DELAYED_RELEASE = [1.5, 1.5, 1.5, 1.0]
PARENT_V8_RESULT_SHA256 = (
    "cade96d287276b7e5346901ed775f3f8cec495c70b7a88f456e08d97f6b77a31"
)
SENTINEL_RECEIPT_SHA256 = (
    "901c74d0fb21931db1e3c74439aec7b901625bcab576b4ef71546d4766686d89"
)


class VideoFinalLedger(v4.RolloutLedger):
    """Record media in the actual final episode instead of replaying afterward."""

    def evaluate_final(self, schedule):
        schedule = list(v4.validate_schedule(schedule))
        controller_hash = v4.schedule_sha256(schedule)
        controller_root = self.root / "final" / "controllers" / controller_hash
        schedule_receipt = {
            "schedule": schedule,
            "schedule_sha256": controller_hash,
        }
        schedule_path = controller_root / "SCHEDULE.json"
        if (
            schedule_path.exists()
            and json.loads(schedule_path.read_text()) != schedule_receipt
        ):
            raise RuntimeError(f"final controller identity mismatch: {schedule_path}")
        v4.write_json(schedule_path, schedule_receipt)

        records = []
        for seed in self.final_seeds:
            path = controller_root / "states" / f"{seed}.json"
            video_path = controller_root / "videos" / f"{seed}.mp4"
            if path.exists():
                record = self._checked_record(path, schedule, seed)
                if not video_path.is_file() or video_path.stat().st_size <= 0:
                    raise RuntimeError(f"final record lacks its actual video: {path}")
                if record.get("video_sha256") != v4.file_sha256(video_path):
                    raise RuntimeError(f"final video identity mismatch: {video_path}")
            else:
                if video_path.exists():
                    raise RuntimeError(
                        f"unreceipted final video requires audit before resume: {video_path}"
                    )
                record = self.runtime.rollout(schedule, seed, video_path=video_path)
                if list(map(float, record.get("schedule", ()))) != schedule:
                    raise RuntimeError("runtime returned a different final schedule")
                if not video_path.is_file() or video_path.stat().st_size <= 0:
                    raise RuntimeError(f"runtime did not write final video: {video_path}")
                record["video_sha256"] = v4.file_sha256(video_path)
                record["video_bytes"] = video_path.stat().st_size
                v4.write_json(path, record)
            records.append(record)
        result = {
            "schedule": schedule,
            "schedule_sha256": controller_hash,
            "summary": v4.summarize(records),
            "actual_episode_videos": len(records),
        }
        v4.write_json(controller_root / "SUMMARY.json", result)
        return result, records


def repair_replaces_uniform(repair: dict, uniform: dict, native: dict) -> bool:
    if not repair["qualified"]:
        return False
    repair_summary = repair["summary"]
    if repair_summary["safety_violations"] or repair_summary["physics_errors"]:
        return False
    if uniform["qualified"]:
        uniform_summary = uniform["summary"]
        return (
            repair_summary["successes"] >= uniform_summary["successes"] + 1
            and repair_summary["achieved_throughput_per_step"]
            >= (1.0 + v4.MIN_THROUGHPUT_GAIN)
            * uniform_summary["achieved_throughput_per_step"]
        )
    return (
        native["qualified"]
        and repair_summary["achieved_throughput_per_step"]
        > native["summary"]["achieved_throughput_per_step"]
    )


def run_v9_search(ledger: VideoFinalLedger, task_label: str) -> dict:
    if task_label != "tea":
        raise ValueError("Tea release v9 supports only tea")
    reports = []
    chronology = []
    for schedule, role in (
        (NATIVE, "native_reference"),
        (UNIFORM, "uniform_1p5_incumbent"),
        (DELAYED_RELEASE, "delayed_release_candidate"),
    ):
        report, _ = ledger.evaluate_search(schedule, role)
        reports.append(report)
        chronology.append(report["schedule_sha256"])
        if role == "native_reference" and not report["qualified"]:
            break

    native = reports[0]
    uniform = next(
        (report for report in reports if report["schedule"] == UNIFORM), None
    )
    repair = next(
        (report for report in reports if report["schedule"] == DELAYED_RELEASE),
        None,
    )
    if not native["qualified"]:
        selected = native
        reason = "native failed the strict reliability gate; stop acceleration"
    elif uniform is None:
        selected = native
        reason = "uniform incumbent was not evaluated"
    else:
        selected = uniform if uniform["qualified"] else native
        reason = (
            "retained qualified uniform 1.5x incumbent"
            if uniform["qualified"]
            else "uniform 1.5x failed the strict gate; retained native"
        )
        if repair is not None and repair_replaces_uniform(repair, uniform, native):
            selected = repair
            reason = (
                "delayed release lifted reliability and materially improved "
                "failure-aware throughput"
            )

    qualified = {
        report["schedule_sha256"]: report["summary"]
        for report in reports
        if report["qualified"]
    }
    frontier = []
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
            frontier.append(candidate_hash)

    return {
        "schema": "act-strider-tea-release-selection-v9",
        "task_label": "tea",
        "selected_schedule": selected["schedule"],
        "selected_schedule_sha256": selected["schedule_sha256"],
        "selected_role": selected["role"],
        "selection_reason": reason,
        "uniform_incumbent_sha256": (
            None if uniform is None else uniform["schedule_sha256"]
        ),
        "native_report": native,
        "uniform_reports": [] if uniform is None else [uniform],
        "adaptive_reports": [] if repair is None else [repair],
        "attribution_receipts": [
            {
                "operation": "one_rung_delayed_release_confirmation",
                "phase": "interaction",
                "source_schedule": UNIFORM,
                "proposed_schedule": DELAYED_RELEASE,
                "candidate_origin": "user_informed_posthoc_v8_diagnostic",
                "parent_v8_result_sha256": PARENT_V8_RESULT_SHA256,
                "sentinel_receipt_sha256": SENTINEL_RECEIPT_SHA256,
            }
        ],
        "chronology": chronology,
        "search_frontier_sha256": sorted(frontier),
        "search_rollouts": ledger.search_rollouts_used(),
        "search_budget": v4.SEARCH_BUDGET,
        "unused_budget": v4.SEARCH_BUDGET - ledger.search_rollouts_used(),
        "adaptive_minimum_throughput_gain": v4.MIN_THROUGHPUT_GAIN,
        "posthoc_development_iteration": True,
    }


def run_video_final(ledger: VideoFinalLedger, selection: dict) -> dict:
    named_schedules = {
        "native_1x": [1.0] * 4,
        "uniform_1p5x": [1.5] * 4,
        "uniform_2x": [2.0] * 4,
        "uniform_2p5x": [2.5] * 4,
        "uniform_3x": [3.0] * 4,
        "uniform_3p5x": [3.5] * 4,
    }
    selected_schedule = list(selection["selected_schedule"])
    selected_hash = v4.schedule_sha256(selected_schedule)
    selected_is_uniform = selected_hash in {
        v4.schedule_sha256(schedule) for schedule in named_schedules.values()
    }
    unique_results = {}
    methods = {}
    for name, schedule in named_schedules.items():
        controller_hash = v4.schedule_sha256(schedule)
        if controller_hash not in unique_results:
            unique_results[controller_hash], _ = ledger.evaluate_final(schedule)
        methods[name] = {
            **unique_results[controller_hash],
            "selected_by_strider": controller_hash == selected_hash,
        }
    if selected_hash not in unique_results:
        unique_results[selected_hash], _ = ledger.evaluate_final(selected_schedule)
        methods["strider_selected"] = {
            **unique_results[selected_hash],
            "selected_by_strider": True,
        }
    else:
        selected_name = next(
            name
            for name, method in methods.items()
            if method["schedule_sha256"] == selected_hash
        )
        methods["strider_selected"] = {
            **methods[selected_name],
            "alias_of": selected_name,
            "selected_by_strider": True,
        }

    native = methods["native_1x"]["summary"]
    native_throughput = native["achieved_throughput_per_step"]
    native_mean = native["successful_mean_first_success_steps"]
    for method in methods.values():
        summary = method["summary"]
        candidate_mean = summary["successful_mean_first_success_steps"]
        summary["successful_rollout_speedup"] = (
            None
            if candidate_mean is None or native_mean is None
            else native_mean / candidate_mean
        )
        summary["throughput_delta_percent_vs_native"] = 100.0 * (
            summary["achieved_throughput_per_step"] / native_throughput - 1.0
        )

    unique_methods = {
        name: method["summary"]
        for name, method in methods.items()
        if name != "strider_selected"
    }
    if "alias_of" not in methods["strider_selected"]:
        unique_methods["strider_selected"] = methods["strider_selected"]["summary"]
    frontier = v4.pareto_names(unique_methods)
    selected_frontier_name = methods["strider_selected"].get(
        "alias_of", "strider_selected"
    )
    return {
        "methods": methods,
        "empirical_frontier": frontier,
        "selected_on_empirical_frontier": selected_frontier_name in frontier,
        "selected_empirical_frontier_name": selected_frontier_name,
        "unique_controllers_evaluated": len(unique_results),
        "new_final_rollouts": len(unique_results) * len(ledger.final_seeds),
        "actual_episode_videos": len(unique_results) * len(ledger.final_seeds),
        "same_gpu_controller_concurrency": False,
        "final_controller_order": list(named_schedules)
        + ([] if selected_is_uniform else ["strider_selected"]),
    }


def main() -> int:
    implementation.VERSION = 9
    implementation.SUCCESS_CRITERION_SCHEMA = "tea-cup-center-success-v1"
    implementation.METRIC_REGRESSION_SCHEMA = (
        "tea-cup-center-semantic-regression-v1"
    )
    implementation.METRIC_REGRESSION_SEEDS = ()
    implementation.run_metric_regression = v7.run_center_semantic_regression
    v4.RolloutLedger = VideoFinalLedger
    v4.run_search = run_v9_search
    v4.run_final = run_video_final
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
