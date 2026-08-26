#!/usr/bin/env python3
"""Run Tea STRIDER center-inside search with parallel final controllers."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts import run_act_strider_frontier_v4 as v4
from scripts import run_act_strider_tea_center_v7 as v7
from scripts import run_act_strider_tea_volume_v5 as implementation


FINAL_WORKERS = 4


def _final_worker(payload: dict) -> dict:
    from scripts.act_vlm_frontier_server import ACTFrontierRuntime

    runtime = ACTFrontierRuntime(
        source_commit=payload["source_commit"],
        run_manifest=Path(payload["run_manifest"]),
        task_label="tea",
        detector_checkpoint=Path(payload["detector_checkpoint"]),
        detector_source=Path(payload["detector_source"]),
        device=payload["device"],
        critical_source_overrides={"sim_tasks.py": payload["sim_tasks_sha256"]},
    )
    ledger = v4.RolloutLedger(
        runtime,
        Path(payload["root"]),
        [],
        list(payload["final_seeds"]),
    )
    result, _ = ledger.evaluate_final(payload["schedule"])
    return result


def _assemble_final(unique_results: dict[str, dict], selection: dict) -> dict:
    named_schedules = {
        "native_1x": [1.0] * 4,
        "uniform_1p5x": [1.5] * 4,
        "uniform_2x": [2.0] * 4,
        "uniform_2p5x": [2.5] * 4,
        "uniform_3x": [3.0] * 4,
    }
    selected_hash = v4.schedule_sha256(selection["selected_schedule"])
    methods = {}
    for name, schedule in named_schedules.items():
        controller_hash = v4.schedule_sha256(schedule)
        methods[name] = {
            **unique_results[controller_hash],
            "selected_by_strider": controller_hash == selected_hash,
        }
    if selected_hash not in {v4.schedule_sha256(s) for s in named_schedules.values()}:
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
        candidate = method["summary"]
        candidate_mean = candidate["successful_mean_first_success_steps"]
        candidate["successful_rollout_speedup"] = (
            None
            if candidate_mean is None or native_mean is None
            else native_mean / candidate_mean
        )
        candidate["throughput_delta_percent_vs_native"] = 100.0 * (
            candidate["achieved_throughput_per_step"] / native_throughput - 1.0
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
        "new_final_rollouts": len(unique_results) * 50,
        "parallel_final_workers": min(FINAL_WORKERS, len(unique_results)),
    }


def run_parallel_final(ledger, selection: dict) -> dict:
    schedules = [[speed] * 4 for speed in (1.0, 1.5, 2.0, 2.5, 3.0)]
    schedules.append(list(selection["selected_schedule"]))
    schedules_by_hash = {
        v4.schedule_sha256(schedule): schedule for schedule in schedules
    }
    runtime = ledger.runtime
    common = {
        "source_commit": runtime.source_commit,
        "run_manifest": str(runtime.run_manifest),
        "detector_checkpoint": runtime.detector["checkpoint_path"],
        "detector_source": runtime.detector["source_root"],
        "device": runtime.detector["device"],
        "sim_tasks_sha256": runtime.critical_source_hashes["sim_tasks.py"],
        "root": str(ledger.root),
        "final_seeds": list(ledger.final_seeds),
    }
    unique_results = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(FINAL_WORKERS, len(schedules_by_hash)),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(_final_worker, {**common, "schedule": schedule}): digest
            for digest, schedule in schedules_by_hash.items()
        }
        for future in as_completed(futures):
            digest = futures[future]
            result = future.result()
            if result["schedule_sha256"] != digest:
                raise RuntimeError("parallel final worker returned wrong controller")
            unique_results[digest] = result
    return _assemble_final(unique_results, selection)


def main() -> int:
    implementation.VERSION = 8
    implementation.SUCCESS_CRITERION_SCHEMA = "tea-cup-center-success-v1"
    implementation.METRIC_REGRESSION_SCHEMA = "tea-cup-center-semantic-regression-v1"
    implementation.METRIC_REGRESSION_SEEDS = ()
    implementation.run_metric_regression = v7.run_center_semantic_regression
    v4.run_final = run_parallel_final
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
