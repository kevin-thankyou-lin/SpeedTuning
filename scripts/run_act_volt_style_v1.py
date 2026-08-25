#!/usr/bin/env python3
"""Run a learned-phase, two-speed VOLT-style frozen-ACT baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_act_strider_frontier_v2 import (  # noqa: E402
    ALLOWED_SPEEDS,
    PHASES,
    SEARCH_BUDGET,
    RolloutLedger,
    bend_replaces_uniform,
    choose_uniform_incumbent,
    comma_ints,
    file_sha256,
    pareto_names,
    schedule_sha256,
    summarize,
    validate_schedule,
    write_json,
)


FAST_PHASES = ("pre_grasp", "transport")
SLOW_PHASES = ("grasp_lift", "interaction")
SHARED_FIXED = {
    "native_1x": [1.0] * 4,
    "uniform_1p5x": [1.5] * 4,
    "uniform_2x": [2.0] * 4,
    "uniform_2p5x": [2.5] * 4,
    "uniform_3x": [3.0] * 4,
}


def adjacent_speed(speed: float, direction: int) -> float:
    index = ALLOWED_SPEEDS.index(float(speed)) + direction
    if not 0 <= index < len(ALLOWED_SPEEDS):
        raise ValueError(f"no adjacent speed from {speed} in direction {direction}")
    return ALLOWED_SPEEDS[index]


def binary_schedule(*, fast_speed: float, slow_speed: float) -> list[float]:
    if fast_speed not in ALLOWED_SPEEDS or slow_speed not in ALLOWED_SPEEDS:
        raise ValueError("VOLT-style speeds must use the frozen speed grid")
    if slow_speed > fast_speed:
        raise ValueError("slow speed cannot exceed fast speed")
    return [
        fast_speed if phase in FAST_PHASES else slow_speed
        for phase in PHASES
    ]


def run_search(ledger: RolloutLedger, task_label: str) -> dict:
    del task_label
    chronology = []
    uniform_reports = []
    rejected = None

    anchor, _ = ledger.evaluate_search([2.0] * 4, "uniform_anchor")
    uniform_reports.append(anchor)
    chronology.append(anchor["schedule_sha256"])
    if anchor["qualified"]:
        for speed in (2.5, 3.0):
            report, _ = ledger.evaluate_search([speed] * 4, "uniform_ladder")
            uniform_reports.append(report)
            chronology.append(report["schedule_sha256"])
            if not report["qualified"]:
                rejected = report
                break
    else:
        rejected = anchor
        fallback, _ = ledger.evaluate_search([1.5] * 4, "uniform_fallback")
        uniform_reports.append(fallback)
        chronology.append(fallback["schedule_sha256"])

    incumbent = choose_uniform_incumbent(uniform_reports)
    if rejected is not None:
        fast_speed = float(rejected["schedule"][0])
        slow_speed = adjacent_speed(fast_speed, -1)
        operation = "backoff both learned critical phases by one rung"
        source_schedule = rejected["schedule"]
    elif incumbent is not None:
        slow_speed = float(incumbent["schedule"][0])
        fast_speed = adjacent_speed(slow_speed, 1)
        operation = "promote both learned noncritical phases by one rung"
        source_schedule = incumbent["schedule"]
    else:
        raise RuntimeError("VOLT-style search has neither rejected nor qualified uniform")

    candidate_schedule = binary_schedule(fast_speed=fast_speed, slow_speed=slow_speed)
    candidate = None
    if schedule_sha256(candidate_schedule) not in {
        report["schedule_sha256"] for report in uniform_reports
    }:
        candidate, _ = ledger.evaluate_search(candidate_schedule, "volt_binary_schedule")
        chronology.append(candidate["schedule_sha256"])

    if candidate is not None and bend_replaces_uniform(candidate, incumbent):
        selected = candidate
        selection_reason = (
            "two-speed schedule matched incumbent reliability and improved "
            "failure-aware throughput"
        )
    elif incumbent is not None:
        selected = incumbent
        selection_reason = "retained best qualified uniform incumbent"
    else:
        selected = {
            "role": "native_fallback",
            "schedule": [1.0] * 4,
            "schedule_sha256": schedule_sha256([1.0] * 4),
            "qualified": True,
            "summary": None,
        }
        selection_reason = "no accelerated schedule qualified; fail closed to native"

    if incumbent is not None and selected is not incumbent and not bend_replaces_uniform(selected, incumbent):
        raise RuntimeError("VOLT-style selection regressed its uniform incumbent")
    if ledger.search_rollouts_used() > SEARCH_BUDGET:
        raise RuntimeError("VOLT-style search exceeded its rollout budget")

    return {
        "schema": "act-volt-style-selection-v1",
        "method": "volt_style_frozen_policy_learned_phase",
        "selected_schedule": selected["schedule"],
        "selected_schedule_sha256": selected["schedule_sha256"],
        "selected_role": selected["role"],
        "selection_reason": selection_reason,
        "uniform_incumbent_sha256": None if incumbent is None else incumbent["schedule_sha256"],
        "binary_candidate_replaced_uniform": candidate is not None and selected is candidate,
        "binary_parameterization": {
            "fast_phases": list(FAST_PHASES),
            "slow_phases": list(SLOW_PHASES),
            "fast_speed": fast_speed,
            "slow_speed": slow_speed,
            "operation": operation,
            "source_schedule": source_schedule,
        },
        "uniform_reports": uniform_reports,
        "binary_candidate_report": candidate,
        "chronology": chronology,
        "search_rollouts": ledger.search_rollouts_used(),
        "search_budget": SEARCH_BUDGET,
    }


def wait_for_shared_complete(root: Path, wait_seconds: int) -> None:
    deadline = time.monotonic() + wait_seconds
    marker = root / "COMPLETE.json"
    while not marker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"shared STRIDER final did not complete: {root}")
        print(f"WAITING_FOR_SHARED_FINAL root={root}", flush=True)
        time.sleep(60)


def shared_records(
    root: Path,
    schedule: list[float],
    seeds: list[int],
    runtime_identity: dict,
) -> list[dict]:
    identity = json.loads((root / "IDENTITY.json").read_text())
    for key in ("task", "task_label", "run_manifest_sha256", "policy_artifacts", "detector"):
        if identity.get(key) != runtime_identity.get(key):
            raise RuntimeError(f"shared final runtime mismatch for {key}: {root}")
    if identity.get("final_seeds") != seeds:
        raise RuntimeError(f"shared final seed mismatch: {root}")
    schedule = list(validate_schedule(schedule))
    controller_root = root / "final" / "controllers" / schedule_sha256(schedule)
    schedule_receipt = json.loads((controller_root / "SCHEDULE.json").read_text())
    if schedule_receipt.get("schedule") != schedule:
        raise RuntimeError(f"shared schedule mismatch: {controller_root}")
    records = []
    for seed in seeds:
        path = controller_root / "states" / f"{seed}.json"
        record = json.loads(path.read_text())
        if record.get("seed") != seed or list(map(float, record.get("schedule", ()))) != schedule:
            raise RuntimeError(f"shared state identity mismatch: {path}")
        records.append(record)
    if len(records) != 50:
        raise RuntimeError("shared final must contain exactly fifty receipts")
    return records


def add_native_comparison(summary: dict, native: dict) -> dict:
    value = dict(summary)
    mean = value["successful_mean_first_success_steps"]
    native_mean = native["successful_mean_first_success_steps"]
    value["successful_rollout_speedup"] = (
        None if mean is None or native_mean is None else native_mean / mean
    )
    value["throughput_delta_percent_vs_native"] = 100.0 * (
        value["achieved_throughput_per_step"]
        / native["achieved_throughput_per_step"]
        - 1.0
    )
    return value


def run_final(
    ledger: RolloutLedger,
    selection: dict,
    shared_root: Path,
    runtime_identity: dict,
    wait_seconds: int,
) -> dict:
    selected_schedule = selection["selected_schedule"]
    selected_hash = schedule_sha256(selected_schedule)
    shared_hashes = {
        schedule_sha256(schedule): name for name, schedule in SHARED_FIXED.items()
    }

    local_selected = None
    if selected_hash not in shared_hashes:
        local_selected, _ = ledger.evaluate_final(selected_schedule)

    wait_for_shared_complete(shared_root, wait_seconds)
    shared_summaries = {
        name: summarize(shared_records(shared_root, schedule, ledger.final_seeds, runtime_identity))
        for name, schedule in SHARED_FIXED.items()
    }
    native = shared_summaries["native_1x"]
    methods = {
        name: {
            "schedule": SHARED_FIXED[name],
            "schedule_sha256": schedule_sha256(SHARED_FIXED[name]),
            "summary": add_native_comparison(value, native),
            "receipt_source": "shared_strider_v2_final",
        }
        for name, value in shared_summaries.items()
    }
    if selected_hash in shared_hashes:
        alias = shared_hashes[selected_hash]
        methods["volt_style"] = {
            **methods[alias],
            "alias_of": alias,
            "selected_by_volt_style": True,
        }
        frontier_name = alias
        new_final_rollouts = 0
    else:
        if local_selected is None:
            raise RuntimeError("missing local VOLT-style final")
        methods["volt_style"] = {
            **local_selected,
            "summary": add_native_comparison(local_selected["summary"], native),
            "receipt_source": "volt_style_v1_final",
            "selected_by_volt_style": True,
        }
        frontier_name = "volt_style"
        new_final_rollouts = 50

    unique = {
        name: value["summary"]
        for name, value in methods.items()
        if name != "volt_style"
    }
    if "alias_of" not in methods["volt_style"]:
        unique["volt_style"] = methods["volt_style"]["summary"]
    frontier = pareto_names(unique)
    return {
        "methods": methods,
        "empirical_frontier": frontier,
        "selected_on_empirical_frontier": frontier_name in frontier,
        "selected_empirical_frontier_name": frontier_name,
        "new_final_rollouts": new_final_rollouts,
        "shared_final_rollouts": 250,
        "shared_final_rollouts_reexecuted": 0,
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
    parser.add_argument("--shared-final-root", type=Path, required=True)
    parser.add_argument("--shared-wait-seconds", type=int, default=21600)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if len(args.search_seeds) != 15 or len(set(args.search_seeds)) != 15:
        raise ValueError("VOLT-style requires fifteen unique search seeds")
    if len(args.final_seeds) != 50 or len(set(args.final_seeds)) != 50:
        raise ValueError("VOLT-style requires fifty unique final seeds")
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
    runtime_identity = runtime.identity()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime_identity,
        "schema": "act-volt-style-identity-v1",
        "method": "volt_style_frozen_policy_learned_phase",
        "paper_faithful_volt": False,
        "contract_sha256": file_sha256(args.contract),
        "search_seeds": args.search_seeds,
        "final_seeds": args.final_seeds,
        "shared_final_root": str(args.shared_final_root.resolve()),
        "search_budget": SEARCH_BUDGET,
        "fast_phases": list(FAST_PHASES),
        "slow_phases": list(SLOW_PHASES),
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("VOLT-style root identity mismatch")
    write_json(identity_path, identity)

    ledger = RolloutLedger(runtime, root, args.search_seeds, args.final_seeds)
    selection = run_search(ledger, args.task_label)
    selection_path = root / "SELECTION.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("sealed VOLT-style selection changed during resume")
    write_json(selection_path, selection)
    selection_sha256 = file_sha256(selection_path)

    final = run_final(
        ledger,
        selection,
        args.shared_final_root.resolve(),
        runtime_identity,
        args.shared_wait_seconds,
    )
    result = {
        "schema": "act-volt-style-result-v1",
        "task_label": args.task_label,
        "identity_sha256": file_sha256(identity_path),
        "selection_sha256_before_final": selection_sha256,
        "selection": selection,
        "final": final,
        "accounting": {
            "search_rollouts": ledger.search_rollouts_used(),
            "search_budget": SEARCH_BUDGET,
            "new_final_rollouts": final["new_final_rollouts"],
            "shared_final_rollouts": final["shared_final_rollouts"],
            "shared_final_rollouts_reexecuted": 0,
            "total_new_rollouts": ledger.search_rollouts_used() + final["new_final_rollouts"],
            "final_bank_opened_only_after_selection": True,
        },
        "method_scope": (
            "VOLT-style binary learned-phase timing over a frozen ACT policy; "
            "not paper-faithful VOLT retraining"
        ),
    }
    result_path = root / "RESULT.json"
    write_json(result_path, result)
    write_json(
        root / "COMPLETE.json",
        {
            "schema": "act-volt-style-completion-v1",
            "identity_sha256": file_sha256(identity_path),
            "selection_sha256": selection_sha256,
            "result_sha256": file_sha256(result_path),
            **result["accounting"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
