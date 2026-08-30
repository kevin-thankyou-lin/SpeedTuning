#!/usr/bin/env python3
"""Run an exact-25 orthogonal phase experiment and finite-horizon DP."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SPEEDTUNING_SPEED_VALUES", "1,1.5,2,2.5,3")

from act_speed_benchmark import canonical_sha256  # noqa: E402
from scripts import run_act_common_grid_strider_causal25_v32 as v32  # noqa: E402
from scripts import run_act_sail_warmstart_v33 as v33  # noqa: E402
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head  # noqa: E402
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256, summarize  # noqa: E402

TASKS = ("pick", "tea", "insertion")
PHASES = tuple(v32.PHASES)
GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
SEARCH_BUDGET = 25
FINAL_METHODS = ("native_1x", "v34_phase_only", "phase_dp")
MIN_VISITS = 3
WILSON_Z = 1.2815515655446004  # preregistered one-sided 80% lower bound


def schedule_sha256(schedule) -> str:
    return canonical_sha256(list(v32.validate_schedule(schedule)))


def orthogonal_schedules() -> list[list[float]]:
    """OA(25, 4, 5, 2): every phase pair sees every action pair once."""

    schedules = []
    for left in range(5):
        for right in range(5):
            levels = (left, right, (left + right) % 5, (left + 2 * right) % 5)
            schedules.append([GRID[index] for index in levels])
    return schedules


def validate_design(schedules: list[list[float]]) -> None:
    if schedules != orthogonal_schedules():
        raise RuntimeError("v35 schedules differ from the preregistered OA construction")
    if len(schedules) != SEARCH_BUDGET or len({tuple(x) for x in schedules}) != SEARCH_BUDGET:
        raise RuntimeError("v35 design must contain 25 unique schedules")
    for phase in range(len(PHASES)):
        counts = {speed: sum(row[phase] == speed for row in schedules) for speed in GRID}
        if set(counts.values()) != {5}:
            raise RuntimeError(f"v35 phase {phase} is not action balanced: {counts}")
    for first in range(len(PHASES)):
        for second in range(first + 1, len(PHASES)):
            pairs = {(row[first], row[second]) for row in schedules}
            if len(pairs) != 25:
                raise RuntimeError(f"v35 phase pair {first}/{second} is not orthogonal")


def metric_steps(record: dict) -> int:
    first = record.get("first_success_step")
    return int(record["physics_steps"] if first is None else first)


def successful(record: dict) -> bool:
    return (
        bool(record.get("success"))
        and record.get("physics_error") is None
        and record.get("safety_violation") is None
    )


def phase_segment(record: dict, phase_index: int) -> dict:
    """Extract the first monotone visit to a registered phase.

    Detector backtracking is ignored after the first matching entry. A phase
    progresses only when the immediately following registered phase is later
    observed; interaction progresses only on terminal task success.
    """

    decisions = list(record.get("phase_decisions", ()))
    previous_step = -1
    for item in decisions:
        step = int(item["physics_step"])
        if step < previous_step:
            raise RuntimeError("v35 phase decisions are not time ordered")
        previous_step = step
    phase = PHASES[phase_index]
    start_at = next(
        (index for index, item in enumerate(decisions) if str(item.get("phase")) == phase),
        None,
    )
    if start_at is None:
        return {"visited": False, "progressed": False, "duration_steps": None}
    start = min(int(decisions[start_at]["physics_step"]), metric_steps(record))
    if phase_index == len(PHASES) - 1:
        end = metric_steps(record)
        progressed = successful(record)
    else:
        target = PHASES[phase_index + 1]
        next_item = next(
            (item for item in decisions[start_at + 1 :] if str(item.get("phase")) == target),
            None,
        )
        progressed = next_item is not None
        end = metric_steps(record) if next_item is None else min(int(next_item["physics_step"]), metric_steps(record))
    return {
        "visited": True,
        "progressed": bool(progressed),
        "duration_steps": max(int(end) - int(start), 0),
    }


def wilson_lower(successes: int, visits: int, z: float = WILSON_Z) -> float:
    if visits <= 0:
        return 0.0
    p = successes / visits
    denominator = 1.0 + z * z / visits
    center = p + z * z / (2.0 * visits)
    radius = z * math.sqrt(p * (1.0 - p) / visits + z * z / (4.0 * visits * visits))
    return max(0.0, (center - radius) / denominator)


def estimate_model(records: list[dict]) -> dict:
    cells = {}
    for phase_index, phase in enumerate(PHASES):
        cells[phase] = {}
        for speed in GRID:
            assigned = [item for item in records if float(item["schedule"][phase_index]) == speed]
            observations = [phase_segment(item, phase_index) for item in assigned]
            visited = [item for item in observations if item["visited"]]
            successes = sum(item["progressed"] for item in visited)
            durations = [int(item["duration_steps"]) for item in visited]
            cells[phase][str(speed)] = {
                "assigned": len(assigned),
                "visits": len(visited),
                "progressions": successes,
                "posterior_mean_progress": (successes + 1.0) / (len(visited) + 2.0),
                "wilson80_lower_progress": wilson_lower(successes, len(visited)),
                "mean_phase_steps": statistics.fmean(durations) if durations else None,
            }
    return {
        "schema": "act-estimated-four-phase-mdp-v35",
        "state_order": list(PHASES),
        "actions": list(GRID),
        "transition": "advance_to_next_registered_phase_or_fail_terminally",
        "terminal_success": "task_success_after_interaction",
        "posterior": "Beta(1,1) mean plus one-sided 80% Wilson lower bound",
        "minimum_visits": MIN_VISITS,
        "cells": cells,
    }


def backward_induction(model: dict) -> dict:
    downstream_lower = 1.0
    downstream_mean = 1.0
    downstream_steps = 0.0
    chosen = [1.0] * len(PHASES)
    stages = []
    for phase_index in reversed(range(len(PHASES))):
        phase = PHASES[phase_index]
        actions = []
        for speed in GRID:
            cell = model["cells"][phase][str(speed)]
            eligible = int(cell["visits"]) >= MIN_VISITS and cell["mean_phase_steps"] is not None
            value = {
                "speed": speed,
                "eligible": eligible,
                "visits": cell["visits"],
                "progressions": cell["progressions"],
                "q_success_lower": float(cell["wilson80_lower_progress"]) * downstream_lower,
                "q_success_mean": float(cell["posterior_mean_progress"]) * downstream_mean,
                "q_expected_steps": (
                    None
                    if not eligible
                    else float(cell["mean_phase_steps"])
                    + float(cell["posterior_mean_progress"]) * downstream_steps
                ),
            }
            actions.append(value)
        eligible = [item for item in actions if item["eligible"]]
        fallback = not eligible
        if fallback:
            selected = next(item for item in actions if item["speed"] == 1.0)
            selected["selection_reason"] = "insufficient_phase_coverage_fail_closed_native"
            selected["q_expected_steps"] = 0.0 if selected["q_expected_steps"] is None else selected["q_expected_steps"]
        else:
            selected = max(
                eligible,
                key=lambda item: (
                    item["q_success_lower"],
                    item["q_success_mean"],
                    -float(item["q_expected_steps"]),
                    -item["speed"],
                ),
            )
            selected["selection_reason"] = "reliability_lower_bound_then_mean_then_expected_steps"
        chosen[phase_index] = float(selected["speed"])
        downstream_lower = float(selected["q_success_lower"])
        downstream_mean = float(selected["q_success_mean"])
        downstream_steps = float(selected["q_expected_steps"])
        stages.append({"phase": phase, "actions": actions, "selected_speed": selected["speed"], "fallback": fallback})
    stages.reverse()
    return {
        "schema": "act-finite-horizon-backward-induction-v35",
        "objective": "maximize_recursive_success_lower_bound_then_posterior_mean_then_minimize_expected_steps",
        "horizon": len(PHASES),
        "selected_schedule": chosen,
        "selected_schedule_sha256": schedule_sha256(chosen),
        "estimated_start_success_lower": downstream_lower,
        "estimated_start_success_mean": downstream_mean,
        "estimated_start_steps": downstream_steps,
        "stages": stages,
        "global_optimum_claimed": False,
        "qualification": "optimal_only_for_the_fitted_coarse_phase_mdp_and_preregistered_estimator",
    }


def search_identity(runtime, contract: Path, banks: Path, design: Path, task: str, seeds: list[int], final: list[int]) -> dict:
    value = {
        **runtime.identity(),
        "schema": "act-phase-dp-search-identity-v35",
        "contract_sha256": file_sha256(contract),
        "banks_sha256": file_sha256(banks),
        "design_sha256": file_sha256(design),
        "task_label": task,
        "search_seeds": seeds,
        "final_seeds_registered_unopened": final,
        "search_budget": SEARCH_BUDGET,
        "historical_outcomes_visible": False,
    }
    value["identity_sha256"] = canonical_sha256(value)
    return value


def run_search(runtime, root: Path, task: str, seeds: list[int], schedules: list[list[float]], identity: dict) -> None:
    output = root / "search" / task
    v33.immutable_or_verify(output / "IDENTITY.json", identity)
    selection_path = output / "SELECTION.json"
    complete_path = output / "SEARCH_COMPLETE.json"
    if complete_path.exists():
        if v33.checked_json(complete_path)["selection_sha256"] != file_sha256(selection_path):
            raise RuntimeError("v35 completed search hash mismatch")
        return
    records = []
    for slot, (seed, schedule) in enumerate(zip(seeds, schedules)):
        path = output / "states" / f"{slot:02d}-{seed}.json"
        if path.exists():
            record = v33.checked_json(path)
        else:
            record = runtime.rollout(schedule, seed, record_attribution_telemetry=False)
            if int(record.get("seed", -1)) != seed or list(map(float, record.get("schedule", ()))) != schedule:
                raise RuntimeError("v35 runtime returned a different search identity")
            record["identity_sha256"] = identity["identity_sha256"]
            record["design_slot"] = slot
            immutable_json(path, record)
        if (
            record.get("identity_sha256") != identity["identity_sha256"]
            or int(record.get("design_slot", -1)) != slot
            or int(record.get("seed", -1)) != seed
            or list(map(float, record.get("schedule", ()))) != schedule
        ):
            raise RuntimeError(f"v35 cached search state mismatch: {path}")
        records.append(record)
        atomic_json(output / "progress.json", {"task": task, "completed": len(records), "successes": sum(successful(x) for x in records)})
        print(json.dumps({"stage": "search", "task": task, "completed": len(records), "successes": sum(successful(x) for x in records)}), flush=True)
    incidents = {
        "physics_errors": sum(item.get("physics_error") is not None for item in records),
        "safety_violations": sum(item.get("safety_violation") is not None for item in records),
    }
    if any(incidents.values()):
        raise RuntimeError(f"v35 search incident requires halt: {incidents}")
    model = estimate_model(records)
    dp = backward_induction(model)
    selection = {
        "schema": "act-phase-dp-selection-v35",
        "task_label": task,
        "search_summary": summarize(records),
        "search_scientific_rollouts": len(records),
        "design_assignment_counts_per_phase_action": 5,
        "estimated_phase_mdp": model,
        "backward_induction": dp,
        "selected_schedule": dp["selected_schedule"],
        "selected_schedule_sha256": dp["selected_schedule_sha256"],
        "incident_totals": incidents,
        "historical_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    }
    v33.immutable_or_verify(selection_path, selection)
    v33.immutable_or_verify(
        complete_path,
        {
            "schema": "act-phase-dp-search-completion-v35",
            "task_label": task,
            "identity_sha256": file_sha256(output / "IDENTITY.json"),
            "selection_sha256": file_sha256(selection_path),
            "search_scientific_rollouts": len(records),
            **incidents,
            "historical_rollouts_reexecuted": 0,
            "final_bank_opened": False,
        },
    )


def require_all_search(root: Path) -> None:
    for task in TASKS:
        receipt = v33.checked_json(root / "search" / task / "SEARCH_COMPLETE.json")
        if int(receipt["search_scientific_rollouts"]) != SEARCH_BUDGET:
            raise RuntimeError(f"v35 search incomplete: {task}")


def method_schedule(root: Path, frozen: dict, task: str, method: str) -> tuple[list[float], str]:
    if method == "native_1x":
        return [1.0] * 4, "preregistered_native"
    if method == "v34_phase_only":
        source = frozen["tasks"][task]
        schedule = list(v32.validate_schedule(source["schedule"]))
        expected = source.get("source_schedule_sha256")
        if expected is not None and schedule_sha256(schedule) != expected:
            raise RuntimeError(f"frozen v34 schedule hash mismatch: {task}")
        return schedule, source["source_selection_sha256"]
    selection_path = root / "search" / task / "SELECTION.json"
    selection = v33.checked_json(selection_path)
    schedule = list(v32.validate_schedule(selection["selected_schedule"]))
    if schedule_sha256(schedule) != selection["selected_schedule_sha256"]:
        raise RuntimeError("v35 selection schedule hash mismatch")
    return schedule, file_sha256(selection_path)


def load_controller_states(directory: Path, seeds: list[int], identity_sha: str) -> list[dict]:
    records = []
    missing = False
    for seed in seeds:
        path = directory / "states" / f"{seed}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError("v35 controller states contain a non-contiguous suffix")
        record = v33.checked_json(path)
        if int(record.get("seed", -1)) != seed or record.get("identity_sha256") != identity_sha:
            raise RuntimeError(f"v35 final state identity mismatch: {path}")
        records.append(record)
    return records


def run_final(runtime, root: Path, task: str, method: str, seeds: list[int], banks_sha: str, frozen: dict) -> None:
    require_all_search(root)
    schedule, provenance = method_schedule(root, frozen, task, method)
    controller_hash = schedule_sha256(schedule)
    controller_root = root / "final" / task / "controllers" / controller_hash
    alias_root = root / "final" / task / "methods" / method
    alias_path = alias_root / "RESULT.json"
    if alias_path.exists():
        alias = v33.checked_json(alias_path)
        if alias.get("controller_sha256") != controller_hash:
            raise RuntimeError(f"v35 final method alias differs: {task}/{method}")
        complete = v33.checked_json(controller_root / "COMPLETE.json")
        if alias.get("controller_result_sha256") != complete.get("result_sha256"):
            raise RuntimeError(f"v35 final method receipt differs: {task}/{method}")
        return
    identity = {
        **runtime.identity(),
        "schema": "act-phase-dp-final-controller-identity-v35",
        "task_label": task,
        "schedule": schedule,
        "schedule_sha256": controller_hash,
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "banks_sha256": banks_sha,
        "search_or_tuning_permitted": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v33.immutable_or_verify(controller_root / "IDENTITY.json", identity)
    was_complete = (controller_root / "COMPLETE.json").exists()
    records = load_controller_states(controller_root, seeds, identity["identity_sha256"])
    for seed in seeds[len(records) :]:
        record = runtime.rollout(schedule, seed, record_attribution_telemetry=False)
        if int(record.get("seed", -1)) != seed or list(map(float, record.get("schedule", ()))) != schedule:
            raise RuntimeError("v35 final runtime returned a different controller")
        record["identity_sha256"] = identity["identity_sha256"]
        immutable_json(controller_root / "states" / f"{seed}.json", record)
        records.append(record)
        atomic_json(controller_root / "progress.json", {"task": task, "controller_sha256": controller_hash, "completed": len(records), "successes": sum(successful(x) for x in records)})
        print(json.dumps({"stage": "final", "task": task, "method": method, "completed": len(records), "successes": sum(successful(x) for x in records)}), flush=True)
    controller_result = {
        "schema": "act-phase-dp-final-controller-result-v35",
        "task_label": task,
        "schedule": schedule,
        "schedule_sha256": controller_hash,
        "episodes": len(records),
        "summary": summarize(records),
        "identity_sha256": identity["identity_sha256"],
    }
    v33.immutable_or_verify(controller_root / "RESULT.json", controller_result)
    v33.immutable_or_verify(controller_root / "COMPLETE.json", {"schema": "act-phase-dp-final-controller-completion-v35", "episodes": len(records), "result_sha256": file_sha256(controller_root / "RESULT.json"), "physics_errors": controller_result["summary"]["physics_errors"], "safety_violations": controller_result["summary"]["safety_violations"]})
    v33.immutable_or_verify(
        alias_path,
        {
            "schema": "act-phase-dp-final-method-result-v35",
            "task_label": task,
            "method": method,
            "controller_schedule": schedule,
            "controller_sha256": controller_hash,
            "controller_result_sha256": file_sha256(controller_root / "RESULT.json"),
            "controller_receipt": str(controller_root / "RESULT.json"),
            "selection_provenance": provenance,
            "controller_cache_hit": was_complete,
            "summary": controller_result["summary"],
        },
    )


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "final"), required=True)
    parser.add_argument("--method", choices=("phase_dp", *FINAL_METHODS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-source-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=TASKS, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--frozen-v34", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if git_head() != args.implementation_commit:
        raise RuntimeError("v35 checked-out source differs from implementation commit")
    banks = v33.checked_json(args.banks)
    design = v33.checked_json(args.design)
    frozen = v33.checked_json(args.frozen_v34)
    all_seeds = []
    for task_spec in banks["tasks"].values():
        if len(task_spec["search"]) != SEARCH_BUDGET or len(task_spec["final"]) != 50:
            raise RuntimeError("v35 banks must register 25 search and 50 final seeds per task")
        if set(task_spec["search"]) & set(task_spec["final"]):
            raise RuntimeError("v35 search and final banks overlap")
        all_seeds.extend(task_spec["search"] + task_spec["final"])
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("v35 task banks overlap")
    if design.get("phase_order") != list(PHASES) or design.get("speed_grid") != list(GRID):
        raise RuntimeError("v35 design phase or speed grid mismatch")
    spec = banks["tasks"][args.task_label]
    schedules = [list(map(float, item)) for item in design["schedules"]]
    validate_design(schedules)
    runtime = ACTFrontierRuntime(
        source_commit=args.base_source_commit,
        checkout_commit=args.implementation_commit,
        run_manifest=args.run_manifest,
        task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
    )
    root = args.root.resolve()
    if args.stage == "search":
        if args.method != "phase_dp":
            raise ValueError("v35 search requires phase_dp")
        identity = search_identity(runtime, args.contract, args.banks, args.design, args.task_label, list(map(int, spec["search"])), list(map(int, spec["final"])))
        run_search(runtime, root, args.task_label, list(map(int, spec["search"])), schedules, identity)
    else:
        run_final(runtime, root, args.task_label, args.method, list(map(int, spec["final"])), file_sha256(args.banks), frozen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
