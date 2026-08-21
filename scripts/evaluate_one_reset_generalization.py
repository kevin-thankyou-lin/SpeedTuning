#!/usr/bin/env python3
"""Evaluate frozen one-reset VLM and tabular schedules on shared random states."""

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

from one_reset_phase_schedule import run_phase_schedule  # noqa: E402
from learned_phase_observation import LearnedPhaseEncoder  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def native_results(ledger: Path, seeds: list[int]) -> list[dict]:
    wanted = set(seeds)
    results = []
    for line in ledger.read_text().splitlines():
        item = json.loads(line)
        if item.get("event") != "rollout_complete" or item.get("role") != "final_native":
            continue
        if int(item["seed"]) in wanted:
            results.append(item)
    by_seed = {int(item["seed"]): item for item in results}
    if set(by_seed) != wanted:
        raise ValueError("native ledger does not exactly cover evaluation seeds")
    return [by_seed[seed] for seed in seeds]


def load_progress(path: Path, identity: dict) -> dict[int, dict]:
    if not path.exists():
        return {}
    progress = json.loads(path.read_text())
    if progress.get("identity") != identity:
        raise ValueError("evaluation checkpoint identity mismatch")
    return {int(item["seed"]): item for item in progress.get("rollouts", [])}


def write_progress(path: Path, identity: dict, seeds: list[int], completed: dict, terminal=False):
    write_json(
        path,
        {
            "schema": "one-reset-evaluation-progress-v1",
            "terminal": bool(terminal),
            "identity": identity,
            "completed_states": len(completed),
            "rollouts": [completed[value] for value in seeds if value in completed],
        },
    )


def summarize(schedule, native: list[dict], candidate: list[dict]) -> dict:
    native_success = [item for item in native if item["success"]]
    candidate_success = [item for item in candidate if item["success"]]
    native_mean = statistics.fmean(item["physics_steps"] for item in native_success)
    candidate_mean = (
        None
        if not candidate_success
        else statistics.fmean(item["physics_steps"] for item in candidate_success)
    )
    return {
        "schedule": list(schedule),
        "candidate_successes": len(candidate_success),
        "candidate_safety_violations": sum(
            item["safety_violation"] is not None for item in candidate
        ),
        "candidate_successful_mean_steps": candidate_mean,
        "native_successes": len(native_success),
        "native_successful_mean_steps": native_mean,
        "success_only_speedup": (
            None if candidate_mean is None else native_mean / candidate_mean
        ),
        "rollouts": candidate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--task", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--method", choices=("vlm", "tabular"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    task = contract["tasks"][args.task]
    detector = contract.get("phase_detector")
    if args.method == "vlm":
        selection = json.loads(Path(task["vlm_selection"]).read_text())
        schedule = selection["schedule"]
    else:
        checkpoint = json.loads(Path(task["tabular_checkpoint"]).read_text())
        schedule = checkpoint["schedule"]
    seeds = [int(seed) for seed in task["evaluation_seeds"]]
    native = native_results(Path(task["native_ledger"]), seeds)
    progress_path = args.output.with_suffix(args.output.suffix + ".progress.json")
    identity = {
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        "task": args.task,
        "method": args.method,
        "schedule": list(schedule),
        "evaluation_seeds": seeds,
        "phase_detector": detector,
    }
    completed = load_progress(progress_path, identity)
    for seed in seeds:
        if seed in completed:
            continue
        completed[seed] = run_phase_schedule(
            task["runtime_task"],
            schedule,
            seed,
            observation_encoder=(
                None if detector is None else LearnedPhaseEncoder(**detector)
            ),
        )
        write_progress(progress_path, identity, seeds, completed)
    candidate = [completed[seed] for seed in seeds]
    result = {
        "schema": "one-reset-generalization-result-v1",
        "task": args.task,
        "runtime_task": task["runtime_task"],
        "method": args.method,
        "learning_states": 1,
        "training_episode_budget": 50,
        "evaluation_states": 100,
        "phase_observation": "oracle" if detector is None else "learned_rgb_proprio",
        **summarize(schedule, native, candidate),
    }
    write_json(args.output, result)
    write_progress(progress_path, identity, seeds, completed, terminal=True)
    print(json.dumps({key: value for key, value in result.items() if key != "rollouts"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
