#!/usr/bin/env python3
"""Evaluate an object/effector-observable Insertion speed region."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from behavior_speed_observation import insertion_speed_observation  # noqa: E402
from ee_sim_env import make_ee_sim_env  # noqa: E402
from observable_behavior_speed_selector import (  # noqa: E402
    BehaviorRegionConfig,
    FixedBehaviorSpeedSelector,
    ObservableBehaviorRegionSelector,
)
from scripted_policy import InsertionPolicy  # noqa: E402
from sim_tasks import TASK_SPECS  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run(seed: int, selector, arm: str, payload: dict) -> dict:
    env = make_ee_sim_env("insertion", render_images=False, seed=seed)
    timestep = env.reset()
    base = InsertionPolicy()
    initial_state = np.asarray(timestep.observation["env_state"], dtype="<f8")
    speed_counts = Counter()
    speed_trace = []
    reward_changes = []
    last_reward = None
    executed_progress = 0.0
    physics_step = 0

    while executed_progress < TASK_SPECS["insertion"].episode_len:
        physics_step += 1
        external_observation = insertion_speed_observation(timestep.observation)
        speed = float(selector.select_speed(external_observation))
        speed_counts[f"{speed:g}"] += 1
        speed_trace.append(
            {
                "physics_step": physics_step,
                "speed": speed,
                "object_pair_distance_m": float(
                    np.linalg.norm(
                        external_observation.object_poses[0, :3]
                        - external_observation.object_poses[1, :3]
                    )
                ),
            }
        )
        timestep = env.step(base(timestep, step_inc=speed))
        executed_progress += speed
        reward = int(timestep.reward or 0)
        if reward != last_reward:
            reward_changes.append(
                {
                    "physics_step": physics_step,
                    "executed_progress": executed_progress,
                    "reward_evaluation_only": reward,
                }
            )
            last_reward = reward
        if reward == env.task.max_reward:
            break

    success = bool(reward_changes and reward_changes[-1]["reward_evaluation_only"] == 4)
    return {
        "arm": arm,
        "seed": seed,
        "success": success,
        "max_reward": max(
            (item["reward_evaluation_only"] for item in reward_changes), default=0
        ),
        "physics_steps": physics_step,
        "executed_progress": executed_progress,
        "terminal_reason": "success" if success else "policy_horizon",
        "initial_state_sha256": hashlib.sha256(initial_state.tobytes()).hexdigest(),
        "controller": payload,
        "controller_sha256": payload_sha256(payload),
        "speed_counts": dict(sorted(speed_counts.items())),
        "speed_trace": speed_trace,
        "entry_events": selector.entry_events,
        "exit_events": selector.exit_events,
        "reward_changes_evaluation_only": reward_changes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--include-fixed2", action="store_true")
    parser.add_argument("--stop-on-candidate-failure", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite")
    return args


def summarize(rollouts: list[dict]) -> dict:
    summary = {}
    for arm in sorted({item["arm"] for item in rollouts}):
        items = [item for item in rollouts if item["arm"] == arm]
        summary[arm] = {
            "episodes": len(items),
            "successes": sum(item["success"] for item in items),
            "success_rate": float(np.mean([item["success"] for item in items])),
            "mean_executed_physics_steps": float(
                np.mean([item["physics_steps"] for item in items])
            ),
        }
    return summary


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True)
    document = json.loads(args.config.read_text())
    config = BehaviorRegionConfig(**document["controller"])
    candidate_payload = config.payload()
    fixed_payload = {"kind": "fixed_behavior", "speed": 2.0}
    rollouts = []

    for seed in args.seeds:
        if args.include_fixed2:
            rollouts.append(
                run(seed, FixedBehaviorSpeedSelector(2.0), "fixed_2x", fixed_payload)
            )
        candidate = run(
            seed,
            ObservableBehaviorRegionSelector(config),
            "candidate",
            candidate_payload,
        )
        rollouts.append(candidate)
        print(json.dumps(candidate, sort_keys=True), flush=True)
        if args.stop_on_candidate_failure and not candidate["success"]:
            break

    for seed in {item["seed"] for item in rollouts}:
        hashes = {
            item["initial_state_sha256"]
            for item in rollouts
            if item["seed"] == seed
        }
        if len(hashes) != 1:
            raise RuntimeError(f"seed {seed} did not produce matched initial states")

    source_files = [
        REPO_ROOT / "scripted_policy.py",
        REPO_ROOT / "behavior_speed_observation.py",
        REPO_ROOT / "observable_behavior_speed_selector.py",
        Path(__file__).resolve(),
        args.config.resolve(),
    ]
    result = {
        "schema": "speedtuning-insertion-observable-speed-region-eval-v1",
        "partition": args.partition,
        "requested_seeds": args.seeds,
        "executed_seeds": sorted({item["seed"] for item in rollouts}),
        "runtime_selector_inputs": candidate_payload["runtime_inputs"],
        "runtime_forbidden_inputs": candidate_payload["runtime_forbidden_inputs"],
        "candidate_controller": candidate_payload,
        "candidate_controller_sha256": payload_sha256(candidate_payload),
        "rollout_count": len(rollouts),
        "summary": summarize(rollouts),
        "rollouts": rollouts,
        "provenance": {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "source_sha256": {
                str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_files
            },
        },
    }
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    complete_path = args.output / "COMPLETE"
    complete_path.write_text(f"{sha256(result_path)}  results.json\n")
    (args.output / "SHA256SUMS").write_text(
        f"{sha256(complete_path)}  COMPLETE\n"
        f"{sha256(result_path)}  results.json\n"
    )
    print(json.dumps({"summary": result["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
