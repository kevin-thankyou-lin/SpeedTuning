#!/usr/bin/env python3
"""Matched open-loop and post-grasp-replan Insertion evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ee_sim_env import make_ee_sim_env  # noqa: E402
from scripted_policy import InsertionPolicy  # noqa: E402
from sim_tasks import TASK_SPECS  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(seed: int, speed: float, adaptive: bool) -> dict:
    env = make_ee_sim_env("insertion", render_images=False, seed=seed)
    timestep = env.reset()
    initial_state = np.asarray(timestep.observation["env_state"], dtype="<f8")
    policy = InsertionPolicy(enable_postgrasp_replan=adaptive)
    rewards = []
    reward_changes = []
    last_reward = None
    first_success_step = None
    steps = math.ceil(TASK_SPECS["insertion"].episode_len / speed)

    for physics_step in range(1, steps + 1):
        timestep = env.step(policy(timestep, step_inc=speed))
        reward = int(timestep.reward or 0)
        rewards.append(reward)
        if reward != last_reward:
            reward_changes.append(
                {
                    "physics_step": physics_step,
                    "policy_time": float(policy.step_count),
                    "reward": reward,
                }
            )
            last_reward = reward
        if reward == env.task.max_reward and first_success_step is None:
            first_success_step = physics_step

    return {
        "arm": "postgrasp_replan" if adaptive else "frozen_open_loop",
        "seed": seed,
        "speed": speed,
        "success": max(rewards, default=0) == env.task.max_reward,
        "max_reward": max(rewards, default=0),
        "target_reward": env.task.max_reward,
        "physics_steps": steps,
        "first_success_physics_step": first_success_step,
        "return": sum(rewards),
        "initial_state_sha256": hashlib.sha256(initial_state.tobytes()).hexdigest(),
        "replan_count": policy.replan_count,
        "replan_event": policy.replan_event,
        "reward_changes": reward_changes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--speeds", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if len(set(args.speeds)) != len(args.speeds) or any(speed <= 0 for speed in args.speeds):
        parser.error("speeds must be unique and positive")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite")
    return args


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True)
    rollouts = [
        run(seed, speed, adaptive)
        for seed in args.seeds
        for speed in args.speeds
        for adaptive in (False, True)
    ]
    for item in rollouts:
        print(json.dumps(item, sort_keys=True), flush=True)

    grouped = {}
    for speed in args.speeds:
        for adaptive in (False, True):
            name = "postgrasp_replan" if adaptive else "frozen_open_loop"
            items = [
                item
                for item in rollouts
                if item["speed"] == speed and item["arm"] == name
            ]
            grouped[f"{name}_{speed:g}x"] = {
                "successes": sum(item["success"] for item in items),
                "episodes": len(items),
                "success_rate": float(np.mean([item["success"] for item in items])),
            }

    for seed in args.seeds:
        hashes = {
            item["initial_state_sha256"] for item in rollouts if item["seed"] == seed
        }
        if len(hashes) != 1:
            raise RuntimeError(f"seed {seed} did not produce matched initial states")

    source_files = [
        REPO_ROOT / "scripted_policy.py",
        REPO_ROOT / "ee_sim_env.py",
        REPO_ROOT / "sim_tasks.py",
        Path(__file__).resolve(),
    ]
    result = {
        "schema": "speedtuning-insertion-postgrasp-replan-v1",
        "task": "insertion",
        "state_source": "privileged_sim_object_pose",
        "correction_mode": "one_shot_translation_only_preserve_demonstrated_orientation",
        "seeds": args.seeds,
        "speeds": args.speeds,
        "rollout_count": len(rollouts),
        "summary": grouped,
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
    complete = args.output / "COMPLETE"
    complete.write_text(f"{sha256(result_path)}  results.json\n")
    sums = args.output / "SHA256SUMS"
    sums.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (complete, result_path)
        )
    )
    print(json.dumps({"summary": grouped}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
