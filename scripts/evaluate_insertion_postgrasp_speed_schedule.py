#!/usr/bin/env python3
"""Evaluate one R3-style speed candidate on the post-grasp Insertion policy."""

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

from ee_sim_env import make_ee_sim_env  # noqa: E402
from postgrasp_speed_schedule import (  # noqa: E402
    FixedSpeedSchedule,
    PostgraspLatchedSpeedSchedule,
    PostgraspScheduleConfig,
)
from scripted_policy import InsertionPolicy  # noqa: E402
from sim_tasks import TASK_SPECS  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def payload_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_rollout(seed: int, schedule, arm: str, controller_payload: dict) -> dict:
    env = make_ee_sim_env("insertion", render_images=False, seed=seed)
    timestep = env.reset()
    initial_state = np.asarray(timestep.observation["env_state"], dtype="<f8")
    policy = InsertionPolicy()
    reward_changes = []
    speed_trace = []
    speed_counts = Counter()
    last_reward = None
    first_success_step = None
    replan_alignment = None
    physics_step = 0

    while policy.step_count < TASK_SPECS["insertion"].episode_len:
        physics_step += 1
        speed = float(schedule.select_speed(timestep, policy, physics_step))
        speed_trace.append(
            {
                "physics_step": physics_step,
                "policy_time_before": float(policy.step_count),
                "observed_reward": int(timestep.reward or 0),
                "speed": speed,
            }
        )
        speed_counts[f"{speed:g}"] += 1
        had_replanned = bool(policy.replan_count)
        timestep = env.step(policy(timestep, step_inc=speed))
        reward = int(timestep.reward or 0)

        if not had_replanned and policy.replan_count:
            replan_alignment = {
                "physics_step": physics_step,
                "speed": speed,
                "policy_time_after": float(policy.step_count),
                "event": policy.replan_event,
            }
        if reward != last_reward:
            reward_changes.append(
                {
                    "physics_step": physics_step,
                    "policy_time": float(policy.step_count),
                    "reward": reward,
                }
            )
            last_reward = reward
        if reward == env.task.max_reward:
            first_success_step = physics_step
            break

    success = first_success_step is not None
    return {
        "arm": arm,
        "seed": seed,
        "success": success,
        "max_reward": max((item["reward"] for item in reward_changes), default=0),
        "target_reward": env.task.max_reward,
        "physics_steps": physics_step,
        "policy_time": float(policy.step_count),
        "first_success_physics_step": first_success_step,
        "terminal_reason": "success" if success else "policy_horizon",
        "initial_state_sha256": hashlib.sha256(initial_state.tobytes()).hexdigest(),
        "controller": controller_payload,
        "controller_sha256": payload_sha256(controller_payload),
        "mean_speed": float(policy.step_count / physics_step),
        "speed_counts": dict(sorted(speed_counts.items())),
        "speed_trace": speed_trace,
        "reward_changes": reward_changes,
        "replan_count": policy.replan_count,
        "replan_alignment": replan_alignment,
        "release_events": list(schedule.release_events),
        "downshift_events": list(schedule.downshift_events),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schedule",
        choices=("postgrasp", "fixed"),
        default="postgrasp",
    )
    parser.add_argument("--pre-replan-speed", type=float, default=2.0)
    parser.add_argument("--post-replan-speed", type=float, default=4.0)
    parser.add_argument("--release-stability", type=int, default=1)
    parser.add_argument("--fixed-speed", type=float, default=2.0)
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--stop-on-candidate-failure", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite")
    return args


def candidate_factory(args):
    if args.schedule == "fixed":
        payload = {"kind": "fixed", "speed": float(args.fixed_speed)}
        return lambda: FixedSpeedSchedule(args.fixed_speed), payload
    config = PostgraspScheduleConfig(
        pre_replan_speed=args.pre_replan_speed,
        post_replan_speed=args.post_replan_speed,
        release_stability=args.release_stability,
    )
    return lambda: PostgraspLatchedSpeedSchedule(config), config.payload()


def summarize(rollouts: list[dict]) -> dict:
    summary = {}
    for arm in sorted({item["arm"] for item in rollouts}):
        items = [item for item in rollouts if item["arm"] == arm]
        successful = [item for item in items if item["success"]]
        summary[arm] = {
            "episodes": len(items),
            "successes": len(successful),
            "success_rate": float(np.mean([item["success"] for item in items])),
            "mean_executed_physics_steps": float(
                np.mean([item["physics_steps"] for item in items])
            ),
            "mean_success_physics_steps": (
                float(np.mean([item["physics_steps"] for item in successful]))
                if successful
                else None
            ),
        }
    if "native_1x" in summary and "candidate" in summary:
        native = [item for item in rollouts if item["arm"] == "native_1x"]
        candidate = [item for item in rollouts if item["arm"] == "candidate"]
        if all(item["success"] for item in native):
            summary["duration_normalized_speedup"] = float(
                np.mean([item["physics_steps"] for item in native])
                / np.mean([item["physics_steps"] for item in candidate])
            )
    return summary


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True)
    make_candidate, candidate_payload = candidate_factory(args)
    native_payload = {"kind": "fixed", "speed": 1.0}
    rollouts = []

    for seed in args.seeds:
        if args.include_native:
            rollouts.append(
                run_rollout(
                    seed,
                    FixedSpeedSchedule(1.0),
                    "native_1x",
                    native_payload,
                )
            )
        candidate = run_rollout(
            seed,
            make_candidate(),
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
        REPO_ROOT / "ee_sim_env.py",
        REPO_ROOT / "sim_tasks.py",
        REPO_ROOT / "postgrasp_speed_schedule.py",
        Path(__file__).resolve(),
    ]
    result = {
        "schema": "speedtuning-insertion-postgrasp-r3-speed-candidate-v1",
        "task": "insertion",
        "partition": args.partition,
        "requested_seeds": args.seeds,
        "executed_seeds": sorted({item["seed"] for item in rollouts}),
        "candidate_controller": candidate_payload,
        "candidate_controller_sha256": payload_sha256(candidate_payload),
        "base_policy": {
            "name": "insertion-postgrasp-base-v1",
            "state_source": "privileged_sim_object_pose",
            "correction_mode": "translation_only_preserve_demonstrated_orientation",
        },
        "success_criterion": "reward == 4",
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
            "assets_tree_sha256": tree_sha256(REPO_ROOT / "assets"),
        },
    }
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    complete = args.output / "COMPLETE"
    complete.write_text(f"{sha256(result_path)}  results.json\n")
    sums = args.output / "SHA256SUMS"
    sums.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in (complete, result_path))
    )
    print(json.dumps({"summary": result["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
