#!/usr/bin/env python3
"""Profile observable Insertion behavior on retired fixed-2x rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ee_sim_env import make_ee_sim_env  # noqa: E402
from scripted_policy import InsertionPolicy  # noqa: E402
from sim_tasks import TASK_SPECS, contact_pairs  # noqa: E402
from behavior_speed_observation import (  # noqa: E402
    behavior_metrics,
    insertion_speed_observation,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_metrics(metrics: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metrics.items()
    }


def run(seed: int, speed: float) -> dict:
    env = make_ee_sim_env("insertion", render_images=False, seed=seed)
    timestep = env.reset()
    policy = InsertionPolicy()
    initial = insertion_speed_observation(timestep.observation)
    previous = None
    frames = []

    for physics_step in range(1, int(np.ceil(TASK_SPECS["insertion"].episode_len / speed)) + 1):
        timestep = env.step(policy(timestep, step_inc=speed))
        current = insertion_speed_observation(timestep.observation)
        metrics = behavior_metrics(current, initial, previous)
        frames.append(
            {
                "physics_step": physics_step,
                "reward_label_only": int(timestep.reward or 0),
                "metrics": json_metrics(metrics),
                "contact_pairs_label_only": [
                    list(pair) for pair in sorted(contact_pairs(env.physics))
                ],
            }
        )
        previous = current

    rewards = [frame["reward_label_only"] for frame in frames]
    milestones = {}
    for reward in (1, 2, 3, 4):
        match = next((frame for frame in frames if frame["reward_label_only"] >= reward), None)
        milestones[str(reward)] = match
    return {
        "seed": seed,
        "speed": speed,
        "success": max(rewards, default=0) == 4,
        "max_reward": max(rewards, default=0),
        "physics_steps": len(frames),
        "milestones": milestones,
        "frames": frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.speed <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("speed must be positive and seeds unique")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite")
    return args


def finite(values):
    return [float(value) for value in values if np.isfinite(value)]


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True)
    rollouts = [run(seed, args.speed) for seed in args.seeds]
    successes = [item for item in rollouts if item["success"]]

    reward2 = [item["milestones"]["2"] for item in successes]
    pre_reward3 = []
    for item in successes:
        milestone = item["milestones"]["3"]
        if milestone is None:
            continue
        index = max(0, milestone["physics_step"] - 3)
        pre_reward3.append(item["frames"][index])

    proposal = None
    if reward2 and pre_reward3:
        lift_values = [
            min(frame["metrics"]["object_lift_m"]) for frame in reward2
        ]
        translation_deltas = finite(
            value
            for item in successes
            for frame in item["frames"]
            if frame["reward_label_only"] >= 2
            for value in frame["metrics"]["object_effector_translation_delta_m"]
        )
        rotation_deltas = finite(
            value
            for item in successes
            for frame in item["frames"]
            if frame["reward_label_only"] >= 2
            for value in frame["metrics"]["object_rotation_delta_deg"]
        )
        terminal_distances = [
            frame["metrics"]["object_pair_distance_m"] for frame in pre_reward3
        ]
        grippers = [
            value
            for frame in reward2
            for value in frame["metrics"]["gripper_positions"]
        ]
        proposal = {
            "derivation": {
                "entry": "minimum successful reward-2 lift with conservative margin",
                "stability": "99th percentile post-lift paired relative motion",
                "exit": "maximum object distance two frames before first insertion contact",
                "reward_and_contacts_available_to_runtime": False,
            },
            "min_object_lift_m": float(max(0.005, min(lift_values) - 0.005)),
            "max_relative_translation_delta_m": float(
                max(0.001, np.percentile(translation_deltas, 99))
            ),
            "max_object_rotation_delta_deg": float(
                max(1.0, np.percentile(rotation_deltas, 99))
            ),
            "max_closed_gripper_position": float(max(grippers) + 0.05),
            "terminal_object_distance_m": float(max(terminal_distances) + 0.005),
            "stable_observations": 2,
            "protected_speed": 2.0,
            "fast_speed": 3.0,
        }

    source_files = [
        REPO_ROOT / "scripted_policy.py",
        REPO_ROOT / "ee_sim_env.py",
        REPO_ROOT / "sim_tasks.py",
        REPO_ROOT / "behavior_speed_observation.py",
        Path(__file__).resolve(),
    ]
    result = {
        "schema": "speedtuning-insertion-observable-region-profile-v1",
        "purpose": "retired-seed offline labeling only",
        "runtime_forbidden_inputs": [
            "base_policy_object",
            "base_policy_time",
            "base_policy_phase",
            "replan_count",
            "replan_event",
            "reward",
            "success_flag",
        ],
        "seeds": args.seeds,
        "speed": args.speed,
        "successes": len(successes),
        "episodes": len(rollouts),
        "proposed_observable_region": proposal,
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
    result_path = args.output / "profile.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    complete_path = args.output / "COMPLETE"
    complete_path.write_text(f"{sha256(result_path)}  profile.json\n")
    (args.output / "SHA256SUMS").write_text(
        f"{sha256(complete_path)}  COMPLETE\n"
        f"{sha256(result_path)}  profile.json\n"
    )
    print(json.dumps({key: result[key] for key in ("episodes", "successes", "proposed_observable_region")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
