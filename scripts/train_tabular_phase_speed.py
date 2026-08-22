#!/usr/bin/env python3
"""Train a tiny phase-entry speed table for exactly N episodes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import defaults_from_argv  # noqa: E402
from policy_speed_env import make_speed_reward  # noqa: E402
from scripts.policy_cli import (  # noqa: E402
    add_base_policy_arguments,
    add_observation_arguments,
    build_speed_env,
    comma_floats,
)
from tabular_phase_speed import TabularTrainingConfig, train_tabular_phase_speed_policy  # noqa: E402


class CyclingSpeedEnv:
    """Cycle deterministically through a small fixed set of scene environments."""

    def __init__(self, envs):
        if not envs:
            raise ValueError("at least one scene environment is required")
        self.envs = list(envs)
        self.index = -1
        self.active = self.envs[0]

    def reset(self):
        self.index = (self.index + 1) % len(self.envs)
        self.active = self.envs[self.index]
        return self.active.reset()

    def step_decision(self, *args, **kwargs):
        return self.active.step_decision(*args, **kwargs)

    def observation_spec(self):
        return self.envs[0].observation_spec()

    def environment_spec(self):
        value = dict(self.envs[0].environment_spec())
        value["fixed_scene_cycle"] = len(self.envs)
        return value

    def close(self):
        for env in self.envs:
            env.close()

    def __getattr__(self, name):
        return getattr(self.active, name)


def main():
    defaults, metadata = defaults_from_argv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--task", choices=("pick_and_place", "tea_bag", "insertion"), required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--object-poses-json",
        type=Path,
        help="Cycle training episodes through this JSON list of fixed object poses.",
    )
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--success-bonus", type=float, default=100.0)
    parser.add_argument("--speed-weight", type=float, default=0.01)
    parser.add_argument("--speed-power", type=float, default=2.0)
    parser.add_argument(
        "--speed-values",
        type=comma_floats,
        default=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    )
    parser.add_argument("--frame-skip", type=int, default=10)
    add_base_policy_arguments(parser)
    add_observation_arguments(parser)
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    reward_fn = make_speed_reward(args.success_bonus, args.speed_weight, args.speed_power)
    if args.object_poses_json is None:
        env = build_speed_env(args, reward_fn=reward_fn)
    else:
        poses = json.loads(args.object_poses_json.read_text())
        if len(poses) != 3:
            raise ValueError("object-poses-json must contain exactly three poses")
        envs = []
        for pose in poses:
            scene_args = copy.copy(args)
            scene_args.object_pose = pose
            scene_args.randomize_object_pose = False
            envs.append(build_speed_env(scene_args, reward_fn=reward_fn))
        env = CyclingSpeedEnv(envs)
    try:
        result = train_tabular_phase_speed_policy(
            env,
            args.output,
            config=TabularTrainingConfig(
                episodes=args.episodes,
                gamma=args.gamma,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
            ),
            seed=args.seed,
        )
    finally:
        env.close()
    report = {"summary": {key: value for key, value in result.items() if key != "episode_history"},
              "episode_history": result["episode_history"], "experiment_config": metadata}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
