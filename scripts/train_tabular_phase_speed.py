#!/usr/bin/env python3
"""Train a tiny phase-entry speed table for exactly N episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import defaults_from_argv  # noqa: E402
from policy_speed_env import make_speed_reward  # noqa: E402
from scripts.policy_cli import add_base_policy_arguments, add_observation_arguments, build_speed_env  # noqa: E402
from tabular_phase_speed import TabularTrainingConfig, train_tabular_phase_speed_policy  # noqa: E402


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
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--success-bonus", type=float, default=100.0)
    parser.add_argument("--speed-weight", type=float, default=0.01)
    parser.add_argument("--speed-power", type=float, default=2.0)
    add_base_policy_arguments(parser)
    add_observation_arguments(parser)
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    reward_fn = make_speed_reward(args.success_bonus, args.speed_weight, args.speed_power)
    env = build_speed_env(args, reward_fn=reward_fn)
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
