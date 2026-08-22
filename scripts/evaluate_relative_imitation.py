"""Evaluate a learned relative-joint policy on fresh randomized poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from relative_imitation import RelativeChunkPredictor
from sim_env import make_sim_env
from sim_tasks import get_task_spec, normalize_task_name


def rollout(task, predictor, seed, replan_interval):
    env = make_sim_env(
        task,
        render_images=True,
        render_camera_names=("angle",),
        seed=int(seed),
        randomize_object_pose=True,
    )
    try:
        timestep = env.reset()
        chunk = None
        maximum_reward = 0
        for step in range(get_task_spec(task).episode_len):
            if chunk is None or step % replan_interval == 0:
                chunk = predictor(timestep.observation)
            action = chunk[step % replan_interval]
            timestep = env.step(action)
            maximum_reward = max(maximum_reward, int(timestep.reward or 0))
            if maximum_reward == env.task.max_reward:
                return {"seed": int(seed), "success": True, "physics_steps": step + 1}
        return {"seed": int(seed), "success": False, "physics_steps": get_task_spec(task).episode_len}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=2600000)
    parser.add_argument("--replan-interval", type=int, default=8)
    parser.add_argument("--speed-condition", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    task = normalize_task_name(args.task)
    predictor = RelativeChunkPredictor(args.checkpoint, args.speed_condition)
    rollouts = []
    for index in range(args.episodes):
        rollouts.append(
            rollout(task, predictor, args.seed_base + index, args.replan_interval)
        )
        print(json.dumps({"completed": index + 1, "successes": sum(item["success"] for item in rollouts)}), flush=True)
    successes = [item for item in rollouts if item["success"]]
    report = {
        "task": task,
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "speed_condition": args.speed_condition,
        "successes": len(successes),
        "success_rate": len(successes) / args.episodes,
        "successful_mean_steps": (
            float(np.mean([item["physics_steps"] for item in successes])) if successes else None
        ),
        "clipping": predictor.clipping_metrics(),
        "rollouts": rollouts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
