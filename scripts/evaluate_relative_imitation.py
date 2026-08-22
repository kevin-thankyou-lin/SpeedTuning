"""Evaluate a learned relative-joint policy on fresh randomized poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from dm_control.rl import control

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
            try:
                timestep = env.step(action)
            except control.PhysicsError as error:
                return {
                    "seed": int(seed),
                    "success": False,
                    "physics_steps": step + 1,
                    "failure_reason": "physics_error",
                    "physics_error": str(error),
                }
            maximum_reward = max(maximum_reward, int(timestep.reward or 0))
            if maximum_reward == env.task.max_reward:
                return {"seed": int(seed), "success": True, "physics_steps": step + 1}
        return {"seed": int(seed), "success": False, "physics_steps": get_task_spec(task).episode_len}
    finally:
        env.close()


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=2600000)
    parser.add_argument("--replan-interval", type=int, default=8)
    parser.add_argument("--speed-condition", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the atomically written per-state partial report",
    )
    args = parser.parse_args()
    task = normalize_task_name(args.task)
    predictor = RelativeChunkPredictor(args.checkpoint, args.speed_condition)
    identity = {
        "task": task,
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "seed_base": args.seed_base,
        "replan_interval": args.replan_interval,
        "speed_condition": args.speed_condition,
    }
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    rollouts = []
    if args.resume and partial_path.exists():
        partial = json.loads(partial_path.read_text())
        if partial.get("identity") != identity:
            raise RuntimeError("evaluation resume identity mismatch")
        rollouts = partial["rollouts"]
    elif partial_path.exists():
        raise RuntimeError(
            f"partial evaluation exists; pass --resume or remove it: {partial_path}"
        )
    for index in range(len(rollouts), args.episodes):
        rollouts.append(
            rollout(task, predictor, args.seed_base + index, args.replan_interval)
        )
        _write_json_atomic(
            partial_path,
            {"schema": "relative-imitation-eval-partial-v1", "identity": identity, "rollouts": rollouts},
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
    _write_json_atomic(args.output, report)
    partial_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
