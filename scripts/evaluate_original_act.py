"""Evaluate original ACT with its paper temporal ensemble."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from dm_control.rl import control

from original_act import create_original_act_policy, set_seed
from sim_env import make_sim_env
from sim_tasks import get_task_spec, normalize_task_name


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def rollout(task, policy, stats, device, seed):
    episode_len = get_task_spec(task).episode_len
    env = make_sim_env(
        task,
        render_images=True,
        render_camera_names=("top",),
        seed=int(seed),
        randomize_object_pose=True,
    )
    try:
        timestep = env.reset()
        all_time_actions = torch.zeros(
            (episode_len, episode_len + 100, 14), device=device
        )
        maximum_reward = 0
        with torch.inference_mode():
            for step in range(episode_len):
                observation = timestep.observation
                qpos = (np.asarray(observation["qpos"]) - stats["qpos_mean"]) / stats["qpos_std"]
                image = np.asarray(observation["images"]["top"])
                qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32, device=device)[None]
                image_tensor = torch.as_tensor(
                    image.transpose(2, 0, 1), dtype=torch.float32, device=device
                )[None, None] / 255.0
                actions = policy(qpos_tensor, image_tensor)
                all_time_actions[step, step : step + 100] = actions[0]
                candidates = all_time_actions[:, step]
                populated = torch.all(candidates != 0, dim=1)
                candidates = candidates[populated]
                weights = np.exp(-0.01 * np.arange(len(candidates)))
                weights = torch.as_tensor(
                    weights / weights.sum(), dtype=torch.float32, device=device
                )[:, None]
                normalized_action = (candidates * weights).sum(dim=0).cpu().numpy()
                action = normalized_action * stats["action_std"] + stats["action_mean"]
                try:
                    timestep = env.step(action)
                except control.PhysicsError as exc:
                    return {"seed": int(seed), "success": False, "steps": step + 1, "physics_error": str(exc)}
                maximum_reward = max(maximum_reward, int(timestep.reward or 0))
        return {
            "seed": int(seed),
            "success": maximum_reward == env.task.max_reward,
            "steps": episode_len,
            "max_reward": maximum_reward,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=1000)
    args = parser.parse_args()
    task = normalize_task_name(args.task)
    set_seed(1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ = create_original_act_policy(device)
    policy.load_state_dict(torch.load(args.checkpoint_dir / "policy_best.ckpt", map_location=device, weights_only=True))
    policy.to(device).eval()
    with (args.checkpoint_dir / "dataset_stats.pkl").open("rb") as stream:
        stats = pickle.load(stream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    records = []
    for index in range(args.num_rollouts):
        records.append(rollout(task, policy, stats, device, args.seed_base + index))
        _atomic_json(partial, {"task": task, "rollouts": records})
        print(json.dumps({"completed": index + 1, "successes": sum(item["success"] for item in records)}), flush=True)
    report = {
        "schema": "original-act-evaluation-v1",
        "task": task,
        "episodes": len(records),
        "successes": sum(item["success"] for item in records),
        "success_rate": float(np.mean([item["success"] for item in records])),
        "temporal_aggregation": {"enabled": True, "m": 0.01},
        "rollouts": records,
    }
    _atomic_json(args.output, report)
    partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
