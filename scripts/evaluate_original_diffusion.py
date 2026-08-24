"""Closed-loop evaluation for the matched multiview Diffusion Policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from dm_control.rl import control

from original_act import normalized_episode_progress, set_seed
from original_diffusion import JointRangeNormalizer, OriginalDiffusionPolicy
from sim_env import make_sim_env
from sim_tasks import get_task_spec, normalize_task_name


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_policy(checkpoint, device):
    value = torch.load(checkpoint, map_location=device, weights_only=False)
    if value.get("schema") != "original-diffusion-checkpoint-v1":
        raise ValueError("unsupported checkpoint schema")
    policy = OriginalDiffusionPolicy(value["config"]).to(device)
    policy.load_state_dict(value["ema"]["model"])
    policy.eval()
    return policy, JointRangeNormalizer.from_state_dict(value["normalizer"]), value


def _observation_tensors(history, normalizer, camera_names, device):
    images, qposes = [], []
    for observation, step, episode_len in history:
        images.append(np.stack([observation["images"][name] for name in camera_names]))
        qpos = normalizer.normalize_qpos(np.asarray(observation["qpos"], dtype=np.float32))
        qposes.append(np.concatenate((qpos, [normalized_episode_progress(step, episode_len)])))
    image = np.stack(images).transpose(0, 1, 4, 2, 3).copy()
    return (
        torch.as_tensor(qposes, dtype=torch.float32, device=device)[None],
        torch.as_tensor(image, dtype=torch.float32, device=device)[None] / 255,
    )


def rollout(task, policy, normalizer, device, seed, camera_names):
    episode_len = get_task_spec(task).episode_len
    env = make_sim_env(
        task,
        render_images=True,
        render_camera_names=camera_names,
        seed=int(seed),
        randomize_object_pose=True,
    )
    maximum_reward = 0
    try:
        timestep = env.reset()
        history = deque(maxlen=policy.observation_horizon)
        history.append((timestep.observation, 0, episode_len))
        history.append((timestep.observation, 0, episode_len))
        generator = torch.Generator(device=device).manual_seed(int(seed) + 10_000_019)
        step = 0
        with torch.inference_mode():
            while step < episode_len:
                qpos, image = _observation_tensors(history, normalizer, camera_names, device)
                normalized = policy.executed_slice(policy.sample(qpos, image, generator=generator))[0]
                actions = normalizer.denormalize_action(normalized.cpu().numpy())
                for action in actions:
                    if step >= episode_len:
                        break
                    try:
                        timestep = env.step(action)
                    except control.PhysicsError as exc:
                        return {
                            "seed": int(seed),
                            "success": False,
                            "steps": step + 1,
                            "max_reward": maximum_reward,
                            "physics_error": str(exc),
                        }
                    step += 1
                    maximum_reward = max(maximum_reward, int(timestep.reward or 0))
                    history.append((timestep.observation, step, episode_len))
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument("--seed-base", type=int, required=True)
    args = parser.parse_args()
    task = normalize_task_name(args.task)
    set_seed(1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, normalizer, state = load_policy(args.checkpoint, device)
    camera_names = tuple(state["config"]["camera_names"])
    identity = {
        "task": task,
        "checkpoint_sha256": checkpoint_identity(args.checkpoint),
        "seed_base": args.seed_base,
        "episodes": args.num_rollouts,
        "camera_names": list(camera_names),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    records = []
    if partial.exists():
        existing = json.loads(partial.read_text())
        if existing.get("identity") != identity:
            raise ValueError("partial evaluation identity mismatch")
        records = existing["rollouts"]
    for index in range(len(records), args.num_rollouts):
        records.append(
            rollout(task, policy, normalizer, device, args.seed_base + index, camera_names)
        )
        _atomic_json(partial, {"identity": identity, "rollouts": records})
        print(json.dumps({"completed": index + 1, "successes": sum(x["success"] for x in records)}), flush=True)
    report = {
        "schema": "original-diffusion-evaluation-v1",
        "identity": identity,
        "successes": sum(item["success"] for item in records),
        "success_rate": float(np.mean([item["success"] for item in records])),
        "replan_interval": policy.action_horizon,
        "prediction_horizon": policy.prediction_horizon,
        "observation_horizon": policy.observation_horizon,
        "rollouts": records,
    }
    _atomic_json(args.output, report)
    partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
