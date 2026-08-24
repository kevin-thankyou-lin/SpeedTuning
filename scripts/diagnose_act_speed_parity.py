#!/usr/bin/env python3
"""Compare retained ACT and the speed adapter on one engineering seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_integration import build_original_act_speed_adapter  # noqa: E402
from original_act import (  # noqa: E402
    create_original_act_policy,
    normalized_episode_progress,
    set_seed,
)
from sim_env import make_sim_env  # noqa: E402
from sim_tasks import get_task_spec, normalize_task_name  # noqa: E402


CAMERAS = ("angle", "left_wrist", "right_wrist")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_retained(task, policy, stats, device, seed):
    episode_len = get_task_spec(task).episode_len
    env = make_sim_env(
        task,
        render_images=True,
        render_camera_names=CAMERAS,
        seed=seed,
        randomize_object_pose=True,
    )
    actions = []
    qpos = []
    states = []
    rewards = []
    try:
        timestep = env.reset()
        all_time_actions = torch.zeros(
            (episode_len, episode_len + 100, 14), device=device
        )
        with torch.inference_mode():
            for step in range(episode_len):
                observation = timestep.observation
                qpos.append(np.asarray(observation["qpos"]).copy())
                states.append(np.asarray(observation["env_state"]).copy())
                normalized_qpos = (
                    np.asarray(observation["qpos"]) - stats["qpos_mean"]
                ) / stats["qpos_std"]
                normalized_qpos = np.concatenate(
                    [
                        normalized_qpos,
                        np.asarray(
                            [normalized_episode_progress(step, episode_len)]
                        ),
                    ]
                )
                image = np.stack(
                    [observation["images"][name] for name in CAMERAS]
                )
                qpos_tensor = torch.as_tensor(
                    normalized_qpos, dtype=torch.float32, device=device
                )[None]
                image_tensor = torch.as_tensor(
                    image.transpose(0, 3, 1, 2).copy(),
                    dtype=torch.float32,
                    device=device,
                )[None] / 255.0
                chunk = policy(qpos_tensor, image_tensor)
                all_time_actions[step, step : step + 100] = chunk[0]
                candidates = all_time_actions[:, step]
                candidates = candidates[torch.all(candidates != 0, dim=1)]
                weights = np.exp(-0.01 * np.arange(len(candidates)))
                weights = torch.as_tensor(
                    weights / weights.sum(), dtype=torch.float32, device=device
                )[:, None]
                normalized_action = (
                    candidates * weights
                ).sum(dim=0).detach().cpu().numpy()
                action = normalized_action * stats["action_std"] + stats["action_mean"]
                actions.append(np.asarray(action).copy())
                timestep = env.step(action)
                rewards.append(float(timestep.reward or 0))
        return {
            "actions": np.stack(actions),
            "qpos": np.stack(qpos),
            "states": np.stack(states),
            "rewards": np.asarray(rewards),
            "max_reward": max(rewards),
        }
    finally:
        env.close()


def trace_adapter(task, adapter, seed):
    episode_len = get_task_spec(task).episode_len
    env = make_sim_env(
        task,
        render_images=True,
        render_camera_names=CAMERAS,
        seed=seed,
        randomize_object_pose=True,
    )
    actions = []
    qpos = []
    states = []
    rewards = []
    try:
        timestep = env.reset()
        adapter.reset()
        for _ in range(episode_len):
            observation = timestep.observation
            qpos.append(np.asarray(observation["qpos"]).copy())
            states.append(np.asarray(observation["env_state"]).copy())
            action = adapter.action(observation, speed=1.0)
            actions.append(np.asarray(action).copy())
            timestep = env.step(action)
            rewards.append(float(timestep.reward or 0))
        return {
            "actions": np.stack(actions),
            "qpos": np.stack(qpos),
            "states": np.stack(states),
            "rewards": np.asarray(rewards),
            "max_reward": max(rewards),
        }
    finally:
        env.close()


def comparison(left, right):
    exact = np.all(left == right, axis=tuple(range(1, left.ndim)))
    differing = np.flatnonzero(~exact)
    first = None if not len(differing) else int(differing[0])
    return {
        "array_equal": bool(np.array_equal(left, right)),
        "first_differing_step": first,
        "maximum_absolute_difference": float(np.max(np.abs(left - right))),
        "first_step_maximum_absolute_difference": (
            None
            if first is None
            else float(np.max(np.abs(left[first] - right[first])))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    args = parser.parse_args()

    task = normalize_task_name(args.task)
    checkpoint = args.checkpoint_root / "checkpoints/policy_best.ckpt"
    stats_path = args.checkpoint_root / "checkpoints/dataset_stats.pkl"
    config_path = args.checkpoint_root / "checkpoints/policy_config.json"
    with stats_path.open("rb") as stream:
        stats = pickle.load(stream)

    set_seed(1000)
    device = torch.device(args.device)
    retained_policy, retained_config = create_original_act_policy(
        device, qpos_dim=15, camera_names=CAMERAS
    )
    retained_policy.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    retained_policy.to(device).eval()
    retained = trace_retained(task, retained_policy, stats, device, args.seed)

    set_seed(1000)
    adapter = build_original_act_speed_adapter(
        task_name=task,
        checkpoint=checkpoint,
        stats_path=stats_path,
        policy_config_path=config_path,
        temporal_ensemble_m=0.01,
        device=args.device,
    )
    adapted = trace_adapter(task, adapter, args.seed)

    args.traces.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.traces,
        retained_actions=retained["actions"],
        adapter_actions=adapted["actions"],
        retained_qpos=retained["qpos"],
        adapter_qpos=adapted["qpos"],
        retained_states=retained["states"],
        adapter_states=adapted["states"],
        retained_rewards=retained["rewards"],
        adapter_rewards=adapted["rewards"],
    )
    report = {
        "schema": "act-speed-parity-diagnostic-v1",
        "task": task,
        "seed": args.seed,
        "engineering_only": True,
        "checkpoint_sha256": sha256(checkpoint),
        "stats_sha256": sha256(stats_path),
        "config_sha256": sha256(config_path),
        "retained_config": retained_config,
        "adapter_config": json.loads(config_path.read_text()),
        "stats_dtypes": {
            key: str(np.asarray(value).dtype) for key, value in stats.items()
        },
        "retained_max_reward": retained["max_reward"],
        "adapter_max_reward": adapted["max_reward"],
        "actions": comparison(retained["actions"], adapted["actions"]),
        "qpos": comparison(retained["qpos"], adapted["qpos"]),
        "states": comparison(retained["states"], adapted["states"]),
        "rewards": comparison(
            retained["rewards"][:, None], adapted["rewards"][:, None]
        ),
        "traces": str(args.traces),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
