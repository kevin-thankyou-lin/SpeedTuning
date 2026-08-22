"""Absolute-command demonstrations and observation-anchored relative chunks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from ee_sim_env import make_ee_sim_env
from scripted_policy import make_scripted_policy
from sim_env import make_sim_env
from sim_tasks import get_task_spec, normalize_task_name


def relative_chunk(target_qpos, first_qpos):
    """Encode every future absolute command relative to one observation qpos."""

    targets = np.asarray(target_qpos, dtype=np.float32)
    anchor = np.asarray(first_qpos, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 14 or anchor.shape != (14,):
        raise ValueError("target_qpos must be [time, 14] and first_qpos must be [14]")
    return targets - anchor[None]


def decode_relative_chunk(delta_qpos, first_qpos):
    return np.asarray(delta_qpos, dtype=np.float32) + np.asarray(
        first_qpos, dtype=np.float32
    )[None]


def _joint_target(timestep):
    target = np.asarray(timestep.observation["qpos"], dtype=np.float32).copy()
    gripper = timestep.observation["gripper_ctrl"]
    target[6] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(gripper[0])
    target[13] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(gripper[2])
    return target


def _trace_speeds(phase_decisions, steps):
    decisions = sorted(phase_decisions, key=lambda item: int(item["physics_step"]))
    if not decisions or int(decisions[0]["physics_step"]) != 0:
        raise ValueError("phase decision trace must begin at physics step zero")
    speeds = []
    cursor = 0
    for step in range(int(steps)):
        while cursor + 1 < len(decisions) and int(
            decisions[cursor + 1]["physics_step"]
        ) <= step:
            cursor += 1
        speeds.append(float(decisions[cursor]["speed"]))
    return speeds


def collect_absolute_targets(task, seed, phase_decisions, physics_steps):
    """Recreate a frozen accelerated EE rollout as absolute joint commands."""

    task = normalize_task_name(task)
    env = make_ee_sim_env(
        task,
        render_images=False,
        seed=int(seed),
        randomize_object_pose=True,
    )
    try:
        timestep = env.reset()
        object_pose = np.asarray(timestep.observation["env_state"], dtype=np.float64)
        policy = make_scripted_policy(task)
        targets = []
        rewards = []
        speeds = _trace_speeds(phase_decisions, physics_steps)
        for speed in speeds:
            timestep = env.step(policy(timestep, step_inc=speed))
            targets.append(_joint_target(timestep))
            rewards.append(int(timestep.reward or 0))
        # The EE benchmark stops on first success. Joint-position replay has
        # actuator lag, so retain the scripted controller's remaining commands
        # through its nominal horizon using the already-frozen final speed.
        while policy.step_count < get_task_spec(task).episode_len:
            timestep = env.step(policy(timestep, step_inc=speeds[-1]))
            targets.append(_joint_target(timestep))
            rewards.append(int(timestep.reward or 0))
        return {
            "target_qpos": np.asarray(targets, dtype=np.float32),
            "object_pose": object_pose,
            "success": max(rewards, default=0) == env.task.max_reward,
        }
    finally:
        env.close()


def record_episode(task, seed, phase_decisions, physics_steps, output_path):
    """Replay one frozen rollout in joint control and store absolute values."""

    import h5py

    task = normalize_task_name(task)
    source = collect_absolute_targets(task, seed, phase_decisions, physics_steps)
    if not source["success"]:
        raise RuntimeError("source EE trace did not reproduce its recorded success")
    env = make_sim_env(
        task,
        render_images=True,
        render_camera_names=("angle",),
        seed=int(seed),
        object_pose=source["object_pose"],
    )
    qpos, targets, images, rewards = [], [], [], []
    try:
        timestep = env.reset()
        for target in source["target_qpos"]:
            qpos.append(np.asarray(timestep.observation["qpos"], dtype=np.float32))
            images.append(np.asarray(timestep.observation["images"]["angle"], dtype=np.uint8))
            targets.append(target)
            timestep = env.step(target)
            rewards.append(int(timestep.reward or 0))
        success = max(rewards, default=0) == env.task.max_reward
    finally:
        env.close()
    if not success:
        raise RuntimeError("absolute joint-command replay did not preserve task success")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as root:
        root.attrs.update(
            task=task,
            seed=int(seed),
            source_physics_steps=int(physics_steps),
            relative_action="target_qpos[t+k] - observations/qpos[t]",
            success=True,
        )
        observations = root.create_group("observations")
        observations.create_dataset("qpos", data=np.asarray(qpos), compression="gzip")
        image_group = observations.create_group("images")
        image_group.create_dataset(
            "angle",
            data=np.asarray(images),
            compression="gzip",
            chunks=(1, *images[0].shape),
        )
        root.create_dataset("target_qpos", data=np.asarray(targets), compression="gzip")
        root.create_dataset("object_pose", data=source["object_pose"])
    return {"path": str(output_path), "steps": len(targets), "success": True}


def fit_normalization(episode_paths, chunk_size):
    """Fit train-only qpos and per-chunk-step delta statistics."""

    import h5py

    qpos_values = []
    delta_values = [[] for _ in range(int(chunk_size))]
    for path in map(Path, episode_paths):
        with h5py.File(path, "r") as root:
            qpos = np.asarray(root["observations/qpos"], dtype=np.float32)
            targets = np.asarray(root["target_qpos"], dtype=np.float32)
        qpos_values.append(qpos)
        for start, anchor in enumerate(qpos):
            end = min(start + chunk_size, len(targets))
            for offset, delta in enumerate(targets[start:end] - anchor[None]):
                delta_values[offset].append(delta)
    qpos = np.concatenate(qpos_values)
    delta_mean, delta_std = [], []
    for values in delta_values:
        array = np.asarray(values, dtype=np.float32)
        delta_mean.append(array.mean(axis=0))
        delta_std.append(np.maximum(array.std(axis=0), 1e-3))
    return {
        "qpos_mean": qpos.mean(axis=0),
        "qpos_std": np.maximum(qpos.std(axis=0), 1e-3),
        "delta_mean": np.asarray(delta_mean),
        "delta_std": np.asarray(delta_std),
        "chunk_size": int(chunk_size),
        "relative_action": "target_qpos[t+k] - observations/qpos[t]",
    }


def save_normalization(stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)


def load_manifest(path):
    return json.loads(Path(path).read_text())
