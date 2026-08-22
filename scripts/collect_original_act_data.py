"""Collect the original ACT simulator's two-pass scripted demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from ee_sim_env import make_ee_sim_env
from scripted_policy import make_scripted_policy
from sim_env import make_sim_env
from sim_tasks import get_task_spec, normalize_task_name


def _target_from_ee_timestep(timestep):
    target = np.asarray(timestep.observation["qpos"], dtype=np.float64).copy()
    control = timestep.observation["gripper_ctrl"]
    target[6] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(control[0])
    target[13] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(control[2])
    return target


def collect_episode(task, seed, output_path):
    """EE-script rollout followed by same-pose joint-target replay, as upstream."""

    task = normalize_task_name(task)
    episode_len = get_task_spec(task).episode_len
    ee_env = make_ee_sim_env(
        task, render_images=False, seed=int(seed), randomize_object_pose=True
    )
    try:
        timestep = ee_env.reset()
        ee_episode = [timestep]
        policy = make_scripted_policy(task)
        for _ in range(episode_len):
            timestep = ee_env.step(policy(timestep))
            ee_episode.append(timestep)
        source_success = max(int(ts.reward or 0) for ts in ee_episode[1:]) == ee_env.task.max_reward
        actions = [_target_from_ee_timestep(ts) for ts in ee_episode]
        object_pose = np.asarray(ee_episode[0].observation["env_state"], dtype=np.float64).copy()
    finally:
        ee_env.close()

    joint_env = make_sim_env(
        task,
        render_images=True,
        render_camera_names=("top",),
        seed=int(seed),
        object_pose=object_pose,
    )
    try:
        timestep = joint_env.reset()
        replay = [timestep]
        for action in actions:
            timestep = joint_env.step(action)
            replay.append(timestep)
        replay_success = max(int(ts.reward or 0) for ts in replay[1:]) == joint_env.task.max_reward
    finally:
        joint_env.close()

    # This is the exact off-by-one truncation in upstream record_sim_episodes.py.
    actions = np.asarray(actions[:-1], dtype=np.float64)
    observations = replay[:-1]
    qpos = np.asarray([ts.observation["qpos"] for ts in observations[:episode_len]])
    qvel = np.asarray([ts.observation["qvel"] for ts in observations[:episode_len]])
    images = np.asarray(
        [ts.observation["images"]["top"] for ts in observations[:episode_len]],
        dtype=np.uint8,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w", rdcc_nbytes=2 * 1024**2) as root:
        root.attrs.update(
            sim=True,
            task=task,
            seed=int(seed),
            source_success=bool(source_success),
            replay_success=bool(replay_success),
            collection="original ACT two-pass EE script to joint-target replay",
        )
        observations_group = root.create_group("observations")
        observations_group.create_dataset("qpos", data=qpos)
        observations_group.create_dataset("qvel", data=qvel)
        image_group = observations_group.create_group("images")
        image_group.create_dataset(
            "top", data=images, chunks=(1, 480, 640, 3), dtype="uint8"
        )
        root.create_dataset("action", data=actions)
        root.create_dataset("object_pose", data=object_pose)
    return {
        "path": str(output_path),
        "seed": int(seed),
        "source_success": bool(source_success),
        "replay_success": bool(replay_success),
        "steps": int(episode_len),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=0)
    args = parser.parse_args()
    records = []
    for index in range(args.num_episodes):
        record = collect_episode(
            args.task,
            args.seed_base + index,
            args.dataset_dir / f"episode_{index}.hdf5",
        )
        records.append(record)
        print(json.dumps({"completed": index + 1, **record}), flush=True)
    summary = {
        "schema": "original-act-two-pass-v1",
        "task": normalize_task_name(args.task),
        "attempted_episodes": len(records),
        "source_successes": sum(item["source_success"] for item in records),
        "replay_successes": sum(item["replay_success"] for item in records),
        "camera_names": ["top"],
        "action": "absolute 14-D target joint positions",
        "records": records,
    }
    (args.dataset_dir / "collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
