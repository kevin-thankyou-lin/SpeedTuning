#!/usr/bin/env python3
"""Reconstruct causal scripted-policy action chunks for an existing image bank.

The existing phase banks intentionally contain images and offline labels only.
For the scripted-policy prototype, this tool deterministically resets each
recorded seed, builds the policy plan from the initial observation, and stores
the same future action chunk that would be available to an online controller.
It does not step physics or create a scientific rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_at(policy, policy_time: float) -> np.ndarray:
    left_current, left_next = policy._waypoint_pair(policy.left_trajectory, policy_time)
    right_current, right_next = policy._waypoint_pair(policy.right_trajectory, policy_time)
    left_xyz, left_quat, left_gripper = policy.interpolate(left_current, left_next, policy_time)
    right_xyz, right_quat, right_gripper = policy.interpolate(right_current, right_next, policy_time)
    return np.concatenate(
        [
            left_xyz,
            left_quat,
            [left_gripper],
            right_xyz,
            right_quat,
            [right_gripper],
        ]
    ).astype(np.float32)


def chunk_feature(policy, policy_time: float, offsets: tuple[int, ...]) -> np.ndarray:
    actions = np.stack([action_at(policy, policy_time + offset) for offset in offsets])
    anchor = actions[:1]
    return np.concatenate([actions[0], (actions[1:] - anchor).reshape(-1)]).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offsets", default="0,5,10,20,40")
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    sys.path.insert(0, str(runtime_root))
    from policy_speed_env import create_speed_env

    dataset = args.dataset.resolve()
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    offsets = tuple(int(item) for item in args.offsets.split(","))
    if not offsets or offsets[0] != 0 or offsets != tuple(sorted(set(offsets))):
        raise ValueError("offsets must be unique, sorted, and begin at zero")
    features = {}
    episode_hashes = {}
    task = manifest["task"]
    environment_task = "tea_bag" if task == "tea_bag_randomized" else task
    for episode in manifest["episodes"]:
        seed = int(episode["seed"])
        labels_path = dataset / episode["labels"]
        records = json.loads(labels_path.read_text())
        env = create_speed_env(
            task_name=environment_task,
            seed=seed,
            render_images=False,
            randomize_object_pose=task == "tea_bag_randomized",
            speed_values=(1.0,),
            terminate_on_success=False,
        )
        try:
            env.reset()
            policy = env.action_source.policy
            policy.generate_trajectory(env.cur_ts)
            values = np.stack(
                [chunk_feature(policy, float(record["policy_time"]), offsets) for record in records]
            )
        finally:
            env.close()
        features[str(seed)] = values
        episode_hashes[str(seed)] = sha256(labels_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **features)
    receipt = {
        "schema": "speedtuning-scripted-action-chunks-v1",
        "task": task,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256(manifest_path),
        "runtime_root": str(runtime_root),
        "policy": "frozen scripted policy",
        "offsets_policy_steps": offsets,
        "feature_contract": "current action plus future-action deltas from current action",
        "runtime_privileged_state": False,
        "offline_reset_used_only_to_reconstruct_policy_plan": True,
        "physics_steps": 0,
        "episode_label_hashes": episode_hashes,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
