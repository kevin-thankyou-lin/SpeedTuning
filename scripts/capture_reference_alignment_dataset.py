#!/usr/bin/env python3
"""Capture segment-free trajectories for reference-position alignment.

Simulator policy time supplies exact correspondence landmarks for this small
benchmark.  Those positions are evaluation-only and never enter an encoder or
alignment update.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(
    os.environ.get("SPEEDTUNING_RUNTIME_ROOT", Path(__file__).resolve().parents[1])
).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import DT  # noqa: E402
from policy_speed_env import create_speed_env  # noqa: E402


QUERY_SPEEDS = (0.85, 1.15, 0.95, 1.10, 0.90)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query_speed(policy_time: float, episode_len: int, query_index: int) -> float:
    progress = float(np.clip(policy_time / episode_len, 0.0, 0.999999))
    bin_index = min(len(QUERY_SPEEDS) - 1, int(progress * len(QUERY_SPEEDS)))
    return QUERY_SPEEDS[(bin_index + query_index - 1) % len(QUERY_SPEEDS)]


def save_observation(
    root: Path,
    camera: str,
    frame_index: int,
    env,
    task_reward: float,
) -> dict:
    relative = Path(camera) / f"{frame_index:05d}.jpg"
    Image.fromarray(env.cur_ts.observation["images"][camera]).save(
        root / relative,
        quality=92,
    )
    return {
        "frame_index": frame_index,
        "image": relative.as_posix(),
        "wall_time_s": float(env.physics_steps * DT),
        "physics_steps": int(env.physics_steps),
        "policy_time": float(env.policy_time),
        "reference_position": float(
            np.clip(env.policy_time / env.episode_len, 0.0, 1.0)
        ),
        "task_reward": float(task_reward),
    }


def capture_trajectory(args, seed: int, trajectory_index: int) -> dict:
    trajectory_id = f"trajectory-{trajectory_index:02d}-seed-{seed}"
    trajectory_root = args.output / trajectory_id
    (trajectory_root / args.camera).mkdir(parents=True, exist_ok=False)
    environment_task = "tea_bag" if args.task == "tea_bag_randomized" else args.task
    env = create_speed_env(
        task_name=environment_task,
        seed=seed,
        render_images=True,
        randomize_object_pose=args.task == "tea_bag_randomized",
        speed_values=(*QUERY_SPEEDS, 1.0),
        terminate_on_success=False,
    )
    env.reset()
    task_reward = float(env.cur_ts.reward or 0.0)
    records = [save_observation(trajectory_root, args.camera, 0, env, task_reward)]
    done = False
    while not done:
        speed = 1.0 if trajectory_index == 0 else query_speed(
            env.policy_time,
            env.episode_len,
            trajectory_index,
        )
        _, _, done, info = env.step(speed, quantized=False)
        task_reward = float(info["task_reward"])
        if env.physics_steps % args.frame_stride == 0 or done:
            if records[-1]["physics_steps"] != env.physics_steps:
                records.append(
                    save_observation(
                        trajectory_root,
                        args.camera,
                        len(records),
                        env,
                        task_reward,
                    )
                )
    success = bool(env.cur_success)
    mean_speed = float(np.mean(env.speed_list))
    env.close()
    record_path = trajectory_root / "trajectory.json"
    record_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    return {
        "trajectory_id": trajectory_id,
        "seed": seed,
        "role": "reference" if trajectory_index == 0 else "query",
        "speed_profile": "constant_1x" if trajectory_index == 0 else "deterministic_piecewise",
        "speed_profile_values": [1.0] if trajectory_index == 0 else list(QUERY_SPEEDS),
        "frames": len(records),
        "success": success,
        "mean_commanded_speed": mean_speed,
        "physics_steps": records[-1]["physics_steps"],
        "record": record_path.relative_to(args.output).as_posix(),
        "record_sha256": sha256(record_path),
    }


def write_landmarks(args, trajectories: list[dict]) -> Path:
    path = args.output / "landmarks.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "query_video",
                "query_time_s",
                "query_frame_index",
                "true_reference_position",
                "annotation_source",
            ),
        )
        writer.writeheader()
        for trajectory in trajectories:
            if trajectory["role"] != "query":
                continue
            records = json.loads((args.output / trajectory["record"]).read_text())
            indices = sorted(
                set(
                    np.linspace(
                        0,
                        len(records) - 1,
                        min(args.landmarks_per_query, len(records)),
                        dtype=int,
                    ).tolist()
                )
            )
            for index in indices:
                record = records[index]
                writer.writerow(
                    {
                        "query_video": trajectory["trajectory_id"],
                        "query_time_s": f'{record["wall_time_s"]:.6f}',
                        "query_frame_index": record["frame_index"],
                        "true_reference_position": f'{record["reference_position"]:.8f}',
                        "annotation_source": "simulator_policy_time",
                    }
                )
    return path


def capture(args) -> dict:
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    trajectories = [
        capture_trajectory(args, seed, index)
        for index, seed in enumerate(args.seeds)
    ]
    landmarks_path = write_landmarks(args, trajectories)
    manifest = {
        "schema": "speedtuning-reference-alignment-dataset-v1",
        "task": args.task,
        "camera": args.camera,
        "frame_stride_physics_steps": args.frame_stride,
        "nominal_frame_rate_hz": 1.0 / (DT * args.frame_stride),
        "semantic_segment_labels_present": False,
        "runtime_privileged_signals": False,
        "evaluation_correspondence_truth": "normalized scripted-policy time",
        "evaluation_truth_used_by_model": False,
        "reference_trajectory": trajectories[0]["trajectory_id"],
        "trajectories": trajectories,
        "landmarks": landmarks_path.name,
        "landmarks_sha256": sha256(landmarks_path),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("pick_and_place", "tea_bag_randomized", "insertion"),
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", choices=("top", "angle", "vis"), default="angle")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--landmarks-per-query", type=int, default=12)
    args = parser.parse_args()
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        parser.error("provide at least three unique seeds")
    if args.frame_stride <= 0 or args.landmarks_per_query < 2:
        parser.error("frame stride must be positive and at least two landmarks are required")
    return args


def main() -> int:
    result = capture(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
