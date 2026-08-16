#!/usr/bin/env python3
"""Capture observation-only frames with offline oracle phase labels.

The labels may use simulator state, but the saved learning inputs contain only
camera frames.  This makes the dataset suitable for testing whether a phase
selector can be distilled without privileged runtime inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(
    os.environ.get("SPEEDTUNING_RUNTIME_ROOT", Path(__file__).resolve().parents[1])
).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy_speed_env import create_speed_env  # noqa: E402


def scalar_feature(state: dict[str, Any], name: str) -> float | bool:
    if name in ("policy_time", "physics_steps", "task_reward", "success"):
        return state[name]
    if name.startswith("distance:"):
        _, mocap_name, object_index = name.split(":")
        mocap = np.asarray(state[f"mocap_{mocap_name}"][:3])
        start = int(object_index) * 7
        target = np.asarray(state["env_state"][start : start + 3])
        return float(np.linalg.norm(mocap - target))
    field, index = name.rsplit(".", 1)
    return float(state[field][int(index)])


def predicate_matches(spec: dict[str, Any] | None, state: dict[str, Any]) -> bool:
    if spec is None:
        return True
    if "all" in spec:
        return all(predicate_matches(item, state) for item in spec["all"])
    if "any" in spec:
        return any(predicate_matches(item, state) for item in spec["any"])
    if "not" in spec:
        return not predicate_matches(spec["not"], state)
    actual = scalar_feature(state, spec["feature"])
    expected = spec.get("value", True)
    operations = {
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
        "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b,
    }
    return bool(operations[spec.get("op", "eq")](actual, expected))


@dataclass
class SegmentState:
    status: str = "pending"
    stable: int = 0


class OfflinePhaseOracle:
    def __init__(self, controller: dict[str, Any]):
        self.segments = list(controller.get("segments", []))
        self.states = [SegmentState() for _ in self.segments]

    def label(self, state: dict[str, Any]) -> str:
        for index, (segment, segment_state) in enumerate(
            zip(self.segments, self.states)
        ):
            if segment_state.status == "released":
                continue
            if segment_state.status == "pending":
                if not predicate_matches(segment["entry"], state):
                    return "fast"
                segment_state.status = "active"
            if predicate_matches(segment["exit"], state):
                segment_state.stable += 1
            else:
                segment_state.stable = 0
            if segment_state.stable >= int(segment.get("release_stability", 1)):
                segment_state.status = "released"
                continue
            return f"segment_{index}"
        return "fast"


def snapshot(env, task_reward: float) -> dict[str, Any]:
    observation = env.cur_ts.observation
    return {
        "policy_time": float(env.policy_time),
        "physics_steps": int(env.physics_steps),
        "task_reward": float(task_reward),
        "success": bool(task_reward >= env.env.task.max_reward),
        "env_state": np.asarray(observation["env_state"]).tolist(),
        "qpos": np.asarray(observation["qpos"]).tolist(),
        "qvel": np.asarray(observation["qvel"]).tolist(),
        "mocap_left": np.asarray(observation["mocap_pose_left"]).tolist(),
        "mocap_right": np.asarray(observation["mocap_pose_right"]).tolist(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_preentry_margin(records: list[dict[str, Any]], margin: int) -> None:
    if margin <= 0:
        return
    for segment in sorted({item["oracle_label"] for item in records} - {"fast"}):
        starts = [
            index
            for index, item in enumerate(records)
            if item["oracle_label"] == segment
            and (index == 0 or records[index - 1]["oracle_label"] != segment)
        ]
        for start in starts:
            for index in range(max(0, start - margin), start):
                if records[index]["oracle_label"] == "fast":
                    records[index]["oracle_label"] = segment
                    records[index]["preentry_margin"] = True


def capture(args: argparse.Namespace) -> dict[str, Any]:
    controller_path = args.controller.resolve()
    controller = json.loads(controller_path.read_text())
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    episodes = []

    for seed in args.seeds:
        episode_root = output_root / f"seed-{seed}"
        frame_root = episode_root / args.camera
        frame_root.mkdir(parents=True, exist_ok=True)
        environment_task = "tea_bag" if args.task == "tea_bag_randomized" else args.task
        env = create_speed_env(
            task_name=environment_task,
            seed=seed,
            render_images=True,
            randomize_object_pose=args.task == "tea_bag_randomized",
            speed_values=(1.0,),
            terminate_on_success=False,
        )
        env.reset()
        oracle = OfflinePhaseOracle(controller)
        task_reward = float(env.cur_ts.reward or 0)
        records = []
        done = False
        frame_index = 0
        while not done:
            state = snapshot(env, task_reward)
            label = oracle.label(state)
            if env.physics_steps % args.stride == 0:
                relative = Path(f"seed-{seed}") / args.camera / f"{frame_index:04d}.jpg"
                imageio.imwrite(output_root / relative, env.cur_ts.observation["images"][args.camera], quality=92)
                records.append(
                    {
                        "frame_index": frame_index,
                        "image": relative.as_posix(),
                        "physics_steps": int(env.physics_steps),
                        "policy_time": float(env.policy_time),
                        "task_reward": task_reward,
                        "oracle_label": label,
                        "preentry_margin": False,
                    }
                )
                frame_index += 1
            _, _, done, info = env.step(1.0, quantized=False)
            task_reward = float(info["task_reward"])
        env.close()
        apply_preentry_margin(records, args.preentry_margin)
        episode_manifest = episode_root / "labels.json"
        episode_manifest.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        episodes.append(
            {
                "seed": seed,
                "frames": len(records),
                "success": bool(task_reward >= env.env.task.max_reward),
                "labels": episode_manifest.relative_to(output_root).as_posix(),
                "label_counts": {
                    label: sum(item["oracle_label"] == label for item in records)
                    for label in sorted({item["oracle_label"] for item in records})
                },
            }
        )

    manifest = {
        "schema": "speedtuning-offline-phase-dataset-v1",
        "task": args.task,
        "controller": str(controller_path),
        "controller_sha256": sha256(controller_path),
        "camera": args.camera,
        "frame_stride": args.stride,
        "preentry_margin_frames": args.preentry_margin,
        "runtime_privileged_signals": False,
        "offline_label_privileged_signals": True,
        "train_seed": args.seeds[0],
        "held_out_seeds": args.seeds[1:],
        "episodes": episodes,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", choices=("top", "angle", "vis"), default="angle")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--preentry-margin", type=int, default=1)
    args = parser.parse_args()
    if args.stride <= 0 or args.preentry_margin < 0:
        parser.error("stride must be positive and preentry-margin non-negative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    return args


def main() -> int:
    manifest = capture(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
