#!/usr/bin/env python3
"""Run a matched privileged/reference-aligned speed-schedule evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference_alignment import OnlineReferenceAligner  # noqa: E402
from reference_schedule import (  # noqa: E402
    CausalTemporalPool,
    EventController,
    expand_protected_speed_map,
    select_aligned_speed,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("embedding server closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class EncoderClient:
    def __init__(self, path: Path):
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.connect(str(path))

    def encode(self, image: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(image, dtype=np.uint8)
        header = json.dumps({"op": "encode", "shape": list(value.shape)}).encode()
        self.connection.sendall(struct.pack("!I", len(header)) + header + value.tobytes())
        size = struct.unpack("!I", read_exact(self.connection, 4))[0]
        return np.frombuffer(read_exact(self.connection, size), dtype="<f4").copy()

    def close(self) -> None:
        self.connection.close()


def contacts_as_strings(physics) -> list[str]:
    from sim_tasks import contact_pairs

    return sorted("|".join(sorted(pair)) for pair in contact_pairs(physics))


def snapshot(env, task_reward: float, success: bool) -> dict[str, Any]:
    obs = env.cur_ts.observation
    return {
        "policy_time": float(env.policy_time),
        "physics_steps": int(env.physics_steps),
        "task_reward": float(task_reward),
        "success": bool(success),
        "env_state": np.asarray(obs["env_state"]).copy(),
        "qpos": np.asarray(obs["qpos"]).copy(),
        "qvel": np.asarray(obs["qvel"]).copy(),
        "mocap_left": np.asarray(obs["mocap_pose_left"]).copy(),
        "mocap_right": np.asarray(obs["mocap_pose_right"]).copy(),
        "gripper_ctrl": np.asarray(obs["gripper_ctrl"]).copy(),
        "contacts": contacts_as_strings(env.env.physics),
    }


def make_env(task: str, seed: int, speeds: tuple[float, ...], render_images: bool):
    from policy_speed_env import create_speed_env

    environment_task = "tea_bag" if task == "tea_bag_randomized" else task
    return create_speed_env(
        task_name=environment_task,
        seed=seed,
        speed_values=speeds,
        render_images=render_images,
        randomize_object_pose=task == "tea_bag_randomized",
        terminate_on_success=False,
    )


def run_reference(
    *,
    task: str,
    seed: int,
    controller_config: dict[str, Any],
    encoder: EncoderClient,
    frame_stride: int,
    pool_frames: int,
) -> dict[str, Any]:
    env = make_env(task, seed, (1.0,), True)
    controller = EventController(controller_config)
    pool = CausalTemporalPool(pool_frames)
    descriptors = []
    speeds = []
    policy_times = []
    first_success = None
    last_reward = 0.0
    last_success = False
    try:
        env.reset()
        done = False
        while not done:
            pre = snapshot(env, last_reward, last_success)
            reference_speed, _, _ = controller.select(pre)
            if env.physics_steps % frame_stride == 0:
                frame = env.cur_ts.observation["images"]["angle"]
                descriptors.append(pool.update(encoder.encode(frame)))
                speeds.append(reference_speed)
                policy_times.append(float(env.policy_time))
            _, _, done, info = env._step_physics(1.0)
            last_reward = float(info["task_reward"])
            last_success = bool(info["success"])
            if first_success is None and last_reward >= env.env.task.max_reward:
                first_success = int(env.physics_steps)
                done = True
        return {
            "seed": seed,
            "success": first_success is not None,
            "first_success_steps": first_success,
            "attempted_steps": int(env.physics_steps),
            "descriptors": np.asarray(descriptors, dtype=np.float32),
            "speeds": np.asarray(speeds, dtype=np.float32),
            "policy_times": np.asarray(policy_times, dtype=np.float32),
            "controller_events": controller.events,
        }
    finally:
        env.close()


def run_arm(
    *,
    task: str,
    seed: int,
    arm: str,
    controller_config: dict[str, Any],
    encoder: EncoderClient,
    reference_descriptors: np.ndarray,
    speed_map: np.ndarray,
    frame_stride: int,
    pool_frames: int,
    confidence_threshold: float,
) -> dict[str, Any]:
    visual = arm.startswith("aligned_")
    ceiling = float(controller_config["ceiling"])
    speeds = tuple(sorted({1.0, ceiling, *map(float, speed_map)}))
    env = make_env(task, seed, speeds, visual)
    privileged = EventController(controller_config)
    pool = CausalTemporalPool(pool_frames)
    aligner = None
    chosen_speed = 1.0
    chosen_reason = "native"
    last_reward = 0.0
    last_success = False
    first_success = None
    speed_counts: Counter[str] = Counter()
    updates = []
    fallback_updates = 0
    started = time.perf_counter()
    try:
        env.reset()
        if visual:
            aligner = OnlineReferenceAligner(
                reference_descriptors,
                max_advance=5,
                max_backtrack=1,
                emission_temperature=0.07,
                expected_advance=1.0,
                transition_scale=1.5,
                backward_penalty=2.0,
                initialization_fraction=0.12,
                updates_per_second=10.0,
            )
        done = False
        while not done:
            pre = snapshot(env, last_reward, last_success)
            oracle_speed, oracle_reason, _ = privileged.select(pre)
            if arm == "native_1x":
                chosen_speed, chosen_reason = 1.0, "native"
            elif arm == "privileged":
                chosen_speed, chosen_reason = oracle_speed, oracle_reason
            elif env.physics_steps % frame_stride == 0:
                assert aligner is not None
                frame = env.cur_ts.observation["images"]["angle"]
                descriptor = pool.update(encoder.encode(frame))
                result = aligner.update_embedding(descriptor)
                threshold = confidence_threshold if arm == "aligned_margin" else None
                chosen_speed, used_fallback = select_aligned_speed(
                    speed_map,
                    result.reference_index,
                    result.confidence,
                    confidence_threshold=threshold,
                    fallback_speed=1.0,
                )
                chosen_reason = "confidence_fallback" if used_fallback else "reference_lookup"
                fallback_updates += int(used_fallback)
                true_position = float(np.clip(env.policy_time / env.episode_len, 0.0, 1.0))
                updates.append(
                    {
                        "physics_steps": int(env.physics_steps),
                        "policy_time": float(env.policy_time),
                        "true_reference_position": true_position,
                        "predicted_reference_position": result.reference_position,
                        "reference_index": result.reference_index,
                        "confidence": result.confidence,
                        "chosen_speed": chosen_speed,
                        "oracle_speed": oracle_speed,
                    }
                )
            speed_counts[str(chosen_speed)] += 1
            _, _, done, info = env._step_physics(chosen_speed)
            last_reward = float(info["task_reward"])
            last_success = bool(info["success"])
            if first_success is None and last_reward >= env.env.task.max_reward:
                first_success = int(env.physics_steps)
                done = True
        errors = [
            abs(item["predicted_reference_position"] - item["true_reference_position"])
            for item in updates
        ]
        return {
            "task": task,
            "seed": seed,
            "arm": arm,
            "success": first_success is not None,
            "first_success_steps": first_success,
            "attempted_steps": int(env.physics_steps),
            "final_reward": last_reward,
            "mean_commanded_speed": float(np.mean(env.speed_list)),
            "speed_counts": dict(speed_counts),
            "fallback_updates": fallback_updates,
            "alignment_updates": len(updates),
            "mean_alignment_error": None if not errors else float(np.mean(errors)),
            "p90_alignment_error": None if not errors else float(np.percentile(errors, 90)),
            "updates": updates,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        env.close()


def summarize(results: list[dict[str, Any]], native_by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    successes = [item for item in results if item["success"]]
    matched = [
        native_by_seed[item["seed"]]["first_success_steps"] / item["first_success_steps"]
        for item in successes
        if native_by_seed[item["seed"]]["success"]
    ]
    alignment_errors = [
        update["predicted_reference_position"] - update["true_reference_position"]
        for item in results
        for update in item["updates"]
    ]
    return {
        "episodes": len(results),
        "successes": len(successes),
        "success_rate": len(successes) / len(results),
        "mean_first_success_steps_successes": (
            None if not successes else float(np.mean([item["first_success_steps"] for item in successes]))
        ),
        "mean_attempted_steps": float(np.mean([item["attempted_steps"] for item in results])),
        "matched_native_speedup_mean": None if not matched else float(np.mean(matched)),
        "matched_native_speedup_median": None if not matched else float(np.median(matched)),
        "matched_successful_pairs": len(matched),
        "fallback_fraction": (
            sum(item["fallback_updates"] for item in results)
            / max(1, sum(item["alignment_updates"] for item in results))
        ),
        "mean_signed_alignment_error": (
            None if not alignment_errors else float(np.mean(alignment_errors))
        ),
        "mean_absolute_alignment_error": (
            None if not alignment_errors else float(np.mean(np.abs(alignment_errors)))
        ),
        "p90_absolute_alignment_error": (
            None if not alignment_errors else float(np.percentile(np.abs(alignment_errors), 90))
        ),
    }


def parse_seeds(value: str) -> list[int]:
    values = []
    for part in value.split(","):
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", 1))
            values.extend(range(start, stop + 1))
        else:
            values.append(int(part))
    if len(set(values)) != len(values):
        raise ValueError("seeds must be unique")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("pick_and_place", "tea_bag_randomized", "insertion"), required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--encoder-socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-seeds", required=True)
    parser.add_argument("--test-seeds", required=True)
    parser.add_argument("--p90-margin", type=float, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--pool-frames", type=int, default=10)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    sys.path.insert(0, str(runtime_root))
    args.output.mkdir(parents=True, exist_ok=False)
    controller_config = json.loads(args.controller.read_text())
    test_seeds = parse_seeds(args.test_seeds)
    reference_seeds = parse_seeds(args.reference_seeds)
    overlap = set(test_seeds) & set(reference_seeds)
    if overlap:
        raise RuntimeError(f"reference/test seed overlap: {sorted(overlap)}")
    preregistration = {
        "schema": "r3-reference-aligned-schedule-eval-v1",
        "task": args.task,
        "arms": ["native_1x", "privileged", "aligned_exact", "aligned_margin"],
        "schedule_arms": ["privileged", "aligned_exact", "aligned_margin"],
        "native_role": "matched speed denominator",
        "reference_seed_candidates": reference_seeds,
        "reference_selection": "first native-success candidate in listed order",
        "test_seeds": test_seeds,
        "controller": str(args.controller),
        "controller_sha256": sha256(args.controller),
        "runtime_root": str(runtime_root),
        "rn18_checkpoint_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        "clip_window_seconds": 1.0,
        "frame_stride_physics_steps": args.frame_stride,
        "p90_margin": args.p90_margin,
        "confidence_threshold": args.confidence_threshold,
        "fallback_speed": 1.0,
        "termination": "first full task reward",
        "no_outcome_tuning": True,
    }
    prereg_path = args.output / "preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")

    encoder = EncoderClient(args.encoder_socket)
    try:
        references = []
        reference = None
        for seed in reference_seeds:
            candidate = run_reference(
                task=args.task,
                seed=seed,
                controller_config=controller_config,
                encoder=encoder,
                frame_stride=args.frame_stride,
                pool_frames=args.pool_frames,
            )
            references.append({key: value for key, value in candidate.items() if key not in {"descriptors", "speeds", "policy_times"}})
            if candidate["success"]:
                reference = candidate
                break
        if reference is None:
            raise RuntimeError("no successful native reference among preregistered candidates")
        np.savez_compressed(
            args.output / "reference.npz",
            descriptors=reference["descriptors"],
            speeds=reference["speeds"],
            policy_times=reference["policy_times"],
        )
        margin_indices = int(np.ceil(args.p90_margin * max(1, len(reference["speeds"]) - 1)))
        expanded_map = expand_protected_speed_map(
            reference["speeds"],
            ceiling=float(controller_config["ceiling"]),
            margin_indices=margin_indices,
        )
        all_results = []
        for seed in test_seeds:
            for arm in ("native_1x", "privileged", "aligned_exact", "aligned_margin"):
                speed_map = expanded_map if arm == "aligned_margin" else reference["speeds"]
                result = run_arm(
                    task=args.task,
                    seed=seed,
                    arm=arm,
                    controller_config=controller_config,
                    encoder=encoder,
                    reference_descriptors=reference["descriptors"],
                    speed_map=speed_map,
                    frame_stride=args.frame_stride,
                    pool_frames=args.pool_frames,
                    confidence_threshold=args.confidence_threshold,
                )
                all_results.append(result)
                with (args.output / "results.jsonl").open("a") as handle:
                    handle.write(json.dumps(result, sort_keys=True) + "\n")
        native_by_seed = {item["seed"]: item for item in all_results if item["arm"] == "native_1x"}
        summaries = {
            arm: summarize(
                [item for item in all_results if item["arm"] == arm], native_by_seed
            )
            for arm in ("native_1x", "privileged", "aligned_exact", "aligned_margin")
        }
        final = {
            "task": args.task,
            "reference_seed": reference["seed"],
            "reference_attempts": references,
            "reference_frames": len(reference["speeds"]),
            "expanded_margin_indices": margin_indices,
            "summaries": summaries,
            "preregistration_sha256": sha256(prereg_path),
            "results_sha256": sha256(args.output / "results.jsonl"),
        }
        result_path = args.output / "summary.json"
        result_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
        (args.output / "COMPLETE").write_text(sha256(result_path) + "  summary.json\n")
        print(json.dumps(final, indent=2, sort_keys=True))
    finally:
        encoder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
