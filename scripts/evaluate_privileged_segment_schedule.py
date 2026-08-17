#!/usr/bin/env python3
"""Replay a frozen privileged event controller against cached matched natives."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference_schedule import EventController  # noqa: E402
from scripts.evaluate_reference_aligned_schedule import make_env, snapshot  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def controller_speeds(config: dict[str, Any]) -> tuple[float, ...]:
    speeds = {1.0, float(config["ceiling"])}
    for segment in config.get("segments", []):
        speeds.add(float(segment.get("speed", 1.0)))
        speeds.update(float(rule["speed"]) for rule in segment.get("speed_rules", []))
    return tuple(sorted(speeds))


def run_privileged(
    task: str,
    seed: int,
    config: dict[str, Any],
    arm_name: str,
) -> dict[str, Any]:
    env = make_env(task, seed, controller_speeds(config), False)
    controller = EventController(config)
    trace = []
    last_reward = 0.0
    last_success = False
    info = {"success": False, "task_reward": 0.0}
    started = time.perf_counter()
    try:
        env.reset()
        done = False
        while not done:
            pre = snapshot(env, last_reward, last_success)
            speed, reason, latch = controller.select(pre)
            trace.append(
                {
                    "physics_steps": int(env.physics_steps),
                    "policy_time": float(env.policy_time),
                    "task_reward": last_reward,
                    "success": last_success,
                    "speed": speed,
                    "reason": reason,
                    "latch": latch,
                }
            )
            _, _, done, info = env.step(speed, quantized=False)
            last_reward = float(info["task_reward"])
            last_success = bool(info["success"])
        return {
            "arm": arm_name,
            "seed": seed,
            "success": bool(info["success"]),
            "physics_steps": int(env.physics_steps),
            "policy_time": float(env.policy_time),
            "task_reward": last_reward,
            "mean_speed": float(np.mean(env.speed_list)),
            "speed_counts": dict(Counter(str(value) for value in env.speed_list)),
            "controller_events": controller.events,
            "trace": trace,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        env.close()


def summarize_privileged(native: list[dict], candidate: list[dict]) -> dict[str, Any]:
    native_success_steps = [item["physics_steps"] for item in native if item["success"]]
    native_mean = None if not native_success_steps else float(np.mean(native_success_steps))
    candidate_mean = float(np.mean([item["physics_steps"] for item in candidate]))
    return {
        "native_1x_success_rate": float(np.mean([item["success"] for item in native])),
        "candidate_success_rate": float(np.mean([item["success"] for item in candidate])),
        "native_successful_mean_physics_steps": native_mean,
        "candidate_all_mean_physics_steps": candidate_mean,
        "duration_normalized_speedup": None if native_mean is None else native_mean / candidate_mean,
        "candidate_mean_executed_speed": float(np.mean([item["mean_speed"] for item in candidate])),
        "cached_native_rollouts": len(native),
        "new_candidate_rollouts": len(candidate),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cached-native-results", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--arm-name", default="privileged_phase_boundaries")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite")
    return args


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.runtime_root.resolve()))
    controller_path = args.controller.resolve()
    controller = json.loads(controller_path.read_text())
    cached_path = (
        None
        if args.cached_native_results is None
        else args.cached_native_results.resolve()
    )
    if cached_path is None:
        native = [
            run_privileged(args.task, seed, {"ceiling": 1.0}, "native_1x")
            for seed in args.seeds
        ]
    else:
        cached = json.loads(cached_path.read_text())
        cached_native = {int(item["seed"]): item for item in cached["native_1x"]}
        missing = [seed for seed in args.seeds if seed not in cached_native]
        if missing:
            raise ValueError(f"cached native results missing seeds: {missing}")
        native = [cached_native[seed] for seed in args.seeds]
        if not all(item["success"] for item in native):
            raise ValueError("cached native reference contains failures")

    candidate = []
    for seed in args.seeds:
        candidate.append(run_privileged(args.task, seed, controller, args.arm_name))
        print(
            json.dumps({key: value for key, value in candidate[-1].items() if key != "trace"}, sort_keys=True),
            flush=True,
        )

    summary = summarize_privileged(native, candidate)
    if cached_path is None:
        summary["new_native_rollouts"] = summary.pop("cached_native_rollouts")
        summary["cached_native_rollouts"] = 0
    else:
        summary["new_native_rollouts"] = 0
    result = {
        "schema": "speedtuning-event-controller-replay-v1",
        "task": args.task,
        "seeds": args.seeds,
        "controller": controller,
        "controller_sha256": sha256(controller_path),
        "runtime_boundary_inputs_privileged": bool(controller.get("segments")),
        "boundary_evaluation_cadence_physics_steps": (
            1 if controller.get("segments") else None
        ),
        "cached_native_results": None if cached_path is None else str(cached_path),
        "cached_native_results_sha256": (
            None if cached_path is None else sha256(cached_path)
        ),
        "native_1x": native,
        "candidate": candidate,
        "summary": summary,
        "provenance": {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "controller": str(controller_path),
            "controller_sha256": sha256(controller_path),
            "runtime_root": str(args.runtime_root.resolve()),
        },
    }
    args.output.mkdir(parents=True)
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "COMPLETE").write_text(f"{sha256(result_path)}  results.json\n")
    print(json.dumps({"summary": result["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
