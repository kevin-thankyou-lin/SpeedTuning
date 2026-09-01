#!/usr/bin/env python3
"""Trace the frozen ACT Rainbow-50 policies on randomized scripted rollouts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import canonical_sha256, sha256  # noqa: E402
from learned_phase_observation import LearnedPhaseEncoder, PHASES  # noqa: E402
from one_reset_phase_schedule import workspace_violation  # noqa: E402
from policy_speed_env import create_speed_env  # noqa: E402
from scripts.run_act_speed_benchmark_cell import (  # noqa: E402
    DETECTOR_HASHES,
    atomic_json,
    immutable_json,
    load_contiguous_states,
)
from scripts.run_act_strider_frontier_v2 import summarize  # noqa: E402
from speed_policy import RainbowSpeedPolicy, SpeedContext  # noqa: E402


TASKS = {"pick": "pick_and_place", "tea": "tea_bag", "insertion": "insertion"}
METHODS = ("native_1x", "frozen_rainbow50")
EPISODES = 10
EXPECTED_SPEEDS = (1.0, 1.25, 1.5, 1.75, 2.0)


def checked_json(path: Path) -> dict:
    return json.loads(path.read_text())


def immutable_or_verify(path: Path, value) -> None:
    if path.exists():
        if checked_json(path) != value:
            raise RuntimeError(f"immutable receipt differs: {path}")
    else:
        immutable_json(path, value)


def bank_seeds(banks: dict, task: str) -> list[int]:
    spec = banks["tasks"][task]
    return list(range(int(spec["start"]), int(spec["start"]) + int(spec["count"])))


def validate_banks(banks: dict) -> None:
    if banks.get("schema") != "act-rainbow50-scripted-correlation-banks-v44":
        raise RuntimeError("v44 seed-bank schema differs")
    if banks.get("methods") != list(METHODS) or int(banks.get("episodes_per_method", -1)) != EPISODES:
        raise RuntimeError("v44 method or episode allocation differs")
    all_seeds = []
    for task in TASKS:
        seeds = bank_seeds(banks, task)
        if len(seeds) != EPISODES or len(seeds) != len(set(seeds)):
            raise RuntimeError(f"v44 {task} bank is not exactly ten unique resets")
        all_seeds.extend(seeds)
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("v44 task seed banks overlap")


def validate_old_artifact(search_root: Path) -> tuple[dict, Path]:
    complete_path = search_root / "COMPLETE.json"
    selected_path = search_root / "selected.json"
    result_path = search_root / "result.json"
    complete = checked_json(complete_path)
    selected = checked_json(selected_path)
    if complete.get("selected_sha256") != sha256(selected_path):
        raise RuntimeError("old Rainbow-50 selected receipt hash differs")
    if complete.get("result_sha256") != sha256(result_path):
        raise RuntimeError("old Rainbow-50 search result hash differs")
    if int(complete.get("episodes", -1)) != 50:
        raise RuntimeError("old Rainbow search was not exactly 50 episodes")
    policy_spec = selected.get("selected_policy", {})
    checkpoint = Path(policy_spec.get("checkpoint", ""))
    if not checkpoint.exists():
        fallback = search_root / "terminal_policy.pt"
        if fallback.exists() and sha256(fallback) == policy_spec.get("sha256"):
            checkpoint = fallback
    if sha256(checkpoint) != policy_spec.get("sha256"):
        raise RuntimeError("old Rainbow-50 terminal checkpoint hash differs")
    return {
        "search_complete_path": str(complete_path),
        "search_complete_sha256": sha256(complete_path),
        "search_result_path": str(result_path),
        "search_result_sha256": sha256(result_path),
        "selected_path": str(selected_path),
        "selected_sha256": sha256(selected_path),
        "terminal_checkpoint_path": str(checkpoint),
        "terminal_checkpoint_sha256": sha256(checkpoint),
    }, checkpoint


def q_values(policy: RainbowSpeedPolicy, observation: np.ndarray) -> list[float]:
    tensor = policy.torch.as_tensor(
        observation, dtype=policy.torch.float32, device=policy.device
    ).unsqueeze(0)
    with policy.torch.inference_mode():
        values = policy.network(tensor)[0].detach().cpu().numpy()
    return [float(value) for value in values]


def decode_phase_map(policy: RainbowSpeedPolicy) -> dict:
    if policy.observation_dim != len(PHASES):
        raise RuntimeError("old Rainbow policy is not the sealed phase-only controller")
    output = {}
    for index, phase in enumerate(PHASES):
        observation = np.eye(len(PHASES), dtype=np.float32)[index]
        values = q_values(policy, observation)
        order = np.argsort(values)
        action = int(order[-1])
        margin = float(values[action] - values[int(order[-2])])
        output[phase] = {
            "action_index": action,
            "speed": float(policy.speed_values[action]),
            "q_values": values,
            "q_margin": margin,
        }
    return output


def environment(task: str, seed: int, policy: RainbowSpeedPolicy, detector_checkpoint: Path,
                detector_source: Path, device: str):
    encoder = LearnedPhaseEncoder(
        checkpoint_path=detector_checkpoint,
        source_root=detector_source,
        checkpoint_sha256=DETECTOR_HASHES["checkpoint"],
        inference_sha256=DETECTOR_HASHES["inference"],
        model_source_sha256=DETECTOR_HASHES["model_source"],
        device=device,
        history_stride=5,
        cpu_threads_per_worker=2,
    )
    env = create_speed_env(
        task_name=TASKS[task],
        chunk_predictor=None,
        seed=int(seed),
        randomize_object_pose=True,
        speed_values=policy.speed_values,
        observation_encoder=encoder,
        decision_frame_skip=10,
        decision_mode="fixed_or_phase_entry",
        terminate_on_success=False,
        safety_monitor=partial(workspace_violation, TASKS[task]),
    )
    env._environment_metadata["v44_transfer_scope"] = (
        "diagnostic_only_ACT_chunked_FK_to_scripted_waypoint_mocap"
    )
    return env


def validate_transfer(policy: RainbowSpeedPolicy, env, observation: np.ndarray) -> dict:
    if tuple(policy.speed_values) != EXPECTED_SPEEDS:
        raise RuntimeError(f"old Rainbow-50 speed grid differs: {policy.speed_values}")
    if policy.observation_dim != observation.size:
        raise RuntimeError("checkpoint observation dimension differs")
    if policy.observation_spec != env.observation_spec():
        raise RuntimeError("learned phase observation preprocessing differs")
    expected = dict(policy.environment_spec or {})
    actual = env.environment_spec()
    stable_keys = ("task", "randomize_object_pose", "speed_decision_mode")
    for key in stable_keys:
        if expected.get(key) != actual.get(key):
            raise RuntimeError(f"unexpected transfer mismatch in {key}")
    if expected.get("base_policy") != "chunked" or actual.get("base_policy") != "scripted":
        raise RuntimeError("v44 did not make the registered ACT-to-scripted transfer")
    return {
        "checkpoint_environment_spec": expected,
        "scripted_environment_spec": actual,
        "allowed_differences": {
            "base_policy": {"checkpoint": "chunked", "diagnostic": "scripted"},
            "effector_position_source": {
                "checkpoint": "joint_fk_body_xpos",
                "diagnostic": "legacy_mocap_pose_fallback",
            },
        },
        "global_checkpoint_guard_weakened": False,
        "local_diagnostic_validation_path": "explicit_expected_difference_only",
        "scientific_scope": "out_of_distribution_mechanism_visualization",
    }


def rollout(task: str, seed: int, method: str, policy: RainbowSpeedPolicy,
            detector_checkpoint: Path, detector_source: Path, device: str,
            identity_sha: str) -> tuple[dict, dict | None]:
    env = environment(task, seed, policy, detector_checkpoint, detector_source, device)
    try:
        observation = env.reset()
        transfer = validate_transfer(policy, env, observation)
        decisions = []
        done = False
        info = {"success": False}
        previous_phase = None
        while not done:
            phase_index = int(np.argmax(observation))
            if phase_index != env.observation_encoder.phase_index:
                raise RuntimeError("phase one-hot and detector state disagree")
            context = SpeedContext(
                policy_time=env.policy_time,
                physics_steps=env.physics_steps,
                episode_len=env.episode_len,
                speed_values=tuple(env.speed_values),
            )
            values = q_values(policy, observation)
            learned_action = int(np.argmax(values))
            learned_speed = float(policy.speed_values[learned_action])
            chosen_speed = 1.0 if method == "native_1x" else learned_speed
            start_policy_time = float(env.policy_time)
            start_physics_steps = int(env.physics_steps)
            next_observation, _reward, done, info = env.step_decision(
                chosen_speed, frame_skip=10, quantized=False
            )
            sorted_values = np.sort(values)
            decisions.append({
                "decision_index": len(decisions),
                "nominal_progress": start_policy_time / env.episode_len,
                "policy_time": start_policy_time,
                "physics_steps": start_physics_steps,
                "phase_index": phase_index,
                "phase": PHASES[phase_index],
                "phase_entry": previous_phase is None or phase_index != previous_phase,
                "rainbow_action_index": learned_action,
                "rainbow_speed": learned_speed,
                "chosen_speed": chosen_speed,
                "q_values": values,
                "q_margin": float(sorted_values[-1] - sorted_values[-2]),
                "decision_physics_steps": int(info["decision_physics_steps"]),
                "next_phase_index": int(np.argmax(next_observation)),
                "next_phase": PHASES[int(np.argmax(next_observation))],
            })
            previous_phase = phase_index
            observation = next_observation
        acceleration = float(env.episode_len / max(info["physics_steps"], 1))
        record = {
            "seed": int(seed),
            "identity_sha256": identity_sha,
            "task_label": task,
            "method": method,
            "success": bool(info["success"]),
            "physics_steps": int(info["physics_steps"]),
            "first_success_step": (
                None if info.get("first_success_step") is None
                else int(info["first_success_step"])
            ),
            "policy_time": float(info["policy_time"]),
            "mean_speed": float(np.mean(env.speed_list)),
            "max_speed": float(np.max(env.speed_list)),
            "acceleration": acceleration,
            "successful_acceleration": acceleration if info["success"] else None,
            "decisions": decisions,
            "decision_count": len(decisions),
            "decision_frame_skip": 10,
            "observation_spec": env.observation_spec(),
            "environment_spec": env.environment_spec(),
            "physics_error": info.get("physics_error"),
            "safety_violation": info.get("safety_violation"),
        }
        return record, transfer
    finally:
        env.close()


def average_ranks(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    left, right = average_ranks(x), average_ranks(y)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def normalized_mutual_information(pairs: list[tuple[str, float]]) -> float | None:
    if not pairs:
        return None
    joint = Counter(pairs)
    phases = Counter(phase for phase, _speed in pairs)
    speeds = Counter(speed for _phase, speed in pairs)
    total = float(len(pairs))
    mi = 0.0
    for (phase, speed), count in joint.items():
        probability = count / total
        mi += probability * math.log(probability / ((phases[phase] / total) * (speeds[speed] / total)))
    phase_entropy = -sum((count / total) * math.log(count / total) for count in phases.values())
    speed_entropy = -sum((count / total) * math.log(count / total) for count in speeds.values())
    denominator = math.sqrt(phase_entropy * speed_entropy)
    return None if denominator == 0 else float(mi / denominator)


def trace_summary(records: list[dict], speed_key: str) -> dict:
    decisions = [decision for record in records for decision in record["decisions"]]
    phase_speed = [(item["phase"], float(item[speed_key])) for item in decisions]
    phase_counts = Counter(item["phase"] for item in decisions)
    transition_counts = Counter(
        (item["phase"], item["next_phase"]) for item in decisions
    )
    nominal_work = {
        phase: sum(
            float(item["decision_physics_steps"]) * float(item["chosen_speed"])
            for item in decisions if item["phase"] == phase
        )
        for phase in PHASES
    }
    total_nominal_work = sum(nominal_work.values())
    phase_speed_counts = {
        phase: dict(sorted(Counter(float(item[speed_key]) for item in decisions if item["phase"] == phase).items()))
        for phase in PHASES
    }
    deciles = {}
    for index in range(10):
        selected = [
            item for item in decisions
            if min(int(float(item["nominal_progress"]) * 10), 9) == index
        ]
        deciles[str(index)] = {
            "range": [index / 10.0, (index + 1) / 10.0],
            "decisions": len(selected),
            "speed_counts": dict(sorted(Counter(float(item[speed_key]) for item in selected).items())),
            "mean_speed": None if not selected else float(np.mean([item[speed_key] for item in selected])),
        }
    episode_correlations = [
        spearman(
            [item["nominal_progress"] for item in record["decisions"]],
            [item[speed_key] for item in record["decisions"]],
        )
        for record in records
    ]
    finite_episode_correlations = [value for value in episode_correlations if value is not None]
    return {
        "decisions": len(decisions),
        "phase_counts": dict(phase_counts),
        "phase_occupancy": {phase: phase_counts[phase] / len(decisions) for phase in PHASES},
        "phase_native_equivalent_work": nominal_work,
        "phase_native_equivalent_work_fraction": {
            phase: nominal_work[phase] / total_nominal_work for phase in PHASES
        },
        "phase_transition_counts": {
            f"{source}->{target}": count
            for (source, target), count in sorted(transition_counts.items())
        },
        "phase_speed_counts": phase_speed_counts,
        "progress_deciles": deciles,
        "normalized_mutual_information_phase_speed": normalized_mutual_information(phase_speed),
        "pooled_spearman_speed_vs_nominal_progress": spearman(
            [item["nominal_progress"] for item in decisions],
            [item[speed_key] for item in decisions],
        ),
        "episode_cluster_spearman": {
            "defined_episodes": len(finite_episode_correlations),
            "mean": None if not finite_episode_correlations else float(np.mean(finite_episode_correlations)),
            "values": episode_correlations,
        },
        "weighting": (
            "decision-weighted correlations plus native-equivalent-work phase occupancy; "
            "the latter weights each decision by executed_physics_steps times chosen_speed"
        ),
    }


def run_task(args, banks: dict, task: str) -> None:
    search_root = args.old_root / task / "learned_phase_rainbow_rl" / "search"
    source, checkpoint = validate_old_artifact(search_root)
    policy = RainbowSpeedPolicy.load(checkpoint, device=args.device)
    phase_map = decode_phase_map(policy)
    seeds = bank_seeds(banks, task)
    task_root = args.output_root / task
    source_receipt = {
        "schema": "act-rainbow50-scripted-transfer-source-v44",
        "task_label": task,
        "old_source_commit": args.old_source_commit,
        "source": source,
        "observation_semantics": "current learned four-phase one-hot only",
        "observation_dim": policy.observation_dim,
        "speed_values": list(policy.speed_values),
        "decision_mode": "fixed_or_phase_entry",
        "decision_frame_skip": policy.frame_skip,
        "decoded_phase_map": phase_map,
        "historical_rollouts_reexecuted": 0,
        "training_or_tuning_permitted": False,
    }
    source_receipt["source_receipt_sha256"] = canonical_sha256(source_receipt)
    immutable_or_verify(task_root / "SOURCE_RECEIPT.json", source_receipt)

    method_results = {}
    method_records = {}
    transfer_receipt = None
    for method in METHODS:
        output = task_root / "methods" / method
        identity = {
            "schema": "act-rainbow50-scripted-correlation-cell-v44",
            "implementation_commit": args.implementation_commit,
            "task_label": task,
            "method": method,
            "seeds": seeds,
            "source_receipt_sha256": sha256(task_root / "SOURCE_RECEIPT.json"),
            "base_policy": "scripted_waypoint",
            "reset_distribution": "fresh_randomized_object_pose",
            "learning_or_tuning_permitted": False,
            "out_of_distribution_transfer": True,
        }
        identity["identity_sha256"] = canonical_sha256(identity)
        immutable_or_verify(output / "IDENTITY.json", identity)
        records = load_contiguous_states(output / "states", seeds, identity["identity_sha256"])
        for seed in seeds[len(records):]:
            record, current_transfer = rollout(
                task, seed, method, policy, args.detector_checkpoint,
                args.detector_source, args.device, identity["identity_sha256"],
            )
            if transfer_receipt is None:
                transfer_receipt = current_transfer
            elif transfer_receipt != current_transfer:
                raise RuntimeError("v44 transfer specification changed across rollouts")
            immutable_json(output / "states" / f"{seed}.json", record)
            records.append(record)
            atomic_json(output / "progress.json", {
                "task": task,
                "method": method,
                "completed": len(records),
                "successes": sum(bool(item["success"]) for item in records),
                "physics_errors": sum(item.get("physics_error") is not None for item in records),
                "safety_violations": sum(item.get("safety_violation") is not None for item in records),
            })
            print(json.dumps({
                "task": task, "method": method, "completed": len(records),
                "successes": sum(bool(item["success"]) for item in records),
            }, sort_keys=True), flush=True)
        result = {
            "schema": "act-rainbow50-scripted-correlation-method-result-v44",
            "task_label": task,
            "method": method,
            "episodes": len(records),
            "summary": summarize(records),
            "trace_summary": trace_summary(
                records, "rainbow_speed" if method == "frozen_rainbow50" else "chosen_speed"
            ),
            "identity_sha256": identity["identity_sha256"],
        }
        immutable_or_verify(output / "RESULT.json", result)
        immutable_or_verify(output / "COMPLETE.json", {
            "schema": "act-rainbow50-scripted-correlation-method-completion-v44",
            "episodes": EPISODES,
            "result_sha256": sha256(output / "RESULT.json"),
        })
        method_results[method] = result
        method_records[method] = records

    native = method_results["native_1x"]["summary"]
    rainbow = method_results["frozen_rainbow50"]["summary"]
    task_result = {
        "schema": "act-rainbow50-scripted-correlation-task-result-v44",
        "task_label": task,
        "source_receipt": source_receipt,
        "transfer_receipt": transfer_receipt,
        "methods": method_results,
        "paired": {
            "pairs": EPISODES,
            "native_successes": native["successes"],
            "rainbow_successes": rainbow["successes"],
            "rainbow_failure_aware_throughput_ratio": (
                None if native["achieved_throughput_per_step"] <= 0
                else rainbow["achieved_throughput_per_step"] / native["achieved_throughput_per_step"]
            ),
        },
        "scientific_scope": "OOD scripted-base mechanism diagnostic; not ACT reliability evidence",
    }
    immutable_or_verify(task_root / "RESULT.json", task_result)
    immutable_or_verify(task_root / "COMPLETE.json", {
        "schema": "act-rainbow50-scripted-correlation-task-completion-v44",
        "episodes": len(METHODS) * EPISODES,
        "result_sha256": sha256(task_root / "RESULT.json"),
        "physics_errors": sum(
            result["summary"]["physics_errors"] for result in method_results.values()
        ),
        "safety_violations": sum(
            result["summary"]["safety_violations"] for result in method_results.values()
        ),
    })


def finalize(args, banks: dict) -> None:
    tasks = {}
    physics_errors = safety_violations = 0
    for task in TASKS:
        complete = checked_json(args.output_root / task / "COMPLETE.json")
        result = checked_json(args.output_root / task / "RESULT.json")
        if complete["result_sha256"] != sha256(args.output_root / task / "RESULT.json"):
            raise RuntimeError(f"v44 task completion hash differs: {task}")
        tasks[task] = result
        physics_errors += int(complete["physics_errors"])
        safety_violations += int(complete["safety_violations"])
    result = {
        "schema": "act-rainbow50-scripted-correlation-result-v44",
        "label": "Frozen ACT Rainbow-50 phase policy transferred to scripted waypoint base",
        "tasks": tasks,
        "accounting": {
            "scientific_rollouts": len(TASKS) * len(METHODS) * EPISODES,
            "historical_rollouts_reexecuted": 0,
            "training_rollouts": 0,
            "physics_errors": physics_errors,
            "safety_violations": safety_violations,
        },
        "interpretation_guard": (
            "The controller is a deterministic phase-to-speed map. Correlation with progress "
            "is mediated by detected phase, and scripted transfer is not ACT generalization evidence."
        ),
    }
    immutable_or_verify(args.output_root / "RESULT.json", result)
    immutable_or_verify(args.output_root / "COMPLETE.json", {
        "schema": "act-rainbow50-scripted-correlation-completion-v44",
        **result["accounting"],
        "banks_sha256": sha256(args.banks),
        "contract_sha256": sha256(args.contract),
        "result_sha256": sha256(args.output_root / "RESULT.json"),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--old-source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import subprocess
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if head != args.implementation_commit:
        raise RuntimeError("v44 checked-out implementation differs")
    banks = checked_json(args.banks)
    validate_banks(banks)
    for task in TASKS:
        run_task(args, banks, task)
    finalize(args, banks)
    print(json.dumps(checked_json(args.output_root / "RESULT.json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
