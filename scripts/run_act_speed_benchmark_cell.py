#!/usr/bin/env python3
"""Run one immutable search, final, or shared-native ACT benchmark cell."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_integration import build_original_act_speed_adapter  # noqa: E402
from act_speed_benchmark import (  # noqa: E402
    METHODS,
    PHASE_METHODS,
    JointEffectorObservationWrapper,
    SWEEP_METHODS,
    SPEED_VALUES,
    build_offline_artifact,
    candidate_for_episode,
    canonical_sha256,
    nonphase_observation_encoder,
    policy_from_candidate,
    preregistration,
    select_candidate,
    sha256,
)
from learned_phase_observation import LearnedPhaseEncoder  # noqa: E402
from one_reset_phase_schedule import workspace_violation  # noqa: E402
from original_act import set_seed  # noqa: E402
from policy_speed_env import create_speed_env, make_speed_reward  # noqa: E402
from speed_policy import (  # noqa: E402
    FixedSpeedPolicy,
    RainbowSpeedPolicy,
    rollout_speed_policy,
    summarize_rollouts,
)
from tabular_phase_speed import TabularPhaseSpeedPolicy, phase_index  # noqa: E402


DETECTOR_HASHES = {
    "checkpoint": "c25c3f530da42eb7c60e5f70405b3a99c56ab72c1e53dfd27055dc3d99c3512d",
    "inference": "1398e1d1b5b4e682f009c6501598e651a516341f6d60822f40fc575a40061815",
    "model_source": "8a47f110f19f4e52a39b7e0e4f2273c2895690f6332ab17a4b71c8eb5ce4ae37",
}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def immutable_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_torch(path: Path, value) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(value, temporary)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def load_contiguous_states(directory: Path, seeds: list[int], identity: str) -> list[dict]:
    records = []
    missing = False
    for seed in seeds:
        path = directory / f"{seed}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError("state receipts contain a non-contiguous suffix")
        value = json.loads(path.read_text())
        if value.get("seed") != seed or value.get("identity_sha256") != identity:
            raise RuntimeError(f"resume identity mismatch in {path}")
        records.append(value)
    return records


def checked_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"artifact hash mismatch: {path}: {actual} != {expected}")
    return actual


def method_source_hashes() -> dict:
    return {
        name: sha256(REPO_ROOT / name)
        for name in (
            "act_integration.py",
            "act_speed_benchmark.py",
            "learned_phase_observation.py",
            "policy_speed_env.py",
            "scripts/run_act_speed_benchmark_cell.py",
            "speed_policy.py",
            "tabular_phase_speed.py",
        )
    }


class CellRuntime:
    def __init__(self, args, contract, manifest, task, prereg, adapter):
        self.args = args
        self.contract = contract
        self.manifest = manifest
        self.task = task
        self.prereg = prereg
        self.adapter = adapter

    def phase_encoder(self):
        if self.args.method not in PHASE_METHODS:
            raise RuntimeError("non-phase method requested detector construction")
        return LearnedPhaseEncoder(
            checkpoint_path=self.args.detector_checkpoint,
            source_root=self.args.detector_source,
            checkpoint_sha256=DETECTOR_HASHES["checkpoint"],
            inference_sha256=DETECTOR_HASHES["inference"],
            model_source_sha256=DETECTOR_HASHES["model_source"],
            device=self.args.device,
            history_stride=5,
            cpu_threads_per_worker=2,
        )

    def environment(self, seed: int, *, training=False):
        phase = self.args.method in PHASE_METHODS
        if phase:
            encoder = self.phase_encoder()
            decision_mode = "fixed_or_phase_entry"
        else:
            encoder = nonphase_observation_encoder(self.args.method)
            decision_mode = "fixed"
        reward = None
        if training:
            config = self.prereg["training"]["reward"]
            reward = make_speed_reward(
                config["success_bonus"], config["speed_weight"], config["speed_power"]
            )
        environment = create_speed_env(
            task_name=self.task["task"],
            reward_fn=reward,
            chunk_predictor=self.adapter,
            seed=int(seed),
            randomize_object_pose=True,
            speed_values=SPEED_VALUES,
            observation_encoder=encoder,
            decision_frame_skip=10,
            decision_mode=decision_mode,
            terminate_on_success=False,
            safety_monitor=partial(workspace_violation, self.task["task"]),
        )
        if phase:
            environment.env = JointEffectorObservationWrapper(environment.env)
            environment._environment_metadata["learned_phase_effector_source"] = (
                "joint_fk_body_xpos"
            )
        return environment


def prepare(args):
    if git_head() != args.source_commit:
        raise RuntimeError("checked-out source does not match requested commit")
    contract = json.loads(args.contract.read_text())
    manifest = json.loads(args.run_manifest.read_text())
    if contract.get("schema") != "act-speed-benchmark-v1":
        raise RuntimeError("unexpected contract schema")
    if manifest.get("source", {}).get("commit") != args.source_commit:
        raise RuntimeError("run manifest source mismatch")
    if manifest.get("contract", {}).get("sha256") != sha256(args.contract):
        raise RuntimeError("run manifest contract mismatch")
    if not manifest.get("parity_gate", {}).get("passed"):
        raise RuntimeError("run manifest does not carry a passed uniform-1x parity gate")
    if args.task_label not in contract["tasks"]:
        raise ValueError("unknown task label")
    if args.stage != "native" and args.method not in METHODS:
        raise ValueError("unknown method")
    if args.stage == "native" and args.method != "native_1x":
        raise ValueError("native stage requires method=native_1x")

    phase = args.method in PHASE_METHODS
    supplied_detector = args.detector_checkpoint is not None or args.detector_source is not None
    if phase and not (
        args.detector_checkpoint is not None and args.detector_source is not None
    ):
        raise RuntimeError("phase method requires the frozen detector paths")
    if not phase and supplied_detector:
        raise RuntimeError("non-phase method must not receive detector information")
    if phase:
        checked_hash(args.detector_checkpoint, DETECTOR_HASHES["checkpoint"])
        checked_hash(
            args.detector_source / "phase_detector/rgb_inference.py",
            DETECTOR_HASHES["inference"],
        )
        checked_hash(
            args.detector_source / "phase_detector/rgb_proprio.py",
            DETECTOR_HASHES["model_source"],
        )

    task = contract["tasks"][args.task_label]
    task_manifest = manifest["tasks"][args.task_label]
    root = Path(task["root"])
    checkpoint = root / "checkpoints/policy_best.ckpt"
    stats = root / "checkpoints/dataset_stats.pkl"
    config = root / "checkpoints/policy_config.json"
    for name, path in (
        ("policy_best.ckpt", checkpoint),
        ("dataset_stats.pkl", stats),
        ("policy_config.json", config),
    ):
        checked_hash(path, task_manifest["artifacts"][name])
    set_seed(1000)
    adapter = build_original_act_speed_adapter(
        task_name=task["task"],
        checkpoint=checkpoint,
        stats_path=stats,
        policy_config_path=config,
        temporal_ensemble_m=0.01,
        device=args.device,
    )
    prereg = (
        None
        if args.stage == "native"
        else preregistration(
            args.method,
            search_rollouts=len(task_manifest["search_bank"]["seeds"]),
            final_rollouts=len(task_manifest["final_bank"]["seeds"]),
        )
    )
    return contract, manifest, task, task_manifest, prereg, adapter


def publish_identity(args, task_manifest, prereg, output, offline_artifact=None):
    bank_name = "final_bank" if args.stage in {"final", "native"} else "search_bank"
    seeds = list(task_manifest[bank_name]["seeds"])
    identity = {
        "schema": "act-speed-cell-identity-v1",
        "source_commit": args.source_commit,
        "run_manifest_sha256": sha256(args.run_manifest),
        "task_label": args.task_label,
        "method": args.method,
        "stage": args.stage,
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "policy_artifacts": task_manifest["artifacts"],
        "method_source_sha256": method_source_hashes(),
        "preregistration_sha256": (
            None if prereg is None else prereg["preregistration_sha256"]
        ),
        "offline_artifact_payload_sha256": (
            None if offline_artifact is None else offline_artifact["artifact_payload_sha256"]
        ),
        "detector": DETECTOR_HASHES if args.method in PHASE_METHODS else None,
        "controller": {
            "cameras": ["angle", "left_wrist", "right_wrist"],
            "progress_clock": "nominal_policy_time",
            "per_physics_step_inference": True,
            "temporal_ensemble_m": 0.01,
            "physics_error_policy": "count_as_failure_and_continue",
            "safety_monitor": "workspace_violation_every_physics_tick",
        },
        "search_receipt_import": (
            None
            if args.import_search_root is None
            else {
                "origin_root": str(args.import_search_root.resolve()),
                "count": args.import_search_count,
                "reason": args.import_reason,
                "origin_identity_sha256": sha256(args.import_search_root / "identity.json"),
                "origin_preregistration_sha256": sha256(
                    args.import_search_root / "preregistration.json"
                ),
            }
        ),
    }
    identity_hash = canonical_sha256(identity)
    identity["identity_sha256"] = identity_hash
    path = output / "identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise RuntimeError("output root contains a different cell identity")
    else:
        immutable_json(path, identity)
    return identity_hash, seeds


def import_search_receipts(args, output, seeds, identity, prereg):
    if args.import_search_root is None:
        return
    if args.stage != "search" or not 0 < args.import_search_count < 50:
        raise RuntimeError("receipt import requires a bounded search prefix")
    origin = args.import_search_root.resolve()
    if any((output / "states").glob("*.json")):
        return
    if json.loads((origin / "preregistration.json").read_text()) != prereg:
        raise RuntimeError("imported search preregistration differs from current preregistration")
    origin_identity = json.loads((origin / "identity.json").read_text())
    if origin_identity.get("method") != args.method or origin_identity.get("task_label") != args.task_label:
        raise RuntimeError("imported search cell identity targets a different cell")
    imported = []
    for seed in seeds[: args.import_search_count]:
        origin_path = origin / "states" / f"{seed}.json"
        if not origin_path.exists():
            raise RuntimeError(f"missing immutable origin receipt {origin_path}")
        value = json.loads(origin_path.read_text())
        if value.get("seed") != seed or value.get("identity_sha256") != origin_identity["identity_sha256"]:
            raise RuntimeError(f"origin receipt identity mismatch: {origin_path}")
        copied = dict(value)
        copied["identity_sha256"] = identity
        copied["imported_rollout_receipt"] = {
            "origin_path": str(origin_path),
            "origin_sha256": sha256(origin_path),
            "origin_identity_sha256": origin_identity["identity_sha256"],
            "origin_source_commit": origin_identity["source_commit"],
            "rollout_reexecuted": False,
        }
        destination = output / "states" / f"{seed}.json"
        immutable_json(destination, copied)
        imported.append({"seed": seed, "origin_sha256": sha256(origin_path), "destination_sha256": sha256(destination)})
    immutable_json(
        output / "IMPORT.json",
        {
            "schema": "act-speed-search-receipt-import-v1",
            "identity_sha256": identity,
            "origin_root": str(origin),
            "reason": args.import_reason,
            "count": len(imported),
            "rollouts_reexecuted": 0,
            "receipts": imported,
        },
    )


def rollout_one(runtime, seed, policy, identity, **extra):
    env = runtime.environment(seed)
    try:
        record = rollout_speed_policy(env, policy, frame_skip=10)
        record.update(
            seed=int(seed),
            identity_sha256=identity,
            observation_spec=env.observation_spec(),
            environment_spec=env.environment_spec(),
            **extra,
        )
        return record
    finally:
        env.close()


def progress(output, identity, records, seeds):
    atomic_json(
        output / "progress.json",
        {
            "schema": "act-speed-cell-progress-v1",
            "identity_sha256": identity,
            "completed": len(records),
            "successes": sum(bool(item["success"]) for item in records),
            "safety_violations": sum(
                item.get("safety_violation") is not None for item in records
            ),
            "physics_errors": sum("physics_error" in item for item in records),
            "next_seed": None if len(records) == len(seeds) else seeds[len(records)],
        },
    )


def run_sweep_search(runtime, output, identity, seeds, records, offline_artifact):
    states = output / "states"
    for index, seed in enumerate(seeds[len(records) :], start=len(records)):
        candidate = candidate_for_episode(runtime.prereg, index)
        effective = candidate
        if offline_artifact is not None:
            effective = offline_artifact["candidates"][index // 10]
        policy = policy_from_candidate(
            runtime.args.method, effective, offline_artifact=offline_artifact
        )
        record = rollout_one(
            runtime,
            seed,
            policy,
            identity,
            candidate_id=candidate["id"],
            candidate_index=index // 10,
        )
        immutable_json(states / f"{seed}.json", record)
        records.append(record)
        progress(output, identity, records, seeds)
        print(json.dumps({"completed": len(records), "successes": sum(r["success"] for r in records)}), flush=True)
    selection = select_candidate(runtime.prereg, records, offline_artifact)
    selected = {
        "schema": "act-speed-selected-method-v1",
        "method": runtime.args.method,
        "task_label": runtime.args.task_label,
        "identity_sha256": identity,
        "selection": selection,
        "selected_policy": selection["selected"]["candidate"],
        "terminal_artifact_only": False,
    }
    immutable_json(output / "selected.json", selected)
    return selected


def tabular_rebuild(records, action_count):
    q_values = np.zeros((4, action_count), dtype=np.float64)
    visits = np.zeros_like(q_values, dtype=np.int64)
    for record in records:
        returns = []
        value = 0.0
        for transition in reversed(record["training_trajectory"]):
            value = float(transition["reward"]) + 0.97 * value
            returns.append((transition["phase"], transition["action"], value))
        for phase, action, value in reversed(returns):
            visits[phase, action] += 1
            q_values[phase, action] += (
                value - q_values[phase, action]
            ) / visits[phase, action]
    return q_values, visits


def run_tabular_search(runtime, output, identity, seeds, records):
    q_values, visits = tabular_rebuild(records, len(SPEED_VALUES))
    rng = np.random.default_rng(runtime.prereg["training"]["seed"])
    if records:
        rng.bit_generator.state = records[-1]["rng_state_after"]
    states = output / "states"
    for index, seed in enumerate(seeds[len(records) :], start=len(records)):
        epsilon = 1.0 + index / max(len(seeds) - 1, 1) * (0.05 - 1.0)
        env = runtime.environment(seed, training=True)
        trajectory = []
        try:
            observation = env.reset()
            done = False
            info = {"success": False}
            total_reward = 0.0
            while not done:
                phase = phase_index(observation)
                if rng.random() < epsilon:
                    action = int(rng.integers(len(SPEED_VALUES)))
                else:
                    maxima = np.flatnonzero(q_values[phase] == q_values[phase].max())
                    action = int(rng.choice(maxima))
                observation, reward, done, info = env.step_decision(action)
                total_reward += float(reward)
                trajectory.append(
                    {"phase": phase, "action": action, "speed": SPEED_VALUES[action], "reward": float(reward)}
                )
            physics_steps = int(info["physics_steps"])
            record = {
                "seed": int(seed),
                "identity_sha256": identity,
                "success": bool(info["success"]),
                "return": total_reward,
                "physics_steps": physics_steps,
                "first_success_step": info.get("first_success_step"),
                "policy_time": float(info["policy_time"]),
                "mean_speed": float(np.mean(env.speed_list)),
                "max_speed": float(np.max(env.speed_list)),
                "acceleration": float(env.episode_len / max(physics_steps, 1)),
                "successful_acceleration": float(env.episode_len / max(physics_steps, 1)) if info["success"] else None,
                "safety_violation": info.get("safety_violation"),
                "epsilon": epsilon,
                "training_trajectory": trajectory,
                "rng_state_after": rng.bit_generator.state,
                "observation_spec": env.observation_spec(),
                "environment_spec": env.environment_spec(),
            }
            if "physics_error" in info:
                record["physics_error"] = str(info["physics_error"])
        finally:
            env.close()
        immutable_json(states / f"{seed}.json", record)
        records.append(record)
        q_values, visits = tabular_rebuild(records, len(SPEED_VALUES))
        progress(output, identity, records, seeds)
        print(json.dumps({"completed": len(records), "successes": sum(r["success"] for r in records)}), flush=True)
    policy = TabularPhaseSpeedPolicy(q_values, SPEED_VALUES)
    selected = {
        "schema": "act-speed-selected-method-v1",
        "method": runtime.args.method,
        "task_label": runtime.args.task_label,
        "identity_sha256": identity,
        "terminal_artifact_only": True,
        "selected_policy": {
            "algorithm": "tabular_monte_carlo_phase_speed",
            "q_values": q_values.tolist(),
            "visits": visits.tolist(),
            "speed_values": list(SPEED_VALUES),
            "schedule": list(policy.schedule),
        },
    }
    immutable_json(output / "selected.json", selected)
    return selected


def normalization_stats(states):
    value = np.asarray(states, dtype=np.float32)
    return {"states_mean": value.mean(axis=0), "states_std": np.maximum(value.std(axis=0), 1e-6)}


def rainbow_snapshot(agent, decision, update_count, history):
    import torch

    return {
        "schema": "act-speed-rainbow-resume-v1",
        "dqn": agent.dqn.state_dict(),
        "target": agent.dqn_target.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "memory": agent.memory.__dict__,
        "memory_n": None if not agent.use_n_step else agent.memory_n.__dict__,
        "epsilon": agent.epsilon,
        "beta": agent.beta,
        "decision": decision,
        "update_count": update_count,
        "history": list(history),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rainbow(agent, snapshot):
    import torch

    agent.dqn.load_state_dict(snapshot["dqn"])
    agent.dqn_target.load_state_dict(snapshot["target"])
    agent.optimizer.load_state_dict(snapshot["optimizer"])
    agent.memory.__dict__.update(snapshot["memory"])
    if agent.use_n_step:
        agent.memory_n.__dict__.update(snapshot["memory_n"])
    agent.epsilon = float(snapshot["epsilon"])
    agent.beta = float(snapshot["beta"])
    random.setstate(snapshot["python_rng"])
    np.random.set_state(snapshot["numpy_rng"])
    torch.set_rng_state(snapshot["torch_rng"])
    if torch.cuda.is_available() and snapshot["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
    return int(snapshot["decision"]), int(snapshot["update_count"]), deque(snapshot["history"], maxlen=4096)


def run_rainbow_search(runtime, output, identity, seeds, records):
    import torch
    from rl.rainbowDQN.dqnAgent import DQNAgent

    config = runtime.prereg["training"]
    probe = runtime.environment(
        seeds[len(records)] if len(records) < len(seeds) else seeds[-1], training=True
    )
    try:
        agent = DQNAgent(
            probe,
            memory_size=config["memory_size"], batch_size=config["batch_size"],
            target_update=config["target_update"], seed=config["seed"],
            lr=config["learning_rate"], gamma=config["gamma"], tau=config["tau"],
            frame_skip=10, epsilon=config["epsilon_start"], epsilon_decay=1.0,
            min_epsilon=config["epsilon_end"], exploration_steps=config["exploration_decisions"],
            alpha=config["per_alpha"], beta=config["per_beta"],
            atom_size=config["atom_size"], v_min=config["v_min"], v_max=config["v_max"],
            n_step=config["n_step"], hidden_dim=config["hidden_dim"], device=runtime.args.device,
        )
    finally:
        probe.close()
    decision = 0
    update_count = 0
    history = deque(maxlen=4096)
    if records:
        resume_path = output / records[-1]["resume_checkpoint"]
        if sha256(resume_path) != records[-1]["resume_checkpoint_sha256"]:
            raise RuntimeError("Rainbow resume checkpoint hash mismatch")
        snapshot = torch.load(resume_path, map_location=runtime.args.device, weights_only=False)
        decision, update_count, history = restore_rainbow(agent, snapshot)

    states = output / "states"
    for episode_index, seed in enumerate(seeds[len(records) :], start=len(records)):
        env = runtime.environment(seed, training=True)
        agent.env = env
        try:
            state = env.reset()
            history.append(state.copy())
            done = False
            total_reward = 0.0
            losses = []
            actions = []
            info = {"success": False}
            while not done:
                decision += 1
                action = agent.select_action(state)
                next_state, reward, done, info = agent.step(action, 10)
                history.append(next_state.copy())
                total_reward += float(reward)
                actions.append(int(action))
                fraction = min(decision / config["exploration_decisions"], 1.0)
                agent.epsilon = config["epsilon_start"] + fraction * (
                    config["epsilon_end"] - config["epsilon_start"]
                )
                agent.beta = config["per_beta"] + fraction * (1.0 - config["per_beta"])
                if len(agent.memory) >= config["learning_starts"]:
                    stats = normalization_stats(history)
                    agent.dqn.update_norm_stats(stats)
                    agent.dqn_target.update_norm_stats(stats)
                    for _ in range(config["gradient_steps"]):
                        losses.append(agent.update_model())
                        update_count += 1
                        if update_count % config["target_update"] == 0:
                            agent._target_soft_update()
                state = next_state
            physics_steps = int(info["physics_steps"])
            record = {
                "seed": int(seed), "identity_sha256": identity,
                "success": bool(info["success"]), "return": total_reward,
                "physics_steps": physics_steps, "first_success_step": info.get("first_success_step"),
                "policy_time": float(info["policy_time"]), "mean_speed": float(np.mean(env.speed_list)),
                "max_speed": float(np.max(env.speed_list)),
                "acceleration": float(env.episode_len / max(physics_steps, 1)),
                "successful_acceleration": float(env.episode_len / max(physics_steps, 1)) if info["success"] else None,
                "safety_violation": info.get("safety_violation"), "decision": decision,
                "update_count": update_count, "epsilon_after": agent.epsilon,
                "loss_last": None if not losses else float(losses[-1]), "actions": actions,
                "observation_spec": env.observation_spec(), "environment_spec": env.environment_spec(),
            }
            if "physics_error" in info:
                record["physics_error"] = str(info["physics_error"])
        finally:
            env.close()
        checkpoint = output / "resume" / f"episode-{episode_index + 1:02d}.pt"
        immutable_torch(checkpoint, rainbow_snapshot(agent, decision, update_count, history))
        record["resume_checkpoint"] = str(checkpoint.relative_to(output))
        record["resume_checkpoint_sha256"] = sha256(checkpoint)
        immutable_json(states / f"{seed}.json", record)
        records.append(record)
        progress(output, identity, records, seeds)
        print(json.dumps({"completed": len(records), "successes": sum(r["success"] for r in records), "decisions": decision}), flush=True)

    agent.dqn.update_norm_stats(normalization_stats(history))
    terminal = output / "terminal_policy.pt"
    payload = {
        "format_version": 2, "algorithm": "rainbow_dqn",
        "model_state_dict": agent.dqn.state_dict(), "observation_dim": int(agent.dqn.in_dim),
        "speed_values": list(SPEED_VALUES), "atom_size": config["atom_size"],
        "v_min": config["v_min"], "v_max": config["v_max"], "hidden_dim": config["hidden_dim"],
        "seed": config["seed"], "training_config": config, "completed_decisions": decision,
        "decision_frame_skip": 10, "reward_aggregation": "undiscounted_sum_per_decision",
        "observation_spec": records[-1]["observation_spec"],
        "environment_spec": records[-1]["environment_spec"],
        "observation_encoder_state_dict": None,
        "metadata": {
            "terminal_after_exact_search_episodes": len(seeds),
            "identity_sha256": identity,
        },
    }
    immutable_torch(terminal, payload)
    selected = {
        "schema": "act-speed-selected-method-v1", "method": runtime.args.method,
        "task_label": runtime.args.task_label, "identity_sha256": identity,
        "terminal_artifact_only": True,
        "selected_policy": {"algorithm": "rainbow_dqn", "checkpoint": str(terminal), "sha256": sha256(terminal), "completed_decisions": decision},
    }
    immutable_json(output / "selected.json", selected)
    return selected


def finish_search(runtime, output, identity, records, selected):
    expected_episodes = len(runtime.manifest["tasks"][runtime.args.task_label]["search_bank"]["seeds"])
    summary = summarize_rollouts(records)
    summary.update(
        schema="act-speed-search-result-v1", method=runtime.args.method,
        task_label=runtime.args.task_label, identity_sha256=identity,
        manifest_path=str(runtime.args.run_manifest),
        policy_checkpoint_sha256=runtime.manifest["tasks"][runtime.args.task_label]["artifacts"]["policy_best.ckpt"],
        selected_path=str(output / "selected.json"), selected_sha256=sha256(output / "selected.json"),
        states_path=str(output / "states"),
        exact_budget_complete=len(records) == expected_episodes,
    )
    immutable_json(output / "result.json", summary)
    immutable_json(
        output / "COMPLETE.json",
        {"schema": "act-speed-search-completion-v1", "identity_sha256": identity, "episodes": expected_episodes, "result_sha256": sha256(output / "result.json"), "selected_sha256": sha256(output / "selected.json")},
    )
    return summary


def load_selected_policy(runtime, search_root, offline_artifact):
    complete = search_root / "COMPLETE.json"
    selected_path = search_root / "selected.json"
    if not complete.exists() or not selected_path.exists():
        raise RuntimeError("final bank cannot open before search completion and selection")
    selected = json.loads(selected_path.read_text())
    method = runtime.args.method
    if method in SWEEP_METHODS:
        policy = policy_from_candidate(method, selected["selected_policy"], offline_artifact)
    elif method == "learned_phase_tabular_rl":
        value = selected["selected_policy"]
        policy = TabularPhaseSpeedPolicy(value["q_values"], value["speed_values"])
    elif method == "learned_phase_rainbow_rl":
        checkpoint = Path(selected["selected_policy"]["checkpoint"])
        checked_hash(checkpoint, selected["selected_policy"]["sha256"])
        policy = RainbowSpeedPolicy.load(checkpoint, device=runtime.args.device)
    else:
        raise ValueError(method)
    return policy, selected_path, selected


def run_evaluation(runtime, output, identity, seeds, records, policy, selected_path=None):
    states = output / "states"
    for seed in seeds[len(records) :]:
        record = rollout_one(runtime, seed, policy, identity)
        immutable_json(states / f"{seed}.json", record)
        records.append(record)
        progress(output, identity, records, seeds)
        print(json.dumps({"completed": len(records), "successes": sum(r["success"] for r in records)}), flush=True)
    summary = summarize_rollouts(records)
    summary.update(
        schema="act-speed-final-result-v1" if runtime.args.stage == "final" else "act-speed-native-result-v1",
        method=runtime.args.method, task_label=runtime.args.task_label,
        identity_sha256=identity, states_path=str(states), exact_budget_complete=len(records) == len(seeds),
        manifest_path=str(runtime.args.run_manifest),
        policy_checkpoint_sha256=runtime.manifest["tasks"][runtime.args.task_label]["artifacts"]["policy_best.ckpt"],
    )
    if selected_path is not None:
        summary.update(method_artifact_path=str(selected_path), method_artifact_sha256=sha256(selected_path))
    immutable_json(output / "result.json", summary)
    immutable_json(
        output / "COMPLETE.json",
        {"schema": "act-speed-final-completion-v1", "identity_sha256": identity, "episodes": len(seeds), "result_sha256": sha256(output / "result.json")},
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--stage", choices=("search", "final", "native"), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--search-root", type=Path)
    parser.add_argument("--detector-checkpoint", type=Path)
    parser.add_argument("--detector-source", type=Path)
    parser.add_argument("--import-search-root", type=Path)
    parser.add_argument("--import-search-count", type=int, default=0)
    parser.add_argument("--import-reason")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    contract, manifest, task, task_manifest, prereg, adapter = prepare(args)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lane.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another healthy process owns {output}") from exc

    offline_artifact = None
    if args.method in {"awe_offline_proxy", "sail_inspired_adaptive"}:
        offline_path = output / "offline_artifact.json"
        if args.stage == "final":
            if args.search_root is None:
                raise RuntimeError("final stage requires search-root")
            offline_path = args.search_root / "offline_artifact.json"
            if not offline_path.exists():
                raise RuntimeError("selected offline artifact is missing")
            offline_artifact = json.loads(offline_path.read_text())
        else:
            generated = build_offline_artifact(Path(task["root"]) / "dataset", args.method)
            if offline_path.exists():
                if json.loads(offline_path.read_text()) != generated:
                    raise RuntimeError("offline artifact identity mismatch")
            else:
                immutable_json(offline_path, generated)
            offline_artifact = generated
    if prereg is not None:
        prereg_path = output / "preregistration.json"
        if args.stage == "search":
            if prereg_path.exists():
                if json.loads(prereg_path.read_text()) != prereg:
                    raise RuntimeError("preregistration mismatch")
            else:
                immutable_json(prereg_path, prereg)

    identity, seeds = publish_identity(args, task_manifest, prereg, output, offline_artifact)
    import_search_receipts(args, output, seeds, identity, prereg)
    records = load_contiguous_states(output / "states", seeds, identity)
    if (output / "COMPLETE.json").exists():
        if len(records) != len(seeds):
            raise RuntimeError("completion marker exists without the exact registered bank")
        print(json.dumps(json.loads((output / "result.json").read_text()), sort_keys=True))
        return 0
    runtime = CellRuntime(args, contract, manifest, task, prereg, adapter)

    if args.stage == "search":
        if args.method in SWEEP_METHODS:
            selected = run_sweep_search(runtime, output, identity, seeds, records, offline_artifact)
        elif args.method == "learned_phase_tabular_rl":
            selected = run_tabular_search(runtime, output, identity, seeds, records)
        else:
            selected = run_rainbow_search(runtime, output, identity, seeds, records)
        result = finish_search(runtime, output, identity, records, selected)
    elif args.stage == "final":
        if args.search_root is None:
            raise RuntimeError("final stage requires search-root")
        policy, selected_path, _ = load_selected_policy(runtime, args.search_root, offline_artifact)
        result = run_evaluation(runtime, output, identity, seeds, records, policy, selected_path)
    else:
        result = run_evaluation(runtime, output, identity, seeds, records, FixedSpeedPolicy(1.0))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
