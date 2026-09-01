#!/usr/bin/env python3
"""Continue V41 Rainbow from episode 18 to 50 on the same three poses."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SPEEDTUNING_SPEED_VALUES", "1,1.5,2,2.5,3")

from act_speed_benchmark import SPEED_VALUES, canonical_sha256, preregistration  # noqa: E402
from scripts import build_three_reset_rainbow_panel_v41 as panel_builder  # noqa: E402
from scripts import run_act_champion_challenger_v37 as v37  # noqa: E402
from scripts import run_act_three_reset_rainbow25_v41 as v41  # noqa: E402
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head  # noqa: E402
from scripts.run_act_speed_benchmark_cell import (  # noqa: E402
    atomic_json,
    immutable_json,
    immutable_torch,
    load_contiguous_states,
    normalization_stats,
    progress,
    rainbow_snapshot,
    restore_rainbow,
    rollout_one,
)
from scripts.run_act_strider_frontier_v4 import file_sha256, summarize  # noqa: E402
from speed_policy import FixedSpeedPolicy, RainbowSpeedPolicy  # noqa: E402


TASKS = v41.TASKS
SEARCH_METHOD = "three_reset_rainbow50_extension"
FINAL_METHODS = v41.FINAL_METHODS
PARENT_TRAINING_EPISODES = 18
EXTENSION_EPISODES = 32
TOTAL_TRAINING_EPISODES = PARENT_TRAINING_EPISODES + EXTENSION_EPISODES
FIXED_PROBE_EPISODES = 3
SCREEN_EPISODES = 10
SCREEN_SUCCESS_FLOOR = 9
FINAL_EPISODES = 50
NEW_PREFINAL_ROLLOUTS = EXTENSION_EPISODES + 2 * FIXED_PROBE_EPISODES + SCREEN_EPISODES


checked_json = v41.checked_json
immutable_or_verify = v41.immutable_or_verify
incidents = v41.incidents


def extension_pose_order() -> list[int]:
    return [index % 3 for index in range(EXTENSION_EPISODES)]


def validate_banks(banks: dict) -> None:
    if banks.get("schema") != "act-three-reset-rainbow50-banks-v42":
        raise RuntimeError("v42 bank schema differs")
    parent_banks = checked_json(
        REPO_ROOT / "experiments/act_three_reset_rainbow25_v41/BANKS.json"
    )
    fresh_values: list[int] = []
    for task in TASKS:
        spec = banks["tasks"][task]
        parent = parent_banks["tasks"][task]
        if list(map(int, spec["parent_training"])) != list(map(int, parent["training"])):
            raise RuntimeError(f"v42 parent training bank differs: {task}")
        if int(spec["pose_design_seed"]) != int(parent["pose_design_seed"]):
            raise RuntimeError(f"v42 pose design differs from v41: {task}")
        expected = {
            "extension": EXTENSION_EPISODES,
            "fixed_probe": FIXED_PROBE_EPISODES,
            "screen": SCREEN_EPISODES,
            "final": FINAL_EPISODES,
        }
        values: list[int] = []
        for key, count in expected.items():
            seeds = list(map(int, spec[key]))
            if len(seeds) != count:
                raise RuntimeError(f"v42 {task}/{key} count differs")
            values.extend(seeds)
        if len(values) != len(set(values)) or min(values) < 420000000:
            raise RuntimeError(f"v42 fresh banks overlap or are not fresh: {task}")
        if set(values).intersection(map(int, spec["parent_training"])):
            raise RuntimeError(f"v42 fresh banks overlap parent training: {task}")
        fresh_values.extend(values)
    if len(fresh_values) != len(set(fresh_values)):
        raise RuntimeError("v42 cross-task fresh banks overlap")


def validate_parent_manifest(parent: dict) -> None:
    if parent.get("schema") != "act-three-reset-rainbow50-parent-v41":
        raise RuntimeError("v42 parent manifest schema differs")
    if parent.get("implementation_commit") != "231fef194fae83d1cc68558c33bc14ea44552b0c":
        raise RuntimeError("v42 parent implementation differs")
    for task in TASKS:
        values = parent["tasks"][task]
        for key in ("search_complete_sha256", "selection_sha256", "three_poses_sha256"):
            value = values.get(key, "")
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise RuntimeError(f"v42 parent hash invalid: {task}/{key}")


def rainbow_qualifies(fixed_successes: int, screen_successes: int,
                      incident_totals: dict) -> bool:
    return (
        int(fixed_successes) == FIXED_PROBE_EPISODES
        and int(screen_successes) >= SCREEN_SUCCESS_FLOOR
        and incident_totals == {"physics_errors": 0, "safety_violations": 0}
    )


class V42Runtime(v41.V41Runtime):
    """V41 runtime with new continuation and fixed-probe seed mappings."""

    def __init__(self, frontier: ACTFrontierRuntime, task_label: str, device: str,
                 extension_seeds: list[int], fixed_probe_seeds: list[int],
                 poses: list[list[float]]):
        super().__init__(frontier, task_label, device, extension_seeds[:18], poses)
        self.prereg = preregistration(
            "learned_phase_rainbow_rl",
            search_rollouts=TOTAL_TRAINING_EPISODES,
            final_rollouts=FINAL_EPISODES,
        )
        self.extension_seed_set = set(map(int, extension_seeds))
        self.fixed_probe_seed_set = set(map(int, fixed_probe_seeds))
        mapping = dict(zip(map(int, extension_seeds), extension_pose_order()))
        mapping.update(dict(zip(map(int, fixed_probe_seeds), (0, 1, 2))))
        self.training_pose_index = mapping

    def environment(self, seed: int, *, training=False):
        env = super().environment(seed, training=training)
        seed = int(seed)
        if seed in self.extension_seed_set:
            regime = "fixed_three_pose_extension_training"
        elif seed in self.fixed_probe_seed_set:
            regime = "fixed_three_pose_matched_probe"
        else:
            regime = "fresh_randomized"
        env._environment_metadata["v42_reset_regime"] = regime
        return env


def verify_parent(parent_root: Path, task: str, spec: dict, parent: dict,
                  panel: dict) -> dict:
    task_root = parent_root / "search" / task
    expected = parent["tasks"][task]
    paths = {
        "search_complete": task_root / "SEARCH_COMPLETE.json",
        "selection": task_root / "SELECTION.json",
        "three_poses": task_root / "THREE_POSES.json",
    }
    for key, path in paths.items():
        hash_key = f"{key}_sha256"
        if file_sha256(path) != expected[hash_key]:
            raise RuntimeError(f"v42 parent hash differs: {task}/{key}")
    if checked_json(paths["three_poses"]) != panel:
        raise RuntimeError(f"v42 rebuilt pose panel differs from v41: {task}")
    complete = checked_json(paths["search_complete"])
    if int(complete.get("search_scientific_rollouts", -1)) != 25:
        raise RuntimeError(f"v42 parent search count differs: {task}")
    last_seed = int(spec["parent_training"][-1])
    state_path = task_root / "training" / "states" / f"{last_seed}.json"
    state = checked_json(state_path)
    if int(state.get("seed", -1)) != last_seed:
        raise RuntimeError(f"v42 parent terminal state seed differs: {task}")
    relative = Path(state["resume_checkpoint"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "resume":
        raise RuntimeError(f"v42 parent resume path is unsafe: {task}")
    resume = task_root / "training" / relative
    if file_sha256(resume) != state["resume_checkpoint_sha256"]:
        raise RuntimeError(f"v42 parent resume checkpoint differs: {task}")
    import torch

    snapshot = torch.load(resume, map_location="cpu", weights_only=False)
    if snapshot.get("schema") != "act-speed-rainbow-resume-v1":
        raise RuntimeError(f"v42 parent resume schema differs: {task}")
    return {
        "schema": "act-three-reset-rainbow50-parent-receipt-v42",
        "task_label": task,
        "parent_implementation_commit": parent["implementation_commit"],
        "parent_training_rollouts": PARENT_TRAINING_EPISODES,
        "parent_search_complete_sha256": expected["search_complete_sha256"],
        "parent_selection_sha256": expected["selection_sha256"],
        "parent_three_poses_sha256": expected["three_poses_sha256"],
        "parent_resume_checkpoint": str(resume),
        "parent_resume_checkpoint_sha256": state["resume_checkpoint_sha256"],
        "parent_completed_decisions": int(snapshot["decision"]),
        "parent_update_count": int(snapshot["update_count"]),
        "parent_screen_or_heldout_outcomes_used": False,
        "parent_rollouts_reexecuted": 0,
    }


def _new_agent(runtime: V42Runtime, seed: int):
    from rl.rainbowDQN.dqnAgent import DQNAgent

    config = runtime.prereg["training"]
    probe = runtime.environment(seed, training=True)
    try:
        return DQNAgent(
            probe,
            memory_size=config["memory_size"], batch_size=config["batch_size"],
            target_update=config["target_update"], seed=config["seed"],
            lr=config["learning_rate"], gamma=config["gamma"], tau=config["tau"],
            frame_skip=10, epsilon=config["epsilon_start"], epsilon_decay=1.0,
            min_epsilon=config["epsilon_end"],
            exploration_steps=config["exploration_decisions"],
            alpha=config["per_alpha"], beta=config["per_beta"],
            atom_size=config["atom_size"], v_min=config["v_min"],
            v_max=config["v_max"], n_step=config["n_step"],
            hidden_dim=config["hidden_dim"], device=runtime.args.device,
        )
    finally:
        probe.close()


def run_extension(runtime: V42Runtime, output: Path, identity: str,
                  seeds: list[int], records: list[dict], parent_receipt: dict) -> None:
    import torch

    config = runtime.prereg["training"]
    agent = _new_agent(runtime, seeds[len(records)] if len(records) < len(seeds) else seeds[-1])
    if records:
        resume = output / records[-1]["resume_checkpoint"]
        if file_sha256(resume) != records[-1]["resume_checkpoint_sha256"]:
            raise RuntimeError("v42 continuation checkpoint hash mismatch")
    else:
        resume = Path(parent_receipt["parent_resume_checkpoint"])
        if file_sha256(resume) != parent_receipt["parent_resume_checkpoint_sha256"]:
            raise RuntimeError("v42 parent checkpoint changed before continuation")
    snapshot = torch.load(resume, map_location=runtime.args.device, weights_only=False)
    decision, update_count, history = restore_rainbow(agent, snapshot)
    if not isinstance(history, deque):
        raise RuntimeError("v42 restored Rainbow history differs")

    states = output / "states"
    for extension_index, seed in enumerate(seeds[len(records):], start=len(records)):
        env = runtime.environment(seed, training=True)
        agent.env = env
        try:
            state = env.reset()
            history.append(state.copy())
            done = False
            total_reward = 0.0
            losses: list[float] = []
            actions: list[int] = []
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
                        losses.append(float(agent.update_model()))
                        update_count += 1
                        if update_count % config["target_update"] == 0:
                            agent._target_soft_update()
                state = next_state
            physics_steps = int(info["physics_steps"])
            record = {
                "seed": int(seed), "identity_sha256": identity,
                "extension_episode": extension_index + 1,
                "total_training_episode": PARENT_TRAINING_EPISODES + extension_index + 1,
                "success": bool(info["success"]), "return": total_reward,
                "physics_steps": physics_steps,
                "first_success_step": info.get("first_success_step"),
                "policy_time": float(info["policy_time"]),
                "mean_speed": float(np.mean(env.speed_list)),
                "max_speed": float(np.max(env.speed_list)),
                "acceleration": float(env.episode_len / max(physics_steps, 1)),
                "successful_acceleration": (
                    float(env.episode_len / max(physics_steps, 1))
                    if info["success"] else None
                ),
                "safety_violation": info.get("safety_violation"),
                "decision": decision, "update_count": update_count,
                "epsilon_after": agent.epsilon,
                "loss_last": None if not losses else losses[-1],
                "actions": actions, "observation_spec": env.observation_spec(),
                "environment_spec": env.environment_spec(),
            }
            if "physics_error" in info:
                record["physics_error"] = str(info["physics_error"])
        finally:
            env.close()
        runtime.validate_search_record(record)
        total_episode = PARENT_TRAINING_EPISODES + extension_index + 1
        checkpoint = output / "resume" / f"episode-{total_episode:02d}.pt"
        immutable_torch(checkpoint, rainbow_snapshot(agent, decision, update_count, history))
        record["resume_checkpoint"] = str(checkpoint.relative_to(output))
        record["resume_checkpoint_sha256"] = file_sha256(checkpoint)
        immutable_json(states / f"{seed}.json", record)
        records.append(record)
        progress(output, identity, records, seeds)
        print(json.dumps({
            "stage": "extension_training", "completed": len(records),
            "total_training_episode": total_episode,
            "successes": sum(bool(item["success"]) for item in records),
            "decisions": decision,
        }, sort_keys=True), flush=True)

    agent.dqn.update_norm_stats(normalization_stats(history))
    terminal = output / "terminal_policy.pt"
    payload = {
        "format_version": 2, "algorithm": "rainbow_dqn",
        "model_state_dict": agent.dqn.state_dict(),
        "observation_dim": int(agent.dqn.in_dim),
        "speed_values": list(SPEED_VALUES), "atom_size": config["atom_size"],
        "v_min": config["v_min"], "v_max": config["v_max"],
        "hidden_dim": config["hidden_dim"], "seed": config["seed"],
        "training_config": config, "completed_decisions": decision,
        "decision_frame_skip": 10,
        "reward_aggregation": "undiscounted_sum_per_decision",
        "observation_spec": records[-1]["observation_spec"],
        "environment_spec": records[-1]["environment_spec"],
        "observation_encoder_state_dict": None,
        "metadata": {
            "parent_v41_resume_sha256": parent_receipt["parent_resume_checkpoint_sha256"],
            "inherited_training_episodes": PARENT_TRAINING_EPISODES,
            "new_extension_episodes": EXTENSION_EPISODES,
            "terminal_after_total_training_episodes": TOTAL_TRAINING_EPISODES,
            "identity_sha256": identity,
        },
    }
    immutable_torch(terminal, payload)


def run_probe(runtime: V42Runtime, output: Path, search_identity: str,
              task: str, stage: str, seeds: list[int], controller: dict, policy):
    identity = {
        "schema": "act-three-reset-rainbow50-probe-identity-v42",
        "search_identity_sha256": search_identity, "task_label": task,
        "stage": stage, "controller": controller,
        "seeds": seeds, "seed_sha256": canonical_sha256(seeds),
        "learning_or_tuning_permitted": False,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    immutable_or_verify(output / "IDENTITY.json", identity)
    records = load_contiguous_states(output / "states", seeds, identity["identity_sha256"])
    for seed in seeds[len(records):]:
        record = rollout_one(runtime, seed, policy, identity["identity_sha256"],
                             controller_sha256=controller["controller_sha256"])
        immutable_json(output / "states" / f"{seed}.json", record)
        records.append(record)
        progress(output, identity["identity_sha256"], records, seeds)
        print(json.dumps({"stage": stage, "task": task, "completed": len(records),
                          "successes": sum(bool(item["success"]) for item in records)},
                         sort_keys=True), flush=True)
        if record.get("physics_error") is not None:
            raise RuntimeError(f"v42 halted on {stage} physics error")
    result = {
        "schema": "act-three-reset-rainbow50-probe-result-v42",
        "task_label": task, "stage": stage, "controller": controller,
        "episodes": len(records), "summary": summarize(records),
        "identity_sha256": identity["identity_sha256"],
    }
    immutable_or_verify(output / "RESULT.json", result)
    immutable_or_verify(output / "COMPLETE.json", {
        "schema": "act-three-reset-rainbow50-probe-completion-v42",
        "episodes": len(records), "result_sha256": file_sha256(output / "RESULT.json"),
    })
    return result, records


def run_search(runtime: V42Runtime, root: Path, parent_root: Path, task: str,
               spec: dict, contract: Path, banks: Path, parent_manifest_path: Path,
               parent_manifest: dict, panel: dict) -> None:
    output = root / "search" / task
    extension_root = output / "extension_training"
    parent_receipt = verify_parent(parent_root, task, spec, parent_manifest, panel)
    immutable_or_verify(output / "PARENT_RECEIPT.json", parent_receipt)
    identity = {
        **runtime.frontier.identity(),
        "schema": "act-three-reset-rainbow50-search-identity-v42",
        "task_label": task, "contract_sha256": file_sha256(contract),
        "banks_sha256": file_sha256(banks),
        "parent_manifest_sha256": file_sha256(parent_manifest_path),
        "parent_receipt_sha256": file_sha256(output / "PARENT_RECEIPT.json"),
        "parent_training_rollouts": PARENT_TRAINING_EPISODES,
        "extension_training_rollouts": EXTENSION_EPISODES,
        "total_training_rollouts": TOTAL_TRAINING_EPISODES,
        "extension_seeds": spec["extension"],
        "extension_pose_order": extension_pose_order(),
        "cumulative_pose_visits": [17, 17, 16],
        "fixed_probe_seeds": spec["fixed_probe"],
        "screen_seeds": spec["screen"],
        "final_seeds_registered_unopened": spec["final"],
        "parent_screen_or_heldout_outcomes_used": False,
        "parent_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    immutable_or_verify(output / "IDENTITY.json", identity)
    if (output / "SEARCH_COMPLETE.json").exists():
        complete = checked_json(output / "SEARCH_COMPLETE.json")
        if complete["selection_sha256"] != file_sha256(output / "SELECTION.json"):
            raise RuntimeError("v42 completed selection hash mismatch")
        return

    extension_seeds = list(map(int, spec["extension"]))
    records = load_contiguous_states(
        extension_root / "states", extension_seeds, identity["identity_sha256"]
    )
    if not (extension_root / "terminal_policy.pt").exists():
        run_extension(runtime, extension_root, identity["identity_sha256"],
                      extension_seeds, records, parent_receipt)
    records = load_contiguous_states(
        extension_root / "states", extension_seeds, identity["identity_sha256"]
    )
    if len(records) != EXTENSION_EPISODES:
        raise RuntimeError("v42 extension did not seal exactly 32 episodes")
    extension_incidents = incidents(records)
    if extension_incidents["physics_errors"]:
        raise RuntimeError("v42 extension contains a physics error")
    deployment, deployment_sha = v41.build_deployment_checkpoint(extension_root)
    rainbow_policy = RainbowSpeedPolicy.load(deployment, device=runtime.args.device)
    native_controller = {"type": "fixed_speed", "speed": 1.0}
    native_controller["controller_sha256"] = canonical_sha256(native_controller)
    rainbow_controller = {
        "type": "phase_conditioned_rainbow50",
        "checkpoint_sha256": deployment_sha,
    }
    rainbow_controller["controller_sha256"] = canonical_sha256(rainbow_controller)

    fixed_seeds = list(map(int, spec["fixed_probe"]))
    fixed_native, fixed_native_records = run_probe(
        runtime, output / "fixed_probe" / "native_1x", identity["identity_sha256"],
        task, "fixed_probe_native_1x", fixed_seeds, native_controller,
        FixedSpeedPolicy(1.0),
    )
    fixed_rainbow, fixed_rainbow_records = run_probe(
        runtime, output / "fixed_probe" / "rainbow", identity["identity_sha256"],
        task, "fixed_probe_rainbow", fixed_seeds, rainbow_controller, rainbow_policy,
    )
    screen, screen_records = run_probe(
        runtime, output / "screen", identity["identity_sha256"], task,
        "fresh_randomized_screen", list(map(int, spec["screen"])),
        rainbow_controller, rainbow_policy,
    )
    all_incidents = {
        "physics_errors": sum(
            incidents(value)["physics_errors"]
            for value in (records, fixed_native_records, fixed_rainbow_records, screen_records)
        ),
        "safety_violations": sum(
            incidents(value)["safety_violations"]
            for value in (records, fixed_native_records, fixed_rainbow_records, screen_records)
        ),
    }
    qualified = rainbow_qualifies(
        fixed_rainbow["summary"]["successes"],
        screen["summary"]["successes"],
        all_incidents,
    )
    selection = {
        "schema": "act-three-reset-rainbow50-selection-v42",
        "task_label": task, "deployment_policy_path": str(deployment),
        "deployment_policy_sha256": deployment_sha,
        "parent_training_rollouts": PARENT_TRAINING_EPISODES,
        "extension_training_summary": summarize(records),
        "total_training_rollouts": TOTAL_TRAINING_EPISODES,
        "fixed_probe": {
            "native_1x": fixed_native, "rainbow": fixed_rainbow,
            "paired": v37.paired_receipt(fixed_rainbow_records, fixed_native_records),
        },
        "screen": screen,
        "selection_rule": "rainbow_only_if_fixed_3_of_3_and_fresh_screen_at_least_9_of_10_and_zero_incidents",
        "rainbow_qualified": qualified,
        "selected_method": "rainbow" if qualified else "native_1x",
        "new_prefinal_scientific_rollouts": NEW_PREFINAL_ROLLOUTS,
        "new_rollout_split": {
            "extension_training": EXTENSION_EPISODES,
            "fixed_native_reference": FIXED_PROBE_EPISODES,
            "fixed_rainbow_probe": FIXED_PROBE_EPISODES,
            "fresh_randomized_screen": SCREEN_EPISODES,
        },
        "incident_totals": all_incidents,
        "parent_screen_or_heldout_outcomes_used": False,
        "parent_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    }
    immutable_or_verify(output / "SELECTION.json", selection)
    immutable_or_verify(output / "SEARCH_COMPLETE.json", {
        "schema": "act-three-reset-rainbow50-search-completion-v42",
        "task_label": task, "parent_training_rollouts": PARENT_TRAINING_EPISODES,
        "new_prefinal_scientific_rollouts": NEW_PREFINAL_ROLLOUTS,
        "selection_sha256": file_sha256(output / "SELECTION.json"),
        "deployment_policy_sha256": deployment_sha, **all_incidents,
        "parent_screen_or_heldout_outcomes_used": False,
        "parent_rollouts_reexecuted": 0, "final_bank_opened": False,
    })


def require_all_search(root: Path) -> None:
    for task in TASKS:
        complete = checked_json(root / "search" / task / "SEARCH_COMPLETE.json")
        if int(complete.get("new_prefinal_scientific_rollouts", -1)) != NEW_PREFINAL_ROLLOUTS:
            raise RuntimeError(f"v42 search incomplete: {task}")


def method_policy(root: Path, task: str, method: str, device: str):
    selection_path = root / "search" / task / "SELECTION.json"
    selection = checked_json(selection_path)
    effective = selection["selected_method"] if method == "selected" else method
    if effective == "native_1x":
        controller = {"type": "fixed_speed", "speed": 1.0}
        policy = FixedSpeedPolicy(1.0)
    elif effective == "uniform_2x":
        controller = {"type": "fixed_speed", "speed": 2.0}
        policy = FixedSpeedPolicy(2.0)
    elif effective == "rainbow":
        checkpoint = Path(selection["deployment_policy_path"])
        if file_sha256(checkpoint) != selection["deployment_policy_sha256"]:
            raise RuntimeError("v42 deployment checkpoint hash differs")
        controller = {
            "type": "phase_conditioned_rainbow50",
            "checkpoint_sha256": selection["deployment_policy_sha256"],
        }
        policy = RainbowSpeedPolicy.load(checkpoint, device=device)
    else:
        raise ValueError(effective)
    controller["controller_sha256"] = canonical_sha256(controller)
    return controller, policy, effective, file_sha256(selection_path)


def run_final(runtime: V42Runtime, root: Path, task: str, method: str,
              seeds: list[int], banks_sha: str) -> None:
    require_all_search(root)
    controller, policy, effective, provenance = method_policy(
        root, task, method, runtime.args.device
    )
    digest = controller["controller_sha256"]
    controller_root = root / "final" / task / "controllers" / digest
    alias_path = root / "final" / task / "methods" / method / "RESULT.json"
    if alias_path.exists():
        return
    identity = {
        **runtime.frontier.identity(),
        "schema": "act-three-reset-rainbow50-final-controller-identity-v42",
        "task_label": task, "controller": controller,
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "banks_sha256": banks_sha, "search_or_tuning_permitted": False,
        "reset_distribution": "fresh_randomized_object_pose",
        "parent_v41_training_state_inherited": True,
        "parent_v41_screen_or_heldout_outcomes_used": False,
        "parent_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    immutable_or_verify(controller_root / "IDENTITY.json", identity)
    was_complete = (controller_root / "COMPLETE.json").exists()
    records = load_contiguous_states(controller_root / "states", seeds, identity["identity_sha256"])
    for seed in seeds[len(records):]:
        record = rollout_one(runtime, seed, policy, identity["identity_sha256"],
                             controller_sha256=digest)
        immutable_json(controller_root / "states" / f"{seed}.json", record)
        records.append(record)
        atomic_json(controller_root / "progress.json", {
            "task": task, "controller_sha256": digest, "completed": len(records),
            "successes": sum(bool(item["success"]) for item in records),
        })
        print(json.dumps({"stage": "final", "task": task, "method": method,
                          "completed": len(records),
                          "successes": sum(bool(item["success"]) for item in records)},
                         sort_keys=True), flush=True)
    result = {
        "schema": "act-three-reset-rainbow50-final-controller-result-v42",
        "task_label": task, "controller": controller, "episodes": len(records),
        "summary": summarize(records), "identity_sha256": identity["identity_sha256"],
    }
    immutable_or_verify(controller_root / "RESULT.json", result)
    immutable_or_verify(controller_root / "COMPLETE.json", {
        "schema": "act-three-reset-rainbow50-final-controller-completion-v42",
        "episodes": len(records),
        "result_sha256": file_sha256(controller_root / "RESULT.json"),
        "physics_errors": result["summary"]["physics_errors"],
        "safety_violations": result["summary"]["safety_violations"],
    })
    immutable_or_verify(alias_path, {
        "schema": "act-three-reset-rainbow50-final-method-result-v42",
        "task_label": task, "method": method, "effective_method": effective,
        "controller": controller, "controller_sha256": digest,
        "controller_result_sha256": file_sha256(controller_root / "RESULT.json"),
        "controller_receipt": str(controller_root / "RESULT.json"),
        "selection_provenance": provenance, "controller_cache_hit": was_complete,
        "summary": result["summary"],
    })


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "final"), required=True)
    parser.add_argument("--method", choices=(SEARCH_METHOD, *FINAL_METHODS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-v41-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--base-source-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=TASKS, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if git_head() != args.implementation_commit:
        raise RuntimeError("v42 checked-out source differs from implementation commit")
    banks = checked_json(args.banks)
    validate_banks(banks)
    parent_manifest = checked_json(args.parent_manifest)
    validate_parent_manifest(parent_manifest)
    spec = banks["tasks"][args.task_label]
    frontier = ACTFrontierRuntime(
        source_commit=args.base_source_commit,
        checkout_commit=args.implementation_commit,
        run_manifest=args.run_manifest, task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source, device=args.device,
    )
    root = args.root.resolve()
    panel = panel_builder.build(args.task_label, int(spec["pose_design_seed"]))
    immutable_or_verify(root / "search" / args.task_label / "THREE_POSES.json", panel)
    runtime = V42Runtime(
        frontier, args.task_label, args.device,
        list(map(int, spec["extension"])), list(map(int, spec["fixed_probe"])),
        panel["object_pose_vectors"],
    )
    if args.stage == "search":
        if args.method != SEARCH_METHOD:
            raise ValueError("v42 search stage requires three_reset_rainbow50_extension")
        run_search(runtime, root, args.parent_v41_root.resolve(), args.task_label,
                   spec, args.contract, args.banks, args.parent_manifest,
                   parent_manifest, panel)
    else:
        if args.method not in FINAL_METHODS:
            raise ValueError("v42 final stage requires a registered final method")
        run_final(runtime, root, args.task_label, args.method,
                  list(map(int, spec["final"])), file_sha256(args.banks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
