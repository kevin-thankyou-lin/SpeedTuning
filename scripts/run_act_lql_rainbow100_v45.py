#!/usr/bin/env python3
"""Run the fresh LQL-Rainbow-100 replication study V45."""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SPEEDTUNING_SPEED_VALUES", "1,1.5,2,2.5,3")

from act_speed_benchmark import (  # noqa: E402
    JointEffectorObservationWrapper,
    SPEED_VALUES,
    canonical_sha256,
    preregistration,
)
from one_reset_phase_schedule import workspace_violation  # noqa: E402
from policy_speed_env import create_speed_env, make_speed_reward  # noqa: E402
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head  # noqa: E402
from scripts.run_act_speed_benchmark_cell import (  # noqa: E402
    atomic_json,
    immutable_json,
    immutable_torch,
    load_contiguous_states,
    progress,
    rollout_one,
    run_rainbow_search,
)
from scripts.run_act_strider_frontier_v4 import file_sha256, summarize  # noqa: E402
from speed_policy import RainbowSpeedPolicy  # noqa: E402


TASKS = ("pick", "tea", "insertion")
SEARCH_METHOD = "lql_rainbow100"
FINAL_METHODS = ("lql_rainbow",)
TRAINING_EPISODES = 100
CHECKPOINT_EPISODES = tuple(range(10, TRAINING_EPISODES + 1, 10))
PROBE_EPISODES = 3
FINAL_EPISODES = 50
PREFINAL_ROLLOUTS = (
    TRAINING_EPISODES + len(CHECKPOINT_EPISODES) * PROBE_EPISODES
)
LQL_TRAJECTORY_LENGTH = 8
LQL_LAMBDA_LB = 1.0
LQL_LAMBDA_UB = 1.0


def checked_json(path: Path) -> dict:
    return json.loads(path.read_text())


def immutable_or_verify(path: Path, value) -> None:
    if path.exists():
        if checked_json(path) != value:
            raise RuntimeError(f"immutable receipt differs: {path}")
    else:
        immutable_json(path, value)


def expand_bank(spec: dict) -> list[int]:
    start = int(spec["start"])
    count = int(spec["count"])
    if count <= 0:
        raise RuntimeError("v45 seed range count must be positive")
    return list(range(start, start + count))


def task_banks(spec: dict) -> dict[str, list[int]]:
    return {name: expand_bank(spec[name]) for name in ("training", "probe", "final")}


def validate_banks(banks: dict) -> None:
    if banks.get("schema") != "act-lql-rainbow100-banks-v45":
        raise RuntimeError("v45 bank schema differs")
    all_values: list[int] = []
    expected = {"training": TRAINING_EPISODES, "probe": PROBE_EPISODES,
                "final": FINAL_EPISODES}
    for task in TASKS:
        values = task_banks(banks["tasks"][task])
        for name, count in expected.items():
            if len(values[name]) != count:
                raise RuntimeError(f"v45 {task}/{name} count differs")
        flattened = [seed for bank in values.values() for seed in bank]
        if len(flattened) != len(set(flattened)) or min(flattened) < 430000000:
            raise RuntimeError(f"v45 {task} banks overlap or are not fresh")
        all_values.extend(flattened)
    if len(all_values) != len(set(all_values)):
        raise RuntimeError("v45 cross-task banks overlap")


def incidents(records: list[dict]) -> dict:
    return {
        "physics_errors": sum(item.get("physics_error") is not None for item in records),
        "safety_violations": sum(
            item.get("safety_violation") is not None for item in records
        ),
    }


def optimizer_diagnostics(records: list[dict]) -> dict:
    values = [
        item["optimizer_diagnostics"] for item in records
        if item.get("optimizer_diagnostics", {}).get("updates", 0) > 0
    ]
    updates = sum(int(item["updates"]) for item in values)

    def weighted(name: str):
        if not updates:
            return None
        return sum(
            float(item[name]) * int(item["updates"])
            for item in values
        ) / updates

    return {
        "updates": updates,
        "mean_td_loss": weighted("mean_td_loss"),
        "mean_lql_lb_loss": weighted("mean_lql_lb_loss"),
        "mean_lql_ub_loss": weighted("mean_lql_ub_loss"),
        "mean_lql_lb_active_fraction": weighted(
            "mean_lql_lb_active_fraction"
        ),
        "mean_lql_ub_active_fraction": weighted(
            "mean_lql_ub_active_fraction"
        ),
    }


class V45Runtime:
    """Public Rainbow runner contract with randomized object poses everywhere."""

    def __init__(self, frontier: ACTFrontierRuntime, task_label: str, device: str,
                 registered: dict[str, list[int]]):
        self.frontier = frontier
        self.args = SimpleNamespace(
            method="learned_phase_rainbow_rl", task_label=task_label, device=device
        )
        self.task = {"task": frontier.task}
        self.manifest = frontier.manifest
        self.prereg = preregistration(
            "learned_phase_rainbow_rl",
            search_rollouts=TRAINING_EPISODES,
            final_rollouts=FINAL_EPISODES,
        )
        self.prereg["training"].update({
            "algorithm": "lql_rainbow_dqn",
            "lql_trajectory_length": LQL_TRAJECTORY_LENGTH,
            "lql_lambda_lb": LQL_LAMBDA_LB,
            "lql_lambda_ub": LQL_LAMBDA_UB,
            "lql_value_scale": "categorical_support_width",
            "lql_future_indices": "all_offsets_at_least_2",
            "lql_past_indices": "all_including_same_state",
        })
        self.training_seeds = set(registered["training"])
        self.probe_seeds = set(registered["probe"])
        self.final_seeds = set(registered["final"])

    def validate_search_record(self, record: dict) -> None:
        if record.get("physics_error") is not None:
            raise RuntimeError("v45 halted immediately on a Rainbow physics error")
        if record["environment_spec"].get("randomize_object_pose") is not True:
            raise RuntimeError("v45 training did not use a randomized reset")

    def environment(self, seed: int, *, training=False):
        seed = int(seed)
        if training and seed not in self.training_seeds:
            raise RuntimeError("v45 training attempted an unregistered reset")
        if not training and seed not in self.probe_seeds | self.final_seeds:
            raise RuntimeError("v45 evaluation attempted an unregistered reset")
        encoder = self.frontier.encoder()
        reward = None
        if training:
            config = self.prereg["training"]["reward"]
            reward = make_speed_reward(
                config["success_bonus"], config["speed_weight"], config["speed_power"]
            )
        env = create_speed_env(
            task_name=self.frontier.task,
            reward_fn=reward,
            chunk_predictor=self.frontier.adapter,
            seed=seed,
            randomize_object_pose=True,
            speed_values=SPEED_VALUES,
            observation_encoder=encoder,
            decision_frame_skip=10,
            decision_mode="fixed_or_phase_entry",
            terminate_on_success=False,
            safety_monitor=partial(workspace_violation, self.frontier.task),
        )
        env.env = JointEffectorObservationWrapper(env.env)
        env._environment_metadata["learned_phase_effector_source"] = (
            "joint_fk_body_xpos"
        )
        env._environment_metadata["v45_reset_regime"] = "randomized_object_pose"
        return env


def build_checkpoint_policy(training_root: Path, episode: int, config: dict) -> tuple[Path, str]:
    import torch

    if episode not in CHECKPOINT_EPISODES:
        raise ValueError(episode)
    if episode == TRAINING_EPISODES:
        path = training_root / "terminal_policy.pt"
        return path, file_sha256(path)
    state_files = sorted((training_root / "states").glob("*.json"))
    if len(state_files) != TRAINING_EPISODES:
        raise RuntimeError("v45 checkpoint conversion requires sealed training")
    record = checked_json(state_files[episode - 1])
    resume = training_root / record["resume_checkpoint"]
    if file_sha256(resume) != record["resume_checkpoint_sha256"]:
        raise RuntimeError(f"v45 episode-{episode} resume checkpoint differs")
    snapshot = torch.load(resume, map_location="cpu", weights_only=False)
    output = training_root / "checkpoint_policies" / f"episode-{episode:03d}.pt"
    payload = {
        "format_version": 2,
        "algorithm": "lql_rainbow_dqn",
        "model_state_dict": snapshot["dqn"],
        "observation_dim": int(snapshot["dqn"]["states_mean"].numel()),
        "speed_values": list(SPEED_VALUES),
        "atom_size": config["atom_size"],
        "v_min": config["v_min"],
        "v_max": config["v_max"],
        "hidden_dim": config["hidden_dim"],
        "seed": config["seed"],
        "training_config": config,
        "completed_decisions": int(snapshot["decision"]),
        "decision_frame_skip": 10,
        "reward_aggregation": "undiscounted_sum_per_decision",
        "observation_spec": record["observation_spec"],
        "environment_spec": record["environment_spec"],
        "observation_encoder_state_dict": None,
        "metadata": {
            "checkpoint_after_training_episode": episode,
            "source_resume_sha256": record["resume_checkpoint_sha256"],
            "learning_or_selection_from_probe_outcomes": False,
        },
    }
    if not output.exists():
        immutable_torch(output, payload)
    return output, file_sha256(output)


def controller(kind: str, *, checkpoint_sha256: str | None = None,
               episode: int | None = None) -> dict:
    if kind == "lql_rainbow":
        value = {
            "type": "phase_conditioned_lql_rainbow100",
            "checkpoint_sha256": checkpoint_sha256,
            "training_episode": episode,
        }
    else:
        raise ValueError(kind)
    value["controller_sha256"] = canonical_sha256(value)
    return value


def run_probe(runtime: V45Runtime, output: Path, search_identity: str, task: str,
              stage: str, seeds: list[int], controller_value: dict, policy) -> tuple[dict, list[dict]]:
    identity = {
        "schema": "act-lql-rainbow100-probe-identity-v45",
        "search_identity_sha256": search_identity,
        "task_label": task,
        "stage": stage,
        "controller": controller_value,
        "seeds": seeds,
        "seed_sha256": canonical_sha256(seeds),
        "learning_or_tuning_permitted": False,
        "checkpoint_selection_permitted": False,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    immutable_or_verify(output / "IDENTITY.json", identity)
    records = load_contiguous_states(output / "states", seeds, identity["identity_sha256"])
    for seed in seeds[len(records):]:
        record = rollout_one(
            runtime, seed, policy, identity["identity_sha256"],
            controller_sha256=controller_value["controller_sha256"],
        )
        immutable_json(output / "states" / f"{seed}.json", record)
        records.append(record)
        progress(output, identity["identity_sha256"], records, seeds)
        print(json.dumps({
            "stage": stage, "task": task, "completed": len(records),
            "successes": sum(bool(item["success"]) for item in records),
        }, sort_keys=True), flush=True)
        if record.get("physics_error") is not None:
            raise RuntimeError(f"v45 halted on {stage} physics error")
    result = {
        "schema": "act-lql-rainbow100-probe-result-v45",
        "task_label": task,
        "stage": stage,
        "controller": controller_value,
        "episodes": len(records),
        "summary": summarize(records),
        "identity_sha256": identity["identity_sha256"],
    }
    immutable_or_verify(output / "RESULT.json", result)
    immutable_or_verify(output / "COMPLETE.json", {
        "schema": "act-lql-rainbow100-probe-completion-v45",
        "episodes": len(records),
        "result_sha256": file_sha256(output / "RESULT.json"),
    })
    return result, records


def run_search(runtime: V45Runtime, root: Path, task: str, spec: dict,
               contract_path: Path, banks_path: Path) -> None:
    output = root / "search" / task
    training_root = output / "training"
    banks = task_banks(spec)
    identity = {
        **runtime.frontier.identity(),
        "schema": "act-lql-rainbow100-search-identity-v45",
        "task_label": task,
        "contract_sha256": file_sha256(contract_path),
        "banks_sha256": file_sha256(banks_path),
        "training_episodes": TRAINING_EPISODES,
        "training_seeds": banks["training"],
        "training_reset_distribution": "fresh_randomized_object_pose_per_episode",
        "paired_baseline": {
            "study": "act_randomized_rainbow100_v43",
            "implementation_commit": "989614daf538bebbbd31fbafea544f6911321683",
            "same_training_probe_and_final_seed_banks": True,
            "baseline_rollouts_reexecuted": 0,
        },
        "lql": {
            "paper": "https://arxiv.org/abs/2605.05812v2",
            "trajectory_length": LQL_TRAJECTORY_LENGTH,
            "lambda_lb": LQL_LAMBDA_LB,
            "lambda_ub": LQL_LAMBDA_UB,
            "failure_localization_labels": False,
            "speed_order_constraints": False,
        },
        "checkpoint_episodes": list(CHECKPOINT_EPISODES),
        "checkpoint_probe_seeds": banks["probe"],
        "checkpoint_probe_outcomes_used_for_training_or_selection": False,
        "final_seeds_registered_unopened": banks["final"],
        "historical_checkpoint_or_rollout_inputs": [],
        "historical_speed_outcomes_used_for_initialization": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    immutable_or_verify(output / "IDENTITY.json", identity)
    if (output / "SEARCH_COMPLETE.json").exists():
        complete = checked_json(output / "SEARCH_COMPLETE.json")
        if complete["terminal_policy_sha256"] != file_sha256(
            training_root / "terminal_policy.pt"
        ):
            raise RuntimeError("v45 completed terminal policy differs")
        return

    training_identity = canonical_sha256({
        "search_identity_sha256": identity["identity_sha256"],
        "stage": "lql_rainbow_training",
    })
    records = load_contiguous_states(
        training_root / "states", banks["training"], training_identity
    )
    if not (training_root / "terminal_policy.pt").exists():
        run_rainbow_search(
            runtime, training_root, training_identity, banks["training"], records
        )
    records = load_contiguous_states(
        training_root / "states", banks["training"], training_identity
    )
    if len(records) != TRAINING_EPISODES:
        raise RuntimeError("v45 training did not seal exactly 100 episodes")
    training_incidents = incidents(records)
    if training_incidents["physics_errors"]:
        raise RuntimeError("v45 training contains a physics error")

    checkpoints = {}
    probe_records: list[dict] = []
    for episode in CHECKPOINT_EPISODES:
        checkpoint_path, checkpoint_sha = build_checkpoint_policy(
            training_root, episode, runtime.prereg["training"]
        )
        value = controller(
            "lql_rainbow", checkpoint_sha256=checkpoint_sha, episode=episode
        )
        result, episode_records = run_probe(
            runtime, output / "checkpoint_probes" / f"episode-{episode:03d}",
            identity["identity_sha256"], task,
            f"deterministic_checkpoint_episode_{episode}", banks["probe"], value,
            RainbowSpeedPolicy.load(checkpoint_path, device=runtime.args.device),
        )
        checkpoints[str(episode)] = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "probe": result,
        }
        probe_records.extend(episode_records)

    all_records = [*records, *probe_records]
    totals = incidents(all_records)
    terminal_path = training_root / "terminal_policy.pt"
    result = {
        "schema": "act-lql-rainbow100-search-result-v45",
        "task_label": task,
        "training_summary": summarize(records),
        "optimizer_diagnostics": optimizer_diagnostics(records),
        "training_episodes": len(records),
        "training_unique_randomized_resets": len({item["seed"] for item in records}),
        "checkpoint_probes": checkpoints,
        "terminal_policy_path": str(terminal_path),
        "terminal_policy_sha256": file_sha256(terminal_path),
        "prefinal_scientific_rollouts": len(all_records),
        "rollout_split": {
            "randomized_training": len(records),
            "deterministic_checkpoint_probes": len(probe_records),
        },
        "incident_totals": totals,
        "checkpoint_probe_outcomes_used_for_training_or_selection": False,
        "historical_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    }
    if result["prefinal_scientific_rollouts"] != PREFINAL_ROLLOUTS:
        raise RuntimeError("v45 prefinal accounting differs")
    immutable_or_verify(output / "SEARCH_RESULT.json", result)
    immutable_or_verify(output / "SEARCH_COMPLETE.json", {
        "schema": "act-lql-rainbow100-search-completion-v45",
        "task_label": task,
        "training_episodes": TRAINING_EPISODES,
        "prefinal_scientific_rollouts": PREFINAL_ROLLOUTS,
        "search_result_sha256": file_sha256(output / "SEARCH_RESULT.json"),
        "terminal_policy_sha256": result["terminal_policy_sha256"],
        **totals,
        "checkpoint_probe_outcomes_used_for_training_or_selection": False,
        "historical_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    })


def require_all_search(root: Path) -> None:
    for task in TASKS:
        complete = checked_json(root / "search" / task / "SEARCH_COMPLETE.json")
        if int(complete.get("training_episodes", -1)) != TRAINING_EPISODES:
            raise RuntimeError(f"v45 search incomplete: {task}")


def method_policy(root: Path, task: str, method: str, device: str):
    search_path = root / "search" / task / "SEARCH_RESULT.json"
    search = checked_json(search_path)
    if method == "lql_rainbow":
        checkpoint = Path(search["terminal_policy_path"])
        if file_sha256(checkpoint) != search["terminal_policy_sha256"]:
            raise RuntimeError("v45 deployment checkpoint differs")
        value = controller(
            method, checkpoint_sha256=search["terminal_policy_sha256"],
            episode=TRAINING_EPISODES,
        )
        policy = RainbowSpeedPolicy.load(checkpoint, device=device)
    else:
        raise ValueError(method)
    return value, policy, file_sha256(search_path)


def run_final(runtime: V45Runtime, root: Path, task: str, method: str,
              seeds: list[int], banks_sha: str) -> None:
    require_all_search(root)
    controller_value, policy, provenance = method_policy(
        root, task, method, runtime.args.device
    )
    digest = controller_value["controller_sha256"]
    output = root / "final" / task / "methods" / method
    identity = {
        **runtime.frontier.identity(),
        "schema": "act-lql-rainbow100-final-identity-v45",
        "task_label": task,
        "method": method,
        "controller": controller_value,
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "banks_sha256": banks_sha,
        "search_result_sha256": provenance,
        "search_or_tuning_permitted": False,
        "reset_distribution": "fresh_randomized_object_pose",
        "historical_rollouts_reexecuted": 0,
        "paired_v43_baseline_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    immutable_or_verify(output / "IDENTITY.json", identity)
    if (output / "COMPLETE.json").exists():
        return
    records = load_contiguous_states(output / "states", seeds, identity["identity_sha256"])
    for seed in seeds[len(records):]:
        record = rollout_one(
            runtime, seed, policy, identity["identity_sha256"],
            controller_sha256=digest,
        )
        immutable_json(output / "states" / f"{seed}.json", record)
        records.append(record)
        atomic_json(output / "progress.json", {
            "task": task, "method": method, "completed": len(records),
            "successes": sum(bool(item["success"]) for item in records),
        })
        print(json.dumps({
            "stage": "final", "task": task, "method": method,
            "completed": len(records),
            "successes": sum(bool(item["success"]) for item in records),
        }, sort_keys=True), flush=True)
        if record.get("physics_error") is not None:
            raise RuntimeError(f"v45 halted on final physics error: {task}/{method}")
    result = {
        "schema": "act-lql-rainbow100-final-result-v45",
        "task_label": task,
        "method": method,
        "controller": controller_value,
        "episodes": len(records),
        "summary": summarize(records),
        "identity_sha256": identity["identity_sha256"],
    }
    immutable_or_verify(output / "RESULT.json", result)
    immutable_or_verify(output / "COMPLETE.json", {
        "schema": "act-lql-rainbow100-final-completion-v45",
        "episodes": len(records),
        "result_sha256": file_sha256(output / "RESULT.json"),
        "physics_errors": result["summary"]["physics_errors"],
        "safety_violations": result["summary"]["safety_violations"],
    })


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "final"), required=True)
    parser.add_argument("--method", choices=(SEARCH_METHOD, *FINAL_METHODS), required=True)
    parser.add_argument("--root", type=Path, required=True)
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
        raise RuntimeError("v45 checked-out source differs from implementation commit")
    banks = checked_json(args.banks)
    validate_banks(banks)
    spec = banks["tasks"][args.task_label]
    registered = task_banks(spec)
    frontier = ACTFrontierRuntime(
        source_commit=args.base_source_commit,
        checkout_commit=args.implementation_commit,
        run_manifest=args.run_manifest,
        task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
    )
    runtime = V45Runtime(frontier, args.task_label, args.device, registered)
    root = args.root.resolve()
    if args.stage == "search":
        if args.method != SEARCH_METHOD:
            raise ValueError("v45 search stage requires lql_rainbow100")
        run_search(
            runtime, root, args.task_label, spec, args.contract, args.banks
        )
    else:
        if args.method not in FINAL_METHODS:
            raise ValueError("v45 final stage requires a registered final method")
        run_final(
            runtime, root, args.task_label, args.method,
            registered["final"], file_sha256(args.banks),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
