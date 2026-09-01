#!/usr/bin/env python3
"""Run the fresh exact-25 three-reset phase-conditioned Rainbow study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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
from scripts import build_three_reset_rainbow_panel_v41 as panel_builder  # noqa: E402
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
from speed_policy import FixedSpeedPolicy, RainbowSpeedPolicy  # noqa: E402

TASKS = ("pick", "tea", "insertion")
SEARCH_METHOD = "three_reset_rainbow25"
FINAL_METHODS = ("native_1x", "uniform_2x", "rainbow", "selected")
TRAINING_EPISODES = 18
SCREEN_EPISODES = 7
SEARCH_BUDGET = TRAINING_EPISODES + SCREEN_EPISODES
FINAL_EPISODES = 50
SCREEN_SUCCESS_FLOOR = 7


def checked_json(path: Path) -> dict:
    return json.loads(path.read_text())


def immutable_or_verify(path: Path, value) -> None:
    if path.exists():
        if checked_json(path) != value:
            raise RuntimeError(f"immutable receipt differs: {path}")
    else:
        immutable_json(path, value)


def validate_banks(banks: dict) -> None:
    if banks.get("schema") != "act-three-reset-rainbow25-banks-v41":
        raise RuntimeError("v41 bank schema differs")
    all_values: list[int] = []
    for task in TASKS:
        spec = banks["tasks"][task]
        if len(spec["training"]) != TRAINING_EPISODES:
            raise RuntimeError("v41 requires exactly 18 training episodes")
        if len(spec["screen"]) != SCREEN_EPISODES:
            raise RuntimeError("v41 requires exactly seven screen episodes")
        if len(spec["final"]) != FINAL_EPISODES:
            raise RuntimeError("v41 requires exactly 50 final episodes")
        values = [int(spec["pose_design_seed"]), *map(int, spec["training"]),
                  *map(int, spec["screen"]), *map(int, spec["final"])]
        if len(values) != len(set(values)):
            raise RuntimeError(f"v41 task banks overlap: {task}")
        all_values.extend(values)
    if len(all_values) != len(set(all_values)):
        raise RuntimeError("v41 cross-task banks overlap")


class V41Runtime:
    """Adapter matching the public Rainbow runner with frozen-pose training."""

    def __init__(self, frontier: ACTFrontierRuntime, task_label: str, device: str,
                 training_seeds: list[int], poses: list[list[float]]):
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
        order = [0, 1, 2] * 6
        self.training_pose_index = dict(zip(map(int, training_seeds), order))
        self.poses = [np.asarray(pose, dtype=np.float64) for pose in poses]

    def validate_search_record(self, record: dict) -> None:
        if record.get("physics_error") is not None:
            raise RuntimeError("v41 halted immediately on a Rainbow physics error")

    def environment(self, seed: int, *, training=False):
        seed = int(seed)
        fixed = seed in self.training_pose_index
        if training and not fixed:
            raise RuntimeError("v41 training attempted an unregistered reset")
        pose = self.poses[self.training_pose_index[seed]] if fixed else None
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
            object_pose=pose,
            seed=seed,
            randomize_object_pose=not fixed,
            speed_values=SPEED_VALUES,
            observation_encoder=encoder,
            decision_frame_skip=10,
            decision_mode="fixed_or_phase_entry",
            terminate_on_success=False,
            safety_monitor=partial(workspace_violation, self.frontier.task),
        )
        env.env = JointEffectorObservationWrapper(env.env)
        env._environment_metadata["learned_phase_effector_source"] = "joint_fk_body_xpos"
        env._environment_metadata["v41_reset_regime"] = (
            "fixed_three_pose_training" if fixed else "fresh_randomized"
        )
        return env


def build_deployment_checkpoint(training_root: Path) -> tuple[Path, str]:
    import torch

    terminal = training_root / "terminal_policy.pt"
    payload = torch.load(terminal, map_location="cpu", weights_only=True)
    payload["environment_spec"] = None
    payload["metadata"] = {
        **dict(payload.get("metadata") or {}),
        "training_terminal_sha256": file_sha256(terminal),
        "deployment_reset_regime": "fresh_randomized_object_pose",
        "environment_spec_relaxation": (
            "reset distribution changes only; observation and action specs remain checked"
        ),
    }
    output = training_root / "DEPLOYMENT_POLICY.pt"
    if not output.exists():
        immutable_torch(output, payload)
    return output, file_sha256(output)


def incidents(records: list[dict]) -> dict:
    return {
        "physics_errors": sum(item.get("physics_error") is not None for item in records),
        "safety_violations": sum(item.get("safety_violation") is not None for item in records),
    }


def run_screen(runtime: V41Runtime, output: Path, identity_sha: str,
               seeds: list[int], policy) -> tuple[dict, list[dict]]:
    records = load_contiguous_states(output / "states", seeds, identity_sha)
    for seed in seeds[len(records):]:
        record = rollout_one(runtime, seed, policy, identity_sha, screen_learning=False)
        if record.get("physics_error") is not None:
            raise RuntimeError("v41 halted immediately on a screen physics error")
        immutable_json(output / "states" / f"{seed}.json", record)
        records.append(record)
        progress(output, identity_sha, records, seeds)
        print(json.dumps({"stage": "screen", "completed": len(records),
                          "successes": sum(bool(item["success"]) for item in records)},
                         sort_keys=True), flush=True)
    result = {
        "schema": "act-three-reset-rainbow25-screen-result-v41",
        "episodes": len(records),
        "summary": summarize(records),
        "identity_sha256": identity_sha,
        "learning_or_tuning_permitted": False,
    }
    immutable_or_verify(output / "RESULT.json", result)
    immutable_or_verify(
        output / "COMPLETE.json",
        {"schema": "act-three-reset-rainbow25-screen-completion-v41",
         "episodes": len(records), "result_sha256": file_sha256(output / "RESULT.json")},
    )
    return result, records


def run_search(runtime: V41Runtime, root: Path, task: str, spec: dict,
               contract: Path, banks: Path, panel: dict) -> None:
    output = root / "search" / task
    training_root = output / "training"
    identity = {
        **runtime.frontier.identity(),
        "schema": "act-three-reset-rainbow25-search-identity-v41",
        "task_label": task,
        "contract_sha256": file_sha256(contract),
        "banks_sha256": file_sha256(banks),
        "search_budget": SEARCH_BUDGET,
        "training_episodes": TRAINING_EPISODES,
        "screen_episodes": SCREEN_EPISODES,
        "training_seeds": spec["training"],
        "training_pose_order": panel["training_pose_order"],
        "three_pose_panel_sha256": canonical_sha256(panel),
        "screen_seeds": spec["screen"],
        "final_seeds_registered_unopened": spec["final"],
        "historical_speed_outcomes_used_for_initialization": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    immutable_or_verify(output / "IDENTITY.json", identity)
    if (output / "SEARCH_COMPLETE.json").exists():
        completion = checked_json(output / "SEARCH_COMPLETE.json")
        if completion["selection_sha256"] != file_sha256(output / "SELECTION.json"):
            raise RuntimeError("v41 completed selection hash mismatch")
        return

    training_identity = canonical_sha256(
        {"search_identity_sha256": identity["identity_sha256"], "stage": "rainbow_training"}
    )
    records = load_contiguous_states(
        training_root / "states", list(map(int, spec["training"])), training_identity
    )
    if not (training_root / "terminal_policy.pt").exists():
        run_rainbow_search(
            runtime, training_root, training_identity,
            list(map(int, spec["training"])), records,
        )
    if len(records) != TRAINING_EPISODES:
        records = load_contiguous_states(
            training_root / "states", list(map(int, spec["training"])), training_identity
        )
    if len(records) != TRAINING_EPISODES:
        raise RuntimeError("v41 Rainbow training did not seal exactly 18 episodes")
    training_incidents = incidents(records)
    if training_incidents["physics_errors"]:
        raise RuntimeError("v41 Rainbow training contains a physics error")

    deployment_path, deployment_sha = build_deployment_checkpoint(training_root)
    policy = RainbowSpeedPolicy.load(deployment_path, device=runtime.args.device)
    screen_identity = canonical_sha256(
        {"search_identity_sha256": identity["identity_sha256"], "stage": "fresh_screen",
         "deployment_policy_sha256": deployment_sha, "seeds": spec["screen"]}
    )
    screen, screen_records = run_screen(
        runtime, output / "screen", screen_identity, list(map(int, spec["screen"])), policy
    )
    screen_incidents = incidents(screen_records)
    qualified = (
        int(screen["summary"]["successes"]) >= SCREEN_SUCCESS_FLOOR
        and screen_incidents == {"physics_errors": 0, "safety_violations": 0}
    )
    selected = "rainbow" if qualified else "native_1x"
    selection = {
        "schema": "act-three-reset-rainbow25-selection-v41",
        "task_label": task,
        "deployment_policy_path": str(deployment_path),
        "deployment_policy_sha256": deployment_sha,
        "training_summary": summarize(records),
        "screen": screen,
        "selection_rule": "rainbow_only_if_7_of_7_fresh_screen_successes_and_zero_incidents",
        "rainbow_qualified": qualified,
        "selected_method": selected,
        "search_scientific_rollouts": len(records) + len(screen_records),
        "search_rollout_split": {"fixed_reset_training": len(records),
                                 "fresh_randomized_screen": len(screen_records)},
        "incident_totals": {
            "physics_errors": training_incidents["physics_errors"] + screen_incidents["physics_errors"],
            "safety_violations": training_incidents["safety_violations"] + screen_incidents["safety_violations"],
        },
        "historical_speed_outcomes_used_for_initialization": False,
        "historical_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    }
    if selection["search_scientific_rollouts"] != SEARCH_BUDGET:
        raise RuntimeError("v41 search accounting differs from exactly 25")
    immutable_or_verify(output / "SELECTION.json", selection)
    immutable_or_verify(
        output / "SEARCH_COMPLETE.json",
        {"schema": "act-three-reset-rainbow25-search-completion-v41",
         "task_label": task, "search_scientific_rollouts": SEARCH_BUDGET,
         "selection_sha256": file_sha256(output / "SELECTION.json"),
         "deployment_policy_sha256": deployment_sha, **selection["incident_totals"],
         "historical_speed_outcomes_used_for_initialization": False,
         "historical_rollouts_reexecuted": 0, "final_bank_opened": False},
    )


def require_all_search(root: Path) -> None:
    for task in TASKS:
        value = checked_json(root / "search" / task / "SEARCH_COMPLETE.json")
        if int(value.get("search_scientific_rollouts", -1)) != SEARCH_BUDGET:
            raise RuntimeError(f"v41 search incomplete: {task}")


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
            raise RuntimeError("v41 deployment checkpoint hash differs")
        controller = {"type": "phase_conditioned_rainbow",
                      "checkpoint_sha256": selection["deployment_policy_sha256"]}
        policy = RainbowSpeedPolicy.load(checkpoint, device=device)
    else:
        raise ValueError(effective)
    controller["controller_sha256"] = canonical_sha256(controller)
    return controller, policy, effective, file_sha256(selection_path)


def run_final(runtime: V41Runtime, root: Path, task: str, method: str,
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
        "schema": "act-three-reset-rainbow25-final-controller-identity-v41",
        "task_label": task, "controller": controller,
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "banks_sha256": banks_sha, "search_or_tuning_permitted": False,
        "reset_distribution": "fresh_randomized_object_pose",
        "historical_speed_outcomes_used_for_initialization": False,
        "historical_rollouts_reexecuted": 0,
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
        atomic_json(controller_root / "progress.json",
                    {"task": task, "controller_sha256": digest,
                     "completed": len(records),
                     "successes": sum(bool(item["success"]) for item in records)})
        print(json.dumps({"stage": "final", "task": task, "method": method,
                          "completed": len(records),
                          "successes": sum(bool(item["success"]) for item in records)},
                         sort_keys=True), flush=True)
    result = {"schema": "act-three-reset-rainbow25-final-controller-result-v41",
              "task_label": task, "controller": controller, "episodes": len(records),
              "summary": summarize(records), "identity_sha256": identity["identity_sha256"]}
    immutable_or_verify(controller_root / "RESULT.json", result)
    immutable_or_verify(
        controller_root / "COMPLETE.json",
        {"schema": "act-three-reset-rainbow25-final-controller-completion-v41",
         "episodes": len(records), "result_sha256": file_sha256(controller_root / "RESULT.json"),
         "physics_errors": result["summary"]["physics_errors"],
         "safety_violations": result["summary"]["safety_violations"]},
    )
    immutable_or_verify(
        alias_path,
        {"schema": "act-three-reset-rainbow25-final-method-result-v41",
         "task_label": task, "method": method, "effective_method": effective,
         "controller": controller, "controller_sha256": digest,
         "controller_result_sha256": file_sha256(controller_root / "RESULT.json"),
         "controller_receipt": str(controller_root / "RESULT.json"),
         "selection_provenance": provenance, "controller_cache_hit": was_complete,
         "summary": result["summary"]},
    )


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
        raise RuntimeError("v41 checked-out source differs from implementation commit")
    banks = checked_json(args.banks)
    validate_banks(banks)
    spec = banks["tasks"][args.task_label]
    frontier = ACTFrontierRuntime(
        source_commit=args.base_source_commit, checkout_commit=args.implementation_commit,
        run_manifest=args.run_manifest, task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint, detector_source=args.detector_source,
        device=args.device,
    )
    root = args.root.resolve()
    panel_path = root / "search" / args.task_label / "THREE_POSES.json"
    panel = panel_builder.build(args.task_label, int(spec["pose_design_seed"]))
    immutable_or_verify(panel_path, panel)
    runtime = V41Runtime(frontier, args.task_label, args.device,
                         list(map(int, spec["training"])), panel["object_pose_vectors"])
    if args.stage == "search":
        if args.method != SEARCH_METHOD:
            raise ValueError("v41 search stage requires three_reset_rainbow25")
        run_search(runtime, root, args.task_label, spec, args.contract, args.banks, panel)
    else:
        if args.method not in FINAL_METHODS:
            raise ValueError("v41 final stage requires a registered final method")
        run_final(runtime, root, args.task_label, args.method,
                  list(map(int, spec["final"])), file_sha256(args.banks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
