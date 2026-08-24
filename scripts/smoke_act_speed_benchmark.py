#!/usr/bin/env python3
"""Exercise all six ACT benchmark families on non-registered engineering seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import (  # noqa: E402
    METHODS,
    SPEED_VALUES,
    build_offline_artifact,
    policy_from_candidate,
    preregistration,
)
from scripts.run_act_speed_benchmark_cell import (  # noqa: E402
    CellRuntime,
    immutable_json,
    prepare,
)
from speed_policy import rollout_speed_policy  # noqa: E402
from tabular_phase_speed import TabularPhaseSpeedPolicy  # noqa: E402


def rainbow_episode(runtime, seed):
    from rl.rainbowDQN.dqnAgent import DQNAgent

    config = runtime.prereg["training"]
    env = runtime.environment(seed, training=True)
    updates = 0
    try:
        state = env.reset()
        agent = DQNAgent(
            env, memory_size=128, batch_size=32, target_update=50,
            seed=config["seed"], lr=config["learning_rate"], gamma=config["gamma"],
            frame_skip=10, atom_size=21, v_min=0, v_max=120,
            n_step=3, hidden_dim=32, device=runtime.args.device,
        )
        done = False
        info = {"success": False}
        while not done:
            action = agent.select_action(state)
            state, _, done, info = agent.step(action, 10)
            if len(agent.memory) >= 32:
                agent.update_model()
                updates += 1
        return {
            "success": bool(info["success"]),
            "physics_steps": int(info["physics_steps"]),
            "safety_violation": info.get("safety_violation"),
            "physics_error": info.get("physics_error"),
            "updates": updates,
            "observation_spec": env.observation_spec(),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--task-label", default="pick")
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base_args = SimpleNamespace(
        contract=args.contract, run_manifest=args.run_manifest,
        task_label=args.task_label, method="uniform_sweep", stage="search",
        source_commit=args.source_commit, detector_checkpoint=None,
        detector_source=None, device=args.device,
    )
    contract, manifest, task, _, _, adapter = prepare(base_args)
    results = {}
    engineering_seed_base = 12_500_000
    registered = set()
    for value in contract["tasks"].values():
        registered.update(range(value["search_seed_base"], value["search_seed_base"] + 50))
        registered.update(range(value["final_seed_base"], value["final_seed_base"] + 50))

    for index, method in enumerate(METHODS):
        seed = engineering_seed_base + index
        if seed in registered:
            raise RuntimeError("engineering smoke seed overlaps a registered bank")
        prereg = preregistration(method)
        method_args = SimpleNamespace(**vars(base_args))
        method_args.method = method
        if prereg["phase_detector_required"]:
            method_args.detector_checkpoint = args.detector_checkpoint
            method_args.detector_source = args.detector_source
        runtime = CellRuntime(method_args, contract, manifest, task, prereg, adapter)
        offline = None
        if method in {"awe_offline_proxy", "sail_inspired_adaptive"}:
            offline = build_offline_artifact(Path(task["root"]) / "dataset", method)
        if method == "learned_phase_rainbow_rl":
            result = rainbow_episode(runtime, seed)
        else:
            if method == "learned_phase_tabular_rl":
                policy = TabularPhaseSpeedPolicy(np.zeros((4, len(SPEED_VALUES))), SPEED_VALUES)
            else:
                candidate = prereg["candidates"][0]
                if offline is not None:
                    candidate = offline["candidates"][0]
                policy = policy_from_candidate(method, candidate, offline)
            env = runtime.environment(seed, training=method == "learned_phase_tabular_rl")
            try:
                result = rollout_speed_policy(env, policy, frame_skip=10)
                result["observation_spec"] = env.observation_spec()
            finally:
                env.close()
        result.update(
            seed=seed,
            registered_budget=False,
            phase_detector_received=prereg["phase_detector_required"],
        )
        results[method] = result
        print(json.dumps({"method": method, "seed": seed, "success": result["success"]}), flush=True)

    receipt = {
        "schema": "act-speed-six-family-engineering-smoke-v1",
        "source_commit": args.source_commit,
        "task_label": args.task_label,
        "registered_rollouts_consumed": 0,
        "methods": results,
    }
    immutable_json(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

