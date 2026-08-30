#!/usr/bin/env python3
"""Run the prospective SAIL-inspired warm-start comparison on ACT."""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pin the prospective study's action space before importing the shared module,
# whose default remains the historical quarter-step grid for old receipts.
os.environ.setdefault("SPEEDTUNING_SPEED_VALUES", "1,1.5,2,2.5,3")

from act_speed_benchmark import (  # noqa: E402
    JointEffectorObservationWrapper,
    SPEED_VALUES,
    build_offline_artifact,
    canonical_sha256,
)
from policy_speed_env import create_speed_env, make_speed_reward  # noqa: E402
from scripts import run_act_common_grid_strider_causal25_v32 as v32  # noqa: E402
from scripts import run_act_common_grid_strider_causal_v30 as v30  # noqa: E402
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head  # noqa: E402
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256, summarize, write_json  # noqa: E402
from one_reset_phase_schedule import workspace_violation  # noqa: E402
from tabular_phase_speed import phase_index  # noqa: E402

TASKS = ("pick", "tea", "insertion")
SEARCH_METHODS = ("sail_causal", "sail_tabular")
FINAL_METHODS = ("native_1x", "strider_v32", *SEARCH_METHODS)
PHASES = v32.PHASES
GRID = v32.GRID
SEARCH_BUDGET = 25


def checked_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing required receipt: {path}")
    return json.loads(path.read_text())


def immutable_or_verify(path: Path, value: dict) -> None:
    if path.exists():
        if checked_json(path) != value:
            raise RuntimeError(f"existing immutable receipt differs: {path}")
    else:
        immutable_json(path, value)


def nearest_grid(value: float, minimum: float = 1.5) -> float:
    allowed = [float(item) for item in GRID if float(item) >= minimum]
    return min(allowed, key=lambda item: (abs(item - float(value)), item))


def sail_phase_prior(artifact: dict) -> dict:
    """Map the preregistered maximum-speed offline profile to four phases."""

    candidates = artifact["candidates"]
    source = max(candidates, key=lambda item: float(item["maximum_speed"]))
    profile = list(map(float, source["profile"]))
    importance = list(map(float, source["importance"]))
    if len(profile) != 8 or len(importance) != 8:
        raise RuntimeError("v33 requires the registered eight-bin offline profile")
    phase_profile = [statistics.fmean(profile[2 * i : 2 * i + 2]) for i in range(4)]
    phase_importance = [
        statistics.fmean(importance[2 * i : 2 * i + 2]) for i in range(4)
    ]
    schedule = [nearest_grid(value) for value in phase_profile]
    result = {
        "schema": "act-sail-inspired-phase-prior-v33",
        "paper_faithful_sail": False,
        "source_artifact_payload_sha256": artifact["artifact_payload_sha256"],
        "source_candidate_id": source["id"],
        "mapping": "mean_each_two_of_eight_nominal_time_bins_then_nearest_common_grid",
        "minimum_phase_speed": 1.5,
        "phase_order": list(PHASES),
        "phase_profile": phase_profile,
        "phase_importance": phase_importance,
        "schedule": schedule,
    }
    result["prior_payload_sha256"] = canonical_sha256(result)
    return result


def build_or_load_prior(runtime: ACTFrontierRuntime, root: Path) -> dict:
    path = root / "offline" / runtime.task_label / "SAIL_PRIOR.json"
    artifact = build_offline_artifact(
        Path(runtime.task_manifest["policy_root"]) / "dataset",
        "sail_inspired_adaptive",
    )
    prior = sail_phase_prior(artifact)
    value = {"offline_artifact": artifact, "phase_prior": prior}
    immutable_or_verify(path, value)
    return value


def sail_ranked_promotions(records: list[dict], schedule: list[float], prior: dict, frozen: set[str]):
    usable = [item for item in records if v32.successful(item)]
    if not usable:
        return []
    workloads = {
        phase: statistics.fmean(v32.prior.base.phase_workloads(item)[phase] for item in usable)
        for phase in PHASES
    }
    proposals = []
    for index, (phase, speed) in enumerate(zip(PHASES, schedule)):
        if phase in frozen or float(speed) == float(GRID[-1]):
            continue
        promoted = v30.adjacent_speed(speed, 1)
        saved = workloads[phase] * (1.0 / speed - 1.0 / promoted)
        sail_safety_weight = 1.0 - 0.5 * float(prior["phase_importance"][index])
        proposal = v30.make_promotion(schedule, phase)
        proposals.append((saved * sail_safety_weight, -index, phase, proposal, workloads[phase]))
    return sorted(proposals, reverse=True)


def sail_guard(schedule: list[float], prior: dict, tried: set[str]):
    ranked = sorted(
        range(len(PHASES)),
        key=lambda index: (float(prior["phase_importance"][index]), -index),
        reverse=True,
    )
    for index in ranked:
        if float(schedule[index]) <= 1.5:
            continue
        proposal = list(schedule)
        proposal[index] = v30.adjacent_speed(schedule[index], -1)
        if v32.schedule_sha256(proposal) not in tried:
            return proposal, PHASES[index]
    return None, None


def run_causal_search(ledger: v32.Ledger, task: str, prior: dict) -> dict:
    reports: list[dict] = []
    records_by_hash: dict[str, list[dict]] = {}
    receipts: list[dict] = []
    frozen: set[str] = set()

    def evaluate(schedule, role):
        report, records = ledger.discovery_report(schedule, role)
        reports.append(report)
        records_by_hash[report["schedule_sha256"]] = records
        return report, records

    native, native_records = evaluate([1.0] * len(PHASES), "native_reference")
    if not v32.safe(native):
        raise RuntimeError("v33 native reference is not safe 3/3")
    last, last_records = evaluate(prior["schedule"], "sail_inspired_warm_start")

    while len(reports) < v32.DISCOVERY_SCHEDULES:
        tried = {item["schedule_sha256"] for item in reports}
        safe_accelerated = [
            item for item in reports
            if v32.safe(item) and item["schedule"] != [1.0] * 4
        ]
        if not v32.safe(last):
            incumbent = (
                max(safe_accelerated, key=lambda item: item["summary"]["achieved_throughput_per_step"])
                if safe_accelerated else native
            )
            incumbent_records = records_by_hash[incumbent["schedule_sha256"]]
            proposed, receipt = v32.causal_backoff(
                last, last_records, incumbent, incumbent_records, native_records
            )
            receipts.append(receipt)
            if receipt.get("phase") is not None:
                frozen.add(receipt["phase"])
            if proposed is not None and v32.schedule_sha256(proposed) in tried:
                proposed = None
        else:
            incumbent = last
            incumbent_records = last_records
            ranked = sail_ranked_promotions(
                incumbent_records, incumbent["schedule"], prior, frozen
            )
            proposed = next(
                (item[3] for item in ranked if v32.schedule_sha256(item[3]) not in tried),
                None,
            )
            if proposed is not None:
                phase = next(item[2] for item in ranked if item[3] == proposed)
                receipts.append(
                    {
                        "operation": "sail_weighted_one_rung_promotion",
                        "source_schedule": incumbent["schedule"],
                        "proposed_schedule": proposed,
                        "phase": phase,
                    }
                )
        if proposed is None:
            base = (
                max(safe_accelerated, key=lambda item: item["summary"]["achieved_throughput_per_step"])
                if safe_accelerated else last
            )
            proposed, phase = sail_guard(base["schedule"], prior, tried)
            receipts.append(
                {
                    "operation": "sail_importance_one_rung_guard",
                    "source_schedule": base["schedule"],
                    "proposed_schedule": proposed,
                    "phase": phase,
                }
            )
        if proposed is None or v32.schedule_sha256(proposed) in tried:
            raise RuntimeError("v33 could not fill five unique causal discovery slots")
        last, last_records = evaluate(proposed, "sail_initialized_causal_update")

    accelerated = [item for item in reports if item["schedule"] != [1.0] * 4]
    ranked = sorted(
        accelerated,
        key=lambda item: (
            v32.safe(item),
            item["summary"]["successes"],
            item["summary"]["achieved_throughput_per_step"],
        ),
        reverse=True,
    )
    finalists = [ledger.confirmation_report(item)[0] for item in ranked[:2]]
    if ledger.used() != SEARCH_BUDGET:
        raise RuntimeError(f"v33 causal search used {ledger.used()}, expected 25")
    eligible = [
        item for item in finalists
        if item["summary"]["successes"] >= 7
        and item["summary"]["physics_errors"] == 0
        and item["summary"]["safety_violations"] == 0
    ]
    selected = max(
        eligible,
        key=lambda item: (
            item["summary"]["successes"],
            item["summary"]["achieved_throughput_per_step"],
            -len(set(item["schedule"])),
        ),
        default=None,
    )
    return {
        "schema": "act-sail-inspired-causal25-selection-v33",
        "task_label": task,
        "offline_prior": prior,
        "discovery_reports": reports,
        "update_receipts": receipts,
        "frozen_causal_phases": sorted(frozen, key=PHASES.index),
        "finalists": finalists,
        "selection_status": "accelerated_schedule_selected" if selected else "no_acceleration_selected",
        "selected_schedule": None if selected is None else selected["schedule"],
        "selected_schedule_sha256": None if selected is None else selected["schedule_sha256"],
        "search_scientific_rollouts": ledger.used(),
        "prior_rollouts_reexecuted": 0,
        "incident_totals": ledger.incident_totals(),
        "final_bank_opened": False,
    }


def prior_q_values(schedule: list[float]) -> np.ndarray:
    values = np.empty((len(PHASES), len(SPEED_VALUES)), dtype=np.float64)
    for phase, target in enumerate(schedule):
        for action, speed in enumerate(SPEED_VALUES):
            values[phase, action] = -1e-3 * abs(float(speed) - float(target))
    return values


def tabular_rebuild(records: list[dict], schedule: list[float]):
    q_values = prior_q_values(schedule)
    visits = np.zeros_like(q_values, dtype=np.int64)
    for record in records:
        value = 0.0
        returns = []
        for transition in reversed(record["training_trajectory"]):
            value = float(transition["reward"]) + 0.97 * value
            returns.append((transition["phase"], transition["action"], value))
        for phase, action, value in reversed(returns):
            visits[phase, action] += 1
            q_values[phase, action] += (value - q_values[phase, action]) / visits[phase, action]
    return q_values, visits


class TabularRuntime:
    def __init__(self, runtime: ACTFrontierRuntime):
        self.runtime = runtime

    def environment(self, seed: int):
        env = create_speed_env(
            task_name=self.runtime.task,
            reward_fn=make_speed_reward(100.0, 0.01, 2.0),
            chunk_predictor=self.runtime.adapter,
            seed=int(seed),
            randomize_object_pose=True,
            speed_values=SPEED_VALUES,
            observation_encoder=self.runtime.encoder(),
            decision_frame_skip=10,
            decision_mode="fixed_or_phase_entry",
            terminate_on_success=False,
            safety_monitor=functools.partial(workspace_violation, self.runtime.task),
        )
        env.env = JointEffectorObservationWrapper(env.env)
        env._environment_metadata["learned_phase_effector_source"] = "joint_fk_body_xpos"
        return env


def load_tabular_states(directory: Path, seeds: list[int], identity: str) -> list[dict]:
    records = []
    missing = False
    for seed in seeds:
        path = directory / f"{seed}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError("v33 tabular states contain a non-contiguous suffix")
        value = checked_json(path)
        if value.get("seed") != seed or value.get("identity_sha256") != identity:
            raise RuntimeError(f"v33 tabular state identity mismatch: {path}")
        records.append(value)
    return records


def greedy_schedule(q_values: np.ndarray) -> list[float]:
    return [float(SPEED_VALUES[int(np.argmax(row))]) for row in q_values]


def run_tabular_search(runtime: TabularRuntime, output: Path, identity: str, seeds: list[int], prior: dict):
    records = load_tabular_states(output / "states", seeds, identity)
    q_values, visits = tabular_rebuild(records, prior["schedule"])
    rng = np.random.default_rng(3301)
    if records:
        rng.bit_generator.state = records[-1]["rng_state_after"]
    for index, seed in enumerate(seeds[len(records):], start=len(records)):
        epsilon = 0.5 + index / max(len(seeds) - 1, 1) * (0.05 - 0.5)
        env = runtime.environment(seed)
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
                "safety_violation": info.get("safety_violation"),
                "physics_error": info.get("physics_error"),
                "epsilon": epsilon,
                "training_trajectory": trajectory,
                "rng_state_after": rng.bit_generator.state,
                "observation_spec": env.observation_spec(),
                "environment_spec": env.environment_spec(),
            }
        finally:
            env.close()
        immutable_json(output / "states" / f"{seed}.json", record)
        records.append(record)
        q_values, visits = tabular_rebuild(records, prior["schedule"])
        atomic_json(
            output / "progress.json",
            {
                "completed": len(records),
                "successes": sum(bool(item["success"]) for item in records),
                "physics_errors": sum(item.get("physics_error") is not None for item in records),
                "safety_violations": sum(item.get("safety_violation") is not None for item in records),
                "greedy_schedule": greedy_schedule(q_values),
            },
        )
        print(json.dumps({"method": "sail_tabular", "completed": len(records)}), flush=True)
        if record.get("physics_error") is not None:
            raise RuntimeError("physics error in v33 Tabular search; receipt preserved")
    return {
        "schema": "act-sail-inspired-tabular25-selection-v33",
        "offline_prior": prior,
        "algorithm": "sail_prior_initialized_first_visit_monte_carlo_phase_speed",
        "epsilon_start": 0.5,
        "epsilon_end": 0.05,
        "q_values": q_values.tolist(),
        "visits": visits.tolist(),
        "selected_schedule": greedy_schedule(q_values),
        "selected_schedule_sha256": v32.schedule_sha256(greedy_schedule(q_values)),
        "search_scientific_rollouts": len(records),
        "incident_totals": {
            "physics_errors": sum(item.get("physics_error") is not None for item in records),
            "safety_violations": sum(item.get("safety_violation") is not None for item in records),
        },
        "prior_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    }


def all_search_complete(root: Path) -> None:
    for task in TASKS:
        for method in SEARCH_METHODS:
            completion = checked_json(root / "search" / task / method / "SEARCH_COMPLETE.json")
            if completion.get("search_scientific_rollouts") != 25:
                raise RuntimeError(f"v33 incomplete search: {task}/{method}")


def selected_schedule(root: Path, v32_root: Path, task: str, method: str, v32_banks_sha: str):
    if method == "native_1x":
        return [1.0] * len(PHASES), None
    if method == "strider_v32":
        checked_json(v32_root / "COMPLETE.json")
        completion = checked_json(v32_root / task / "SEARCH_COMPLETE.json")
        identity_path = v32_root / task / "IDENTITY.json"
        identity = checked_json(identity_path)
        path = v32_root / task / "SELECTION.json"
        if completion["identity_sha256"] != file_sha256(identity_path):
            raise RuntimeError("v32 identity receipt hash mismatch")
        if completion["selection_sha256"] != file_sha256(path):
            raise RuntimeError("v32 selection receipt hash mismatch")
        if (
            identity["banks_sha256"] != v32_banks_sha
            or completion.get("final_bank_opened") is not False
        ):
            raise RuntimeError("v32 final-bank identity mismatch")
    else:
        path = root / "search" / task / method / "SELECTION.json"
    selection = checked_json(path)
    schedule = list(v30.validate_schedule(selection["selected_schedule"]))
    if v32.schedule_sha256(schedule) != selection["selected_schedule_sha256"]:
        raise RuntimeError(f"selected schedule hash mismatch: {path}")
    return schedule, path


def run_final(
    runtime: ACTFrontierRuntime,
    root: Path,
    v32_root: Path,
    task: str,
    method: str,
    seeds: list[int],
    banks_sha: str,
    v32_banks_sha: str,
):
    all_search_complete(root)
    schedule, selection_path = selected_schedule(root, v32_root, task, method, v32_banks_sha)
    output = root / "final" / task / method
    identity = {
        **runtime.identity(),
        "schema": "act-sail-warmstart-final-identity-v33",
        "method": method,
        "schedule": schedule,
        "schedule_sha256": v32.schedule_sha256(schedule),
        "selection_sha256": None if selection_path is None else file_sha256(selection_path),
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "banks_sha256": banks_sha,
        "v32_banks_sha256": v32_banks_sha,
        "search_or_tuning_permitted": False,
        "prior_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    immutable_or_verify(output / "IDENTITY.json", identity)
    records = load_tabular_states(output / "states", seeds, identity["identity_sha256"])
    if (output / "COMPLETE.json").exists():
        if len(records) != 50:
            raise RuntimeError("v33 final completion exists without 50 states")
        return checked_json(output / "RESULT.json")
    for seed in seeds[len(records):]:
        record = runtime.rollout(schedule, seed, record_attribution_telemetry=False)
        if record.get("seed") != seed or list(map(float, record.get("schedule", ()))) != schedule:
            raise RuntimeError("v33 final runtime returned a different identity")
        record["identity_sha256"] = identity["identity_sha256"]
        immutable_json(output / "states" / f"{seed}.json", record)
        records.append(record)
        atomic_json(
            output / "progress.json",
            {
                "task": task,
                "method": method,
                "completed": len(records),
                "successes": sum(bool(item["success"]) for item in records),
                "physics_errors": sum(item.get("physics_error") is not None for item in records),
                "safety_violations": sum(item.get("safety_violation") is not None for item in records),
            },
        )
        print(json.dumps({"stage": "final", "task": task, "method": method, "completed": len(records)}), flush=True)
    result = {
        "schema": "act-sail-warmstart-final-result-v33",
        "task_label": task,
        "method": method,
        "schedule": schedule,
        "episodes": 50,
        "summary": summarize(records),
        "identity_sha256": identity["identity_sha256"],
    }
    immutable_json(output / "RESULT.json", result)
    immutable_json(
        output / "COMPLETE.json",
        {
            "schema": "act-sail-warmstart-final-completion-v33",
            "episodes": 50,
            "result_sha256": file_sha256(output / "RESULT.json"),
            "physics_errors": result["summary"]["physics_errors"],
            "safety_violations": result["summary"]["safety_violations"],
            "prior_rollouts_reexecuted": 0,
        },
    )
    return result


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "final"), required=True)
    parser.add_argument("--method", choices=(*SEARCH_METHODS, *FINAL_METHODS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v32-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=TASKS, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--v32-banks", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if git_head() != args.source_commit:
        raise RuntimeError("v33 checked-out source differs from requested commit")
    if args.stage == "search" and args.method not in SEARCH_METHODS:
        raise ValueError("v33 search requires a warm-start method")
    banks = checked_json(args.banks)
    v32_banks = checked_json(args.v32_banks)
    spec = banks["tasks"][args.task_label]
    final_spec = v32_banks["tasks"][args.task_label]["final"]
    final_seeds = list(range(final_spec["start"], final_spec["start"] + final_spec["count"]))
    runtime = ACTFrontierRuntime(
        source_commit=args.source_commit,
        run_manifest=args.run_manifest,
        task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
    )
    root = args.root.resolve()
    if args.stage == "final":
        run_final(
            runtime,
            root,
            args.v32_root.resolve(),
            args.task_label,
            args.method,
            final_seeds,
            file_sha256(args.banks),
            file_sha256(args.v32_banks),
        )
        return 0

    output = root / "search" / args.task_label / args.method
    prior_bundle = build_or_load_prior(runtime, root)
    prior = prior_bundle["phase_prior"]
    method_bank = spec[args.method]
    if args.method == "sail_causal":
        discovery = list(map(int, method_bank["discovery"]))
        confirmation = list(map(int, method_bank["confirmation"]))
        if (
            len(discovery) != 3
            or len(set(discovery)) != 3
            or len(confirmation) != 5
            or len(set(confirmation)) != 5
            or set(discovery) & set(confirmation)
        ):
            raise RuntimeError("v33 causal search requires three discovery and five confirmation seeds")
        seeds = discovery + confirmation
    else:
        seeds = list(map(int, method_bank))
        if len(seeds) != SEARCH_BUDGET or len(set(seeds)) != SEARCH_BUDGET:
            raise RuntimeError("v33 tabular search requires exactly 25 unique episode seeds")
    identity = {
        **runtime.identity(),
        "schema": "act-sail-warmstart-search-identity-v33",
        "method": args.method,
        "contract_sha256": file_sha256(args.contract),
        "banks_sha256": file_sha256(args.banks),
        "v32_banks_sha256": file_sha256(args.v32_banks),
        "search_seed_bank": method_bank,
        "search_budget": SEARCH_BUDGET,
        "offline_prior_payload_sha256": prior["prior_payload_sha256"],
        "historical_schedule_outcomes_visible": False,
        "final_seeds_registered_unopened": final_seeds,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    immutable_or_verify(output / "IDENTITY.json", identity)
    selection_path = output / "SELECTION.json"
    complete_path = output / "SEARCH_COMPLETE.json"
    if complete_path.exists():
        completion = checked_json(complete_path)
        if completion["selection_sha256"] != file_sha256(selection_path):
            raise RuntimeError("v33 completed selection hash mismatch")
        print(json.dumps(checked_json(selection_path), sort_keys=True))
        return 0
    if args.method == "sail_causal":
        selection = run_causal_search(
            v32.Ledger(runtime, output / "ledger", discovery, confirmation),
            args.task_label,
            prior,
        )
    else:
        selection = run_tabular_search(TabularRuntime(runtime), output, identity["identity_sha256"], seeds, prior)
    immutable_or_verify(selection_path, selection)
    incidents = selection["incident_totals"]
    immutable_json(
        complete_path,
        {
            "schema": "act-sail-warmstart-search-completion-v33",
            "method": args.method,
            "identity_sha256": file_sha256(output / "IDENTITY.json"),
            "selection_sha256": file_sha256(selection_path),
            "search_scientific_rollouts": SEARCH_BUDGET,
            "physics_errors": incidents["physics_errors"],
            "safety_violations": incidents["safety_violations"],
            "prior_rollouts_reexecuted": 0,
            "final_bank_opened": False,
        },
    )
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
