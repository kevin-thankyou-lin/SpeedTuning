#!/usr/bin/env python3
"""Run exact-25 prior-guided causal speed search with a causal risk gate."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SPEEDTUNING_SPEED_VALUES", "1,1.5,2,2.5,3")

from act_speed_benchmark import JointEffectorObservationWrapper, canonical_sha256  # noqa: E402
from learned_phase_observation import PHASES  # noqa: E402
from one_reset_phase_schedule import TASK_OBJECTS, workspace_violation  # noqa: E402
from policy_speed_env import create_speed_env  # noqa: E402
from scripts import run_act_common_grid_strider_causal25_v32 as v32  # noqa: E402
from scripts import run_act_common_grid_strider_causal_v30 as v30  # noqa: E402
from scripts import run_act_sail_warmstart_v33 as v33  # noqa: E402
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head  # noqa: E402
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256, summarize  # noqa: E402
from sim_tasks import normalize_task_name  # noqa: E402
from speed_policy import SpeedContext  # noqa: E402

TASKS = ("pick", "tea", "insertion")
FINAL_METHODS = ("native_1x", "phase_only", "risk_gated")
GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
SEARCH_BUDGET = 25
DISCOVERY_SCHEDULES = 5
CONTACT_CRITICAL_PHASES = frozenset(("grasp_lift", "interaction"))


def combined_prior(path: Path, task: str) -> dict:
    """Combine the zero-rollout semantic schedule with the sealed SAIL prior."""

    bundle, sail = v33.load_trained_prior(path, task)
    semantic = v33.agent_semantic_prior()
    importance = [
        max(float(left), float(right))
        for left, right in zip(semantic["phase_importance"], sail["phase_importance"])
    ]
    result = {
        "schema": "act-semantic-sail-combined-prior-v34",
        "prior_kind": "semantic_schedule_with_sail_precision_ranking",
        "phase_order": list(PHASES),
        "schedule": list(map(float, semantic["schedule"])),
        "phase_importance": importance,
        "semantic_prior_sha256": semantic["prior_payload_sha256"],
        "sail_prior_sha256": sail["prior_payload_sha256"],
        "sail_bundle_payload_sha256": bundle["payload_sha256"],
        "offline_training_rollouts": int(bundle["offline_training_rollouts"]),
        "online_rollouts": 0,
        "historical_speed_outcomes_used_by_runtime": False,
        "study_design_informed_by_v33_results": True,
        "paper_faithful_sail": False,
        "gripper_delta_threshold": float(
            sail["precision_target"]["gripper_delta_threshold"]
        ),
    }
    result["prior_payload_sha256"] = canonical_sha256(result)
    return result


def risk_gate_spec(prior: dict) -> dict:
    protected = sorted(
        CONTACT_CRITICAL_PHASES
        | {
            phase
            for phase, importance in zip(PHASES, prior["phase_importance"])
            if float(importance) >= 0.5
        },
        key=PHASES.index,
    )
    result = {
        "schema": "act-causal-observation-risk-gate-v34",
        "inputs": [
            "current_learned_phase_argmax",
            "current_observed_gripper_qpos",
        ],
        "protected_phases": protected,
        "gripper_indices": [6, 13],
        "gripper_delta_threshold": float(prior["gripper_delta_threshold"]),
        "override_speed": 1.0,
        "release_rule": "one_consecutive_stable_observation",
        "future_action_or_terminal_signal_visible": False,
    }
    result["controller_sha256"] = canonical_sha256(result)
    return result


@dataclass
class CausalRiskGate:
    spec: dict
    last_phase: int | None = None
    last_grippers: np.ndarray | None = None
    latched: bool = False
    stable_observations: int = 0
    events: list[dict] = field(default_factory=list)

    def choose(self, base_speed: float, phase: int, qpos, physics_step: int) -> tuple[float, list[str]]:
        grippers = np.asarray(qpos, dtype=np.float64)[self.spec["gripper_indices"]]
        phase_changed = self.last_phase is None or int(phase) != int(self.last_phase)
        gripper_delta = (
            0.0
            if self.last_grippers is None
            else float(np.max(np.abs(grippers - self.last_grippers)))
        )
        gripper_changed = gripper_delta >= float(self.spec["gripper_delta_threshold"])
        protected = PHASES[int(phase)] in self.spec["protected_phases"]
        reasons = []
        if phase_changed:
            reasons.append("learned_phase_transition")
        if gripper_changed:
            reasons.append("observed_gripper_transition")
        if (phase_changed and protected) or gripper_changed:
            self.latched = True
            self.stable_observations = 0
        elif self.latched:
            self.stable_observations += 1
            if self.stable_observations >= 1:
                self.latched = False
        effective = float(self.spec["override_speed"] if self.latched else base_speed)
        if effective != float(base_speed):
            self.events.append(
                {
                    "physics_step": int(physics_step),
                    "phase": PHASES[int(phase)],
                    "base_speed": float(base_speed),
                    "effective_speed": effective,
                    "reasons": reasons or ["latched_risk"],
                    "observed_gripper_delta": gripper_delta,
                }
            )
        self.last_phase = int(phase)
        self.last_grippers = grippers.copy()
        return effective, reasons


def run_risk_gated_schedule(
    runtime: ACTFrontierRuntime,
    schedule,
    seed: int,
    gate_spec: dict,
    *,
    record_attribution_telemetry: bool = False,
) -> dict:
    """Run one phase schedule with an observation-only per-tick downshift."""

    task = normalize_task_name(runtime.task)
    schedule = list(v30.validate_schedule(schedule))
    env = create_speed_env(
        task_name=task,
        seed=int(seed),
        randomize_object_pose=True,
        chunk_predictor=runtime.adapter,
        speed_values=GRID,
        observation_encoder=runtime.encoder(),
        decision_mode="phase_entry",
        decision_frame_skip=1,
        terminate_on_success=False,
    )
    env.env = JointEffectorObservationWrapper(env.env)
    env._environment_metadata["learned_phase_effector_source"] = "joint_fk_body_xpos"
    gate = CausalRiskGate(gate_spec)
    safety = None
    telemetry = []
    decisions = []
    last_decision = None
    info = {"success": False, "physics_steps": 0, "policy_time": 0.0}
    try:
        observation = env.reset()
        done = False
        while not done:
            phase = int(np.argmax(np.asarray(observation, dtype=np.float64)))
            base_speed = float(schedule[phase])
            effective_speed, reasons = gate.choose(
                base_speed,
                phase,
                env.cur_ts.observation["qpos"],
                env.physics_steps,
            )
            decision = (phase, base_speed, effective_speed, tuple(reasons))
            if decision != last_decision:
                decisions.append(
                    {
                        "phase": PHASES[phase],
                        "physics_step": int(env.physics_steps),
                        "base_speed": base_speed,
                        "speed": effective_speed,
                        "risk_reasons": list(reasons),
                    }
                )
                last_decision = decision
            observation, reward, done, info = env.step(effective_speed, quantized=False)
            safety = safety or workspace_violation(task, env.cur_ts.observation)
            if record_attribution_telemetry:
                env_state = np.asarray(env.cur_ts.observation["env_state"], dtype=np.float64)
                telemetry.append(
                    {
                        "physics_step": int(env.physics_steps),
                        "policy_time": float(env.policy_time),
                        "observed_phase": PHASES[
                            int(np.argmax(np.asarray(observation, dtype=np.float64)))
                        ],
                        "task_reward": float(0.0 if reward is None else reward),
                        "robot_qpos": [
                            float(value) for value in env.cur_ts.observation["qpos"]
                        ],
                        "object_positions": [
                            [float(value) for value in env_state[i * 7 : i * 7 + 3]]
                            for i in range(TASK_OBJECTS[task])
                        ],
                    }
                )
        first = info.get("first_success_step")
        metric_steps = int(info["physics_steps"] if first is None else first)
        result = {
            "task": task,
            "seed": int(seed),
            "schedule": schedule,
            "controller_sha256": gate_spec["controller_sha256"],
            "success": bool(info["success"]) and safety is None,
            "raw_task_success": bool(info["success"]),
            "physics_steps": int(info["physics_steps"]),
            "first_success_step": first,
            "success_only_acceleration": (
                float(env.episode_len / max(metric_steps, 1))
                if bool(info["success"]) and safety is None
                else None
            ),
            "safety_violation": safety,
            "phase_decisions": decisions,
            "risk_gate_events": gate.events,
            "risk_gate_event_count": len(gate.events),
        }
        if record_attribution_telemetry:
            result["attribution_telemetry"] = telemetry
        if info.get("physics_error") is not None:
            result["physics_error"] = str(info["physics_error"])
        return result
    finally:
        env.close()


class RiskGatedRuntime:
    def __init__(self, runtime: ACTFrontierRuntime, gate_spec: dict):
        self.runtime = runtime
        self.gate_spec = gate_spec

    def rollout(self, schedule, seed: int, *, record_attribution_telemetry=False):
        return run_risk_gated_schedule(
            self.runtime,
            schedule,
            seed,
            self.gate_spec,
            record_attribution_telemetry=record_attribution_telemetry,
        )


def ranked_promotions(records, schedule, prior, frozen):
    usable = [item for item in records if v32.successful(item)]
    if not usable:
        return []
    workloads = {
        phase: statistics.fmean(v32.prior.base.phase_workloads(item)[phase] for item in usable)
        for phase in PHASES
    }
    ranked = []
    for index, (phase, speed) in enumerate(zip(PHASES, schedule)):
        if phase in frozen or float(speed) == GRID[-1]:
            continue
        promoted = v30.adjacent_speed(speed, 1)
        saved = workloads[phase] * (1.0 / speed - 1.0 / promoted)
        precision_penalty = 1.0 - 0.5 * float(prior["phase_importance"][index])
        ranked.append(
            (saved * precision_penalty, -index, phase, v30.make_promotion(schedule, phase))
        )
    return sorted(ranked, reverse=True)


def run_search(ledger: v32.Ledger, task: str, prior: dict, gate_spec: dict) -> dict:
    reports = []
    records_by_hash = {}
    receipts = []
    frozen = set()

    def evaluate(schedule, role):
        report, records = ledger.discovery_report(schedule, role)
        reports.append(report)
        records_by_hash[report["schedule_sha256"]] = records
        return report, records

    native, native_records = evaluate([1.0] * 4, "native_reference")
    last, last_records = evaluate(prior["schedule"], "combined_prior_warm_start")
    while len(reports) < DISCOVERY_SCHEDULES:
        tried = {item["schedule_sha256"] for item in reports}
        safe_accelerated = [
            item for item in reports if v32.safe(item) and item["schedule"] != [1.0] * 4
        ]
        incumbent = (
            max(safe_accelerated, key=lambda item: item["summary"]["achieved_throughput_per_step"])
            if safe_accelerated
            else native
        )
        incumbent_records = records_by_hash[incumbent["schedule_sha256"]]
        proposed = None
        if not v32.safe(last):
            proposed, receipt = v32.causal_backoff(
                last, last_records, incumbent, incumbent_records, native_records
            )
            receipts.append(receipt)
            if receipt.get("phase") is not None:
                frozen.add(receipt["phase"])
        else:
            ranked = ranked_promotions(last_records, last["schedule"], prior, frozen)
            proposed = next(
                (item[3] for item in ranked if v32.schedule_sha256(item[3]) not in tried),
                None,
            )
            if proposed is not None:
                phase = next(item[2] for item in ranked if item[3] == proposed)
                receipts.append(
                    {
                        "operation": "precision_weighted_workload_promotion",
                        "phase": phase,
                        "source_schedule": last["schedule"],
                        "proposed_schedule": proposed,
                    }
                )
        if proposed is None or v32.schedule_sha256(proposed) in tried:
            # If causal backoff reaches the grid floor for the attributed
            # phase, keep repairing the failed schedule rather than jumping
            # to native.  Native has no accelerated dimensions for
            # ``sail_guard`` to lower, which previously left an exact-25
            # search unable to fill its fifth unique discovery slot.
            base = last["schedule"] if not v32.safe(last) else incumbent["schedule"]
            candidates = v33.sail_guard(base, prior, tried)
            proposed, phase = candidates
            receipts.append(
                {
                    "operation": "precision_ranked_one_rung_guard",
                    "phase": phase,
                    "source_schedule": base,
                    "proposed_schedule": proposed,
                }
            )
        if proposed is None or v32.schedule_sha256(proposed) in tried:
            raise RuntimeError("v34 could not fill five unique discovery schedules")
        last, last_records = evaluate(proposed, "causal_risk_frontier")

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
        raise RuntimeError(f"v34 search used {ledger.used()}, expected exactly 25")
    eligible = [
        item
        for item in finalists
        if item["summary"]["successes"] >= 7
        and item["summary"]["physics_errors"] == 0
        and item["summary"]["safety_violations"] == 0
    ]
    selected = max(
        eligible,
        key=lambda item: (
            item["summary"]["successes"] == 8,
            item["summary"]["achieved_throughput_per_step"],
            item["summary"]["successes"],
        ),
        default=None,
    )
    return {
        "schema": "act-prior-causal-risk25-selection-v34",
        "task_label": task,
        "prior": prior,
        "risk_gate": gate_spec,
        "discovery_reports": reports,
        "update_receipts": receipts,
        "frozen_causal_phases": sorted(frozen, key=PHASES.index),
        "finalists": finalists,
        "selection_rule": "prefer_8_of_8_then_failure_aware_throughput;_7_of_8_provisional_floor",
        "selection_status": "accelerated_selected" if selected else "no_acceleration_selected",
        "selected_schedule": None if selected is None else selected["schedule"],
        "selected_schedule_sha256": None if selected is None else selected["schedule_sha256"],
        "search_scientific_rollouts": ledger.used(),
        "incident_totals": ledger.incident_totals(),
        "historical_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    }


def load_states(directory: Path, seeds: list[int], identity: str) -> list[dict]:
    records = []
    missing = False
    for seed in seeds:
        path = directory / f"{seed}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError("v34 states contain a non-contiguous suffix")
        value = v33.checked_json(path)
        if value.get("seed") != seed or value.get("identity_sha256") != identity:
            raise RuntimeError(f"v34 state identity mismatch: {path}")
        records.append(value)
    return records


def selected_schedule(root: Path, task: str) -> tuple[list[float], Path]:
    path = root / "search" / task / "SELECTION.json"
    selection = v33.checked_json(path)
    if selection.get("selected_schedule") is None:
        return [1.0] * 4, path
    schedule = list(v30.validate_schedule(selection["selected_schedule"]))
    if v32.schedule_sha256(schedule) != selection["selected_schedule_sha256"]:
        raise RuntimeError("v34 selection schedule hash mismatch")
    return schedule, path


def all_search_complete(root: Path) -> None:
    for task in TASKS:
        complete = v33.checked_json(root / "search" / task / "SEARCH_COMPLETE.json")
        if complete.get("search_scientific_rollouts") != SEARCH_BUDGET:
            raise RuntimeError(f"v34 incomplete search for {task}")


def run_final(runtime, root, task, method, seeds, banks_sha, prior):
    all_search_complete(root)
    selected, selection_path = selected_schedule(root, task)
    gate_spec = risk_gate_spec(prior)
    if method == "native_1x":
        schedule, gated = [1.0] * 4, False
    elif method == "phase_only":
        schedule, gated = selected, False
    else:
        schedule, gated = selected, True
    controller = {
        "schedule": schedule,
        "schedule_sha256": v32.schedule_sha256(schedule),
        "risk_gate_enabled": gated,
        "risk_gate_sha256": gate_spec["controller_sha256"] if gated else None,
    }
    output = root / "final" / task / method
    identity = {
        **runtime.identity(),
        "schema": "act-prior-causal-risk-final-identity-v34",
        "method": method,
        "controller": controller,
        "selection_sha256": file_sha256(selection_path),
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "banks_sha256": banks_sha,
        "search_or_tuning_permitted": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v33.immutable_or_verify(output / "IDENTITY.json", identity)
    records = load_states(output / "states", seeds, identity["identity_sha256"])
    runner = RiskGatedRuntime(runtime, gate_spec) if gated else runtime
    for seed in seeds[len(records):]:
        record = runner.rollout(schedule, seed, record_attribution_telemetry=False)
        if record.get("seed") != seed or list(map(float, record.get("schedule", ()))) != schedule:
            raise RuntimeError("v34 final runtime identity mismatch")
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
            },
        )
        print(json.dumps({"stage": "final", "task": task, "method": method, "completed": len(records)}), flush=True)
    result = {
        "schema": "act-prior-causal-risk-final-result-v34",
        "task_label": task,
        "method": method,
        "controller": controller,
        "episodes": len(records),
        "summary": summarize(records),
        "risk_gate_events": sum(int(item.get("risk_gate_event_count", 0)) for item in records),
        "identity_sha256": identity["identity_sha256"],
    }
    immutable_json(output / "RESULT.json", result)
    immutable_json(
        output / "COMPLETE.json",
        {
            "schema": "act-prior-causal-risk-final-completion-v34",
            "episodes": len(records),
            "result_sha256": file_sha256(output / "RESULT.json"),
            "physics_errors": result["summary"]["physics_errors"],
            "safety_violations": result["summary"]["safety_violations"],
            "historical_rollouts_reexecuted": 0,
        },
    )


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "final"), required=True)
    parser.add_argument("--method", choices=("prior_causal_risk", *FINAL_METHODS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-source-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=TASKS, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--offline-priors", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if git_head() != args.implementation_commit:
        raise RuntimeError("v34 checked-out source differs from implementation commit")
    banks = v33.checked_json(args.banks)
    spec = banks["tasks"][args.task_label]
    runtime = ACTFrontierRuntime(
        source_commit=args.base_source_commit,
        checkout_commit=args.implementation_commit,
        run_manifest=args.run_manifest,
        task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
    )
    prior = combined_prior(args.offline_priors, args.task_label)
    gate_spec = risk_gate_spec(prior)
    root = args.root.resolve()
    if args.stage == "final":
        if args.method not in FINAL_METHODS:
            raise ValueError("v34 final stage requires a final method")
        run_final(
            runtime,
            root,
            args.task_label,
            args.method,
            list(map(int, spec["final"])),
            file_sha256(args.banks),
            prior,
        )
        return 0
    if args.method != "prior_causal_risk":
        raise ValueError("v34 search stage requires prior_causal_risk")
    output = root / "search" / args.task_label
    identity = {
        **runtime.identity(),
        "schema": "act-prior-causal-risk-search-identity-v34",
        "contract_sha256": file_sha256(args.contract),
        "banks_sha256": file_sha256(args.banks),
        "task_label": args.task_label,
        "search_budget": SEARCH_BUDGET,
        "discovery_seeds": spec["discovery"],
        "confirmation_seeds": spec["confirmation"],
        "final_seeds_registered_unopened": spec["final"],
        "prior_payload_sha256": prior["prior_payload_sha256"],
        "risk_gate_sha256": gate_spec["controller_sha256"],
        "historical_speed_outcomes_used_by_runtime": False,
        "study_design_informed_by_v33_results": True,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v33.immutable_or_verify(output / "IDENTITY.json", identity)
    selection_path = output / "SELECTION.json"
    complete_path = output / "SEARCH_COMPLETE.json"
    if complete_path.exists():
        complete = v33.checked_json(complete_path)
        if complete["selection_sha256"] != file_sha256(selection_path):
            raise RuntimeError("v34 completed selection hash mismatch")
        print(json.dumps(v33.checked_json(selection_path), sort_keys=True))
        return 0
    ledger = v32.Ledger(
        RiskGatedRuntime(runtime, gate_spec),
        output / "search",
        list(map(int, spec["discovery"])),
        list(map(int, spec["confirmation"])),
    )
    selection = run_search(ledger, args.task_label, prior, gate_spec)
    immutable_json(selection_path, selection)
    completion = {
        "schema": "act-prior-causal-risk-search-completion-v34",
        "task_label": args.task_label,
        "identity_sha256": file_sha256(output / "IDENTITY.json"),
        "selection_sha256": file_sha256(selection_path),
        "search_scientific_rollouts": SEARCH_BUDGET,
        **selection["incident_totals"],
        "historical_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    }
    immutable_json(complete_path, completion)
    print(json.dumps({"selection": selection, "completion": completion}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
