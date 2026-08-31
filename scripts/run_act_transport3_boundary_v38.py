#!/usr/bin/env python3
"""Run the exact-25 transport-first terminal-boundary speed study."""

from __future__ import annotations

import argparse
import json
import os
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
from scripts import run_act_champion_challenger_v37 as v37  # noqa: E402
from scripts import run_act_common_grid_strider_causal25_v32 as v32  # noqa: E402
from scripts import run_act_sail_warmstart_v33 as v33  # noqa: E402
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head  # noqa: E402
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256, summarize  # noqa: E402
from sim_tasks import normalize_task_name  # noqa: E402

TASKS = ("pick", "tea", "insertion")
SEARCH_METHOD = "transport3_boundary"
FINAL_METHODS = ("native_1x", "champion", "selected")
DISCOVERY_EPISODES = 5
PAIRED_EPISODES = 10
SEARCH_BUDGET = DISCOVERY_EPISODES + 2 * PAIRED_EPISODES
DISCOVERY_SUCCESS_FLOOR = 4
PAIRED_SUCCESS_FLOOR = 9
CLEAR_FAILURE_CEILING = 7
SUCCESS_GAIN_THROUGHPUT_FLOOR = 0.95
TIED_SUCCESS_THROUGHPUT_RATIO = 1.05
TRANSPORT_INDEX = PHASES.index("transport")


def controller_sha256(controller: dict) -> str:
    return canonical_sha256(controller)


def static_controller(schedule) -> dict:
    result = {
        "type": "static_phase_schedule",
        "phase_order": list(PHASES),
        "schedule": list(v32.validate_schedule(schedule)),
    }
    result["controller_sha256"] = canonical_sha256(result)
    return result


def transport_controller(task: str, spec: dict) -> dict:
    schedule = list(v32.validate_schedule(spec["champion_schedule"]))
    schedule[TRANSPORT_INDEX] = 3.0
    gate = {
        "phase": "transport",
        "entry_metric": str(spec["approach_metric"]),
        "entry_threshold_lte": float(spec["approach_threshold"]),
        "protected_speed": float(spec["protected_transport_speed"]),
        "latch_until_phase_exit": True,
        "current_observation_only": True,
        "reward_visible": False,
        "object_state_visible": False,
        "future_or_terminal_signal_visible": False,
    }
    if "approach_target" in spec:
        gate["fixed_target_xy"] = list(map(float, spec["approach_target"]))
    result = {
        "type": "transport3_terminal_approach_gate",
        "task_label": task,
        "phase_order": list(PHASES),
        "schedule": schedule,
        "gate": gate,
    }
    result["controller_sha256"] = canonical_sha256(result)
    return result


def validate_controller(controller: dict) -> dict:
    value = dict(controller)
    expected = value.pop("controller_sha256", None)
    schedule = list(v32.validate_schedule(value["schedule"]))
    value["schedule"] = schedule
    if value["phase_order"] != list(PHASES):
        raise RuntimeError("v38 controller phase order differs")
    if value["type"] == "transport3_terminal_approach_gate":
        if schedule[TRANSPORT_INDEX] != 3.0:
            raise RuntimeError("v38 challenger must use 3x transport")
        gate = value["gate"]
        if gate["phase"] != "transport" or not gate["current_observation_only"]:
            raise RuntimeError("v38 gate is not current-observation transport-only")
        if gate["reward_visible"] or gate["object_state_visible"]:
            raise RuntimeError("v38 gate exposes forbidden outcome or object state")
        if gate["future_or_terminal_signal_visible"]:
            raise RuntimeError("v38 gate exposes future or terminal information")
    elif value["type"] != "static_phase_schedule":
        raise RuntimeError("v38 controller type differs")
    digest = canonical_sha256(value)
    value["controller_sha256"] = digest
    if expected is not None and expected != digest:
        raise RuntimeError("v38 controller hash mismatch")
    return value


def load_controllers(path: Path, task: str) -> tuple[dict, dict, str]:
    bundle = v33.checked_json(path)
    if bundle.get("schema") != "act-transport3-boundary-controllers-v38":
        raise RuntimeError("v38 controller bundle schema differs")
    if bundle.get("historical_rollouts_reexecuted") != 0:
        raise RuntimeError("v38 controller bundle re-executes history")
    if bundle.get("phase_order") != list(PHASES):
        raise RuntimeError("v38 controller bundle phase order differs")
    spec = dict(bundle["tasks"][task])
    champion = static_controller(spec["champion_schedule"])
    challenger = transport_controller(task, spec)
    if champion["controller_sha256"] == challenger["controller_sha256"]:
        raise RuntimeError("v38 challenger must differ from champion")
    return champion, challenger, file_sha256(path)


@dataclass
class TerminalApproachGate:
    spec: dict
    latched: bool = False
    events: list[dict] = field(default_factory=list)

    def measure(self, observation: dict) -> float:
        left = np.asarray(observation["effector_position_left"], dtype=np.float64)
        right = np.asarray(observation["effector_position_right"], dtype=np.float64)
        metric = self.spec["entry_metric"]
        if metric == "effector_pair_distance_3d":
            return float(np.linalg.norm(left - right))
        if metric == "right_effector_to_fixed_target_xy":
            target = np.asarray(self.spec["fixed_target_xy"], dtype=np.float64)
            return float(np.linalg.norm(right[:2] - target))
        raise RuntimeError(f"unknown v38 approach metric: {metric}")

    def choose(
        self, base_speed: float, phase: int, observation: dict, physics_step: int
    ) -> float:
        if phase != TRANSPORT_INDEX:
            self.latched = False
            return float(base_speed)
        measure = self.measure(observation)
        if not self.latched and measure <= float(self.spec["entry_threshold_lte"]):
            self.latched = True
            self.events.append(
                {
                    "physics_step": int(physics_step),
                    "phase": "transport",
                    "event": "terminal_approach_entry",
                    "metric": self.spec["entry_metric"],
                    "observed_value": measure,
                    "threshold_lte": float(self.spec["entry_threshold_lte"]),
                    "base_speed": float(base_speed),
                    "effective_speed": float(self.spec["protected_speed"]),
                }
            )
        return float(self.spec["protected_speed"] if self.latched else base_speed)


def run_transport_controller(
    runtime: ACTFrontierRuntime,
    controller: dict,
    seed: int,
    *,
    record_attribution_telemetry: bool = False,
) -> dict:
    controller = validate_controller(controller)
    task = normalize_task_name(runtime.task)
    schedule = controller["schedule"]
    env = create_speed_env(
        task_name=task,
        seed=int(seed),
        randomize_object_pose=True,
        chunk_predictor=runtime.adapter,
        speed_values=(1.0, 1.5, 2.0, 2.5, 3.0),
        observation_encoder=runtime.encoder(),
        decision_mode="phase_entry",
        decision_frame_skip=1,
        terminate_on_success=False,
    )
    env.env = JointEffectorObservationWrapper(env.env)
    env._environment_metadata["learned_phase_effector_source"] = "joint_fk_body_xpos"
    gate = TerminalApproachGate(controller["gate"])
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
            effective_speed = gate.choose(
                base_speed, phase, env.cur_ts.observation, env.physics_steps
            )
            decision = (phase, base_speed, effective_speed, gate.latched)
            if decision != last_decision:
                decisions.append(
                    {
                        "phase": PHASES[phase],
                        "physics_step": int(env.physics_steps),
                        "base_speed": base_speed,
                        "speed": effective_speed,
                        "terminal_approach_latched": bool(gate.latched),
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
            "controller_sha256": controller["controller_sha256"],
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
            "terminal_approach_events": gate.events,
            "terminal_approach_event_count": len(gate.events),
        }
        if record_attribution_telemetry:
            result["attribution_telemetry"] = telemetry
        if info.get("physics_error") is not None:
            result["physics_error"] = str(info["physics_error"])
        return result
    finally:
        env.close()


class ControllerRuntime:
    def __init__(self, runtime: ACTFrontierRuntime):
        self.runtime = runtime

    def rollout(self, controller: dict, seed: int, *, record_attribution_telemetry=False):
        controller = validate_controller(controller)
        if controller["type"] == "static_phase_schedule":
            record = self.runtime.rollout(
                controller["schedule"],
                seed,
                record_attribution_telemetry=record_attribution_telemetry,
            )
            record["controller_sha256"] = controller["controller_sha256"]
            return record
        return run_transport_controller(
            self.runtime,
            controller,
            seed,
            record_attribution_telemetry=record_attribution_telemetry,
        )


class SearchLedger:
    def __init__(self, runtime, root: Path):
        self.runtime = runtime
        self.root = root

    def used(self) -> int:
        return len(list((self.root / "states").glob("*/*.json")))

    def one(self, controller: dict, seed: int, role: str) -> dict:
        controller = validate_controller(controller)
        digest = controller["controller_sha256"]
        path = self.root / "states" / digest / f"{seed}.json"
        if path.exists():
            record = v33.checked_json(path)
            if int(record.get("seed", -1)) != int(seed):
                raise RuntimeError("v38 cached search seed differs")
            if record.get("controller_sha256") != digest:
                raise RuntimeError("v38 cached controller differs")
            return record
        if self.used() >= SEARCH_BUDGET:
            raise RuntimeError("v38 exact-25 search budget exhausted")
        record = self.runtime.rollout(
            controller, int(seed), record_attribution_telemetry=True
        )
        if int(record.get("seed", -1)) != int(seed):
            raise RuntimeError("v38 runtime returned a different seed")
        if record.get("controller_sha256") != digest:
            raise RuntimeError("v38 runtime returned a different controller")
        record["search_role"] = role
        immutable_json(path, record)
        print(
            json.dumps(
                {
                    "stage": "search",
                    "role": role,
                    "controller_sha256": digest,
                    "seed": int(seed),
                    "success": v32.successful(record),
                    "search_rollouts_used": self.used(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return record

    def report(self, controller: dict, seeds: list[int], role: str):
        controller = validate_controller(controller)
        records = [self.one(controller, seed, role) for seed in seeds]
        report = {
            "role": role,
            "controller": controller,
            "controller_sha256": controller["controller_sha256"],
            "seed_order": list(map(int, seeds)),
            "summary": summarize(records),
            "terminal_approach_events": sum(
                int(item.get("terminal_approach_event_count", 0)) for item in records
            ),
        }
        immutable_json(self.root / "reports" / f"{role}.json", report)
        return report, records

    def incidents(self) -> dict:
        records = [
            v33.checked_json(path) for path in (self.root / "states").glob("*/*.json")
        ]
        return {
            "physics_errors": sum(item.get("physics_error") is not None for item in records),
            "safety_violations": sum(
                item.get("safety_violation") is not None for item in records
            ),
        }


def run_search(ledger: SearchLedger, task: str, spec: dict, champion: dict, challenger: dict) -> dict:
    discovery, _ = ledger.report(
        challenger, list(map(int, spec["challenger_discovery"])), "challenger_discovery"
    )
    champion_paired, champion_records = ledger.report(
        champion, list(map(int, spec["paired"])), "champion_paired"
    )
    challenger_paired, challenger_records = ledger.report(
        challenger, list(map(int, spec["paired"])), "challenger_paired"
    )
    if ledger.used() != SEARCH_BUDGET:
        raise RuntimeError(f"v38 search used {ledger.used()}, expected exactly 25")
    pair = v37.paired_receipt(challenger_records, champion_records)
    challenger_incidents = {
        "physics_errors": int(discovery["summary"]["physics_errors"])
        + int(challenger_paired["summary"]["physics_errors"]),
        "safety_violations": int(discovery["summary"]["safety_violations"])
        + int(challenger_paired["summary"]["safety_violations"]),
    }
    champion_incidents = {
        "physics_errors": int(champion_paired["summary"]["physics_errors"]),
        "safety_violations": int(champion_paired["summary"]["safety_violations"]),
    }
    champion_successes = int(champion_paired["summary"]["successes"])
    challenger_successes = int(challenger_paired["summary"]["successes"])
    ratio = pair["challenger_throughput_ratio"]
    champion_clear = (
        champion_successes > CLEAR_FAILURE_CEILING
        and champion_incidents == {"physics_errors": 0, "safety_violations": 0}
    )
    challenger_qualified = (
        int(discovery["summary"]["successes"]) >= DISCOVERY_SUCCESS_FLOOR
        and challenger_successes >= PAIRED_SUCCESS_FLOOR
        and challenger_incidents == {"physics_errors": 0, "safety_violations": 0}
    )
    challenger_preferred = challenger_qualified and (
        champion_successes < PAIRED_SUCCESS_FLOOR
        or (
            challenger_successes > champion_successes
            and ratio is not None
            and ratio >= SUCCESS_GAIN_THROUGHPUT_FLOOR
        )
        or (
            challenger_successes == champion_successes
            and ratio is not None
            and ratio >= TIED_SUCCESS_THROUGHPUT_RATIO
        )
    )
    if challenger_preferred:
        selected, status = challenger, "transport3_boundary_promoted"
    elif champion_clear:
        selected, status = champion, "accelerated_champion_retained"
    else:
        selected, status = static_controller([1.0] * len(PHASES)), "native_fallback_clear_failure"
    return {
        "schema": "act-transport3-boundary25-selection-v38",
        "task_label": task,
        "challenger_discovery": discovery,
        "champion_paired": champion_paired,
        "challenger_paired": challenger_paired,
        "paired_receipt": pair,
        "selection_rule": {
            "challenger_discovery_success_floor": DISCOVERY_SUCCESS_FLOOR,
            "paired_success_floor": PAIRED_SUCCESS_FLOOR,
            "clear_failure_ceiling": CLEAR_FAILURE_CEILING,
            "success_gain_throughput_floor": SUCCESS_GAIN_THROUGHPUT_FLOOR,
            "tied_success_throughput_ratio": TIED_SUCCESS_THROUGHPUT_RATIO,
            "ambiguous_result": "retain_accelerated_champion",
        },
        "champion_clear_of_failure": champion_clear,
        "challenger_qualified": challenger_qualified,
        "challenger_preferred": challenger_preferred,
        "selection_status": status,
        "selected_controller": selected,
        "selected_controller_sha256": selected["controller_sha256"],
        "search_scientific_rollouts": ledger.used(),
        "incident_totals": ledger.incidents(),
        "historical_speed_outcomes_used_for_initialization": True,
        "historical_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    }


def require_all_search(root: Path) -> None:
    for task in TASKS:
        complete = v33.checked_json(root / "search" / task / "SEARCH_COMPLETE.json")
        if int(complete.get("search_scientific_rollouts", -1)) != SEARCH_BUDGET:
            raise RuntimeError(f"v38 search incomplete: {task}")


def method_controller(root: Path, controllers_path: Path, task: str, method: str):
    champion, _, provenance = load_controllers(controllers_path, task)
    if method == "native_1x":
        return static_controller([1.0] * len(PHASES)), "preregistered_native"
    if method == "champion":
        return champion, provenance
    path = root / "search" / task / "SELECTION.json"
    selection = v33.checked_json(path)
    controller = validate_controller(selection["selected_controller"])
    if controller["controller_sha256"] != selection["selected_controller_sha256"]:
        raise RuntimeError("v38 selected controller hash mismatch")
    return controller, file_sha256(path)


def load_final_states(directory: Path, seeds: list[int], identity_sha: str):
    records = []
    missing = False
    for seed in seeds:
        path = directory / "states" / f"{seed}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError("v38 final states contain a non-contiguous suffix")
        record = v33.checked_json(path)
        if int(record.get("seed", -1)) != seed or record.get("identity_sha256") != identity_sha:
            raise RuntimeError(f"v38 final state identity mismatch: {path}")
        records.append(record)
    return records


def run_final(runtime, root, controllers_path, task, method, seeds, banks_sha):
    require_all_search(root)
    controller, provenance = method_controller(root, controllers_path, task, method)
    digest = controller["controller_sha256"]
    controller_root = root / "final" / task / "controllers" / digest
    alias_path = root / "final" / task / "methods" / method / "RESULT.json"
    if alias_path.exists():
        alias = v33.checked_json(alias_path)
        complete = v33.checked_json(controller_root / "COMPLETE.json")
        if alias.get("controller_sha256") != digest:
            raise RuntimeError(f"v38 method alias differs: {task}/{method}")
        if alias.get("controller_result_sha256") != complete.get("result_sha256"):
            raise RuntimeError(f"v38 method completion differs: {task}/{method}")
        return
    identity = {
        **runtime.identity(),
        "schema": "act-transport3-boundary-final-controller-identity-v38",
        "task_label": task,
        "controller": controller,
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "banks_sha256": banks_sha,
        "search_or_tuning_permitted": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v33.immutable_or_verify(controller_root / "IDENTITY.json", identity)
    was_complete = (controller_root / "COMPLETE.json").exists()
    records = load_final_states(controller_root, seeds, identity["identity_sha256"])
    runner = ControllerRuntime(runtime)
    for seed in seeds[len(records) :]:
        record = runner.rollout(controller, seed, record_attribution_telemetry=False)
        if int(record.get("seed", -1)) != seed or record.get("controller_sha256") != digest:
            raise RuntimeError("v38 final runtime returned a different controller")
        record["identity_sha256"] = identity["identity_sha256"]
        immutable_json(controller_root / "states" / f"{seed}.json", record)
        records.append(record)
        atomic_json(
            controller_root / "progress.json",
            {
                "task": task,
                "controller_sha256": digest,
                "completed": len(records),
                "successes": sum(v32.successful(item) for item in records),
            },
        )
        print(
            json.dumps(
                {
                    "stage": "final",
                    "task": task,
                    "method": method,
                    "completed": len(records),
                    "successes": sum(v32.successful(item) for item in records),
                }
            ),
            flush=True,
        )
    result = {
        "schema": "act-transport3-boundary-final-controller-result-v38",
        "task_label": task,
        "controller": controller,
        "episodes": len(records),
        "summary": summarize(records),
        "terminal_approach_events": sum(
            int(item.get("terminal_approach_event_count", 0)) for item in records
        ),
        "identity_sha256": identity["identity_sha256"],
    }
    v33.immutable_or_verify(controller_root / "RESULT.json", result)
    v33.immutable_or_verify(
        controller_root / "COMPLETE.json",
        {
            "schema": "act-transport3-boundary-final-controller-completion-v38",
            "episodes": len(records),
            "result_sha256": file_sha256(controller_root / "RESULT.json"),
            "physics_errors": result["summary"]["physics_errors"],
            "safety_violations": result["summary"]["safety_violations"],
        },
    )
    v33.immutable_or_verify(
        alias_path,
        {
            "schema": "act-transport3-boundary-final-method-result-v38",
            "task_label": task,
            "method": method,
            "controller": controller,
            "controller_sha256": digest,
            "controller_result_sha256": file_sha256(controller_root / "RESULT.json"),
            "controller_receipt": str(controller_root / "RESULT.json"),
            "selection_provenance": provenance,
            "controller_cache_hit": was_complete,
            "summary": result["summary"],
            "terminal_approach_events": result["terminal_approach_events"],
        },
    )


def validate_banks(banks: dict) -> None:
    all_seeds = []
    for task in TASKS:
        spec = banks["tasks"][task]
        if len(spec["challenger_discovery"]) != DISCOVERY_EPISODES:
            raise RuntimeError("v38 banks require five challenger discovery seeds")
        if len(spec["paired"]) != PAIRED_EPISODES or len(spec["final"]) != 50:
            raise RuntimeError("v38 banks require ten paired and fifty final seeds")
        task_seeds = spec["challenger_discovery"] + spec["paired"] + spec["final"]
        if len(task_seeds) != len(set(task_seeds)):
            raise RuntimeError(f"v38 task banks overlap: {task}")
        all_seeds.extend(task_seeds)
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("v38 cross-task banks overlap")


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
    parser.add_argument("--controllers", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if git_head() != args.implementation_commit:
        raise RuntimeError("v38 checked-out source differs from implementation commit")
    banks = v33.checked_json(args.banks)
    validate_banks(banks)
    champion, challenger, controllers_sha = load_controllers(
        args.controllers, args.task_label
    )
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
    root = args.root.resolve()
    if args.stage == "final":
        if args.method not in FINAL_METHODS:
            raise ValueError("v38 final stage requires a final method")
        run_final(
            runtime,
            root,
            args.controllers,
            args.task_label,
            args.method,
            list(map(int, spec["final"])),
            file_sha256(args.banks),
        )
        return 0
    if args.method != SEARCH_METHOD:
        raise ValueError("v38 search stage requires transport3_boundary")
    output = root / "search" / args.task_label
    identity = {
        **runtime.identity(),
        "schema": "act-transport3-boundary-search-identity-v38",
        "contract_sha256": file_sha256(args.contract),
        "banks_sha256": file_sha256(args.banks),
        "controllers_sha256": controllers_sha,
        "task_label": args.task_label,
        "search_budget": SEARCH_BUDGET,
        "challenger_discovery_seeds": spec["challenger_discovery"],
        "paired_seeds": spec["paired"],
        "final_seeds_registered_unopened": spec["final"],
        "champion_controller_sha256": champion["controller_sha256"],
        "challenger_controller_sha256": challenger["controller_sha256"],
        "historical_speed_outcomes_used_for_initialization": True,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v33.immutable_or_verify(output / "IDENTITY.json", identity)
    selection_path = output / "SELECTION.json"
    complete_path = output / "SEARCH_COMPLETE.json"
    if complete_path.exists():
        complete = v33.checked_json(complete_path)
        if complete["selection_sha256"] != file_sha256(selection_path):
            raise RuntimeError("v38 completed selection hash mismatch")
        return 0
    ledger = SearchLedger(ControllerRuntime(runtime), output / "search")
    selection = run_search(ledger, args.task_label, spec, champion, challenger)
    immutable_json(selection_path, selection)
    immutable_json(
        complete_path,
        {
            "schema": "act-transport3-boundary-search-completion-v38",
            "task_label": args.task_label,
            "identity_sha256": file_sha256(output / "IDENTITY.json"),
            "selection_sha256": file_sha256(selection_path),
            "search_scientific_rollouts": SEARCH_BUDGET,
            **selection["incident_totals"],
            "historical_rollouts_reexecuted": 0,
            "final_bank_opened": False,
        },
    )
    print(json.dumps({"selection": selection}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
