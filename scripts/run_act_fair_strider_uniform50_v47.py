#!/usr/bin/env python3
"""Run the fair fresh 50-rollout STRIDER-versus-uniform study."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SPEEDTUNING_SPEED_VALUES", "1,1.5,2,2.5,3")

from act_speed_benchmark import canonical_sha256  # noqa: E402
from learned_phase_observation import PHASES  # noqa: E402
from scripts import run_act_fresh_transport25_v39 as v39  # noqa: E402
from scripts import run_act_strider_frontier_v2 as v2  # noqa: E402
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head  # noqa: E402
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256, summarize  # noqa: E402

TASKS = ("pick", "tea", "insertion")
SEARCH_ARMS = ("uniform", "strider")
FINAL_METHODS = ("native_1x", "uniform_selected", "strider_selected")
GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
ANCHOR = (2.0, 2.0, 2.0, 2.0)
ROUNDS = 2
DIAGNOSTIC_EPISODES = 5
PAIRED_EPISODES = 10
ROUND_BUDGET = DIAGNOSTIC_EPISODES + 2 * PAIRED_EPISODES
SEARCH_BUDGET = ROUNDS * ROUND_BUDGET
FINAL_EPISODES = 100
PAIRED_SUCCESS_FLOOR = 9
SUCCESS_GAIN_THROUGHPUT_FLOOR = 0.95
TIED_SUCCESS_THROUGHPUT_RATIO = 1.05


def static_controller(schedule) -> dict:
    return v39.static_controller(schedule)


def validate_controller(controller: dict) -> dict:
    value = v39.validate_controller(controller)
    if value["type"] != "static_phase_schedule":
        raise RuntimeError("v47 permits learned-phase static schedules only")
    return value


def successful(record: dict) -> bool:
    return v39.v38.v32.successful(record)


def adjacent(value: float, direction: int) -> float | None:
    index = GRID.index(float(value)) + int(direction)
    return None if index < 0 or index >= len(GRID) else GRID[index]


class SearchLedger:
    def __init__(self, runtime, root: Path):
        self.runtime = runtime
        self.root = root

    def used(self) -> int:
        return len(list((self.root / "states").glob("*/*.json")))

    def one(self, controller: dict, seed: int, role: str) -> dict:
        controller = validate_controller(controller)
        digest = controller["controller_sha256"]
        path = self.root / "states" / digest / f"{int(seed)}.json"
        if path.exists():
            record = v39.v38.v33.checked_json(path)
            if int(record.get("seed", -1)) != int(seed):
                raise RuntimeError("v47 cached search seed differs")
            if record.get("controller_sha256") != digest:
                raise RuntimeError("v47 cached search controller differs")
            return record
        if self.used() >= SEARCH_BUDGET:
            raise RuntimeError("v47 exact-50 search budget exhausted")
        record = self.runtime.rollout(
            controller, int(seed), record_attribution_telemetry=True
        )
        if int(record.get("seed", -1)) != int(seed):
            raise RuntimeError("v47 runtime returned a different seed")
        if record.get("controller_sha256") != digest:
            raise RuntimeError("v47 runtime returned a different controller")
        record["search_role"] = role
        immutable_json(path, record)
        print(
            json.dumps(
                {
                    "stage": "search",
                    "role": role,
                    "controller_sha256": digest,
                    "seed": int(seed),
                    "success": successful(record),
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
        }
        immutable_json(self.root / "reports" / f"{role}.json", report)
        return report, records

    def incidents(self) -> dict:
        records = [
            v39.v38.v33.checked_json(path)
            for path in (self.root / "states").glob("*/*.json")
        ]
        return {
            "physics_errors": sum(
                item.get("physics_error") is not None for item in records
            ),
            "safety_violations": sum(
                item.get("safety_violation") is not None for item in records
            ),
        }


def qualified(report: dict) -> bool:
    summary = report["summary"]
    return (
        int(summary["successes"]) >= PAIRED_SUCCESS_FLOOR
        and int(summary["physics_errors"]) == 0
        and int(summary["safety_violations"]) == 0
    )


def paired_receipt(challenger_records: list[dict], incumbent_records: list[dict]):
    return v39.v38.v37.paired_receipt(challenger_records, incumbent_records)


def choose_round_winner(incumbent_report: dict, challenger_report: dict, pair: dict):
    incumbent_ok = qualified(incumbent_report)
    challenger_ok = qualified(challenger_report)
    incumbent_successes = int(incumbent_report["summary"]["successes"])
    challenger_successes = int(challenger_report["summary"]["successes"])
    ratio = pair["challenger_throughput_ratio"]
    challenger_preferred = challenger_ok and (
        not incumbent_ok
        or (
            challenger_successes > incumbent_successes
            and ratio is not None
            and ratio >= SUCCESS_GAIN_THROUGHPUT_FLOOR
        )
        or (
            challenger_successes == incumbent_successes
            and ratio is not None
            and ratio >= TIED_SUCCESS_THROUGHPUT_RATIO
        )
    )
    if challenger_preferred:
        return challenger_report["controller"], True, "challenger_promoted"
    if incumbent_ok:
        return incumbent_report["controller"], True, "incumbent_retained"
    return incumbent_report["controller"], False, "unqualified_incumbent_retained"


def uniform_proposal(round_index: int, incumbent: dict, previous: dict | None):
    speed = float(incumbent["schedule"][0])
    if len(set(map(float, incumbent["schedule"]))) != 1:
        raise RuntimeError("v47 uniform arm incumbent is nonuniform")
    if round_index == 0:
        proposed_speed = adjacent(speed, 1)
        operation = "registered_uniform_upward_challenger"
    elif (
        previous is not None
        and previous["winner_qualified"]
        and previous["selection_status"] == "challenger_promoted"
    ):
        proposed_speed = adjacent(speed, 1)
        operation = "second_adjacent_uniform_promotion"
    else:
        proposed_speed = adjacent(speed, -1)
        operation = "registered_uniform_reliability_backoff"
    if proposed_speed is None:
        proposed_speed = adjacent(speed, -1 if speed == GRID[-1] else 1)
    if proposed_speed is None:
        raise RuntimeError("v47 uniform arm exhausted the registered grid")
    return static_controller([proposed_speed] * len(PHASES)), {
        "operation": operation,
        "source_schedule": incumbent["schedule"],
        "proposed_schedule": [proposed_speed] * len(PHASES),
    }


def mean_phase_workload(records: list[dict]) -> dict[str, float]:
    usable = [record for record in records if successful(record)]
    if not usable:
        return {phase: 0.0 for phase in PHASES}
    return {
        phase: statistics.fmean(v2.phase_workloads(record)[phase] for record in usable)
        for phase in PHASES
    }


def failed_phase(records: list[dict]) -> str:
    return v2.earliest_failed_phase(records, PHASES[0])[0]


def strider_proposal(
    incumbent: dict,
    diagnostic_records: list[dict],
    tried: set[str],
    previous: dict | None,
):
    schedule = list(map(float, incumbent["schedule"]))
    if previous is not None and not previous["winner_qualified"]:
        phase = failed_phase(previous["incumbent_records"] + previous["challenger_records"])
        index = PHASES.index(phase)
        lower = adjacent(schedule[index], -1)
        if lower is not None:
            proposal = list(schedule)
            proposal[index] = lower
            controller = static_controller(proposal)
            if controller["controller_sha256"] not in tried:
                return controller, {
                    "operation": "one_rung_failed_phase_backoff",
                    "phase": phase,
                    "source_schedule": schedule,
                    "proposed_schedule": proposal,
                }

    workloads = mean_phase_workload(diagnostic_records)
    candidates = []
    for index, (phase, speed) in enumerate(zip(PHASES, schedule)):
        higher = adjacent(speed, 1)
        if higher is None:
            continue
        proposal = list(schedule)
        proposal[index] = higher
        controller = static_controller(proposal)
        if controller["controller_sha256"] in tried:
            continue
        saving = workloads[phase] * (1.0 / speed - 1.0 / higher)
        candidates.append((saving, -index, phase, controller, proposal))
    if not candidates:
        raise RuntimeError("v47 STRIDER arm exhausted one-rung proposals")
    saving, _, phase, controller, proposal = max(candidates)
    return controller, {
        "operation": "one_rung_current_run_bang_for_buck_promotion",
        "phase": phase,
        "mean_native_equivalent_phase_workload": workloads[phase],
        "predicted_saved_steps": saving,
        "source_schedule": schedule,
        "proposed_schedule": proposal,
    }


def run_search(ledger: SearchLedger, arm: str, spec: dict) -> dict:
    incumbent = static_controller(ANCHOR)
    previous = None
    rounds = []
    tried = {incumbent["controller_sha256"]}
    qualified_controllers = []
    for round_index in range(ROUNDS):
        bank = spec["rounds"][round_index]
        diagnostic, diagnostic_records = ledger.report(
            incumbent,
            list(map(int, bank["diagnostic"])),
            f"round{round_index + 1}_incumbent_diagnostic",
        )
        if arm == "uniform":
            challenger, proposal = uniform_proposal(round_index, incumbent, previous)
        else:
            challenger, proposal = strider_proposal(
                incumbent, diagnostic_records, tried, previous
            )
        tried.add(challenger["controller_sha256"])
        incumbent_paired, incumbent_records = ledger.report(
            incumbent,
            list(map(int, bank["paired"])),
            f"round{round_index + 1}_incumbent_paired",
        )
        challenger_paired, challenger_records = ledger.report(
            challenger,
            list(map(int, bank["paired"])),
            f"round{round_index + 1}_challenger_paired",
        )
        pair = paired_receipt(challenger_records, incumbent_records)
        winner, winner_ok, status = choose_round_winner(
            incumbent_paired, challenger_paired, pair
        )
        if winner_ok:
            qualified_controllers.append(
                challenger_paired
                if winner["controller_sha256"]
                == challenger_paired["controller_sha256"]
                else incumbent_paired
            )
        value = {
            "round": round_index + 1,
            "diagnostic": diagnostic,
            "proposal_receipt": proposal,
            "incumbent_paired": incumbent_paired,
            "challenger_paired": challenger_paired,
            "paired_receipt": pair,
            "selection_status": status,
            "winner_controller": winner,
            "winner_qualified": winner_ok,
            "incumbent_records": incumbent_records,
            "challenger_records": challenger_records,
        }
        rounds.append(value)
        previous = value
        incumbent = winner

    if ledger.used() != SEARCH_BUDGET:
        raise RuntimeError(f"v47 {arm} search used {ledger.used()}, expected 50")
    if qualified_controllers:
        selected_report = max(
            qualified_controllers,
            key=lambda item: (
                int(item["summary"]["successes"]),
                float(item["summary"]["achieved_throughput_per_step"]),
                -len(set(item["controller"]["schedule"])),
            ),
        )
        selected = selected_report["controller"]
        selection_status = "qualified_search_controller_selected"
    else:
        selected = static_controller([1.0] * len(PHASES))
        selection_status = "native_fallback_no_qualified_search_controller"
    # Raw rollout records are persisted separately; omit them from the sealed summary.
    clean_rounds = []
    for value in rounds:
        clean_rounds.append(
            {key: item for key, item in value.items() if not key.endswith("_records")}
        )
    return {
        "schema": "act-fair-strider-uniform50-selection-v47",
        "arm": arm,
        "rounds": clean_rounds,
        "selection_status": selection_status,
        "selected_controller": selected,
        "selected_controller_sha256": selected["controller_sha256"],
        "search_scientific_rollouts": ledger.used(),
        "incident_totals": ledger.incidents(),
        "historical_speed_outcomes_used_for_initialization": False,
        "historical_speed_schedules_used_for_initialization": False,
        "historical_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    }


def expand_task_banks(spec: dict) -> dict:
    def arm(start: int) -> dict:
        start = int(start)
        return {
            "rounds": [
                {
                    "diagnostic": list(range(start, start + 5)),
                    "paired": list(range(start + 5, start + 15)),
                },
                {
                    "diagnostic": list(range(start + 15, start + 20)),
                    "paired": list(range(start + 20, start + 30)),
                },
            ]
        }

    return {
        "uniform": arm(spec["uniform_search_start"]),
        "strider": arm(spec["strider_search_start"]),
        "final": list(
            range(int(spec["final_start"]), int(spec["final_start"]) + FINAL_EPISODES)
        ),
    }


def validate_banks(banks: dict) -> None:
    if banks.get("schema") != "act-fair-strider-uniform50-banks-v47":
        raise RuntimeError("v47 bank schema differs")
    all_seeds = []
    for task in TASKS:
        task_seeds = []
        spec = expand_task_banks(banks["tasks"][task])
        for arm in SEARCH_ARMS:
            rounds = spec[arm]["rounds"]
            if len(rounds) != ROUNDS:
                raise RuntimeError("v47 requires two search rounds")
            for value in rounds:
                if len(value["diagnostic"]) != DIAGNOSTIC_EPISODES:
                    raise RuntimeError("v47 requires five diagnostics per round")
                if len(value["paired"]) != PAIRED_EPISODES:
                    raise RuntimeError("v47 requires ten incumbent comparisons")
                task_seeds.extend(value["diagnostic"])
                task_seeds.extend(value["paired"])
        if len(spec["final"]) != FINAL_EPISODES:
            raise RuntimeError("v47 requires 100 final seeds")
        task_seeds.extend(spec["final"])
        if len(task_seeds) != len(set(task_seeds)):
            raise RuntimeError(f"v47 task banks overlap: {task}")
        all_seeds.extend(task_seeds)
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("v47 cross-task banks overlap")


def require_all_search(root: Path) -> None:
    for task in TASKS:
        for arm in SEARCH_ARMS:
            complete = v39.v38.v33.checked_json(
                root / "search" / task / arm / "SEARCH_COMPLETE.json"
            )
            if int(complete.get("search_scientific_rollouts", -1)) != SEARCH_BUDGET:
                raise RuntimeError(f"v47 search incomplete: {task}/{arm}")


def method_controller(root: Path, task: str, method: str):
    if method == "native_1x":
        return static_controller([1.0] * len(PHASES)), "preregistered_native"
    arm = "uniform" if method == "uniform_selected" else "strider"
    path = root / "search" / task / arm / "SELECTION.json"
    selection = v39.v38.v33.checked_json(path)
    controller = validate_controller(selection["selected_controller"])
    if controller["controller_sha256"] != selection["selected_controller_sha256"]:
        raise RuntimeError("v47 selected controller hash mismatch")
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
            raise RuntimeError("v47 final states contain a non-contiguous suffix")
        record = v39.v38.v33.checked_json(path)
        if int(record.get("seed", -1)) != seed:
            raise RuntimeError(f"v47 final seed mismatch: {path}")
        if record.get("identity_sha256") != identity_sha:
            raise RuntimeError(f"v47 final state identity mismatch: {path}")
        records.append(record)
    return records


def run_final(runtime, root, task, method, seeds, banks_sha):
    require_all_search(root)
    controller, provenance = method_controller(root, task, method)
    digest = controller["controller_sha256"]
    controller_root = root / "final" / task / "controllers" / digest
    alias_path = root / "final" / task / "methods" / method / "RESULT.json"
    if alias_path.exists():
        alias = v39.v38.v33.checked_json(alias_path)
        complete = v39.v38.v33.checked_json(controller_root / "COMPLETE.json")
        if alias.get("controller_sha256") != digest:
            raise RuntimeError(f"v47 method alias differs: {task}/{method}")
        if alias.get("controller_result_sha256") != complete.get("result_sha256"):
            raise RuntimeError(f"v47 method completion differs: {task}/{method}")
        return
    identity = {
        **runtime.identity(),
        "schema": "act-fair-strider-uniform50-final-controller-identity-v47",
        "task_label": task,
        "controller": controller,
        "phase_speed_selector": "learned_phase_detector_argmax",
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "banks_sha256": banks_sha,
        "search_or_tuning_permitted": False,
        "historical_speed_outcomes_used_for_initialization": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v39.v38.v33.immutable_or_verify(controller_root / "IDENTITY.json", identity)
    was_complete = (controller_root / "COMPLETE.json").exists()
    records = load_final_states(controller_root, seeds, identity["identity_sha256"])
    runner = v39.v38.ControllerRuntime(runtime)
    for seed in seeds[len(records) :]:
        record = runner.rollout(controller, seed, record_attribution_telemetry=False)
        if int(record.get("seed", -1)) != seed:
            raise RuntimeError("v47 final runtime returned a different seed")
        if record.get("controller_sha256") != digest:
            raise RuntimeError("v47 final runtime returned a different controller")
        record["identity_sha256"] = identity["identity_sha256"]
        immutable_json(controller_root / "states" / f"{seed}.json", record)
        records.append(record)
        atomic_json(
            controller_root / "progress.json",
            {
                "task": task,
                "controller_sha256": digest,
                "completed": len(records),
                "successes": sum(successful(item) for item in records),
            },
        )
        print(
            json.dumps(
                {
                    "stage": "final",
                    "task": task,
                    "method": method,
                    "completed": len(records),
                    "successes": sum(successful(item) for item in records),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    result = {
        "schema": "act-fair-strider-uniform50-final-controller-result-v47",
        "task_label": task,
        "controller": controller,
        "episodes": len(records),
        "summary": summarize(records),
        "identity_sha256": identity["identity_sha256"],
    }
    v39.v38.v33.immutable_or_verify(controller_root / "RESULT.json", result)
    v39.v38.v33.immutable_or_verify(
        controller_root / "COMPLETE.json",
        {
            "schema": "act-fair-strider-uniform50-final-controller-completion-v47",
            "episodes": len(records),
            "result_sha256": file_sha256(controller_root / "RESULT.json"),
            "physics_errors": result["summary"]["physics_errors"],
            "safety_violations": result["summary"]["safety_violations"],
        },
    )
    v39.v38.v33.immutable_or_verify(
        alias_path,
        {
            "schema": "act-fair-strider-uniform50-final-method-result-v47",
            "task_label": task,
            "method": method,
            "controller": controller,
            "controller_sha256": digest,
            "controller_result_sha256": file_sha256(controller_root / "RESULT.json"),
            "selection_provenance": provenance,
            "controller_cache_hit": was_complete,
            "summary": result["summary"],
        },
    )


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "final"), required=True)
    parser.add_argument("--method", choices=(*SEARCH_ARMS, *FINAL_METHODS), required=True)
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
        raise RuntimeError("v47 checked-out source differs from implementation commit")
    banks = v39.v38.v33.checked_json(args.banks)
    validate_banks(banks)
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
    spec = expand_task_banks(banks["tasks"][args.task_label])
    if args.stage == "final":
        if args.method not in FINAL_METHODS:
            raise ValueError("v47 final stage requires a final method")
        run_final(
            runtime,
            root,
            args.task_label,
            args.method,
            list(map(int, spec["final"])),
            file_sha256(args.banks),
        )
        return 0
    if args.method not in SEARCH_ARMS:
        raise ValueError("v47 search stage requires uniform or strider")
    output = root / "search" / args.task_label / args.method
    identity = {
        **runtime.identity(),
        "schema": "act-fair-strider-uniform50-search-identity-v47",
        "contract_sha256": file_sha256(args.contract),
        "banks_sha256": file_sha256(args.banks),
        "task_label": args.task_label,
        "arm": args.method,
        "phase_speed_selector": "learned_phase_detector_argmax",
        "search_budget": SEARCH_BUDGET,
        "round_seed_banks": spec[args.method]["rounds"],
        "final_seeds_registered_unopened": spec["final"],
        "initial_anchor": list(ANCHOR),
        "historical_speed_outcomes_used_for_initialization": False,
        "historical_speed_schedules_used_for_initialization": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v39.v38.v33.immutable_or_verify(output / "IDENTITY.json", identity)
    selection_path = output / "SELECTION.json"
    complete_path = output / "SEARCH_COMPLETE.json"
    if complete_path.exists():
        complete = v39.v38.v33.checked_json(complete_path)
        if complete["selection_sha256"] != file_sha256(selection_path):
            raise RuntimeError("v47 completed selection hash mismatch")
        return 0
    ledger = SearchLedger(v39.v38.ControllerRuntime(runtime), output / "search")
    selection = run_search(ledger, args.method, spec[args.method])
    immutable_json(selection_path, selection)
    immutable_json(
        complete_path,
        {
            "schema": "act-fair-strider-uniform50-search-completion-v47",
            "task_label": args.task_label,
            "arm": args.method,
            "identity_sha256": file_sha256(output / "IDENTITY.json"),
            "selection_sha256": file_sha256(selection_path),
            "search_scientific_rollouts": SEARCH_BUDGET,
            **selection["incident_totals"],
            "historical_speed_outcomes_used_for_initialization": False,
            "historical_speed_schedules_used_for_initialization": False,
            "historical_rollouts_reexecuted": 0,
            "final_bank_opened": False,
        },
    )
    print(json.dumps({"selection": selection}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
