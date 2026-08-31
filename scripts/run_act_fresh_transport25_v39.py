#!/usr/bin/env python3
"""Run the fresh exact-25 uniform-2x versus transport-2.5x study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SPEEDTUNING_SPEED_VALUES", "1,1.5,2,2.5,3")

from act_speed_benchmark import canonical_sha256  # noqa: E402
from learned_phase_observation import PHASES  # noqa: E402
from scripts import run_act_transport3_boundary_v38 as v38  # noqa: E402
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head  # noqa: E402
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256, summarize  # noqa: E402

TASKS = ("pick", "tea", "insertion")
SEARCH_METHOD = "fresh_transport25"
FINAL_METHODS = ("native_1x", "uniform_2x", "transport_2p5", "selected")
DISCOVERY_EPISODES = 5
PAIRED_EPISODES = 10
FINAL_EPISODES = 50
SEARCH_BUDGET = DISCOVERY_EPISODES + 2 * PAIRED_EPISODES
DISCOVERY_SUCCESS_FLOOR = 4
PAIRED_SUCCESS_FLOOR = 9
CLEAR_FAILURE_CEILING = 7
SUCCESS_GAIN_THROUGHPUT_FLOOR = 0.95
TIED_SUCCESS_THROUGHPUT_RATIO = 1.05
UNIFORM_SCHEDULE = (2.0, 2.0, 2.0, 2.0)
TRANSPORT_SCHEDULE = (2.0, 2.0, 2.5, 2.0)


def static_controller(schedule) -> dict:
    return v38.static_controller(schedule)


def validate_controller(controller: dict) -> dict:
    value = v38.validate_controller(controller)
    if value["type"] != "static_phase_schedule":
        raise RuntimeError("v39 permits learned-phase schedules only")
    return value


def load_controllers(path: Path) -> tuple[dict, dict, str]:
    bundle = v38.v33.checked_json(path)
    if bundle.get("schema") != "act-fresh-transport25-controllers-v39":
        raise RuntimeError("v39 controller bundle schema differs")
    if bundle.get("historical_speed_outcomes_used_for_initialization") is not False:
        raise RuntimeError("v39 must not initialize from historical speed outcomes")
    if bundle.get("historical_rollouts_reexecuted") != 0:
        raise RuntimeError("v39 controller bundle re-executes history")
    if bundle.get("phase_order") != list(PHASES):
        raise RuntimeError("v39 controller phase order differs")
    if bundle.get("phase_speed_selector") != "learned_phase_detector_argmax":
        raise RuntimeError("v39 requires the learned phase detector selector")
    uniform = static_controller(bundle["uniform_2x_schedule"])
    challenger = static_controller(bundle["transport_2p5_schedule"])
    if tuple(uniform["schedule"]) != UNIFORM_SCHEDULE:
        raise RuntimeError("v39 uniform anchor differs")
    if tuple(challenger["schedule"]) != TRANSPORT_SCHEDULE:
        raise RuntimeError("v39 transport challenger differs")
    return uniform, challenger, file_sha256(path)


class SearchLedger(v38.SearchLedger):
    """The v38 ledger is outcome-agnostic and already enforces exact-25 receipts."""


def run_search(
    ledger: SearchLedger,
    task: str,
    spec: dict,
    uniform: dict,
    challenger: dict,
) -> dict:
    discovery, _ = ledger.report(
        challenger,
        list(map(int, spec["challenger_discovery"])),
        "transport_2p5_discovery",
    )
    uniform_paired, uniform_records = ledger.report(
        uniform, list(map(int, spec["paired"])), "uniform_2x_paired"
    )
    challenger_paired, challenger_records = ledger.report(
        challenger, list(map(int, spec["paired"])), "transport_2p5_paired"
    )
    if ledger.used() != SEARCH_BUDGET:
        raise RuntimeError(f"v39 search used {ledger.used()}, expected exactly 25")

    pair = v38.v37.paired_receipt(challenger_records, uniform_records)
    challenger_incidents = {
        "physics_errors": int(discovery["summary"]["physics_errors"])
        + int(challenger_paired["summary"]["physics_errors"]),
        "safety_violations": int(discovery["summary"]["safety_violations"])
        + int(challenger_paired["summary"]["safety_violations"]),
    }
    uniform_incidents = {
        "physics_errors": int(uniform_paired["summary"]["physics_errors"]),
        "safety_violations": int(uniform_paired["summary"]["safety_violations"]),
    }
    uniform_successes = int(uniform_paired["summary"]["successes"])
    challenger_successes = int(challenger_paired["summary"]["successes"])
    ratio = pair["challenger_throughput_ratio"]
    uniform_clear = (
        uniform_successes > CLEAR_FAILURE_CEILING
        and uniform_incidents == {"physics_errors": 0, "safety_violations": 0}
    )
    challenger_qualified = (
        int(discovery["summary"]["successes"]) >= DISCOVERY_SUCCESS_FLOOR
        and challenger_successes >= PAIRED_SUCCESS_FLOOR
        and challenger_incidents == {"physics_errors": 0, "safety_violations": 0}
    )
    challenger_preferred = challenger_qualified and (
        not uniform_clear
        or (
            challenger_successes > uniform_successes
            and ratio is not None
            and ratio >= SUCCESS_GAIN_THROUGHPUT_FLOOR
        )
        or (
            challenger_successes == uniform_successes
            and ratio is not None
            and ratio >= TIED_SUCCESS_THROUGHPUT_RATIO
        )
    )
    if challenger_preferred:
        selected, status = challenger, "transport_2p5_promoted"
    elif uniform_clear:
        selected, status = uniform, "uniform_2x_retained"
    else:
        selected, status = static_controller([1.0] * len(PHASES)), "native_fallback_clear_failure"

    return {
        "schema": "act-fresh-transport25-selection-v39",
        "task_label": task,
        "transport_2p5_discovery": discovery,
        "uniform_2x_paired": uniform_paired,
        "transport_2p5_paired": challenger_paired,
        "paired_receipt": pair,
        "selection_rule": {
            "challenger_discovery_success_floor": DISCOVERY_SUCCESS_FLOOR,
            "paired_success_floor": PAIRED_SUCCESS_FLOOR,
            "clear_failure_ceiling": CLEAR_FAILURE_CEILING,
            "success_gain_throughput_floor": SUCCESS_GAIN_THROUGHPUT_FLOOR,
            "tied_success_throughput_ratio": TIED_SUCCESS_THROUGHPUT_RATIO,
            "ambiguous_result": "retain_uniform_2x",
        },
        "uniform_clear_of_failure": uniform_clear,
        "challenger_qualified": challenger_qualified,
        "challenger_preferred": challenger_preferred,
        "selection_status": status,
        "selected_controller": selected,
        "selected_controller_sha256": selected["controller_sha256"],
        "search_scientific_rollouts": ledger.used(),
        "incident_totals": ledger.incidents(),
        "historical_speed_outcomes_used_for_initialization": False,
        "historical_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    }


def require_all_search(root: Path) -> None:
    for task in TASKS:
        complete = v38.v33.checked_json(root / "search" / task / "SEARCH_COMPLETE.json")
        if int(complete.get("search_scientific_rollouts", -1)) != SEARCH_BUDGET:
            raise RuntimeError(f"v39 search incomplete: {task}")


def method_controller(root: Path, controllers_path: Path, task: str, method: str):
    uniform, challenger, provenance = load_controllers(controllers_path)
    fixed = {
        "native_1x": static_controller([1.0] * len(PHASES)),
        "uniform_2x": uniform,
        "transport_2p5": challenger,
    }
    if method in fixed:
        return fixed[method], "preregistered_controller_bundle:" + provenance
    path = root / "search" / task / "SELECTION.json"
    selection = v38.v33.checked_json(path)
    controller = validate_controller(selection["selected_controller"])
    if controller["controller_sha256"] != selection["selected_controller_sha256"]:
        raise RuntimeError("v39 selected controller hash mismatch")
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
            raise RuntimeError("v39 final states contain a non-contiguous suffix")
        record = v38.v33.checked_json(path)
        if int(record.get("seed", -1)) != seed or record.get("identity_sha256") != identity_sha:
            raise RuntimeError(f"v39 final state identity mismatch: {path}")
        records.append(record)
    return records


def run_final(runtime, root, controllers_path, task, method, seeds, banks_sha):
    require_all_search(root)
    controller, provenance = method_controller(root, controllers_path, task, method)
    digest = controller["controller_sha256"]
    controller_root = root / "final" / task / "controllers" / digest
    alias_path = root / "final" / task / "methods" / method / "RESULT.json"
    if alias_path.exists():
        alias = v38.v33.checked_json(alias_path)
        complete = v38.v33.checked_json(controller_root / "COMPLETE.json")
        if alias.get("controller_sha256") != digest:
            raise RuntimeError(f"v39 method alias differs: {task}/{method}")
        if alias.get("controller_result_sha256") != complete.get("result_sha256"):
            raise RuntimeError(f"v39 method completion differs: {task}/{method}")
        return
    identity = {
        **runtime.identity(),
        "schema": "act-fresh-transport25-final-controller-identity-v39",
        "task_label": task,
        "controller": controller,
        "phase_speed_selector": "learned_phase_detector_argmax",
        "secondary_speed_override": None,
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "banks_sha256": banks_sha,
        "search_or_tuning_permitted": False,
        "historical_speed_outcomes_used_for_initialization": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v38.v33.immutable_or_verify(controller_root / "IDENTITY.json", identity)
    was_complete = (controller_root / "COMPLETE.json").exists()
    records = load_final_states(controller_root, seeds, identity["identity_sha256"])
    runner = v38.ControllerRuntime(runtime)
    for seed in seeds[len(records) :]:
        record = runner.rollout(controller, seed, record_attribution_telemetry=False)
        if int(record.get("seed", -1)) != seed or record.get("controller_sha256") != digest:
            raise RuntimeError("v39 final runtime returned a different controller")
        record["identity_sha256"] = identity["identity_sha256"]
        immutable_json(controller_root / "states" / f"{seed}.json", record)
        records.append(record)
        atomic_json(
            controller_root / "progress.json",
            {
                "task": task,
                "controller_sha256": digest,
                "completed": len(records),
                "successes": sum(v38.v32.successful(item) for item in records),
            },
        )
        print(
            json.dumps(
                {
                    "stage": "final",
                    "task": task,
                    "method": method,
                    "completed": len(records),
                    "successes": sum(v38.v32.successful(item) for item in records),
                },
                sort_keys=True,
        ),
            flush=True,
        )
    result = {
        "schema": "act-fresh-transport25-final-controller-result-v39",
        "task_label": task,
        "controller": controller,
        "episodes": len(records),
        "summary": summarize(records),
        "identity_sha256": identity["identity_sha256"],
    }
    v38.v33.immutable_or_verify(controller_root / "RESULT.json", result)
    v38.v33.immutable_or_verify(
        controller_root / "COMPLETE.json",
        {
            "schema": "act-fresh-transport25-final-controller-completion-v39",
            "episodes": len(records),
            "result_sha256": file_sha256(controller_root / "RESULT.json"),
            "physics_errors": result["summary"]["physics_errors"],
            "safety_violations": result["summary"]["safety_violations"],
        },
    )
    v38.v33.immutable_or_verify(
        alias_path,
        {
            "schema": "act-fresh-transport25-final-method-result-v39",
            "task_label": task,
            "method": method,
            "controller": controller,
            "controller_sha256": digest,
            "controller_result_sha256": file_sha256(controller_root / "RESULT.json"),
            "controller_receipt": str(controller_root / "RESULT.json"),
            "selection_provenance": provenance,
            "controller_cache_hit": was_complete,
            "summary": result["summary"],
        },
    )


def validate_banks(banks: dict) -> None:
    if banks.get("schema") != "act-fresh-transport25-banks-v39":
        raise RuntimeError("v39 bank schema differs")
    all_seeds = []
    for task in TASKS:
        spec = banks["tasks"][task]
        if len(spec["challenger_discovery"]) != DISCOVERY_EPISODES:
            raise RuntimeError("v39 banks require five challenger discovery seeds")
        if len(spec["paired"]) != PAIRED_EPISODES or len(spec["final"]) != FINAL_EPISODES:
            raise RuntimeError("v39 banks require ten paired and fifty final seeds")
        task_seeds = spec["challenger_discovery"] + spec["paired"] + spec["final"]
        if len(task_seeds) != len(set(task_seeds)):
            raise RuntimeError(f"v39 task banks overlap: {task}")
        all_seeds.extend(task_seeds)
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("v39 cross-task banks overlap")


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
        raise RuntimeError("v39 checked-out source differs from implementation commit")
    banks = v38.v33.checked_json(args.banks)
    validate_banks(banks)
    uniform, challenger, controllers_sha = load_controllers(args.controllers)
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
            raise ValueError("v39 final stage requires a final method")
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
        raise ValueError("v39 search stage requires fresh_transport25")
    output = root / "search" / args.task_label
    identity = {
        **runtime.identity(),
        "schema": "act-fresh-transport25-search-identity-v39",
        "contract_sha256": file_sha256(args.contract),
        "banks_sha256": file_sha256(args.banks),
        "controllers_sha256": controllers_sha,
        "task_label": args.task_label,
        "phase_speed_selector": "learned_phase_detector_argmax",
        "secondary_speed_override": None,
        "search_budget": SEARCH_BUDGET,
        "challenger_discovery_seeds": spec["challenger_discovery"],
        "paired_seeds": spec["paired"],
        "final_seeds_registered_unopened": spec["final"],
        "uniform_controller_sha256": uniform["controller_sha256"],
        "challenger_controller_sha256": challenger["controller_sha256"],
        "historical_speed_outcomes_used_for_initialization": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v38.v33.immutable_or_verify(output / "IDENTITY.json", identity)
    selection_path = output / "SELECTION.json"
    complete_path = output / "SEARCH_COMPLETE.json"
    if complete_path.exists():
        complete = v38.v33.checked_json(complete_path)
        if complete["selection_sha256"] != file_sha256(selection_path):
            raise RuntimeError("v39 completed selection hash mismatch")
        return 0
    ledger = SearchLedger(v38.ControllerRuntime(runtime), output / "search")
    selection = run_search(ledger, args.task_label, spec, uniform, challenger)
    immutable_json(selection_path, selection)
    immutable_json(
        complete_path,
        {
            "schema": "act-fresh-transport25-search-completion-v39",
            "task_label": args.task_label,
            "identity_sha256": file_sha256(output / "IDENTITY.json"),
            "selection_sha256": file_sha256(selection_path),
            "search_scientific_rollouts": SEARCH_BUDGET,
            **selection["incident_totals"],
            "historical_speed_outcomes_used_for_initialization": False,
            "historical_rollouts_reexecuted": 0,
            "final_bank_opened": False,
        },
    )
    print(json.dumps({"selection": selection}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
