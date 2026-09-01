#!/usr/bin/env python3
"""Run the fresh exact-25 phase-workload bandit study."""

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
from one_reset_phase_schedule import estimate_phase_workload  # noqa: E402
from scripts import run_act_fresh_transport25_v39 as v39  # noqa: E402
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head  # noqa: E402
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256, summarize  # noqa: E402

TASKS = ("pick", "tea", "insertion")
SEARCH_METHOD = "phase_bandit25"
FINAL_METHODS = ("native_1x", "uniform_2x", "phase_bump_3x", "selected")
DIAGNOSTIC_EPISODES = 5
PAIRED_EPISODES = 10
FINAL_EPISODES = 50
SEARCH_BUDGET = DIAGNOSTIC_EPISODES + 2 * PAIRED_EPISODES
PAIRED_SUCCESS_FLOOR = 9
CLEAR_FAILURE_CEILING = 7
SUCCESS_GAIN_THROUGHPUT_FLOOR = 0.95
TIED_SUCCESS_THROUGHPUT_RATIO = 1.05
UNIFORM_SCHEDULE = (2.0, 2.0, 2.0, 2.0)
PROPOSED_PHASE_SPEED = 3.0


static_controller = v39.static_controller
validate_controller = v39.validate_controller
SearchLedger = v39.SearchLedger


def load_controllers(path: Path) -> tuple[dict, str]:
    bundle = v39.v38.v33.checked_json(path)
    if bundle.get("schema") != "act-phase-bandit25-controllers-v40":
        raise RuntimeError("v40 controller bundle schema differs")
    if bundle.get("historical_speed_outcomes_used_for_initialization") is not False:
        raise RuntimeError("v40 must not initialize from historical speed outcomes")
    if bundle.get("historical_rollouts_reexecuted") != 0:
        raise RuntimeError("v40 controller bundle re-executes history")
    if bundle.get("phase_order") != list(PHASES):
        raise RuntimeError("v40 controller phase order differs")
    if bundle.get("phase_speed_selector") != "learned_phase_detector_argmax":
        raise RuntimeError("v40 requires the learned phase detector selector")
    if bundle.get("proposal_rule") != "max_mean_predicted_steps_saved":
        raise RuntimeError("v40 proposal rule differs")
    if float(bundle.get("proposed_phase_speed", -1)) != PROPOSED_PHASE_SPEED:
        raise RuntimeError("v40 proposed phase speed differs")
    uniform = static_controller(bundle["uniform_2x_schedule"])
    if tuple(uniform["schedule"]) != UNIFORM_SCHEDULE:
        raise RuntimeError("v40 uniform anchor differs")
    return uniform, file_sha256(path)


def choose_phase_bump(diagnostic_records: list[dict], uniform: dict) -> tuple[dict, dict]:
    eligible = [
        item
        for item in diagnostic_records
        if v39.v38.v32.successful(item)
        and item.get("physics_error") is None
        and item.get("safety_violation") is None
    ]
    totals = {phase: 0.0 for phase in PHASES}
    for item in eligible:
        workload = estimate_phase_workload(item)
        for phase in PHASES:
            totals[phase] += float(workload[phase])
    means = {
        phase: (totals[phase] / len(eligible) if eligible else 0.0)
        for phase in PHASES
    }
    old_speed = float(UNIFORM_SCHEDULE[0])
    predicted_saved = {
        phase: means[phase] * (1.0 / old_speed - 1.0 / PROPOSED_PHASE_SPEED)
        for phase in PHASES
    }
    phase_index = max(
        range(len(PHASES)),
        key=lambda index: (predicted_saved[PHASES[index]], -index),
    )
    schedule = list(UNIFORM_SCHEDULE)
    schedule[phase_index] = PROPOSED_PHASE_SPEED
    challenger = static_controller(schedule)
    proposal = {
        "schema": "act-phase-bandit25-proposal-v40",
        "diagnostic_controller_sha256": uniform["controller_sha256"],
        "diagnostic_successful_incident_free_episodes": len(eligible),
        "aggregation": "mean_native_equivalent_phase_workload_on_successful_incident_free_diagnostics",
        "score": "D_i * (1/current_speed_i - 1/proposed_speed_i)",
        "phase_order_tie_break": list(PHASES),
        "insufficient_success_fallback": "phase_order_first",
        "mean_native_equivalent_phase_workload": means,
        "predicted_steps_saved": predicted_saved,
        "selected_phase": PHASES[phase_index],
        "selected_phase_index": phase_index,
        "uniform_schedule": list(UNIFORM_SCHEDULE),
        "proposed_schedule": schedule,
        "proposed_controller": challenger,
        "proposed_controller_sha256": challenger["controller_sha256"],
        "historical_speed_outcomes_used_for_initialization": False,
    }
    return challenger, proposal


def run_search(
    ledger: SearchLedger,
    task: str,
    spec: dict,
    uniform: dict,
    proposal_path: Path,
) -> dict:
    diagnostic, diagnostic_records = ledger.report(
        uniform, list(map(int, spec["diagnostic"])), "uniform_2x_diagnostic"
    )
    if int(diagnostic["summary"]["physics_errors"]) > 0:
        raise RuntimeError("v40 halts on a diagnostic physics error")
    challenger, proposal = choose_phase_bump(diagnostic_records, uniform)
    v39.v38.v33.immutable_or_verify(proposal_path, proposal)
    uniform_paired, uniform_records = ledger.report(
        uniform, list(map(int, spec["paired"])), "uniform_2x_paired"
    )
    if int(uniform_paired["summary"]["physics_errors"]) > 0:
        raise RuntimeError("v40 halts on a uniform paired physics error")
    challenger_paired, challenger_records = ledger.report(
        challenger, list(map(int, spec["paired"])), "phase_bump_3x_paired"
    )
    if ledger.used() != SEARCH_BUDGET:
        raise RuntimeError(f"v40 search used {ledger.used()}, expected exactly 25")
    if int(challenger_paired["summary"]["physics_errors"]) > 0:
        raise RuntimeError("v40 halts on a phase-bump paired physics error")

    pair = v39.v38.v37.paired_receipt(challenger_records, uniform_records)
    uniform_incidents = {
        "physics_errors": int(diagnostic["summary"]["physics_errors"])
        + int(uniform_paired["summary"]["physics_errors"]),
        "safety_violations": int(diagnostic["summary"]["safety_violations"])
        + int(uniform_paired["summary"]["safety_violations"]),
    }
    challenger_incidents = {
        "physics_errors": int(challenger_paired["summary"]["physics_errors"]),
        "safety_violations": int(challenger_paired["summary"]["safety_violations"]),
    }
    uniform_successes = int(uniform_paired["summary"]["successes"])
    challenger_successes = int(challenger_paired["summary"]["successes"])
    ratio = pair["challenger_throughput_ratio"]
    uniform_clear = (
        uniform_successes > CLEAR_FAILURE_CEILING
        and uniform_incidents == {"physics_errors": 0, "safety_violations": 0}
    )
    challenger_qualified = (
        challenger_successes >= PAIRED_SUCCESS_FLOOR
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
        selected, status = challenger, "phase_bump_3x_promoted"
    elif uniform_clear:
        selected, status = uniform, "uniform_2x_retained"
    else:
        selected, status = static_controller([1.0] * len(PHASES)), "native_fallback_clear_failure"

    return {
        "schema": "act-phase-bandit25-selection-v40",
        "task_label": task,
        "uniform_2x_diagnostic": diagnostic,
        "proposal": proposal,
        "proposal_sha256": file_sha256(proposal_path),
        "uniform_2x_paired": uniform_paired,
        "phase_bump_3x_paired": challenger_paired,
        "paired_receipt": pair,
        "selection_rule": {
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
        complete = v39.v38.v33.checked_json(root / "search" / task / "SEARCH_COMPLETE.json")
        if int(complete.get("search_scientific_rollouts", -1)) != SEARCH_BUDGET:
            raise RuntimeError(f"v40 search incomplete: {task}")


def method_controller(root: Path, controllers_path: Path, task: str, method: str):
    uniform, provenance = load_controllers(controllers_path)
    if method == "native_1x":
        return static_controller([1.0] * len(PHASES)), "preregistered_native"
    if method == "uniform_2x":
        return uniform, "preregistered_controller_bundle:" + provenance
    selection_path = root / "search" / task / "SELECTION.json"
    selection = v39.v38.v33.checked_json(selection_path)
    key = "proposed_controller" if method == "phase_bump_3x" else "selected_controller"
    controller = validate_controller(
        selection["proposal"][key] if method == "phase_bump_3x" else selection[key]
    )
    expected = (
        selection["proposal"]["proposed_controller_sha256"]
        if method == "phase_bump_3x"
        else selection["selected_controller_sha256"]
    )
    if controller["controller_sha256"] != expected:
        raise RuntimeError("v40 selected controller hash mismatch")
    return controller, file_sha256(selection_path)


def load_final_states(directory: Path, seeds: list[int], identity_sha: str):
    records = []
    missing = False
    for seed in seeds:
        path = directory / "states" / f"{seed}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError("v40 final states contain a non-contiguous suffix")
        record = v39.v38.v33.checked_json(path)
        if int(record.get("seed", -1)) != seed or record.get("identity_sha256") != identity_sha:
            raise RuntimeError(f"v40 final state identity mismatch: {path}")
        records.append(record)
    return records


def run_final(runtime, root, controllers_path, task, method, seeds, banks_sha):
    require_all_search(root)
    controller, provenance = method_controller(root, controllers_path, task, method)
    digest = controller["controller_sha256"]
    controller_root = root / "final" / task / "controllers" / digest
    alias_path = root / "final" / task / "methods" / method / "RESULT.json"
    if alias_path.exists():
        alias = v39.v38.v33.checked_json(alias_path)
        complete = v39.v38.v33.checked_json(controller_root / "COMPLETE.json")
        if alias.get("controller_sha256") != digest:
            raise RuntimeError(f"v40 method alias differs: {task}/{method}")
        if alias.get("controller_result_sha256") != complete.get("result_sha256"):
            raise RuntimeError(f"v40 method completion differs: {task}/{method}")
        return
    identity = {
        **runtime.identity(),
        "schema": "act-phase-bandit25-final-controller-identity-v40",
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
    v39.v38.v33.immutable_or_verify(controller_root / "IDENTITY.json", identity)
    was_complete = (controller_root / "COMPLETE.json").exists()
    records = load_final_states(controller_root, seeds, identity["identity_sha256"])
    runner = v39.v38.ControllerRuntime(runtime)
    for seed in seeds[len(records) :]:
        record = runner.rollout(controller, seed, record_attribution_telemetry=False)
        if int(record.get("seed", -1)) != seed or record.get("controller_sha256") != digest:
            raise RuntimeError("v40 final runtime returned a different controller")
        record["identity_sha256"] = identity["identity_sha256"]
        immutable_json(controller_root / "states" / f"{seed}.json", record)
        records.append(record)
        atomic_json(
            controller_root / "progress.json",
            {
                "task": task,
                "controller_sha256": digest,
                "completed": len(records),
                "successes": sum(v39.v38.v32.successful(item) for item in records),
            },
        )
        print(json.dumps({"stage": "final", "task": task, "method": method,
                          "completed": len(records),
                          "successes": sum(v39.v38.v32.successful(item) for item in records)},
                         sort_keys=True), flush=True)
    result = {
        "schema": "act-phase-bandit25-final-controller-result-v40",
        "task_label": task,
        "controller": controller,
        "episodes": len(records),
        "summary": summarize(records),
        "identity_sha256": identity["identity_sha256"],
    }
    v39.v38.v33.immutable_or_verify(controller_root / "RESULT.json", result)
    v39.v38.v33.immutable_or_verify(
        controller_root / "COMPLETE.json",
        {"schema": "act-phase-bandit25-final-controller-completion-v40",
         "episodes": len(records), "result_sha256": file_sha256(controller_root / "RESULT.json"),
         "physics_errors": result["summary"]["physics_errors"],
         "safety_violations": result["summary"]["safety_violations"]},
    )
    v39.v38.v33.immutable_or_verify(
        alias_path,
        {"schema": "act-phase-bandit25-final-method-result-v40", "task_label": task,
         "method": method, "controller": controller, "controller_sha256": digest,
         "controller_result_sha256": file_sha256(controller_root / "RESULT.json"),
         "controller_receipt": str(controller_root / "RESULT.json"),
         "selection_provenance": provenance, "controller_cache_hit": was_complete,
         "summary": result["summary"]},
    )


def validate_banks(banks: dict) -> None:
    if banks.get("schema") != "act-phase-bandit25-banks-v40":
        raise RuntimeError("v40 bank schema differs")
    all_seeds = []
    for task in TASKS:
        spec = banks["tasks"][task]
        if len(spec["diagnostic"]) != DIAGNOSTIC_EPISODES:
            raise RuntimeError("v40 banks require five diagnostic seeds")
        if len(spec["paired"]) != PAIRED_EPISODES or len(spec["final"]) != FINAL_EPISODES:
            raise RuntimeError("v40 banks require ten paired and fifty final seeds")
        task_seeds = spec["diagnostic"] + spec["paired"] + spec["final"]
        if len(task_seeds) != len(set(task_seeds)):
            raise RuntimeError(f"v40 task banks overlap: {task}")
        all_seeds.extend(task_seeds)
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("v40 cross-task banks overlap")


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
        raise RuntimeError("v40 checked-out source differs from implementation commit")
    banks = v39.v38.v33.checked_json(args.banks)
    validate_banks(banks)
    uniform, controllers_sha = load_controllers(args.controllers)
    spec = banks["tasks"][args.task_label]
    runtime = ACTFrontierRuntime(
        source_commit=args.base_source_commit, checkout_commit=args.implementation_commit,
        run_manifest=args.run_manifest, task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint, detector_source=args.detector_source,
        device=args.device,
    )
    root = args.root.resolve()
    if args.stage == "final":
        if args.method not in FINAL_METHODS:
            raise ValueError("v40 final stage requires a final method")
        run_final(runtime, root, args.controllers, args.task_label, args.method,
                  list(map(int, spec["final"])), file_sha256(args.banks))
        return 0
    if args.method != SEARCH_METHOD:
        raise ValueError("v40 search stage requires phase_bandit25")
    output = root / "search" / args.task_label
    identity = {
        **runtime.identity(), "schema": "act-phase-bandit25-search-identity-v40",
        "contract_sha256": file_sha256(args.contract), "banks_sha256": file_sha256(args.banks),
        "controllers_sha256": controllers_sha, "task_label": args.task_label,
        "phase_speed_selector": "learned_phase_detector_argmax", "secondary_speed_override": None,
        "search_budget": SEARCH_BUDGET, "diagnostic_seeds": spec["diagnostic"],
        "paired_seeds": spec["paired"], "final_seeds_registered_unopened": spec["final"],
        "uniform_controller_sha256": uniform["controller_sha256"],
        "proposal_rule": "max_mean_predicted_steps_saved", "proposed_phase_speed": 3.0,
        "historical_speed_outcomes_used_for_initialization": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v39.v38.v33.immutable_or_verify(output / "IDENTITY.json", identity)
    selection_path = output / "SELECTION.json"
    complete_path = output / "SEARCH_COMPLETE.json"
    if complete_path.exists():
        complete = v39.v38.v33.checked_json(complete_path)
        if complete["selection_sha256"] != file_sha256(selection_path):
            raise RuntimeError("v40 completed selection hash mismatch")
        return 0
    ledger = SearchLedger(v39.v38.ControllerRuntime(runtime), output / "search")
    selection = run_search(ledger, args.task_label, spec, uniform, output / "PROPOSAL.json")
    immutable_json(selection_path, selection)
    immutable_json(
        complete_path,
        {"schema": "act-phase-bandit25-search-completion-v40", "task_label": args.task_label,
         "identity_sha256": file_sha256(output / "IDENTITY.json"),
         "proposal_sha256": file_sha256(output / "PROPOSAL.json"),
         "selection_sha256": file_sha256(selection_path),
         "search_scientific_rollouts": SEARCH_BUDGET, **selection["incident_totals"],
         "historical_speed_outcomes_used_for_initialization": False,
         "historical_rollouts_reexecuted": 0, "final_bank_opened": False},
    )
    print(json.dumps({"selection": selection}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
