#!/usr/bin/env python3
"""Run exact-25 full-schedule confirmation and a fresh held-out evaluation."""

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
from scripts import run_act_common_grid_strider_causal25_v32 as v32  # noqa: E402
from scripts import run_act_prior_causal_risk_v34 as v34  # noqa: E402
from scripts import run_act_sail_warmstart_v33 as v33  # noqa: E402
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head  # noqa: E402
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json  # noqa: E402
from scripts.run_act_strider_frontier_v4 import file_sha256, summarize  # noqa: E402

TASKS = ("pick", "tea", "insertion")
FINAL_METHODS = ("native_1x", "confirmed_phase_schedule")
SEARCH_METHOD = "end_to_end_confirm"
SEARCH_BUDGET = 25


def schedule_sha256(schedule) -> str:
    return canonical_sha256(list(v32.validate_schedule(schedule)))


def run_search(ledger: v32.Ledger, task: str, prior: dict) -> dict:
    """Use v34's candidate logic, but evaluate every schedule without a gate."""

    selection = v34.run_search(
        ledger,
        task,
        prior,
        {"controller_sha256": "not-executed-in-v36"},
    )
    selection.pop("risk_gate", None)
    selection.update(
        {
            "schema": "act-end-to-end-confirm25-selection-v36",
            "method": SEARCH_METHOD,
            "candidate_execution": "complete_schedule_end_to_end_without_runtime_gate",
            "discovery_design": "five_complete_schedules_x_three_matched_seeds",
            "confirmation_design": "two_complete_finalists_x_five_fresh_seeds",
            "phase_dp_estimator_used": False,
            "runtime_risk_gate_used": False,
            "offline_prior_training_rollouts_reused": int(prior["offline_training_rollouts"]),
            "study_design_informed_by_v34_v35_results": True,
            "historical_speed_outcomes_used_by_runtime": False,
        }
    )
    for report in selection["discovery_reports"]:
        if report.get("role") == "causal_risk_frontier":
            report["role"] = "end_to_end_causal_frontier"
        if int(report["summary"]["episodes"]) != 3:
            raise RuntimeError("v36 discovery candidate lacks three end-to-end trials")
    for finalist in selection["finalists"]:
        if int(finalist["summary"]["episodes"]) != 8:
            raise RuntimeError("v36 finalist lacks eight end-to-end trials")
    if ledger.used() != SEARCH_BUDGET:
        raise RuntimeError(f"v36 search used {ledger.used()}, expected 25")
    return selection


def require_all_search(root: Path) -> None:
    for task in TASKS:
        complete = v33.checked_json(root / "search" / task / "SEARCH_COMPLETE.json")
        if int(complete.get("search_scientific_rollouts", -1)) != SEARCH_BUDGET:
            raise RuntimeError(f"v36 search incomplete: {task}")


def method_schedule(root: Path, task: str, method: str) -> tuple[list[float], str]:
    if method == "native_1x":
        return [1.0] * 4, "preregistered_native"
    selection_path = root / "search" / task / "SELECTION.json"
    selection = v33.checked_json(selection_path)
    if selection.get("selected_schedule") is None:
        return [1.0] * 4, file_sha256(selection_path)
    schedule = list(v32.validate_schedule(selection["selected_schedule"]))
    if schedule_sha256(schedule) != selection["selected_schedule_sha256"]:
        raise RuntimeError("v36 selected schedule hash mismatch")
    return schedule, file_sha256(selection_path)


def load_states(directory: Path, seeds: list[int], identity_sha: str) -> list[dict]:
    records = []
    missing = False
    for seed in seeds:
        path = directory / "states" / f"{seed}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError("v36 final states contain a non-contiguous suffix")
        record = v33.checked_json(path)
        if int(record.get("seed", -1)) != seed or record.get("identity_sha256") != identity_sha:
            raise RuntimeError(f"v36 final state identity mismatch: {path}")
        records.append(record)
    return records


def successful(record: dict) -> bool:
    return (
        bool(record.get("success"))
        and record.get("physics_error") is None
        and record.get("safety_violation") is None
    )


def run_final(runtime, root: Path, task: str, method: str, seeds: list[int], banks_sha: str) -> None:
    require_all_search(root)
    schedule, provenance = method_schedule(root, task, method)
    controller_hash = schedule_sha256(schedule)
    controller_root = root / "final" / task / "controllers" / controller_hash
    alias_path = root / "final" / task / "methods" / method / "RESULT.json"
    if alias_path.exists():
        alias = v33.checked_json(alias_path)
        complete = v33.checked_json(controller_root / "COMPLETE.json")
        if alias.get("controller_sha256") != controller_hash:
            raise RuntimeError(f"v36 method alias differs: {task}/{method}")
        if alias.get("controller_result_sha256") != complete.get("result_sha256"):
            raise RuntimeError(f"v36 method completion differs: {task}/{method}")
        return
    identity = {
        **runtime.identity(),
        "schema": "act-end-to-end-confirm-final-controller-identity-v36",
        "task_label": task,
        "schedule": schedule,
        "schedule_sha256": controller_hash,
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "banks_sha256": banks_sha,
        "search_or_tuning_permitted": False,
        "runtime_risk_gate_used": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v33.immutable_or_verify(controller_root / "IDENTITY.json", identity)
    was_complete = (controller_root / "COMPLETE.json").exists()
    records = load_states(controller_root, seeds, identity["identity_sha256"])
    for seed in seeds[len(records) :]:
        record = runtime.rollout(schedule, seed, record_attribution_telemetry=False)
        if int(record.get("seed", -1)) != seed or list(map(float, record.get("schedule", ()))) != schedule:
            raise RuntimeError("v36 final runtime returned a different controller")
        record["identity_sha256"] = identity["identity_sha256"]
        immutable_json(controller_root / "states" / f"{seed}.json", record)
        records.append(record)
        atomic_json(
            controller_root / "progress.json",
            {
                "task": task,
                "controller_sha256": controller_hash,
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
                }
            ),
            flush=True,
        )
    result = {
        "schema": "act-end-to-end-confirm-final-controller-result-v36",
        "task_label": task,
        "schedule": schedule,
        "schedule_sha256": controller_hash,
        "episodes": len(records),
        "summary": summarize(records),
        "identity_sha256": identity["identity_sha256"],
    }
    v33.immutable_or_verify(controller_root / "RESULT.json", result)
    v33.immutable_or_verify(
        controller_root / "COMPLETE.json",
        {
            "schema": "act-end-to-end-confirm-final-controller-completion-v36",
            "episodes": len(records),
            "result_sha256": file_sha256(controller_root / "RESULT.json"),
            "physics_errors": result["summary"]["physics_errors"],
            "safety_violations": result["summary"]["safety_violations"],
        },
    )
    v33.immutable_or_verify(
        alias_path,
        {
            "schema": "act-end-to-end-confirm-final-method-result-v36",
            "task_label": task,
            "method": method,
            "controller_schedule": schedule,
            "controller_sha256": controller_hash,
            "controller_result_sha256": file_sha256(controller_root / "RESULT.json"),
            "controller_receipt": str(controller_root / "RESULT.json"),
            "selection_provenance": provenance,
            "controller_cache_hit": was_complete,
            "summary": result["summary"],
        },
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
    parser.add_argument("--offline-priors", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if git_head() != args.implementation_commit:
        raise RuntimeError("v36 checked-out source differs from implementation commit")
    banks = v33.checked_json(args.banks)
    all_seeds = []
    for spec in banks["tasks"].values():
        if len(spec["discovery"]) != 3 or len(spec["confirmation"]) != 5 or len(spec["final"]) != 50:
            raise RuntimeError("v36 banks must register 3 discovery, 5 confirmation, and 50 final seeds")
        task_seeds = spec["discovery"] + spec["confirmation"] + spec["final"]
        if len(task_seeds) != len(set(task_seeds)):
            raise RuntimeError("v36 task banks overlap")
        all_seeds.extend(task_seeds)
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("v36 cross-task banks overlap")
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
    prior = v34.combined_prior(args.offline_priors, args.task_label)
    root = args.root.resolve()
    if args.stage == "final":
        if args.method not in FINAL_METHODS:
            raise ValueError("v36 final stage requires a final method")
        run_final(runtime, root, args.task_label, args.method, list(map(int, spec["final"])), file_sha256(args.banks))
        return 0
    if args.method != SEARCH_METHOD:
        raise ValueError("v36 search stage requires end_to_end_confirm")
    output = root / "search" / args.task_label
    identity = {
        **runtime.identity(),
        "schema": "act-end-to-end-confirm-search-identity-v36",
        "contract_sha256": file_sha256(args.contract),
        "banks_sha256": file_sha256(args.banks),
        "task_label": args.task_label,
        "search_budget": SEARCH_BUDGET,
        "discovery_seeds": spec["discovery"],
        "confirmation_seeds": spec["confirmation"],
        "final_seeds_registered_unopened": spec["final"],
        "prior_payload_sha256": prior["prior_payload_sha256"],
        "candidate_execution": "complete_schedule_end_to_end_without_runtime_gate",
        "phase_dp_estimator_used": False,
        "historical_speed_outcomes_used_by_runtime": False,
        "study_design_informed_by_v34_v35_results": True,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v33.immutable_or_verify(output / "IDENTITY.json", identity)
    selection_path = output / "SELECTION.json"
    complete_path = output / "SEARCH_COMPLETE.json"
    if complete_path.exists():
        complete = v33.checked_json(complete_path)
        if complete["selection_sha256"] != file_sha256(selection_path):
            raise RuntimeError("v36 completed selection hash mismatch")
        return 0
    ledger = v32.Ledger(
        runtime,
        output / "search",
        list(map(int, spec["discovery"])),
        list(map(int, spec["confirmation"])),
    )
    selection = run_search(ledger, args.task_label, prior)
    immutable_json(selection_path, selection)
    immutable_json(
        complete_path,
        {
            "schema": "act-end-to-end-confirm-search-completion-v36",
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
