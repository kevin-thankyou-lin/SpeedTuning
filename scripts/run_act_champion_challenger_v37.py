#!/usr/bin/env python3
"""Run the exact-25 accelerated champion-challenger speed study."""

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

from act_speed_benchmark import canonical_sha256
from scripts import run_act_common_grid_strider_causal25_v32 as v32
from scripts import run_act_sail_warmstart_v33 as v33
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json
from scripts.run_act_strider_frontier_v4 import file_sha256, summarize

TASKS = ("pick", "tea", "insertion")
SEARCH_METHOD = "champion_challenger"
FINAL_METHODS = ("native_1x", "champion", "selected")
DISCOVERY_EPISODES = 5
PAIRED_EPISODES = 10
SEARCH_BUDGET = DISCOVERY_EPISODES + 2 * PAIRED_EPISODES
DISCOVERY_SUCCESS_FLOOR = 4
PAIRED_SUCCESS_FLOOR = 9
THROUGHPUT_PROMOTION_RATIO = 1.10


def schedule_sha256(schedule) -> str:
    return canonical_sha256(list(v32.validate_schedule(schedule)))


def successful(record: dict) -> bool:
    return v32.successful(record)


def load_champion(path: Path, task: str) -> dict:
    bundle = v33.checked_json(path)
    if bundle.get("schema") != "act-champion-challenger-offline-incumbents-v37":
        raise RuntimeError("v37 champion bundle schema differs")
    if bundle.get("historical_rollouts_reexecuted") != 0:
        raise RuntimeError("v37 champion bundle re-executes historical rollouts")
    value = dict(bundle["tasks"][task])
    champion = list(v32.validate_schedule(value["champion_schedule"]))
    challenger = list(v32.validate_schedule(value["challenger_schedule"]))
    if schedule_sha256(champion) != value["champion_schedule_sha256"]:
        raise RuntimeError(f"v37 champion hash mismatch: {task}")
    if schedule_sha256(challenger) != value["challenger_schedule_sha256"]:
        raise RuntimeError(f"v37 challenger hash mismatch: {task}")
    changed = [
        index
        for index, pair in enumerate(zip(champion, challenger))
        if pair[0] != pair[1]
    ]
    if changed != [v32.PHASES.index(value["proposal_phase"])]:
        raise RuntimeError(f"v37 challenger is not a one-phase update: {task}")
    index = changed[0]
    grid = list(map(float, v32.GRID))
    if grid.index(challenger[index]) != grid.index(champion[index]) + 1:
        raise RuntimeError(f"v37 challenger is not one adjacent promotion: {task}")
    value["champion_schedule"] = champion
    value["challenger_schedule"] = challenger
    value["bundle_sha256"] = file_sha256(path)
    return value


class SearchLedger:
    def __init__(self, runtime, root: Path):
        self.runtime = runtime
        self.root = root

    def used(self) -> int:
        return len(list((self.root / "states").glob("*/*.json")))

    def one(self, schedule, seed: int, role: str) -> dict:
        schedule = list(v32.validate_schedule(schedule))
        path = self.root / "states" / schedule_sha256(schedule) / f"{seed}.json"
        if path.exists():
            record = v33.checked_json(path)
            if int(record.get("seed", -1)) != int(seed):
                raise RuntimeError("v37 cached search seed differs")
            if list(map(float, record.get("schedule", ()))) != schedule:
                raise RuntimeError("v37 cached search schedule differs")
            return record
        if self.used() >= SEARCH_BUDGET:
            raise RuntimeError("v37 exact-25 search budget exhausted")
        record = self.runtime.rollout(
            schedule, int(seed), record_attribution_telemetry=True
        )
        if int(record.get("seed", -1)) != int(seed):
            raise RuntimeError("v37 runtime returned a different seed")
        if list(map(float, record.get("schedule", ()))) != schedule:
            raise RuntimeError("v37 runtime returned a different schedule")
        record["search_role"] = role
        immutable_json(path, record)
        print(
            json.dumps(
                {
                    "stage": "search",
                    "role": role,
                    "schedule": schedule,
                    "seed": int(seed),
                    "success": successful(record),
                    "search_rollouts_used": self.used(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return record

    def report(self, schedule, seeds: list[int], role: str) -> tuple[dict, list[dict]]:
        schedule = list(v32.validate_schedule(schedule))
        records = [self.one(schedule, seed, role) for seed in seeds]
        report = {
            "role": role,
            "schedule": schedule,
            "schedule_sha256": schedule_sha256(schedule),
            "seed_order": list(map(int, seeds)),
            "summary": summarize(records),
        }
        immutable_json(self.root / "reports" / f"{role}.json", report)
        return report, records

    def incidents(self) -> dict:
        records = [
            v33.checked_json(path) for path in (self.root / "states").glob("*/*.json")
        ]
        return {
            "physics_errors": sum(
                item.get("physics_error") is not None for item in records
            ),
            "safety_violations": sum(
                item.get("safety_violation") is not None for item in records
            ),
        }


def metric_steps(record: dict) -> int:
    first = record.get("first_success_step")
    return int(record["physics_steps"] if first is None else first)


def paired_receipt(
    challenger_records: list[dict], champion_records: list[dict]
) -> dict:
    if [item["seed"] for item in challenger_records] != [
        item["seed"] for item in champion_records
    ]:
        raise RuntimeError("v37 paired seed order differs")
    pairs = list(zip(challenger_records, champion_records))
    common = [
        (left, right) for left, right in pairs if successful(left) and successful(right)
    ]
    challenger_summary = summarize(challenger_records)
    champion_summary = summarize(champion_records)
    champion_throughput = float(champion_summary["achieved_throughput_per_step"])
    challenger_throughput = float(challenger_summary["achieved_throughput_per_step"])
    return {
        "pairs": len(pairs),
        "challenger_successes": int(challenger_summary["successes"]),
        "champion_successes": int(champion_summary["successes"]),
        "challenger_only_success": sum(
            successful(left) and not successful(right) for left, right in pairs
        ),
        "champion_only_success": sum(
            not successful(left) and successful(right) for left, right in pairs
        ),
        "both_success": len(common),
        "challenger_failure_aware_throughput": challenger_throughput,
        "champion_failure_aware_throughput": champion_throughput,
        "challenger_throughput_ratio": (
            None
            if champion_throughput <= 0
            else challenger_throughput / champion_throughput
        ),
        "challenger_speedup_on_common_success": (
            None
            if not common
            else (sum(metric_steps(right) for _, right in common) / len(common))
            / (sum(metric_steps(left) for left, _ in common) / len(common))
        ),
    }


def run_search(ledger: SearchLedger, task: str, spec: dict, incumbent: dict) -> dict:
    champion = incumbent["champion_schedule"]
    challenger = incumbent["challenger_schedule"]
    discovery, _discovery_records = ledger.report(
        challenger,
        list(map(int, spec["challenger_discovery"])),
        "challenger_discovery",
    )
    champion_paired, champion_records = ledger.report(
        champion,
        list(map(int, spec["paired"])),
        "champion_paired",
    )
    challenger_paired, challenger_records = ledger.report(
        challenger,
        list(map(int, spec["paired"])),
        "challenger_paired",
    )
    if ledger.used() != SEARCH_BUDGET:
        raise RuntimeError(f"v37 search used {ledger.used()}, expected exactly 25")
    pair = paired_receipt(challenger_records, champion_records)
    challenger_incidents = {
        "physics_errors": int(discovery["summary"]["physics_errors"])
        + int(challenger_paired["summary"]["physics_errors"]),
        "safety_violations": int(discovery["summary"]["safety_violations"])
        + int(challenger_paired["summary"]["safety_violations"]),
    }
    champion_qualified = (
        champion_paired["summary"]["successes"] >= PAIRED_SUCCESS_FLOOR
        and champion_paired["summary"]["physics_errors"] == 0
        and champion_paired["summary"]["safety_violations"] == 0
    )
    challenger_qualified = (
        discovery["summary"]["successes"] >= DISCOVERY_SUCCESS_FLOOR
        and challenger_paired["summary"]["successes"] >= PAIRED_SUCCESS_FLOOR
        and challenger_incidents == {"physics_errors": 0, "safety_violations": 0}
    )
    challenger_dominates = (
        challenger_qualified
        and pair["challenger_successes"] >= pair["champion_successes"]
        and pair["challenger_throughput_ratio"] is not None
        and pair["challenger_throughput_ratio"] >= THROUGHPUT_PROMOTION_RATIO
    )
    if challenger_qualified and (not champion_qualified or challenger_dominates):
        selected = challenger
        status = "challenger_promoted"
    elif champion_qualified:
        selected = champion
        status = "champion_retained"
    else:
        selected = [1.0] * len(v32.PHASES)
        status = "native_fallback"
    incidents = ledger.incidents()
    return {
        "schema": "act-champion-challenger25-selection-v37",
        "task_label": task,
        "champion_provenance": incumbent,
        "challenger_discovery": discovery,
        "champion_paired": champion_paired,
        "challenger_paired": challenger_paired,
        "paired_receipt": pair,
        "selection_rule": {
            "challenger_discovery_success_floor": DISCOVERY_SUCCESS_FLOOR,
            "paired_success_floor": PAIRED_SUCCESS_FLOOR,
            "throughput_promotion_ratio": THROUGHPUT_PROMOTION_RATIO,
            "ambiguous_result": "retain_accelerated_champion",
            "native_fallback": "only_when_neither_controller_reaches_paired_floor",
        },
        "champion_qualified": champion_qualified,
        "challenger_qualified": challenger_qualified,
        "challenger_dominates": challenger_dominates,
        "selection_status": status,
        "selected_schedule": selected,
        "selected_schedule_sha256": schedule_sha256(selected),
        "search_scientific_rollouts": ledger.used(),
        "incident_totals": incidents,
        "historical_speed_outcomes_used_for_initialization": True,
        "historical_rollouts_reexecuted": 0,
        "final_bank_opened": False,
    }


def require_all_search(root: Path) -> None:
    for task in TASKS:
        complete = v33.checked_json(root / "search" / task / "SEARCH_COMPLETE.json")
        if int(complete.get("search_scientific_rollouts", -1)) != SEARCH_BUDGET:
            raise RuntimeError(f"v37 search incomplete: {task}")


def method_schedule(
    root: Path, champions_path: Path, task: str, method: str
) -> tuple[list[float], str]:
    if method == "native_1x":
        return [1.0] * len(v32.PHASES), "preregistered_native"
    if method == "champion":
        incumbent = load_champion(champions_path, task)
        return incumbent["champion_schedule"], incumbent["bundle_sha256"]
    selection_path = root / "search" / task / "SELECTION.json"
    selection = v33.checked_json(selection_path)
    schedule = list(v32.validate_schedule(selection["selected_schedule"]))
    if schedule_sha256(schedule) != selection["selected_schedule_sha256"]:
        raise RuntimeError("v37 selected schedule hash mismatch")
    return schedule, file_sha256(selection_path)


def load_final_states(
    directory: Path, seeds: list[int], identity_sha: str
) -> list[dict]:
    records = []
    missing = False
    for seed in seeds:
        path = directory / "states" / f"{seed}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError("v37 final states contain a non-contiguous suffix")
        record = v33.checked_json(path)
        if (
            int(record.get("seed", -1)) != seed
            or record.get("identity_sha256") != identity_sha
        ):
            raise RuntimeError(f"v37 final state identity mismatch: {path}")
        records.append(record)
    return records


def run_final(
    runtime,
    root: Path,
    champions_path: Path,
    task: str,
    method: str,
    seeds: list[int],
    banks_sha: str,
) -> None:
    require_all_search(root)
    schedule, provenance = method_schedule(root, champions_path, task, method)
    controller_hash = schedule_sha256(schedule)
    controller_root = root / "final" / task / "controllers" / controller_hash
    alias_path = root / "final" / task / "methods" / method / "RESULT.json"
    if alias_path.exists():
        alias = v33.checked_json(alias_path)
        complete = v33.checked_json(controller_root / "COMPLETE.json")
        if alias.get("controller_sha256") != controller_hash:
            raise RuntimeError(f"v37 method alias differs: {task}/{method}")
        if alias.get("controller_result_sha256") != complete.get("result_sha256"):
            raise RuntimeError(f"v37 method completion differs: {task}/{method}")
        return
    identity = {
        **runtime.identity(),
        "schema": "act-champion-challenger-final-controller-identity-v37",
        "task_label": task,
        "schedule": schedule,
        "schedule_sha256": controller_hash,
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "banks_sha256": banks_sha,
        "search_or_tuning_permitted": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    v33.immutable_or_verify(controller_root / "IDENTITY.json", identity)
    was_complete = (controller_root / "COMPLETE.json").exists()
    records = load_final_states(controller_root, seeds, identity["identity_sha256"])
    for seed in seeds[len(records) :]:
        record = runtime.rollout(schedule, seed, record_attribution_telemetry=False)
        if (
            int(record.get("seed", -1)) != seed
            or list(map(float, record.get("schedule", ()))) != schedule
        ):
            raise RuntimeError("v37 final runtime returned a different controller")
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
        "schema": "act-champion-challenger-final-controller-result-v37",
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
            "schema": "act-champion-challenger-final-controller-completion-v37",
            "episodes": len(records),
            "result_sha256": file_sha256(controller_root / "RESULT.json"),
            "physics_errors": result["summary"]["physics_errors"],
            "safety_violations": result["summary"]["safety_violations"],
        },
    )
    v33.immutable_or_verify(
        alias_path,
        {
            "schema": "act-champion-challenger-final-method-result-v37",
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


def validate_banks(banks: dict) -> None:
    all_seeds = []
    for task in TASKS:
        spec = banks["tasks"][task]
        if len(spec["challenger_discovery"]) != DISCOVERY_EPISODES:
            raise RuntimeError("v37 banks require five challenger discovery seeds")
        if len(spec["paired"]) != PAIRED_EPISODES or len(spec["final"]) != 50:
            raise RuntimeError("v37 banks require ten paired and fifty final seeds")
        task_seeds = spec["challenger_discovery"] + spec["paired"] + spec["final"]
        if len(task_seeds) != len(set(task_seeds)):
            raise RuntimeError(f"v37 task banks overlap: {task}")
        all_seeds.extend(task_seeds)
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("v37 cross-task banks overlap")


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "final"), required=True)
    parser.add_argument(
        "--method", choices=(SEARCH_METHOD, *FINAL_METHODS), required=True
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-source-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=TASKS, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--champions", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if git_head() != args.implementation_commit:
        raise RuntimeError("v37 checked-out source differs from implementation commit")
    banks = v33.checked_json(args.banks)
    validate_banks(banks)
    incumbent = load_champion(args.champions, args.task_label)
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
            raise ValueError("v37 final stage requires a final method")
        run_final(
            runtime,
            root,
            args.champions,
            args.task_label,
            args.method,
            list(map(int, spec["final"])),
            file_sha256(args.banks),
        )
        return 0
    if args.method != SEARCH_METHOD:
        raise ValueError("v37 search stage requires champion_challenger")
    output = root / "search" / args.task_label
    identity = {
        **runtime.identity(),
        "schema": "act-champion-challenger-search-identity-v37",
        "contract_sha256": file_sha256(args.contract),
        "banks_sha256": file_sha256(args.banks),
        "champions_sha256": file_sha256(args.champions),
        "task_label": args.task_label,
        "search_budget": SEARCH_BUDGET,
        "challenger_discovery_seeds": spec["challenger_discovery"],
        "paired_seeds": spec["paired"],
        "final_seeds_registered_unopened": spec["final"],
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
            raise RuntimeError("v37 completed selection hash mismatch")
        return 0
    ledger = SearchLedger(runtime, output / "search")
    selection = run_search(ledger, args.task_label, spec, incumbent)
    immutable_json(selection_path, selection)
    immutable_json(
        complete_path,
        {
            "schema": "act-champion-challenger-search-completion-v37",
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
