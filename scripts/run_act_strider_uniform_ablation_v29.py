#!/usr/bin/env python3
"""Replay the missing fixed-uniform comparators on STRIDER v28's paired bank."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import canonical_sha256, sha256
from scripts.act_vlm_frontier_server import ACTFrontierRuntime
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json
from scripts.run_act_strider_frontier_v4 import schedule_sha256, summarize

TASKS = ("pick", "tea")
UNIFORM = [2.0] * 4


def checked(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing sealed input: {path}")
    return json.loads(path.read_text())


def immutable_or_equal(path: Path, value: dict) -> None:
    if path.exists():
        if checked(path) != value:
            raise RuntimeError(f"sealed receipt differs: {path}")
    else:
        immutable_json(path, value)


def valid_success(record: dict) -> bool:
    return (
        bool(record.get("success"))
        and record.get("physics_error") is None
        and record.get("safety_violation") is None
    )


def paired_summary(strider: list[dict], uniform: list[dict]) -> dict:
    if len(strider) != len(uniform) or not strider:
        raise ValueError("paired records must be nonempty and equal length")
    pairs = []
    both_success_steps = []
    counts = {"both_success": 0, "strider_only": 0, "uniform_only": 0, "both_fail": 0}
    for left, right in zip(strider, uniform):
        if int(left["seed"]) != int(right["seed"]):
            raise RuntimeError("paired seed mismatch")
        ls, rs = valid_success(left), valid_success(right)
        key = (
            "both_success" if ls and rs else
            "strider_only" if ls else
            "uniform_only" if rs else
            "both_fail"
        )
        counts[key] += 1
        if ls and rs:
            left_steps = int(left["first_success_step"])
            right_steps = int(right["first_success_step"])
            both_success_steps.append({
                "seed": int(left["seed"]),
                "strider_steps": left_steps,
                "uniform_steps": right_steps,
                "uniform_over_strider_step_ratio": right_steps / left_steps,
            })
        pairs.append({"seed": int(left["seed"]), "strider_success": ls, "uniform_success": rs})
    return {
        "episodes": len(strider),
        "success_contingency": counts,
        "success_delta_strider_minus_uniform": counts["strider_only"] - counts["uniform_only"],
        "both_success_step_pairs": both_success_steps,
        "pair_outcomes": pairs,
    }


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--v26-root", type=Path, required=True)
    parser.add_argument("--v28-root", type=Path, required=True)
    parser.add_argument("--task-label", choices=TASKS, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    contract = checked(args.contract)
    if contract.get("schema") != "act-strider-uniform-ablation-contract-v29":
        raise RuntimeError("unexpected ablation contract")
    comparison = contract["comparisons"][args.task_label]
    if comparison["uniform_schedule"] != UNIFORM or comparison["new_rollouts"] != 50:
        raise RuntimeError("task is not a registered missing uniform comparator")

    v26_manifest_path = args.v26_root / "RUN_MANIFEST.json"
    v26_complete_path = args.v26_root / "COMPLETE.json"
    v26_manifest = checked(v26_manifest_path)
    v26_complete = checked(v26_complete_path)
    if v26_complete.get("new_final_rollouts") != 300 or v26_complete.get("simulator_invalid_attempts") != 0:
        raise RuntimeError("v26 bank is not cleanly sealed")

    v28_aggregate_path = args.v28_root / "RESULT.json"
    v28_complete_path = args.v28_root / "COMPLETE.json"
    v28_aggregate = checked(v28_aggregate_path)
    v28_complete = checked(v28_complete_path)
    if v28_complete.get("result_sha256") != sha256(v28_aggregate_path):
        raise RuntimeError("v28 aggregate hash mismatch")
    v28_task = args.v28_root / args.task_label
    v28_task_result_path = v28_task / "RESULT.json"
    v28_task_complete = checked(v28_task / "COMPLETE.json")
    v28_task_result = checked(v28_task_result_path)
    if v28_task_complete.get("result_sha256") != sha256(v28_task_result_path):
        raise RuntimeError("v28 task result hash mismatch")
    if v28_task_result.get("schedule") != comparison["strider_schedule"]:
        raise RuntimeError("registered STRIDER schedule differs from sealed v28 selection")
    if v28_aggregate["tasks"][args.task_label]["result_sha256"] != sha256(v28_task_result_path):
        raise RuntimeError("v28 task is not the one sealed by its aggregate")

    runtime = ACTFrontierRuntime(
        source_commit=args.source_commit,
        run_manifest=v26_manifest_path,
        task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
    )
    root = args.root.resolve() / args.task_label
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".lane.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another process owns this ablation task") from exc

    seeds = list(v26_manifest["tasks"][args.task_label]["final_bank"]["seeds"])
    identity = {
        **runtime.identity(),
        "schema": "act-strider-uniform-ablation-identity-v29",
        "method": "fixed_uniform_backbone_ablation",
        "task_label": args.task_label,
        "uniform_schedule": UNIFORM,
        "uniform_schedule_sha256": schedule_sha256(UNIFORM),
        "strider_schedule": comparison["strider_schedule"],
        "contract_sha256": sha256(args.contract),
        "v26_manifest_sha256": sha256(v26_manifest_path),
        "v26_completion_sha256": sha256(v26_complete_path),
        "v28_aggregate_sha256": sha256(v28_aggregate_path),
        "v28_task_result_sha256": sha256(v28_task_result_path),
        "final_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "uses_already_opened_final_bank": True,
        "new_rollout_budget": 50,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    identity_path = root / "IDENTITY.json"
    immutable_or_equal(identity_path, identity)
    complete_path = root / "COMPLETE.json"
    if complete_path.exists():
        complete = checked(complete_path)
        if complete.get("identity_sha256") != sha256(identity_path):
            raise RuntimeError("completed identity hash mismatch")
        if complete.get("result_sha256") != sha256(root / "RESULT.json"):
            raise RuntimeError("completed result hash mismatch")
        print(json.dumps(checked(root / "RESULT.json"), sort_keys=True))
        return 0

    states = root / "uniform" / "states"
    uniform_records = []
    missing = False
    for seed in seeds:
        path = states / f"{seed}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError("non-contiguous uniform states")
        value = checked(path)
        if value.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError("uniform state identity mismatch")
        uniform_records.append(value)
    for seed in seeds[len(uniform_records):]:
        value = runtime.rollout(UNIFORM, seed, record_attribution_telemetry=False)
        value["identity_sha256"] = identity["identity_sha256"]
        immutable_json(states / f"{seed}.json", value)
        uniform_records.append(value)
        atomic_json(root / "progress.json", {
            "completed": len(uniform_records),
            "successes": sum(valid_success(item) for item in uniform_records),
            "physics_errors": sum(item.get("physics_error") is not None for item in uniform_records),
            "safety_violations": sum(item.get("safety_violation") is not None for item in uniform_records),
            "new_rollouts": len(uniform_records),
        })
        print(json.dumps({
            "task": args.task_label,
            "completed": len(uniform_records),
            "successes": sum(valid_success(item) for item in uniform_records),
        }), flush=True)

    strider_identity = checked(v28_task / "IDENTITY.json")
    strider_records = []
    for seed in seeds:
        path = v28_task / "final" / "states" / f"{seed}.json"
        value = checked(path)
        if value.get("identity_sha256") != strider_identity["identity_sha256"]:
            raise RuntimeError("v28 STRIDER state identity mismatch")
        strider_records.append(value)
    result = {
        "schema": "act-strider-uniform-ablation-result-v29",
        "task_label": args.task_label,
        "strider_schedule": comparison["strider_schedule"],
        "uniform_schedule": UNIFORM,
        "strider_summary": summarize(strider_records),
        "uniform_summary": summarize(uniform_records),
        "paired": paired_summary(strider_records, uniform_records),
        "episodes_per_controller": 50,
        "new_rollouts": 50,
    }
    result_path = root / "RESULT.json"
    immutable_or_equal(result_path, result)
    immutable_or_equal(complete_path, {
        "schema": "act-strider-uniform-ablation-completion-v29",
        "identity_sha256": sha256(identity_path),
        "result_sha256": sha256(result_path),
        "new_rollouts": 50,
        "v20_v26_v27_v28_rollouts_reexecuted": 0,
        "physics_errors": result["uniform_summary"]["physics_errors"],
        "safety_violations": result["uniform_summary"]["safety_violations"],
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
