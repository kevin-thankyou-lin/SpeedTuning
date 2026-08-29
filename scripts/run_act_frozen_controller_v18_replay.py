#!/usr/bin/env python3
"""Replay three audited frozen speed controllers on the exact v18 final bank."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

METHOD_SOURCES = {
    "learned_phase_subtask": "866c9f436caf0a73e5e08ef83be38cbe89a23a61",
    "learned_phase_tabular_rl": "298c6d16784f228df0b1f455d0e41b4276ec5184",
    "learned_phase_rainbow_rl": "298c6d16784f228df0b1f455d0e41b4276ec5184",
}
TASKS = ("pick", "tea", "insertion")
V18_SCHEMA = "act-strider-representative-final-banks-v18"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def checked_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing required receipt: {path}")
    return json.loads(path.read_text())


def primary_seeds(banks: dict, task_label: str) -> list[int]:
    if banks.get("schema") != V18_SCHEMA:
        raise RuntimeError("unexpected v18 bank schema")
    if task_label not in TASKS:
        raise ValueError(f"unknown task: {task_label}")
    bank = banks["tasks"][task_label]["final_primary"]
    seeds = list(range(int(bank["start"]), int(bank["start"]) + int(bank["count"])))
    if len(seeds) != 50 or len(set(seeds)) != 50:
        raise RuntimeError("v19 requires exactly 50 unique v18 primary seeds")
    return seeds


def checked_selected(search_root: Path, method: str, task_label: str) -> dict:
    selected_path = search_root / "selected.json"
    complete_path = search_root / "COMPLETE.json"
    selected = checked_json(selected_path)
    complete = checked_json(complete_path)
    if selected.get("method") != method or selected.get("task_label") != task_label:
        raise RuntimeError("frozen selected artifact targets another replay cell")
    if complete.get("selected_sha256") != sha256(selected_path):
        raise RuntimeError("frozen selected artifact hash does not match completion receipt")
    if not selected.get("selected_policy"):
        raise RuntimeError("frozen selected artifact lacks a selected policy")
    return selected


def build_runtime_and_policy(args, source_commit: str):
    from act_integration import build_original_act_speed_adapter
    from act_speed_benchmark import preregistration
    from original_act import set_seed
    from scripts.run_act_speed_benchmark_cell import (
        DETECTOR_HASHES,
        CellRuntime,
        checked_hash,
        load_selected_policy,
    )

    contract = checked_json(args.benchmark_root / "contract.json")
    manifest_path = args.benchmark_root / "attempts" / source_commit / "run_manifest.json"
    manifest = checked_json(manifest_path)
    if manifest.get("source", {}).get("commit") != source_commit:
        raise RuntimeError("source manifest commit mismatch")
    if not manifest.get("parity_gate", {}).get("passed"):
        raise RuntimeError("source manifest lacks its passed parity gate")
    task = contract["tasks"][args.task_label]
    task_manifest = manifest["tasks"][args.task_label]
    task_root = Path(task["root"])
    artifact_paths = {
        "policy_best.ckpt": task_root / "checkpoints/policy_best.ckpt",
        "dataset_stats.pkl": task_root / "checkpoints/dataset_stats.pkl",
        "policy_config.json": task_root / "checkpoints/policy_config.json",
    }
    for name, path in artifact_paths.items():
        checked_hash(path, task_manifest["artifacts"][name])
    checked_hash(args.detector_checkpoint, DETECTOR_HASHES["checkpoint"])
    checked_hash(
        args.detector_source / "phase_detector/rgb_inference.py",
        DETECTOR_HASHES["inference"],
    )
    checked_hash(
        args.detector_source / "phase_detector/rgb_proprio.py",
        DETECTOR_HASHES["model_source"],
    )
    set_seed(1000)
    adapter = build_original_act_speed_adapter(
        task_name=task["task"],
        checkpoint=artifact_paths["policy_best.ckpt"],
        stats_path=artifact_paths["dataset_stats.pkl"],
        policy_config_path=artifact_paths["policy_config.json"],
        temporal_ensemble_m=0.01,
        device=args.device,
    )
    runtime_args = SimpleNamespace(
        method=args.method,
        task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
        run_manifest=manifest_path,
    )
    runtime = CellRuntime(
        runtime_args,
        contract,
        manifest,
        task,
        preregistration(args.method),
        adapter,
    )
    search_root = (
        args.benchmark_root / "runs" / source_commit / args.task_label / args.method / "search"
    )
    selected = checked_selected(search_root, args.method, args.task_label)
    policy, selected_path, _ = load_selected_policy(runtime, search_root, None)
    return runtime, policy, manifest_path, selected_path, selected, task_manifest


def main() -> int:
    from scripts.run_act_speed_benchmark_cell import (
        DETECTOR_HASHES,
        immutable_json,
        load_contiguous_states,
        method_source_hashes,
        progress,
        rollout_one,
    )
    from speed_policy import summarize_rollouts

    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--task-label", choices=TASKS, required=True)
    parser.add_argument("--method", choices=tuple(METHOD_SOURCES), required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = args.root.resolve() / args.task_label / args.method
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lane.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another process owns replay cell {output}") from exc

    banks = checked_json(args.banks)
    seeds = primary_seeds(banks, args.task_label)
    source = METHOD_SOURCES[args.method]
    runtime, policy, manifest_path, selected_path, selected, task_manifest = (
        build_runtime_and_policy(args, source)
    )
    identity = {
        "schema": "act-frozen-controller-v18-replay-identity-v19",
        "replay_source_commit": args.source_commit,
        "task_label": args.task_label,
        "method": args.method,
        "v18_primary_seeds": seeds,
        "v18_primary_seeds_sha256": canonical_sha256(seeds),
        "v18_banks_sha256": sha256(args.banks),
        "replay_contract_sha256": sha256(args.contract),
        "frozen_source_commit": source,
        "frozen_run_manifest": str(manifest_path),
        "frozen_run_manifest_sha256": sha256(manifest_path),
        "frozen_selected_path": str(selected_path),
        "frozen_selected_sha256": sha256(selected_path),
        "frozen_selected_policy": selected["selected_policy"],
        "policy_artifacts": task_manifest["artifacts"],
        "detector": DETECTOR_HASHES,
        "replay_method_source_sha256": method_source_hashes(),
        "training_or_selection_permitted": False,
        "rerun_v18_controllers": False,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    identity_path = output / "IDENTITY.json"
    if identity_path.exists() and checked_json(identity_path) != identity:
        raise RuntimeError("replay cell identity mismatch")
    if not identity_path.exists():
        immutable_json(identity_path, identity)

    states = output / "states"
    records = load_contiguous_states(states, seeds, identity["identity_sha256"])
    if (output / "COMPLETE.json").exists():
        if len(records) != 50:
            raise RuntimeError("completion receipt exists without 50 immutable states")
        print(json.dumps(checked_json(output / "RESULT.json"), sort_keys=True))
        return 0
    for seed in seeds[len(records) :]:
        record = rollout_one(runtime, seed, policy, identity["identity_sha256"])
        immutable_json(states / f"{seed}.json", record)
        records.append(record)
        progress(output, identity["identity_sha256"], records, seeds)
        print(
            json.dumps(
                {
                    "task": args.task_label,
                    "method": args.method,
                    "completed": len(records),
                    "successes": sum(bool(item["success"]) for item in records),
                }
            ),
            flush=True,
        )

    summary = summarize_rollouts(records)
    simulator_invalid = sum("physics_error" in item for item in records)
    result = {
        "schema": "act-frozen-controller-v18-replay-result-v19",
        "identity_sha256": identity["identity_sha256"],
        "task_label": args.task_label,
        "method": args.method,
        "episodes": 50,
        "exact_v18_primary_bank_complete": True,
        "paired_claim_valid": simulator_invalid == 0,
        "simulator_invalid_attempts": simulator_invalid,
        "summary": summary,
    }
    immutable_json(output / "RESULT.json", result)
    complete = {
        "schema": "act-frozen-controller-v18-replay-completion-v19",
        "identity_sha256": identity["identity_sha256"],
        "result_sha256": sha256(output / "RESULT.json"),
        "episodes": 50,
        "paired_claim_valid": result["paired_claim_valid"],
        "new_physical_attempts": 50,
    }
    immutable_json(output / "COMPLETE.json", complete)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
