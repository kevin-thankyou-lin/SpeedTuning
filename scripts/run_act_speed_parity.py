#!/usr/bin/env python3
"""Run one resumable uniform-1x ACT parity lane on its accepted 50-state bank."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_integration import build_original_act_speed_adapter  # noqa: E402
from one_reset_phase_schedule import workspace_violation  # noqa: E402
from original_act import set_seed  # noqa: E402
from policy_speed_env import create_speed_env  # noqa: E402
from speed_policy import FixedSpeedPolicy, rollout_speed_policy, summarize_rollouts  # noqa: E402


PARITY = {
    "pick": {
        "seed_base": 9_150_000,
        "expected_successes": 49,
        "policy_checkpoint_sha256": "01f73838acd4c50b4b0db815f2ae9c845d343fb7f00983ee30736d13f34dbd89",
        "dataset_stats_sha256": "1aa06430677e631c6aabb082f6c27b21cba4d287a4e49fb48379e8ad206299c8",
        "policy_config_sha256": "994e00f5d8ba6f26d7ef067d2819470d551b087b407248f737c723230936b180",
    },
    "tea": {
        "seed_base": 9_250_000,
        "expected_successes": 50,
        "policy_checkpoint_sha256": "f6ed29c07bd4a840fd05ca0b6308c729d81ed3d703ec4f9a29f12a3b0504f596",
        "dataset_stats_sha256": "6f6a9e2e8a75a3194e3215200e575da2cda296e56413ae57f0c5be24c678cae0",
        "policy_config_sha256": "994e00f5d8ba6f26d7ef067d2819470d551b087b407248f737c723230936b180",
    },
    "insertion": {
        "seed_base": 9_350_000,
        "expected_successes": 49,
        "policy_checkpoint_sha256": "013ae8dfb88383fb3ed01498285d82a35dd19de1d12ad0ddeb3758151907e0ca",
        "dataset_stats_sha256": "35ef807f30ba564f713a326b93a5c6b1e7200a2bb3c759b1529621d5f0c3222a",
        "policy_config_sha256": "994e00f5d8ba6f26d7ef067d2819470d551b087b407248f737c723230936b180",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def immutable_json(path: Path, value) -> None:
    """Atomically publish a file without ever replacing an existing receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checked_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"artifact hash mismatch for {path}: {actual} != {expected}")
    return actual


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def load_completed_states(states_dir: Path, seeds: list[int], identity_hash: str) -> list[dict]:
    records = []
    missing_seen = False
    for seed in seeds:
        path = states_dir / f"{seed}.json"
        if not path.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise RuntimeError("resume state contains a non-contiguous seed suffix")
        record = json.loads(path.read_text())
        if record.get("seed") != seed or record.get("identity_sha256") != identity_hash:
            raise RuntimeError(f"resume identity mismatch in {path}")
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=tuple(PARITY), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if git_head() != args.source_commit:
        raise RuntimeError(f"source commit mismatch: {git_head()} != {args.source_commit}")
    contract = json.loads(args.contract.read_text())
    if contract.get("schema") != "act-speed-benchmark-v1":
        raise RuntimeError("unexpected benchmark contract schema")
    manifest = json.loads(args.run_manifest.read_text())
    if manifest.get("source", {}).get("commit") != args.source_commit:
        raise RuntimeError("run manifest source identity mismatch")
    if manifest.get("contract", {}).get("sha256") != sha256(args.contract):
        raise RuntimeError("run manifest contract identity mismatch")

    label = args.task_label
    task = contract["tasks"][label]
    parity = PARITY[label]
    root = Path(task["root"])
    checkpoint = root / "checkpoints/policy_best.ckpt"
    stats = root / "checkpoints/dataset_stats.pkl"
    config = root / "checkpoints/policy_config.json"
    artifact_hashes = {
        "policy_best.ckpt": checked_hash(checkpoint, parity["policy_checkpoint_sha256"]),
        "dataset_stats.pkl": checked_hash(stats, parity["dataset_stats_sha256"]),
        "policy_config.json": checked_hash(config, parity["policy_config_sha256"]),
    }
    policy_config = json.loads(config.read_text())
    if policy_config.get("camera_names") != ["angle", "left_wrist", "right_wrist"]:
        raise RuntimeError("frozen ACT camera order mismatch")
    if policy_config.get("qpos_dim") != 15 or policy_config.get("num_queries") != 100:
        raise RuntimeError("frozen ACT qpos/chunk contract mismatch")

    seeds = list(range(parity["seed_base"], parity["seed_base"] + 50))
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_stream = (output / ".lane.lock").open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another healthy parity process owns {output}") from exc

    identity = {
        "schema": "act-speed-parity-identity-v1",
        "task_label": label,
        "task": task["task"],
        "source_commit": args.source_commit,
        "contract_sha256": sha256(args.contract),
        "run_manifest_sha256": sha256(args.run_manifest),
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "expected_successes": parity["expected_successes"],
        "device": args.device,
        "policy_root": str(root),
        "artifacts": artifact_hashes,
        "controller": {
            "speed": 1.0,
            "cameras": ["angle", "left_wrist", "right_wrist"],
            "qpos_dim": 15,
            "action_dim": 14,
            "chunk_horizon": 100,
            "progress_clock": "nominal_policy_time",
            "per_physics_step_inference": True,
            "temporal_ensemble": {"enabled": True, "m": 0.01},
            "safety_monitor": "one_reset_phase_schedule.workspace_violation_every_physics_tick",
            "physics_error_policy": "count_as_failure_and_continue_bank",
        },
    }
    identity_hash = canonical_sha256(identity)
    identity["identity_sha256"] = identity_hash
    identity_path = output / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("output root has a different parity identity")
    else:
        immutable_json(identity_path, identity)

    states_dir = output / "states"
    records = load_completed_states(states_dir, seeds, identity_hash)
    if (output / "COMPLETE.json").exists():
        if len(records) != 50:
            raise RuntimeError("completion marker exists without 50 state receipts")
        print(json.dumps(json.loads((output / "result.json").read_text()), sort_keys=True))
        return 0

    set_seed(1000)
    adapter = build_original_act_speed_adapter(
        task_name=task["task"],
        checkpoint=checkpoint,
        stats_path=stats,
        policy_config_path=config,
        temporal_ensemble_m=0.01,
        device=args.device,
    )
    for seed in seeds[len(records):]:
        env = create_speed_env(
            task_name=task["task"],
            chunk_predictor=adapter,
            seed=seed,
            randomize_object_pose=True,
            speed_values=(1.0,),
            decision_frame_skip=10,
            terminate_on_success=False,
            safety_monitor=partial(workspace_violation, task["task"]),
        )
        try:
            # SpeedPolicyEnv converts only dm_control PhysicsError into a failed
            # state receipt.  Any other runner exception escapes and stops the
            # lane so an unclassified fault is never silently counted as physics.
            record = rollout_speed_policy(env, FixedSpeedPolicy(1.0), frame_skip=10)
        finally:
            env.close()
        record.update(seed=seed, identity_sha256=identity_hash)
        immutable_json(states_dir / f"{seed}.json", record)
        records.append(record)
        atomic_json(
            output / "progress.json",
            {
                "schema": "act-speed-parity-progress-v1",
                "identity_sha256": identity_hash,
                "completed": len(records),
                "successes": sum(bool(item["success"]) for item in records),
                "safety_violations": sum(item.get("safety_violation") is not None for item in records),
                "physics_errors": sum("physics_error" in item for item in records),
                "next_seed": None if len(records) == 50 else seeds[len(records)],
            },
        )
        print(
            json.dumps(
                {
                    "task": label,
                    "completed": len(records),
                    "successes": sum(bool(item["success"]) for item in records),
                }
            ),
            flush=True,
        )

    summary = summarize_rollouts(records)
    summary.update(
        schema="act-speed-uniform-1x-parity-v1",
        task_label=label,
        task=task["task"],
        identity_sha256=identity_hash,
        expected_successes=parity["expected_successes"],
        parity_passed=(
            summary["successes"] == parity["expected_successes"]
            and summary["safety_violations"] == 0
            and summary["physics_errors"] == 0
        ),
        manifest_path=str(args.run_manifest),
        states_path=str(states_dir),
    )
    immutable_json(output / "result.json", summary)
    marker = {
        "schema": "act-speed-parity-completion-v1",
        "identity_sha256": identity_hash,
        "result_sha256": sha256(output / "result.json"),
        "episodes": 50,
        "successes": summary["successes"],
        "parity_passed": summary["parity_passed"],
    }
    immutable_json(
        output / ("COMPLETE.json" if summary["parity_passed"] else "PARITY_FAILED.json"),
        marker,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["parity_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
