#!/usr/bin/env python3
"""Create the immutable source, artifact, detector, and seed manifest for ACT speed v1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

from run_act_speed_parity import PARITY, canonical_sha256, immutable_json, sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
DETECTOR_HASHES = {
    "checkpoint": "c25c3f530da42eb7c60e5f70405b3a99c56ab72c1e53dfd27055dc3d99c3512d",
    "inference": "1398e1d1b5b4e682f009c6501598e651a516341f6d60822f40fc575a40061815",
    "model_source": "8a47f110f19f4e52a39b7e0e4f2273c2895690f6332ab17a4b71c8eb5ce4ae37",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--parity-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if git("rev-parse", "HEAD") != args.source_commit:
        raise RuntimeError("checked-out commit does not match requested manifest source")
    if git("status", "--porcelain"):
        raise RuntimeError("manifest source worktree is not clean")
    contract = json.loads(args.contract.read_text())
    if contract.get("schema") != "act-speed-benchmark-v1":
        raise RuntimeError("unexpected benchmark contract schema")

    detector_paths = {
        "checkpoint": args.detector_checkpoint,
        "inference": args.detector_source / "phase_detector/rgb_inference.py",
        "model_source": args.detector_source / "phase_detector/rgb_proprio.py",
    }
    detector_actual = {name: sha256(path) for name, path in detector_paths.items()}
    if detector_actual != DETECTOR_HASHES:
        raise RuntimeError(f"learned detector hash mismatch: {detector_actual}")
    if contract["learned_phase_detector"]["sha256"] != detector_actual["checkpoint"]:
        raise RuntimeError("contract learned-detector hash mismatch")

    tracked = git("ls-files").splitlines()
    source_files = {
        path: sha256(REPO_ROOT / path)
        for path in tracked
        if (REPO_ROOT / path).is_file()
    }
    parity_attempt = args.benchmark_root / "attempts" / args.parity_source_commit
    parity_manifest_path = parity_attempt / "run_manifest.json"
    if not parity_manifest_path.exists():
        raise RuntimeError("passed parity source manifest is missing")
    parity_manifest = json.loads(parity_manifest_path.read_text())
    critical_sources = (
        "act_integration.py",
        "detr/models/backbone.py",
        "detr/models/detr_vae.py",
        "detr/models/transformer.py",
        "detr/util/misc.py",
        "original_act.py",
        "policy.py",
        "policy_speed_env.py",
        "sim_env.py",
        "sim_tasks.py",
        "speed_policy.py",
    )
    parity_tracked = parity_manifest["source"]["tracked_file_sha256"]
    critical_hashes = {}
    for path in critical_sources:
        current = source_files.get(path)
        previous = parity_tracked.get(path)
        if current is None or current != previous:
            raise RuntimeError(
                f"method source changes parity-critical file {path}: {current} != {previous}"
            )
        critical_hashes[path] = current

    parity_gate = {}
    for label, expected in (("pick", 49), ("tea", 50), ("insertion", 49)):
        root = parity_attempt / "parity" / label
        for name in ("identity.json", "result.json", "COMPLETE.json"):
            if not (root / name).exists():
                raise RuntimeError(f"missing passed parity receipt: {root / name}")
        result = json.loads((root / "result.json").read_text())
        marker = json.loads((root / "COMPLETE.json").read_text())
        if not (
            result.get("parity_passed") is True
            and result.get("successes") == expected
            and result.get("episodes") == 50
            and result.get("safety_violations") == 0
            and result.get("physics_errors") == 0
            and marker.get("parity_passed") is True
        ):
            raise RuntimeError(f"{label} does not carry the exact passed parity gate")
        parity_gate[label] = {
            "successes": expected,
            "episodes": 50,
            "identity_sha256": result["identity_sha256"],
            "identity_path": str(root / "identity.json"),
            "identity_file_sha256": sha256(root / "identity.json"),
            "result_path": str(root / "result.json"),
            "result_file_sha256": sha256(root / "result.json"),
            "completion_path": str(root / "COMPLETE.json"),
            "completion_file_sha256": sha256(root / "COMPLETE.json"),
        }
    tasks = {}
    for label, task in contract["tasks"].items():
        parity = PARITY[label]
        root = Path(task["root"])
        artifacts = {
            "policy_best.ckpt": sha256(root / "checkpoints/policy_best.ckpt"),
            "dataset_stats.pkl": sha256(root / "checkpoints/dataset_stats.pkl"),
            "policy_config.json": sha256(root / "checkpoints/policy_config.json"),
        }
        expected = {
            "policy_best.ckpt": parity["policy_checkpoint_sha256"],
            "dataset_stats.pkl": parity["dataset_stats_sha256"],
            "policy_config.json": parity["policy_config_sha256"],
        }
        if artifacts != expected:
            raise RuntimeError(f"{label} frozen ACT artifact hash mismatch: {artifacts}")
        search_seeds = list(range(task["search_seed_base"], task["search_seed_base"] + 50))
        final_seeds = list(range(task["final_seed_base"], task["final_seed_base"] + 50))
        parity_seeds = list(range(parity["seed_base"], parity["seed_base"] + 50))
        tasks[label] = {
            "task": task["task"],
            "policy_root": str(root),
            "artifacts": artifacts,
            "accepted_parity": {
                "expected_successes": parity["expected_successes"],
                "seeds": parity_seeds,
                "seed_bank_sha256": canonical_sha256(parity_seeds),
            },
            "search_bank": {"seeds": search_seeds, "sha256": canonical_sha256(search_seeds)},
            "final_bank": {"seeds": final_seeds, "sha256": canonical_sha256(final_seeds)},
        }

    packages = {}
    for name in ("numpy", "torch", "torchvision", "dm-control", "mujoco"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "schema": "act-speed-benchmark-run-manifest-v1",
        "source": {
            "repository": "https://github.com/kevin-thankyou-lin/SpeedTuning.git",
            "commit": args.source_commit,
            "tree": git("rev-parse", "HEAD^{tree}"),
            "tracked_file_sha256": source_files,
        },
        "contract": {
            "path": str(args.contract),
            "sha256": sha256(args.contract),
            "payload": contract,
        },
        "runtime": {"python": platform.python_version(), "packages": packages},
        "learned_phase_detector": {
            "canonical_checkpoint": contract["learned_phase_detector"]["checkpoint"],
            "staged_checkpoint": str(args.detector_checkpoint),
            "staged_source": str(args.detector_source),
            "sha256": detector_actual,
            "inputs": contract["learned_phase_detector"]["inputs"],
            "postprocessing": contract["learned_phase_detector"]["postprocessing"],
        },
        "parity_gate": {
            "passed": True,
            "source_commit": args.parity_source_commit,
            "source_manifest_path": str(parity_manifest_path),
            "source_manifest_sha256": sha256(parity_manifest_path),
            "critical_source_sha256": critical_hashes,
            "tasks": parity_gate,
        },
        "tasks": tasks,
        "rollout_accounting": {
            "parity_is_engineering_not_search": True,
            "search_rollouts_per_task_method": 50,
            "final_rollouts_per_task_method": 50,
            "shared_native_final_rollouts_per_task": 50,
            "physics_errors": "count_as_failure_and_continue",
            "resume": "requires_exact_identity_match",
            "progress": "atomic_per_state",
        },
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    immutable_json(args.output, manifest)
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
