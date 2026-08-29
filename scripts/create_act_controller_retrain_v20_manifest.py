#!/usr/bin/env python3
"""Create the hash-bound manifest for v20 Tabular/Rainbow reproductions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


EXPECTED_ARTIFACTS = {
    "pick": {
        "policy_best.ckpt": "01f73838acd4c50b4b0db815f2ae9c845d343fb7f00983ee30736d13f34dbd89",
        "dataset_stats.pkl": "1aa06430677e631c6aabb082f6c27b21cba4d287a4e49fb48379e8ad206299c8",
        "policy_config.json": "994e00f5d8ba6f26d7ef067d2819470d551b087b407248f737c723230936b180",
    },
    "tea": {
        "policy_best.ckpt": "f6ed29c07bd4a840fd05ca0b6308c729d81ed3d703ec4f9a29f12a3b0504f596",
        "dataset_stats.pkl": "6f6a9e2e8a75a3194e3215200e575da2cda296e56413ae57f0c5be24c678cae0",
        "policy_config.json": "994e00f5d8ba6f26d7ef067d2819470d551b087b407248f737c723230936b180",
    },
    "insertion": {
        "policy_best.ckpt": "013ae8dfb88383fb3ed01498285d82a35dd19de1d12ad0ddeb3758151907e0ca",
        "dataset_stats.pkl": "35ef807f30ba564f713a326b93a5c6b1e7200a2bb3c759b1529621d5f0c3222a",
        "policy_config.json": "994e00f5d8ba6f26d7ef067d2819470d551b087b407248f737c723230936b180",
    },
}
DETECTOR_HASHES = {
    "checkpoint": "c25c3f530da42eb7c60e5f70405b3a99c56ab72c1e53dfd27055dc3d99c3512d",
    "inference": "1398e1d1b5b4e682f009c6501598e651a516341f6d60822f40fc575a40061815",
    "model_source": "8a47f110f19f4e52a39b7e0e4f2273c2895690f6332ab17a4b71c8eb5ce4ae37",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def checked_v18_receipt(root: Path, task: str, final_seeds: list[int]) -> dict:
    task_root = root / task
    paths = {name: task_root / name for name in ("IDENTITY.json", "RESULT.json", "COMPLETE.json")}
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError(f"missing sealed v18 receipt for {task}")
    values = {name: json.loads(path.read_text()) for name, path in paths.items()}
    identity = values["IDENTITY.json"]
    result = values["RESULT.json"]
    complete = values["COMPLETE.json"]
    if identity.get("task_label") != task or result.get("task_label") != task:
        raise RuntimeError(f"v18 task receipt mismatch for {task}")
    if complete.get("identity_sha256") != sha256(paths["IDENTITY.json"]):
        raise RuntimeError(f"v18 identity hash mismatch for {task}")
    if complete.get("result_sha256") != sha256(paths["RESULT.json"]):
        raise RuntimeError(f"v18 result hash mismatch for {task}")
    if identity.get("task_final_seed_pool", [])[:50] != final_seeds:
        raise RuntimeError(f"v18 final seed mismatch for {task}")
    final = result.get("final", {})
    if final.get("valid_pair_seeds") != final_seeds:
        raise RuntimeError(f"v18 valid pair seed mismatch for {task}")
    if complete.get("simulator_invalid_pairs") != 0:
        raise RuntimeError(f"v18 contains simulator-invalid pairs for {task}")
    return {
        "identity_sha256": sha256(paths["IDENTITY.json"]),
        "result_sha256": sha256(paths["RESULT.json"]),
        "completion_sha256": sha256(paths["COMPLETE.json"]),
        "scientific_rollouts": complete.get("final_scientific_rollouts"),
    }


def write_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError("existing run manifest differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded)
    os.link(temporary, path)
    temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--v18-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    if head != args.source_commit:
        raise RuntimeError("manifest source commit mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip():
        raise RuntimeError("manifest requires a clean source worktree")
    contract = json.loads(args.contract.read_text())
    if contract.get("schema") != "act-speed-benchmark-v1":
        raise RuntimeError("unexpected contract schema")
    observed_detector = {
        "checkpoint": sha256(args.detector_checkpoint),
        "inference": sha256(args.detector_source / "phase_detector/rgb_inference.py"),
        "model_source": sha256(args.detector_source / "phase_detector/rgb_proprio.py"),
    }
    if observed_detector != DETECTOR_HASHES:
        raise RuntimeError(f"detector hash mismatch: {observed_detector}")

    tasks = {}
    v18 = {}
    for task, config in contract["tasks"].items():
        artifact_root = Path(config["root"]) / "checkpoints"
        observed = {name: sha256(artifact_root / name) for name in EXPECTED_ARTIFACTS[task]}
        if observed != EXPECTED_ARTIFACTS[task]:
            raise RuntimeError(f"ACT artifact hash mismatch for {task}: {observed}")
        search = list(range(config["search_seed_base"], config["search_seed_base"] + 50))
        final = list(range(config["final_seed_base"], config["final_seed_base"] + 50))
        v18[task] = checked_v18_receipt(args.v18_receipts, task, final)
        tasks[task] = {
            "task": config["task"],
            "policy_root": config["root"],
            "artifacts": observed,
            "search_bank": {"seeds": search, "sha256": canonical_sha256(search)},
            "final_bank": {"seeds": final, "sha256": canonical_sha256(final)},
        }
    tracked = subprocess.check_output(["git", "ls-files"], cwd=repository, text=True).splitlines()
    manifest = {
        "schema": "act-controller-retrain-run-manifest-v20",
        "source": {
            "repository": "https://github.com/kevin-thankyou-lin/SpeedTuning.git",
            "commit": args.source_commit,
            "tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repository, text=True).strip(),
            "tracked_file_sha256": {
                name: sha256(repository / name) for name in tracked if (repository / name).is_file()
            },
        },
        "contract": {"path": str(args.contract), "sha256": sha256(args.contract), "payload": contract},
        "learned_phase_detector": {"sha256": observed_detector},
        "tasks": tasks,
        "parity_gate": {
            "passed": True,
            "basis": "sealed_v18_exact_ACT_and_detector_receipts",
            "v18_receipts": v18,
        },
        "provenance": {
            "original_training_source": "298c6d16784f228df0b1f455d0e41b4276ec5184",
            "controller_status": "retrained_reproduction_not_original_frozen_bytes",
            "v18_outcomes_available_to_training": False,
        },
    }
    write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
