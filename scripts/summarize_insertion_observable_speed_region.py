#!/usr/bin/env python3
"""Audit the policy-agnostic observable Insertion speed-region study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EVAL_STAGES = (
    "sentinel-seed4403000",
    "sentinel-seed4403002",
    "discovery-retired5",
    "counterexamples-retired2",
)
FRESH_CALIBRATION = frozenset(range(5103000, 5103020))
FRESH_FINAL = frozenset(range(5203000, 5203100))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sums(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        entries[name] = checksum
    return entries


def validate_receipt(directory: Path, payload_name: str) -> dict:
    for name, expected in parse_sums(directory / "SHA256SUMS").items():
        if sha256(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory.name}/{name}")
    complete = parse_sums(directory / "COMPLETE")
    payload_path = directory / payload_name
    if complete != {payload_name: sha256(payload_path)}:
        raise RuntimeError(f"invalid COMPLETE: {directory.name}")
    return json.loads(payload_path.read_text())


def arm(result: dict, name: str) -> list[dict]:
    return [item for item in result["rollouts"] if item["arm"] == name]


def row(items: list[dict]) -> dict:
    return {
        "episodes": len(items),
        "successes": sum(item["success"] for item in items),
        "success_rate": float(np.mean([item["success"] for item in items])),
        "mean_physics_steps": float(np.mean([item["physics_steps"] for item in items])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = validate_receipt(args.root / "profile-fixed2-retired14", "profile.json")
    results = {
        name: validate_receipt(args.root / name, "results.json")
        for name in EVAL_STAGES
    }
    source_commits = {result["provenance"]["source_commit"] for result in results.values()}
    if len(source_commits) != 1:
        raise RuntimeError("evaluation source changed across stages")
    controller_hashes = {result["candidate_controller_sha256"] for result in results.values()}
    if len(controller_hashes) != 1:
        raise RuntimeError("candidate controller changed across stages")
    forbidden = results[EVAL_STAGES[0]]["runtime_forbidden_inputs"]
    for result in results.values():
        if result["runtime_forbidden_inputs"] != forbidden:
            raise RuntimeError("runtime information boundary changed")

    sentinel_candidate = []
    sentinel_fixed = []
    for name in EVAL_STAGES[:2]:
        sentinel_candidate.extend(arm(results[name], "candidate"))
        sentinel_fixed.extend(arm(results[name], "fixed_2x"))
    discovery_candidate = arm(results["discovery-retired5"], "candidate")
    discovery_fixed = arm(results["discovery-retired5"], "fixed_2x")
    counter_candidate = arm(results["counterexamples-retired2"], "candidate")
    counter_fixed = arm(results["counterexamples-retired2"], "fixed_2x")

    if [item["seed"] for item in sentinel_candidate] != [4403000, 4403002]:
        raise RuntimeError("unexpected sentinel execution")
    if [item["seed"] for item in discovery_candidate] != list(range(4703000, 4703005)):
        raise RuntimeError("unexpected retired discovery execution")
    if [item["seed"] for item in counter_candidate] != [4903001]:
        raise RuntimeError("counterexample gate did not stop on first failure")

    eval_rollouts = [
        item
        for result in results.values()
        for item in result["rollouts"]
    ]
    executed = {item["seed"] for item in eval_rollouts}
    if executed.intersection(FRESH_CALIBRATION | FRESH_FINAL):
        raise RuntimeError("fresh partition was consumed")

    sentinel_candidate_row = row(sentinel_candidate)
    sentinel_fixed_row = row(sentinel_fixed)
    discovery_candidate_row = row(discovery_candidate)
    discovery_fixed_row = row(discovery_fixed)
    summary = {
        "schema": "speedtuning-insertion-observable-speed-region-summary-v1",
        "status": "retired_discovery_passed_counterexample_failed_base_policy",
        "information_boundary": {
            "runtime_inputs": results[EVAL_STAGES[0]]["runtime_selector_inputs"],
            "runtime_forbidden_inputs": forbidden,
            "firewall_passed": True,
        },
        "profile": {
            "episodes": profile["episodes"],
            "successes": profile["successes"],
            "purpose": profile["purpose"],
            "proposed_observable_region": profile["proposed_observable_region"],
            "sha256": sha256(args.root / "profile-fixed2-retired14" / "profile.json"),
        },
        "retired_sentinels": {
            "candidate": sentinel_candidate_row,
            "fixed_2x": sentinel_fixed_row,
            "incremental_speedup_vs_fixed_2x": (
                sentinel_fixed_row["mean_physics_steps"]
                / sentinel_candidate_row["mean_physics_steps"]
            ),
        },
        "retired_discovery": {
            "candidate": discovery_candidate_row,
            "fixed_2x": discovery_fixed_row,
            "incremental_speedup_vs_fixed_2x": (
                discovery_fixed_row["mean_physics_steps"]
                / discovery_candidate_row["mean_physics_steps"]
            ),
        },
        "retired_counterexample": {
            "seed": 4903001,
            "candidate": row(counter_candidate),
            "fixed_2x": row(counter_fixed),
            "stopping_reason": "observable terminal slowdown did not repair base-policy insertion loss",
        },
        "rollout_accounting": {
            "offline_telemetry_replays": profile["episodes"],
            "candidate_evaluation_rollouts": sum(
                item["arm"] == "candidate" for item in eval_rollouts
            ),
            "fixed_2x_evaluation_rollouts": sum(
                item["arm"] == "fixed_2x" for item in eval_rollouts
            ),
            "fresh_calibration_rollouts": 0,
            "final_test_rollouts": 0,
            "media_replays": 0,
        },
        "fresh_partitions": {
            "calibration_seed_count": len(FRESH_CALIBRATION),
            "final_seed_count": len(FRESH_FINAL),
            "executed": 0,
        },
    }
    manifest = {
        "schema": "speedtuning-insertion-observable-speed-region-manifest-v1",
        "base_policy_commit": "2332ee126eb72e7dad73702dceee976e6f9e211a",
        "profile_source_commit": profile["provenance"]["source_commit"],
        "evaluation_source_commit": next(iter(source_commits)),
        "candidate_controller": results[EVAL_STAGES[0]]["candidate_controller"],
        "candidate_controller_sha256": next(iter(controller_hashes)),
        "fresh_calibration_seeds": sorted(FRESH_CALIBRATION),
        "fresh_final_seeds": sorted(FRESH_FINAL),
        "stages": [
            {
                "name": name,
                "requested_seeds": results[name]["requested_seeds"],
                "executed_seeds": results[name]["executed_seeds"],
                "rollout_count": results[name]["rollout_count"],
                "results_sha256": sha256(args.root / name / "results.json"),
            }
            for name in EVAL_STAGES
        ],
    }

    summary_path = args.root / "summary.json"
    manifest_path = args.root / "manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete_path = args.root / "COMPLETE"
    complete_path.write_text(
        f"{sha256(manifest_path)}  manifest.json\n"
        f"{sha256(summary_path)}  summary.json\n"
    )
    covered = [complete_path, manifest_path, summary_path]
    covered.append(args.root / "profile-fixed2-retired14" / "profile.json")
    covered.extend(args.root / name / "results.json" for name in EVAL_STAGES)
    (args.root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(args.root)}\n"
            for path in covered
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
