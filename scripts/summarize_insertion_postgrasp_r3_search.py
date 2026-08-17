#!/usr/bin/env python3
"""Validate and summarize the bounded post-grasp R3 speed search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


STAGES = (
    "sentinel-4x-stability1-seed4403000",
    "sentinel-4x-stability1-seed4403002",
    "discovery-4x-stability1",
    "discovery-repair-4x-stability2-seed4803000",
    "discovery-repair-3x-stability1-seed4803000",
    "sentinel-confirmation-3x-stability1-seed4403000",
    "discovery-3x-stability1-remaining4",
    "calibration-3x-stability1",
)
DISCOVERY_SEEDS = tuple(range(4803000, 4803005))
CALIBRATION_SEEDS = tuple(range(4903000, 4903020))
FINAL_SEEDS = tuple(range(5003000, 5003100))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sum_file(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        entries[name] = checksum
    return entries


def validate_stage(root: Path, name: str) -> dict:
    stage = root / name
    result_path = stage / "results.json"
    for relative, expected in parse_sum_file(stage / "SHA256SUMS").items():
        if sha256(stage / relative) != expected:
            raise RuntimeError(f"checksum mismatch: {name}/{relative}")
    complete = parse_sum_file(stage / "COMPLETE")
    if complete != {"results.json": sha256(result_path)}:
        raise RuntimeError(f"invalid COMPLETE receipt: {name}")
    return json.loads(result_path.read_text())


def candidates(result: dict) -> list[dict]:
    return [item for item in result["rollouts"] if item["arm"] == "candidate"]


def natives(result: dict) -> list[dict]:
    return [item for item in result["rollouts"] if item["arm"] == "native_1x"]


def assert_seeds(items: list[dict], expected: tuple[int, ...], label: str):
    seeds = tuple(item["seed"] for item in items)
    if seeds != expected:
        raise RuntimeError(f"{label} seeds {seeds} != {expected}")


def summary_row(items: list[dict]) -> dict:
    return {
        "episodes": len(items),
        "successes": sum(item["success"] for item in items),
        "success_rate": float(np.mean([item["success"] for item in items])),
        "mean_executed_physics_steps": float(
            np.mean([item["physics_steps"] for item in items])
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-discovery", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_results = {
        name: validate_stage(args.root, name)
        for name in STAGES
    }
    source_commits = {
        result["provenance"]["source_commit"] for result in stage_results.values()
    }
    if len(source_commits) != 1:
        raise RuntimeError(f"multiple source commits: {sorted(source_commits)}")

    prior = json.loads(args.prior_discovery.read_text())
    prior_policy_sha = prior["provenance"]["source_sha256"]["scripted_policy.py"]
    current_policy_shas = {
        result["provenance"]["source_sha256"]["scripted_policy.py"]
        for result in stage_results.values()
    }
    if current_policy_shas != {prior_policy_sha}:
        raise RuntimeError("cached hard-seed native references use a different base policy")
    hard_native_cache = [
        item
        for item in prior["rollouts"]
        if item["arm"] == "postgrasp_replan"
        and item["speed"] == 1
        and item["seed"] in (4403000, 4403002)
    ]
    assert_seeds(hard_native_cache, (4403000, 4403002), "hard native cache")

    selected_first = stage_results[
        "discovery-repair-3x-stability1-seed4803000"
    ]
    selected_remaining = stage_results["discovery-3x-stability1-remaining4"]
    first_native = natives(stage_results["discovery-4x-stability1"])
    discovery_candidate = candidates(selected_first) + candidates(selected_remaining)
    discovery_native = first_native + natives(selected_remaining)
    assert_seeds(discovery_candidate, DISCOVERY_SEEDS, "discovery candidate")
    assert_seeds(discovery_native, DISCOVERY_SEEDS, "discovery native")

    selected_payload = selected_first["candidate_controller"]
    selected_hash = selected_first["candidate_controller_sha256"]
    for result in (selected_remaining, stage_results["calibration-3x-stability1"]):
        if result["candidate_controller_sha256"] != selected_hash:
            raise RuntimeError("selected controller changed across stages")

    calibration = stage_results["calibration-3x-stability1"]
    calibration_candidate = candidates(calibration)
    calibration_native = natives(calibration)
    assert_seeds(calibration_candidate, (4903000, 4903001), "calibration candidate")
    assert_seeds(calibration_native, (4903000, 4903001), "calibration native")

    all_rollouts = [
        item
        for result in stage_results.values()
        for item in result["rollouts"]
    ]
    executed_seeds = {item["seed"] for item in all_rollouts}
    if executed_seeds.intersection(FINAL_SEEDS):
        raise RuntimeError("final-test partition was consumed")
    untouched_calibration = sorted(set(CALIBRATION_SEEDS) - executed_seeds)

    discovery_candidate_row = summary_row(discovery_candidate)
    discovery_native_row = summary_row(discovery_native)
    discovery_speedup = (
        discovery_native_row["mean_executed_physics_steps"]
        / discovery_candidate_row["mean_executed_physics_steps"]
    )
    calibration_candidate_row = summary_row(calibration_candidate)
    calibration_native_row = summary_row(calibration_native)

    attempts = []
    for name in STAGES:
        result = stage_results[name]
        attempts.append(
            {
                "stage": name,
                "partition": result["partition"],
                "controller": result["candidate_controller"],
                "controller_sha256": result["candidate_controller_sha256"],
                "requested_seeds": result["requested_seeds"],
                "executed_seeds": result["executed_seeds"],
                "rollout_count": result["rollout_count"],
                "candidate_successes": sum(
                    item["success"] for item in candidates(result)
                ),
                "candidate_episodes": len(candidates(result)),
                "results_sha256": sha256(args.root / name / "results.json"),
            }
        )

    manifest = {
        "schema": "speedtuning-insertion-postgrasp-r3-search-manifest-v1",
        "base_policy": {
            "name": "insertion-postgrasp-base-v1",
            "source_commit": "2332ee126eb72e7dad73702dceee976e6f9e211a",
            "scripted_policy_sha256": prior_policy_sha,
        },
        "search_source_commit": next(iter(source_commits)),
        "candidate_speed_set": [4.0, 3.0, 2.0],
        "sentinel_seeds": [4403000, 4403002],
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "selected_controller": selected_payload,
        "selected_controller_sha256": selected_hash,
        "attempts": attempts,
        "rollout_accounting": {
            "new_rollouts": len(all_rollouts),
            "cache_hits": 4,
            "cache_entries": [
                {
                    "use": "sentinel matched native 1x",
                    "seeds": [4403000, 4403002],
                    "artifact": str(args.prior_discovery),
                    "artifact_sha256": sha256(args.prior_discovery),
                },
                {
                    "use": "selected-controller discovery seed 4803000",
                    "seeds": [4803000],
                    "native_stage": "discovery-4x-stability1",
                    "candidate_stage": "discovery-repair-3x-stability1-seed4803000",
                },
            ],
            "failed_candidate_rollouts": 3,
            "failed_native_rollouts": 1,
            "final_test_rollouts": 0,
            "media_replays": 0,
        },
    }
    summary = {
        "schema": "speedtuning-insertion-postgrasp-r3-search-summary-v1",
        "status": "calibration_failed_base_policy_unreliable",
        "selected_controller": selected_payload,
        "discovery": {
            "candidate": discovery_candidate_row,
            "native_1x": discovery_native_row,
            "matched_1x_speedup": discovery_speedup,
        },
        "calibration": {
            "gate": "20/20",
            "executed_before_fail_closed_stop": 2,
            "candidate": calibration_candidate_row,
            "native_1x": calibration_native_row,
            "candidate_failure_seed": 4903001,
            "native_failure_seed": 4903000,
            "untouched_seeds": untouched_calibration,
        },
        "final_test": {
            "executed": 0,
            "untouched_seed_count": len(FINAL_SEEDS),
        },
        "rollout_accounting": manifest["rollout_accounting"],
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
    checksum_paths = [complete_path, manifest_path, summary_path]
    checksum_paths.extend(args.root / name / "results.json" for name in STAGES)
    (args.root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(args.root)}\n"
            for path in checksum_paths
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
