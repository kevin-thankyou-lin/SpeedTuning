#!/usr/bin/env python3
"""Freeze exact episode-25 Tabular and Rainbow policies from sealed v20 cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import SPEED_VALUES, canonical_sha256, preregistration, sha256
from scripts.run_act_speed_benchmark_cell import immutable_json, immutable_torch, tabular_rebuild
from tabular_phase_speed import TabularPhaseSpeedPolicy


TASKS = ("pick", "tea", "insertion")
METHODS = ("learned_phase_tabular_rl", "learned_phase_rainbow_rl")
EXPECTED_TABULAR_SCHEDULES = {
    "pick": [1.75, 1.5, 1.5, 2.0],
    "tea": [1.25, 2.0, 1.25, 2.0],
    "insertion": [1.0, 1.75, 1.75, 2.0],
}
EXPECTED_RAINBOW_EPISODE25_SHA256 = {
    "pick": "4cda5b3e24a1db90a9e45f4887e6f7c347c698a43a5d2dada17e34d624ac1c45",
    "tea": "9a51c3acd04ed58051a8e7c737e369f76c043326ea6d817040f6451b5586cead",
    "insertion": "0ab0ac00bf3182a25413bb8d67dbd77e7efb7a0ba74426ba2f51e2375b13a5b6",
}


def checked_prefix(
    source_root: Path,
    seeds: list[int],
    method: str,
    task: str,
    episodes: int = 25,
) -> tuple[list[dict], dict]:
    complete = json.loads((source_root / "COMPLETE.json").read_text())
    identity = json.loads((source_root / "identity.json").read_text())
    stored_hash = identity.pop("identity_sha256")
    if canonical_sha256(identity) != stored_hash:
        raise RuntimeError(f"v20 identity payload mismatch for {task}/{method}")
    identity["identity_sha256"] = stored_hash
    if identity.get("method") != method or identity.get("task_label") != task:
        raise RuntimeError(f"v20 cell identity mismatch for {task}/{method}")
    if complete.get("identity_sha256") != stored_hash or complete.get("episodes") != 50:
        raise RuntimeError(f"v20 completion mismatch for {task}/{method}")
    expected_prereg = preregistration(method)
    if json.loads((source_root / "preregistration.json").read_text()) != expected_prereg:
        raise RuntimeError(f"v20 preregistration mismatch for {task}/{method}")
    records = []
    for seed in seeds[:episodes]:
        path = source_root / "states" / f"{seed}.json"
        record = json.loads(path.read_text())
        if record.get("seed") != seed or record.get("identity_sha256") != stored_hash:
            raise RuntimeError(f"v20 state identity mismatch: {path}")
        records.append(record)
    return records, identity


def freeze_tabular(records: list[dict], expected_schedule: list[float] | None = None) -> tuple[dict, dict]:
    q_values, visits = tabular_rebuild(records, len(SPEED_VALUES))
    policy = TabularPhaseSpeedPolicy(q_values, SPEED_VALUES)
    if expected_schedule is not None and list(policy.schedule) != expected_schedule:
        raise RuntimeError(f"Tabular episode-25 schedule mismatch: {list(policy.schedule)}")
    selected = {
        "algorithm": "tabular_monte_carlo_phase_speed",
        "q_values": q_values.tolist(),
        "visits": visits.tolist(),
        "speed_values": list(SPEED_VALUES),
        "schedule": list(policy.schedule),
    }
    evidence = {"q_values_sha256": canonical_sha256(q_values.tolist()), "visits_sha256": canonical_sha256(visits.tolist())}
    return selected, evidence


def freeze_rainbow(source_root: Path, records: list[dict], destination: Path, expected_sha256: str | None = None) -> tuple[dict, dict]:
    import torch

    record = records[-1]
    checkpoint = source_root / record["resume_checkpoint"]
    observed = sha256(checkpoint)
    if observed != record["resume_checkpoint_sha256"]:
        raise RuntimeError(f"Rainbow episode-25 checkpoint mismatch: {checkpoint}")
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"unexpected Rainbow episode-25 checkpoint: {observed}")
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if snapshot.get("schema") != "act-speed-rainbow-resume-v1":
        raise RuntimeError("unexpected Rainbow resume schema")
    config = preregistration("learned_phase_rainbow_rl")["training"]
    state = dict(snapshot["dqn"])
    history = np.asarray(snapshot["history"], dtype=np.float32)
    state["states_mean"] = torch.as_tensor(history.mean(axis=0), dtype=state["states_mean"].dtype)
    state["states_std"] = torch.as_tensor(np.maximum(history.std(axis=0), 1e-6), dtype=state["states_std"].dtype)
    observation_dim = int(state["feature_layer.0.weight"].shape[1])
    terminal = destination / "terminal_policy.pt"
    payload = {
        "format_version": 2,
        "algorithm": "rainbow_dqn",
        "model_state_dict": state,
        "observation_dim": observation_dim,
        "speed_values": list(SPEED_VALUES),
        "atom_size": config["atom_size"],
        "v_min": config["v_min"],
        "v_max": config["v_max"],
        "hidden_dim": config["hidden_dim"],
        "seed": config["seed"],
        "training_config": config,
        "completed_decisions": int(snapshot["decision"]),
        "decision_frame_skip": 10,
        "reward_aggregation": "undiscounted_sum_per_decision",
        "observation_spec": record["observation_spec"],
        "environment_spec": record["environment_spec"],
        "observation_encoder_state_dict": None,
        "metadata": {
            "terminal_after_exact_search_episodes": 25,
            "source_resume_checkpoint_sha256": observed,
            "terminal_normalization_from_snapshot_history": True,
        },
    }
    immutable_torch(terminal, payload)
    selected = {"algorithm": "rainbow_dqn", "checkpoint": str(terminal.resolve()), "sha256": sha256(terminal), "completed_decisions": int(snapshot["decision"])}
    evidence = {"source_resume_checkpoint": str(checkpoint), "source_resume_checkpoint_sha256": observed, "terminal_policy_sha256": selected["sha256"]}
    return selected, evidence


def freeze_cell(source_root: Path, destination: Path, seeds: list[int], method: str, task: str, manifest_sha256: str) -> dict:
    if (destination / "COMPLETE.json").exists():
        return json.loads((destination / "COMPLETE.json").read_text())
    records, source_identity = checked_prefix(source_root, seeds, method, task)
    destination.mkdir(parents=True, exist_ok=True)
    receipt_hashes = [sha256(source_root / "states" / f"{seed}.json") for seed in seeds[:25]]
    identity = {
        "schema": "act-controller-budget25-frozen-identity-v22",
        "task_label": task,
        "method": method,
        "training_episodes": 25,
        "training_rollouts_reexecuted": 0,
        "source_manifest_sha256": manifest_sha256,
        "source_identity_sha256": source_identity["identity_sha256"],
        "source_prefix_seeds": seeds[:25],
        "source_prefix_receipt_sha256": receipt_hashes,
        "source_prefix_sha256": canonical_sha256(receipt_hashes),
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    immutable_json(destination / "identity.json", identity)
    if method == "learned_phase_tabular_rl":
        selected_policy, evidence = freeze_tabular(records, EXPECTED_TABULAR_SCHEDULES[task])
    else:
        selected_policy, evidence = freeze_rainbow(
            source_root, records, destination, EXPECTED_RAINBOW_EPISODE25_SHA256[task]
        )
    selected = {
        "schema": "act-speed-selected-method-v1",
        "method": method,
        "task_label": task,
        "identity_sha256": identity["identity_sha256"],
        "terminal_artifact_only": True,
        "selected_policy": selected_policy,
        "episode_25_evidence": evidence,
    }
    immutable_json(destination / "selected.json", selected)
    complete = {
        "schema": "act-controller-budget25-frozen-completion-v22",
        "task_label": task,
        "method": method,
        "episodes": 25,
        "training_rollouts_reexecuted": 0,
        "identity_sha256": sha256(destination / "identity.json"),
        "selected_sha256": sha256(destination / "selected.json"),
        "simulator_invalid_attempts_in_prefix": sum("physics_error" in item for item in records),
        "safety_incidents_in_prefix": sum(item.get("safety_violation") is not None for item in records),
    }
    immutable_json(destination / "COMPLETE.json", complete)
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--v20-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.run_manifest.read_text())
    manifest_hash = sha256(args.run_manifest)
    completions = []
    for task in TASKS:
        seeds = manifest["tasks"][task]["search_bank"]["seeds"]
        for method in METHODS:
            source = args.v20_run / "cells" / task / method / "search"
            destination = args.output_root / task / method
            completions.append(freeze_cell(source, destination, seeds, method, task, manifest_hash))
    marker = {
        "schema": "act-controller-budget25-all-frozen-v22",
        "controllers": 6,
        "training_rollouts_reexecuted": 0,
        "all_frozen_before_final_bank": True,
        "completion_sha256": [sha256(args.output_root / item["task_label"] / item["method"] / "COMPLETE.json") for item in completions],
    }
    marker_path = args.output_root / "FROZEN_CONTROLLERS_COMPLETE.json"
    if marker_path.exists():
        if json.loads(marker_path.read_text()) != marker:
            raise RuntimeError("existing all-frozen marker differs")
    else:
        immutable_json(marker_path, marker)
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
