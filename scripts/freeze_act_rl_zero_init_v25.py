#!/usr/bin/env python3
"""Freeze zero-training Tabular and seed-fixed untrained Rainbow controllers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import SPEED_VALUES, canonical_sha256, preregistration, sha256
from scripts.run_act_speed_benchmark_cell import immutable_json, immutable_torch

TASKS = ("pick", "tea", "insertion")
METHODS = ("learned_phase_tabular_rl", "learned_phase_rainbow_rl")


def tabular_selected() -> dict:
    q_values = [[0.0] * len(SPEED_VALUES) for _ in range(4)]
    return {
        "algorithm": "tabular_monte_carlo_phase_speed",
        "q_values": q_values,
        "visits": [[0] * len(SPEED_VALUES) for _ in range(4)],
        "speed_values": list(SPEED_VALUES),
        "schedule": [float(SPEED_VALUES[0])] * 4,
    }


def rainbow_selected(task: str, v22_root: Path, destination: Path, seed: int) -> tuple[dict, dict]:
    import torch
    from rl.rainbowDQN.network import Network

    source_selected = json.loads((v22_root / "frozen" / task / "learned_phase_rainbow_rl" / "selected.json").read_text())
    source_checkpoint = Path(source_selected["selected_policy"]["checkpoint"])
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    config = preregistration("learned_phase_rainbow_rl")["training"]
    torch.manual_seed(seed)
    support = torch.linspace(config["v_min"], config["v_max"], config["atom_size"])
    network = Network(
        int(source_payload["observation_dim"]), len(SPEED_VALUES), config["atom_size"], support,
        hidden_dim=config["hidden_dim"],
    )
    network.eval()
    terminal = destination / "terminal_policy.pt"
    payload = {
        "format_version": 2,
        "algorithm": "rainbow_dqn",
        "model_state_dict": {key: value.detach().cpu() for key, value in network.state_dict().items()},
        "observation_dim": int(source_payload["observation_dim"]),
        "speed_values": list(SPEED_VALUES),
        "atom_size": config["atom_size"],
        "v_min": config["v_min"],
        "v_max": config["v_max"],
        "hidden_dim": config["hidden_dim"],
        "seed": seed,
        "training_config": config,
        "completed_decisions": 0,
        "decision_frame_skip": 10,
        "reward_aggregation": "none_no_training",
        "observation_spec": source_payload["observation_spec"],
        "environment_spec": source_payload["environment_spec"],
        "observation_encoder_state_dict": None,
        "metadata": {
            "training_episodes": 0,
            "explicit_torch_seed": seed,
            "normalization": "initial_zero_mean_unit_std",
            "historical_episode0_reconstruction": False,
        },
    }
    immutable_torch(terminal, payload)
    selected = {"algorithm": "rainbow_dqn", "checkpoint": str(terminal.resolve()), "sha256": sha256(terminal), "completed_decisions": 0}
    evidence = {
        "architecture_metadata_source_sha256": sha256(source_checkpoint),
        "torch_seed": seed,
        "terminal_policy_sha256": selected["sha256"],
    }
    return selected, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--v22-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.run_manifest.read_text())
    seed = int(manifest["contract"]["payload"]["initialization"]["rainbow_torch_seed"])
    completions = []
    for task in TASKS:
        for method in METHODS:
            destination = args.output_root / task / method
            if (destination / "COMPLETE.json").exists():
                completions.append(json.loads((destination / "COMPLETE.json").read_text()))
                continue
            destination.mkdir(parents=True, exist_ok=True)
            identity = {
                "schema": "act-rl-zero-init-frozen-identity-v25",
                "task_label": task,
                "method": method,
                "training_episodes": 0,
                "training_rollouts": 0,
                "run_manifest_sha256": sha256(args.run_manifest),
            }
            identity["identity_sha256"] = canonical_sha256(identity)
            immutable_json(destination / "identity.json", identity)
            if method == "learned_phase_tabular_rl":
                policy = tabular_selected()
                evidence = {"q_values_sha256": canonical_sha256(policy["q_values"]), "deterministic_schedule": policy["schedule"]}
            else:
                policy, evidence = rainbow_selected(task, args.v22_root, destination, seed)
            selected = {
                "schema": "act-speed-selected-method-v1",
                "method": method,
                "task_label": task,
                "identity_sha256": identity["identity_sha256"],
                "terminal_artifact_only": True,
                "selected_policy": policy,
                "zero_training_evidence": evidence,
            }
            immutable_json(destination / "selected.json", selected)
            complete = {
                "schema": "act-rl-zero-init-frozen-completion-v25",
                "task_label": task,
                "method": method,
                "episodes": 0,
                "training_rollouts": 0,
                "identity_sha256": sha256(destination / "identity.json"),
                "selected_sha256": sha256(destination / "selected.json"),
            }
            immutable_json(destination / "COMPLETE.json", complete)
            completions.append(complete)
    marker = {
        "schema": "act-rl-zero-init-all-frozen-v25",
        "controllers": 6,
        "training_rollouts": 0,
        "all_frozen_before_final_bank": True,
        "completion_sha256": [sha256(args.output_root / item["task_label"] / item["method"] / "COMPLETE.json") for item in completions],
    }
    marker_path = args.output_root / "FROZEN_CONTROLLERS_COMPLETE.json"
    if marker_path.exists():
        if json.loads(marker_path.read_text()) != marker:
            raise RuntimeError("existing freeze marker differs")
    else:
        immutable_json(marker_path, marker)
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
