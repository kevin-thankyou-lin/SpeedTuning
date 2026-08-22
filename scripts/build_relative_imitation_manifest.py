"""Select the fastest successful stored rollout for every randomized pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim_tasks import normalize_task_name


def build_manifest(round_root):
    round_root = Path(round_root)
    contract = json.loads((round_root / "CONTRACT.json").read_text())
    tasks = {}
    for short_name in ("pick", "tea", "insertion"):
        candidates = []
        for method in ("vlm", "tabular"):
            report = json.loads((round_root / "results" / short_name / f"{method}.json").read_text())
            for rollout in report["rollouts"]:
                if rollout["success"]:
                    candidates.append({**rollout, "source_method": method})
        by_seed = {}
        for rollout in candidates:
            seed = int(rollout["seed"])
            current = by_seed.get(seed)
            if current is None or int(rollout["physics_steps"]) < int(current["physics_steps"]):
                by_seed[seed] = rollout
        native_by_seed = {}
        native_ledger = Path(contract["tasks"][short_name]["native_ledger"])
        for line in native_ledger.read_text().splitlines():
            event = json.loads(line)
            if (
                event.get("event") == "rollout_complete"
                and event.get("role") == "final_native"
                and event.get("success")
            ):
                native_by_seed[int(event["seed"])] = event
        for seed in contract["tasks"][short_name]["evaluation_seeds"]:
            seed = int(seed)
            if seed not in by_seed and seed in native_by_seed:
                native = native_by_seed[seed]
                by_seed[seed] = {
                    "seed": seed,
                    "schedule": [1.0, 1.0, 1.0, 1.0],
                    "physics_steps": int(native["physics_steps"]),
                    "phase_decisions": [
                        {"phase": "pre_grasp", "physics_step": 0, "speed": 1.0}
                    ],
                    "source_method": "native_fallback",
                }
        episodes = []
        for seed, rollout in sorted(by_seed.items()):
            episodes.append(
                {
                    "seed": seed,
                    "schedule": rollout["schedule"],
                    "physics_steps": rollout["physics_steps"],
                    "phase_decisions": rollout["phase_decisions"],
                    "source_method": rollout["source_method"],
                }
            )
        runtime_task = json.loads(
            (round_root / "results" / short_name / "vlm.json").read_text()
        )["runtime_task"]
        tasks[short_name] = {
            "runtime_task": normalize_task_name(runtime_task),
            "episodes": episodes,
        }
    return {
        "schema": "relative-joint-imitation-manifest-v1",
        "source_round": str(round_root),
        "selection": "fastest successful final rollout per shared seed across vlm and tabular",
        "stored_values": ["observations/qpos", "target_qpos", "observations/images/angle"],
        "training_action": "target_qpos[t+k] - observations/qpos[t]",
        "tasks": tasks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.round_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({task: len(value["episodes"]) for task, value in manifest["tasks"].items()}))


if __name__ == "__main__":
    main()
