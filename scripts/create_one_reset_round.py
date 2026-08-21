#!/usr/bin/env python3
"""Create a blind one-reset VLM versus tabular generalization round."""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
from pathlib import Path


TASKS = {
    "pick": ("pick_and_place", "scripted_pick_and_place.json"),
    "tea": ("tea_bag", "scripted_tea_bag_randomized.json"),
    "insertion": ("insertion", "scripted_insertion.json"),
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-round", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = args.source_round.resolve()
    if root.exists():
        raise SystemExit(f"refusing existing root: {root}")
    root.mkdir(parents=True)
    used = set()
    source_contracts = {}
    for lane in TASKS:
        contract = json.loads((source / "manager" / "contracts" / f"{lane}.json").read_text())
        source_contracts[lane] = contract
        used.update(contract["learning_seeds"])
        used.update(contract["evaluation_seeds"])
    tasks = {}
    for lane, (runtime_task, config_name) in TASKS.items():
        while True:
            training_seed = secrets.randbelow(1_900_000_000) + 10_000_000
            if training_seed not in used:
                used.add(training_seed)
                break
        agent_root = root / "vlm_agents" / lane
        runner_root = root / "vlm_runners" / lane
        (agent_root / "analysis").mkdir(parents=True)
        (runner_root / "public" / "media").mkdir(parents=True)
        shutil.copy2(Path(__file__).with_name("one_reset_client.py"), agent_root / "one_reset.py")
        task_contract = f"""# One-reset oracle-phase generalization study

Task: `{runtime_task}`
Learning evidence: exactly one frozen initial scene, reused for every schedule.
Learning budget: 50 total episodes, including native `1x`.
Runtime phases: `pre_grasp`, `grasp_lift`, `transport`, `interaction` from oracle labels.
Allowed speeds: `1, 1.5, 2, 2.5, 3, 3.5, 4`.
Client: `python3 /workspace/one_reset.py --api /workspace/runner_api`.

Start with `info` and inspect `media/native.mp4`. Test schedules with `test A,B,C,D`; each unique schedule costs one episode and returns its phase-entry trace, phase workload estimate, and MP4. Before choosing among plausible challengers, call `score ANCHOR_HASH A,B,C,D --safe-success-probability P` for each, where `P` is your conservative MP4- and evidence-based probability of safe success. Prioritize the largest `expected_absolute_steps_saved = P * predicted_absolute_steps_saved`, not the largest multiplier or relative speedup. The score is only an acquisition estimate; protect weak phases and retain a safely successful anchor. You may stop early. Before terminating, call `select HASH` for one tested schedule that succeeded without a workspace violation. Optimize for likely randomized-pose generalization first and absolute expected time reduction second; one-scene success is not a reliability claim. Do not access other experiments, seeds, private state, or prior schedules.

Write only `analysis/STATUS.json` and `analysis/FINAL_REPORT.md`. `STATUS.json` must set `terminal: true` and record the selected schedule hash, schedule, budget used, and the limitation that learning used one state.
"""
        (agent_root / "TASK_CONTRACT.md").write_text(task_contract)
        (agent_root / "AGENTS.md").write_text(
            "Read TASK_CONTRACT.md completely. Execute the experiment, not merely a plan. "
            "Use only the one-reset runner and its current media. Do not write scripts or inspect host paths.\n"
        )
        tasks[lane] = {
            "runtime_task": runtime_task,
            "config": config_name,
            "training_seed": training_seed,
            "evaluation_seeds": source_contracts[lane]["evaluation_seeds"],
            "native_ledger": str(source / "manager" / "artifacts" / lane / "ledger.jsonl"),
            "vlm_runner_root": str(runner_root),
            "vlm_agent_root": str(agent_root),
            "vlm_selection": str(runner_root / "public" / "SELECTION.json"),
            "tabular_checkpoint": str(root / "tabular" / lane / "checkpoint.json"),
            "tabular_report": str(root / "tabular" / lane / "training.json"),
        }
    write_json(
        root / "CONTRACT.json",
        {
            "schema": "one-reset-oracle-generalization-v1",
            "source_round": str(source),
            "learning_states_per_task": 1,
            "learning_episodes_per_method": 50,
            "methods": ["vlm", "tabular"],
            "phase_labels": "oracle four-phase entry",
            "evaluation_states_per_task": 100,
            "metric": "success_only_speedup",
            "tasks": tasks,
        },
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
