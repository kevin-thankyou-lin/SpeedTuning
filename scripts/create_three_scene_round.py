#!/usr/bin/env python3
"""Create a blind three-scene VLM versus tabular learned-phase round."""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import subprocess
from pathlib import Path


TASKS = {
    "pick": ("pick_and_place", "scripted_pick_and_place.json"),
    "tea": ("tea_bag", "scripted_tea_bag_randomized.json"),
    "insertion": ("insertion", "scripted_insertion.json"),
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fresh_seeds(count: int, used: set[int]) -> list[int]:
    result = []
    while len(result) < count:
        value = secrets.randbelow(1_900_000_000) + 10_000_000
        if value not in used:
            used.add(value)
            result.append(value)
    return result


def task_contract(runtime_task: str) -> str:
    return f"""# Three-scene learned-phase speed search

Task: `{runtime_task}`
Learning budget: at most 50 candidate/native episodes.
Runtime phases: `pre_grasp`, `grasp_lift`, `transport`, `interaction` from the sealed RGB/proprio detector.
Allowed speeds: `1, 1.5, 2, 2.5, 3, 3.5, 4`.
Client: `python3 /workspace/three_scene.py --api /workspace/runner_api`.

The runner has three frozen discovery scenes and ten additional randomized ranking poses. Start with `info` and inspect all three native MP4s. Use `probe A,B,C,D` for at most five distinct candidate schedules. Every probe always runs all three discovery poses, including after a failure; compare the three outcomes before proposing another schedule.

Keep native `1x` only as an external deployment fallback; it never occupies an accelerated finalist slot. Preserve the runner's six-episode causal-ladder reserve. Before ranking, always call `backoff HASH PHASE EVIDENCE` exactly once on the fastest safe accelerated `3/3` schedule. Attribute the phase from the three pose-aligned MP4s and telemetry, and give concise evidence. The runner holds every other phase fixed and tests one-rung and two-rung slowdowns of only that phase on all three scenes. It designates the fast base and the first safe causal backoff; if neither backoff is safe, it ranks the base alone. Do not invent another ladder after reading outcomes.

Do not assign success probabilities. Call `rank` exactly once with the runner-designated ladder finalists. Normally there are two. If every downward rung collapses to native, the runner may designate the sole minimally accelerated base and rank it alone; never duplicate its hash or invent a second schedule. The runner evaluates each finalist on the same ten fresh randomized poses. A schedule is provisionally qualified only with zero safety violations and at least `9/10`; among qualifiers it selects by successes and successful mean steps. If none qualifies, deployment falls back to native while the least-bad accelerated schedule is still frozen for a descriptive 100-state benchmark. Preserve twenty episodes for ranking. Do not treat discovery success or `9/10` as certification, and do not access other experiments, seeds, private state, oracle phases, or prior schedules.

Write only `analysis/STATUS.json` and `analysis/FINAL_REPORT.md`. `STATUS.json` must set `terminal: true` and record accelerated qualification, deployment schedule, benchmark schedule, budget used, detector type, attributed phase/evidence, finalist results, and the three-scene limitation.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = args.source_root.resolve()
    if root.exists():
        raise SystemExit(f"refusing existing root: {root}")
    source_contract = json.loads((source / "CONTRACT.json").read_text())
    detector = json.loads((source / "DETECTOR.json").read_text())
    used = set()
    for task in source_contract["tasks"].values():
        used.update(int(value) for value in task["evaluation_seeds"])
    root.mkdir(parents=True)
    repository_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()
    tasks = {}
    for lane, (runtime_task, config_name) in TASKS.items():
        source_task = source_contract["tasks"][lane]
        discovery = fresh_seeds(3, used)
        ranking = fresh_seeds(10, used)
        agent_root = root / "vlm_agents" / lane
        runner_root = root / "vlm_runners" / lane
        (agent_root / "analysis").mkdir(parents=True)
        (runner_root / "public" / "media").mkdir(parents=True)
        shutil.copy2(Path(__file__).with_name("three_scene_client.py"), agent_root / "three_scene.py")
        (agent_root / "TASK_CONTRACT.md").write_text(task_contract(runtime_task))
        (agent_root / "AGENTS.md").write_text(
            "Read TASK_CONTRACT.md completely. Execute the experiment, not merely a plan. "
            "Use only this lane's runner and media. Do not write scripts or inspect host paths.\n"
        )
        tasks[lane] = {
            "runtime_task": runtime_task,
            "config": config_name,
            "discovery_seeds": discovery,
            "ranking_seeds": ranking,
            "evaluation_seeds": source_task["evaluation_seeds"],
            "native_ledger": source_task["native_ledger"],
            "vlm_runner_root": str(runner_root),
            "vlm_agent_root": str(agent_root),
            "vlm_selection": str(runner_root / "public" / "SELECTION.json"),
            "tabular_checkpoint": str(root / "tabular" / lane / "checkpoint.json"),
            "tabular_report": str(root / "tabular" / lane / "training.json"),
            "tabular_poses": str(root / "tabular" / lane / "poses.json"),
        }
    write_json(
        root / "CONTRACT.json",
        {
            "schema": "three-scene-learned-phase-generalization-v1",
            "repository_commit": repository_commit,
            "source_round": str(source),
            "learning_states_per_task": 3,
            "learning_episodes_per_method": 50,
            "methods": ["vlm", "tabular"],
            "phase_labels": "sealed learned RGB/proprio four-phase raw argmax",
            "phase_detector": detector,
            "evaluation_states_per_task": 100,
            "metric": "success_only_speedup",
            "tasks": tasks,
        },
    )
    write_json(root / "DETECTOR.json", detector)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
