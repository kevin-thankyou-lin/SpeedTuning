#!/usr/bin/env python3
"""Run four registered fixed-uniform schedules on one ACT final bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import sha256
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head
from scripts.run_act_strider_baseline import load_native_records, summary
from scripts.three_scene_server import comma_ints, write_json


SPEEDS = (1.5, 2.0, 2.5, 3.0)


def speed_slug(speed: float) -> str:
    return f"uniform_{str(float(speed)).replace('.', 'p')}x"


def evaluate(runtime, root: Path, seeds: list[int], speed: float) -> list[dict]:
    schedule = [float(speed)] * 4
    state_root = root / speed_slug(speed) / "states"
    values = []
    for seed in seeds:
        path = state_root / f"{seed}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("seed") != seed or record.get("schedule") != schedule:
                raise RuntimeError(f"cached fixed-uniform identity mismatch: {path}")
        else:
            record = runtime.rollout(schedule, seed)
            write_json(path, record)
        values.append(record)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--final-seeds", type=comma_ints, required=True)
    parser.add_argument("--native-final-root", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if git_head() != args.source_commit:
        raise RuntimeError("checked-out source does not match requested commit")
    if len(args.final_seeds) != 50 or len(set(args.final_seeds)) != 50:
        raise ValueError("fixed-uniform final evaluation requires 50 unique seeds")

    runtime = ACTFrontierRuntime(
        source_commit=args.source_commit,
        run_manifest=args.run_manifest,
        task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
    )
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": "act-fixed-uniform-final-identity-v1",
        "method": "fixed_uniform_ladder",
        "speeds": list(SPEEDS),
        "final_seeds": args.final_seeds,
        "contract_sha256": sha256(args.contract),
        "native_final_root": str(args.native_final_root.resolve()),
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("fixed-uniform root identity mismatch")
    write_json(identity_path, identity)

    native_records = load_native_records(args.native_final_root, args.final_seeds)
    native_summary = summary(native_records)
    methods = []
    for speed in SPEEDS:
        records = evaluate(runtime, root, args.final_seeds, speed)
        value = summary(records)
        candidate_mean = value["successful_mean_first_success_steps"]
        native_mean = native_summary["successful_mean_first_success_steps"]
        value["successful_rollout_speedup"] = (
            None if candidate_mean is None else native_mean / candidate_mean
        )
        value["throughput_delta_percent_vs_native"] = 100.0 * (
            value["achieved_throughput_per_step"]
            / native_summary["achieved_throughput_per_step"]
            - 1.0
        )
        methods.append({
            "method": speed_slug(speed),
            "schedule": [speed] * 4,
            **value,
        })

    result = {
        "schema": "act-fixed-uniform-final-result-v1",
        "task_label": args.task_label,
        "identity_sha256": sha256(identity_path),
        "final_rollouts_per_speed": 50,
        "new_candidate_rollouts": 50 * len(SPEEDS),
        "native_rollouts_reexecuted": 0,
        "native_reused": native_summary,
        "methods": methods,
    }
    result_path = root / "RESULT.json"
    write_json(result_path, result)
    write_json(root / "COMPLETE.json", {
        "schema": "act-fixed-uniform-final-completion-v1",
        "identity_sha256": sha256(identity_path),
        "result_sha256": sha256(result_path),
        "speeds": list(SPEEDS),
        "final_rollouts_per_speed": 50,
        "new_candidate_rollouts": 200,
        "native_rollouts_reexecuted": 0,
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

