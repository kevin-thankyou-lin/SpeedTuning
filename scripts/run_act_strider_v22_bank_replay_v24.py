#!/usr/bin/env python3
"""Replay one frozen v17 STRIDER schedule on the exact v22 final bank."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import canonical_sha256, sha256
from scripts.act_vlm_frontier_server import ACTFrontierRuntime
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json
from scripts.run_act_strider_frontier_v4 import schedule_sha256, summarize, validate_schedule


TASKS = ("pick", "tea", "insertion")


def checked_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing required receipt: {path}")
    return json.loads(path.read_text())


def load_contiguous_states(root: Path, seeds: list[int], identity: str) -> list[dict]:
    records = []
    missing = False
    for seed in seeds:
        path = root / f"{seed}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError("state receipts contain a non-contiguous suffix")
        value = checked_json(path)
        if value.get("seed") != seed or value.get("identity_sha256") != identity:
            raise RuntimeError(f"state identity mismatch: {path}")
        records.append(value)
    return records


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--v22-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=TASKS, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = checked_json(args.run_manifest)
    if manifest.get("source", {}).get("commit") != args.source_commit:
        raise RuntimeError("v24 source identity mismatch")
    task = manifest["tasks"][args.task_label]
    schedule = list(validate_schedule(task["strider"]["schedule"]))
    if schedule_sha256(schedule) != task["strider"]["schedule_sha256"]:
        raise RuntimeError("frozen schedule hash mismatch")
    seeds = list(task["final_bank"]["seeds"])
    if len(seeds) != 50 or task["final_bank"]["sha256"] != canonical_sha256(seeds):
        raise RuntimeError("v24 requires exactly 50 hash-bound seeds")

    output = args.root.resolve() / "final" / args.task_label / "strider_v17"
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lane.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another process owns replay cell {output}") from exc

    runtime = ACTFrontierRuntime(
        source_commit=args.source_commit,
        run_manifest=args.v22_manifest,
        task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
    )
    identity = {
        **runtime.identity(),
        "schema": "act-strider-v22-bank-replay-identity-v24",
        "method": "strider_v17_frozen",
        "v24_manifest_sha256": sha256(args.run_manifest),
        "v22_manifest_sha256": sha256(args.v22_manifest),
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "schedule": schedule,
        "schedule_sha256": schedule_sha256(schedule),
        "v17_selection_sha256": task["strider"]["v17_selection_sha256"],
        "v18_identity_sha256": task["strider"]["v18_identity_sha256"],
        "search_or_tuning_permitted": False,
        "cached_v20_v22_v23_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    identity_path = output / "identity.json"
    if identity_path.exists():
        if checked_json(identity_path) != identity:
            raise RuntimeError("existing replay identity differs")
    else:
        immutable_json(identity_path, identity)

    states = output / "states"
    records = load_contiguous_states(states, seeds, identity["identity_sha256"])
    if (output / "COMPLETE.json").exists():
        if len(records) != 50:
            raise RuntimeError("completion receipt exists without 50 states")
        print(json.dumps(checked_json(output / "result.json"), sort_keys=True))
        return 0

    for seed in seeds[len(records) :]:
        record = runtime.rollout(
            schedule, int(seed), record_attribution_telemetry=False
        )
        if record.get("seed") != seed or list(map(float, record.get("schedule", ()))) != schedule:
            raise RuntimeError("runtime returned a different rollout identity")
        record["identity_sha256"] = identity["identity_sha256"]
        immutable_json(states / f"{seed}.json", record)
        records.append(record)
        atomic_json(
            output / "progress.json",
            {
                "schema": "act-strider-v22-bank-replay-progress-v24",
                "identity_sha256": identity["identity_sha256"],
                "completed": len(records),
                "successes": sum(bool(item["success"]) for item in records),
                "physics_errors": sum(item.get("physics_error") is not None for item in records),
                "safety_violations": sum(item.get("safety_violation") is not None for item in records),
                "next_seed": None if len(records) == 50 else seeds[len(records)],
            },
        )
        print(
            json.dumps(
                {
                    "task": args.task_label,
                    "completed": len(records),
                    "successes": sum(bool(item["success"]) for item in records),
                }
            ),
            flush=True,
        )

    summary = summarize(records)
    result = {
        "schema": "act-strider-v22-bank-replay-result-v24",
        "task_label": args.task_label,
        "identity_sha256": identity["identity_sha256"],
        "schedule": schedule,
        "schedule_sha256": schedule_sha256(schedule),
        "episodes": 50,
        "exact_v22_bank_complete": True,
        "summary": summary,
    }
    immutable_json(output / "result.json", result)
    immutable_json(
        output / "COMPLETE.json",
        {
            "schema": "act-strider-v22-bank-replay-completion-v24",
            "identity_sha256": identity["identity_sha256"],
            "episodes": 50,
            "result_sha256": sha256(output / "result.json"),
            "new_physical_attempts": 50,
            "physics_errors": summary["physics_errors"],
            "safety_violations": summary["safety_violations"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
