"""Build matched slow-only and slow/fast joint-control datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py

from imitation_data import record_scheduled_joint_episode
from sim_tasks import normalize_task_name


def _parse_schedule(value):
    schedule = tuple(float(item) for item in value.split(","))
    if len(schedule) != 4:
        raise argparse.ArgumentTypeError("a schedule must contain four comma-separated speeds")
    return schedule


def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    os.link(source, destination)


def _audit_dataset(path):
    records = []
    for episode in sorted(Path(path).glob("episode_*.hdf5")):
        with h5py.File(episode, "r") as root:
            condition = int(root.attrs["speed_condition"])
            schedule = json.loads(root.attrs["schedule"])
            replay = json.loads(root.attrs["replay_validation"])
            record = {
                "seed": int(root.attrs["seed"]),
                "condition": condition,
                "schedule": schedule,
                "absolute_replay": bool(replay["absolute_success"]),
                "relative_replay": bool(replay["relative_success"]),
                "max_command_error": float(replay["max_absolute_command_error"]),
            }
        expected = int(any(abs(float(value) - 1.0) > 1e-8 for value in schedule))
        if condition != expected:
            raise RuntimeError(f"condition/schedule mismatch in {episode}")
        if not record["absolute_replay"] or not record["relative_replay"]:
            raise RuntimeError(f"unvalidated replay in {episode}")
        if record["max_command_error"] > 1e-6:
            raise RuntimeError(f"relative command round-trip exceeded tolerance in {episode}")
        records.append(record)
    return records


def validate_conditioned_dataset_contract(output_root, task):
    slow = _audit_dataset(Path(output_root) / "slow_150" / task)
    mixed = _audit_dataset(Path(output_root) / "mixed_100slow_50fast" / task)
    if len(slow) != 150 or len(mixed) != 150:
        raise RuntimeError("both datasets must contain exactly 150 episodes")
    if sum(item["condition"] for item in slow) != 0:
        raise RuntimeError("slow_150 must contain only native 1x episodes")
    if sum(item["condition"] for item in mixed) != 50:
        raise RuntimeError("mixed dataset must contain exactly 50 fast episodes")
    if {item["seed"] for item in slow} != {item["seed"] for item in mixed}:
        raise RuntimeError("the two datasets must use the same 150 randomized poses")
    return {
        "slow_150": {"episodes": 150, "slow": 150, "fast": 0},
        "mixed_100slow_50fast": {"episodes": 150, "slow": 100, "fast": 50},
        "shared_pose_seeds": 150,
        "absolute_replays_passed": 300,
        "relative_replays_passed": 300,
        "command_roundtrip_atol": 1e-6,
    }


def collect(task, output_root, seed_base, fast_schedules, max_attempts):
    task = normalize_task_name(task)
    output_root = Path(output_root)
    source_root = output_root / "source" / task
    source_root.mkdir(parents=True, exist_ok=True)
    slow, paired_fast, attempts = [], [], []

    for offset in range(int(max_attempts)):
        if len(slow) >= 150 and len(paired_fast) >= 50:
            break
        seed = int(seed_base) + offset
        slow_path = source_root / f"slow_seed_{seed}.hdf5"
        try:
            slow_record = record_scheduled_joint_episode(
                task, seed, (1.0, 1.0, 1.0, 1.0), slow_path
            )
        except RuntimeError as exc:
            attempts.append({"seed": seed, "mode": "slow", "success": False, "error": str(exc)})
            continue
        slow.append(slow_record)
        attempts.append({"seed": seed, "mode": "slow", "success": True})

        if len(paired_fast) < 50:
            for schedule in fast_schedules:
                fast_path = source_root / f"fast_seed_{seed}.hdf5"
                try:
                    fast_record = record_scheduled_joint_episode(
                        task, seed, schedule, fast_path
                    )
                except RuntimeError as exc:
                    attempts.append(
                        {
                            "seed": seed,
                            "mode": "fast",
                            "schedule": list(schedule),
                            "success": False,
                            "error": str(exc),
                        }
                    )
                    continue
                paired_fast.append(fast_record)
                attempts.append(
                    {"seed": seed, "mode": "fast", "schedule": list(schedule), "success": True}
                )
                break

        print(
            json.dumps(
                {
                    "task": task,
                    "attempted_seed": seed,
                    "slow_successes": len(slow),
                    "paired_fast_successes": len(paired_fast),
                }
            ),
            flush=True,
        )

    if len(paired_fast) < 50:
        raise RuntimeError(f"only found {len(paired_fast)}/50 paired fast successes")
    fast_seeds = {int(item["seed"]) for item in paired_fast[:50]}
    unpaired_slow = [item for item in slow if int(item["seed"]) not in fast_seeds]
    if len(unpaired_slow) < 100:
        raise RuntimeError(f"only found {len(unpaired_slow)}/100 unpaired slow successes")
    paired_slow = [item for item in slow if int(item["seed"]) in fast_seeds][:50]
    selected_slow = paired_slow + unpaired_slow[:100]

    slow_dir = output_root / "slow_150" / task
    mixed_dir = output_root / "mixed_100slow_50fast" / task
    for index, record in enumerate(selected_slow):
        _link(record["path"], slow_dir / f"episode_{index:04d}.hdf5")
    for index, record in enumerate(unpaired_slow[:100]):
        _link(record["path"], mixed_dir / f"episode_{index:04d}.hdf5")
    for index, record in enumerate(paired_fast[:50], start=100):
        _link(record["path"], mixed_dir / f"episode_{index:04d}.hdf5")

    audit = validate_conditioned_dataset_contract(output_root, task)
    summary = {
        "schema": "conditioned-relative-joint-datasets-v1",
        "task": task,
        "pose_contract": "same 150 seeds in both datasets; 50 paired slow episodes replaced by fast",
        "slow_150": {"slow": 150, "fast": 0},
        "mixed_100slow_50fast": {"slow": 100, "fast": 50},
        "fast_schedules": [list(value) for value in fast_schedules],
        "relative_action": "target_qpos[t+k] - observations/qpos[t]",
        "audit": audit,
        "attempts": attempts,
    }
    (output_root / f"{task}_collection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--fast-schedule", type=_parse_schedule, action="append", required=True)
    parser.add_argument("--max-attempts", type=int, default=1000)
    args = parser.parse_args()
    result = collect(
        args.task,
        args.output_root,
        args.seed_base,
        args.fast_schedule,
        args.max_attempts,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
