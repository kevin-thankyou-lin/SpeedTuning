"""Materialize ACT-style HDF5 episodes from a frozen rollout manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imitation_data import load_manifest, record_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    task = load_manifest(args.manifest)["tasks"][args.task]
    episodes = task["episodes"][: args.limit]
    records = []
    for index, episode in enumerate(episodes):
        output = args.output_root / args.task / f"episode_{index:04d}.hdf5"
        if output.exists():
            records.append({"path": str(output), "skipped": True})
            continue
        records.append(
            record_episode(
                task["runtime_task"],
                episode["seed"],
                episode["phase_decisions"],
                episode["physics_steps"],
                output,
            )
        )
        print(json.dumps({"task": args.task, "episode": index + 1, "total": len(episodes)}), flush=True)
    summary = args.output_root / args.task / "replay_summary.json"
    summary.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
