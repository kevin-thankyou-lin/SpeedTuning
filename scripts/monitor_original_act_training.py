"""Evaluate immutable ACT training checkpoints on one fixed pose bank."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _ready(checkpoint_dir, epoch):
    checkpoint = checkpoint_dir / f"policy_epoch_{epoch}_seed_0.ckpt"
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        return False
    # Training writes checkpoints directly. Waiting for the following checkpoint
    # proves this one is closed; the final checkpoint is released by completion.
    successor = checkpoint_dir / f"policy_epoch_{epoch + 100}_seed_0.ckpt"
    return successor.exists() or (epoch == 1900 and (checkpoint_dir / "training_complete.json").exists())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--epochs", type=int, nargs="+", default=(0, 500, 1000, 1500, 1900))
    args = parser.parse_args()
    checkpoints = args.training_root / "checkpoints"
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.json"
    progress = {
        "schema": "original-act-training-eval-v1",
        "task": args.task,
        "episodes_per_checkpoint": args.episodes,
        "seed_base": args.seed_base,
        "results": [],
    }
    _atomic_json(progress_path, progress)
    for epoch in args.epochs:
        while not _ready(checkpoints, epoch):
            time.sleep(args.poll_seconds)
        checkpoint_name = f"policy_epoch_{epoch}_seed_0.ckpt"
        output = args.output_root / f"epoch-{epoch:04d}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.evaluate_original_act",
                "--task",
                args.task,
                "--checkpoint-dir",
                str(checkpoints),
                "--checkpoint-name",
                checkpoint_name,
                "--output",
                str(output),
                "--num-rollouts",
                str(args.episodes),
                "--seed-base",
                str(args.seed_base),
            ],
            check=True,
        )
        result = json.loads(output.read_text())
        progress["results"].append(
            {
                "epoch": epoch,
                "checkpoint": checkpoint_name,
                "successes": result["successes"],
                "episodes": result["episodes"],
                "success_rate": result["success_rate"],
            }
        )
        _atomic_json(progress_path, progress)
        print(json.dumps(progress["results"][-1]), flush=True)
    (args.output_root / "monitor_complete.json").write_text(
        json.dumps(progress, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
