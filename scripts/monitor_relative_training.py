"""Evaluate stable best checkpoints while conditioned imitation training runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from relative_imitation import RelativeChunkPredictor
from scripts.evaluate_relative_imitation import rollout
from sim_tasks import normalize_task_name


MODEL_SPECS = (
    ("slow_150", "act", (0,)),
    ("slow_150", "diffusion", (0,)),
    ("mixed_100slow_50fast", "act", (0, 1)),
    ("mixed_100slow_50fast", "diffusion", (0, 1)),
)


def _latest_training_step(log_path, training_complete):
    """Return the latest trainer step recorded by its append-only JSON log."""

    log_path = Path(log_path)
    latest = 0
    if log_path.exists():
        for line in log_path.read_text(errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "step" in record:
                latest = max(latest, int(record["step"]))
    if Path(training_complete).exists():
        record = json.loads(Path(training_complete).read_text())
        latest = max(latest, int(record["steps"]))
    return latest


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _stable_snapshot(source, destination, attempts=10):
    """Copy one checkpoint only when its source identity remains unchanged."""

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        before = source.stat()
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            temporary.unlink(missing_ok=True)
            time.sleep(2)
            continue
        try:
            payload = torch.load(temporary, map_location="cpu", weights_only=False)
        except Exception:
            temporary.unlink(missing_ok=True)
            time.sleep(2)
            continue
        os.replace(temporary, destination)
        return payload
    raise RuntimeError(f"could not obtain a stable checkpoint snapshot: {source}")


def _evaluate(task, checkpoint, condition, episodes, seed_base):
    predictor = RelativeChunkPredictor(checkpoint, speed_condition=condition)
    rollouts = []
    for index in range(int(episodes)):
        rollouts.append(rollout(task, predictor, int(seed_base) + index, 8))
    successes = [item for item in rollouts if item["success"]]
    return {
        "task": task,
        "checkpoint": str(checkpoint),
        "speed_condition": int(condition),
        "seed_base": int(seed_base),
        "episodes": int(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / int(episodes),
        "successful_mean_steps": (
            float(np.mean([item["physics_steps"] for item in successes]))
            if successes
            else None
        ),
        "clipping": predictor.clipping_metrics(),
        "rollouts": rollouts,
    }


def monitor(task, training_root, output_root, seed_base, episodes, poll_seconds):
    task = normalize_task_name(task)
    training_root = Path(training_root)
    output_root = Path(output_root)
    thresholds = (5000, 10000, 15000, 20000)
    progress_path = output_root / "progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {"schema": "conditioned-training-monitor-v1", "models": {}}
    )

    while True:
        all_complete = True
        changed = False
        for dataset, kind, conditions in MODEL_SPECS:
            key = f"{dataset}/{kind}"
            model_root = training_root / "checkpoints" / dataset / kind
            checkpoint = model_root / "best.pt"
            training_complete = model_root / "training_complete.json"
            training_step = _latest_training_step(
                training_root / f"train-{dataset}-{kind}.log", training_complete
            )
            all_complete &= training_complete.exists()
            state = progress["models"].setdefault(
                key,
                {
                    "evaluated_actual_steps": [],
                    "threshold_crossings": {},
                    "final_best_step": None,
                },
            )
            state["observed_training_step"] = training_step
            if not checkpoint.exists():
                continue
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            except Exception:
                continue
            actual_step = int(payload["step"])
            crossed = [
                threshold
                for threshold in thresholds
                if threshold <= training_step
                and str(threshold) not in state["threshold_crossings"]
            ]
            needs_final = (
                training_complete.exists() and state.get("final_best_step") != actual_step
            )
            if not crossed and not needs_final:
                continue
            if actual_step in state["evaluated_actual_steps"]:
                for threshold in crossed:
                    state["threshold_crossings"][str(threshold)] = actual_step
                if needs_final:
                    state["final_best_step"] = actual_step
                changed = True
                _write_json(progress_path, progress)
                continue
            snapshot = (
                output_root
                / "checkpoints"
                / dataset
                / kind
                / f"best-step-{actual_step:05d}.pt"
            )
            stable_payload = _stable_snapshot(checkpoint, snapshot)
            stable_step = int(stable_payload["step"])
            if stable_step != actual_step:
                continue
            digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            evaluations = {}
            for condition in conditions:
                report = _evaluate(
                    task, snapshot, condition, episodes, seed_base
                )
                report.update(
                    dataset=dataset,
                    kind=kind,
                    checkpoint_step=actual_step,
                    checkpoint_sha256=digest,
                    threshold_crossings=crossed,
                    partition="fixed online-monitor bank; never final selection",
                )
                result_path = (
                    output_root
                    / "results"
                    / dataset
                    / kind
                    / f"best-step-{actual_step:05d}-mode{condition}.json"
                )
                _write_json(result_path, report)
                evaluations[str(condition)] = str(result_path)
            state["evaluated_actual_steps"].append(actual_step)
            for threshold in crossed:
                state["threshold_crossings"][str(threshold)] = actual_step
            if needs_final:
                state["final_best_step"] = actual_step
            state["latest"] = {
                "actual_step": actual_step,
                "observed_training_step": training_step,
                "checkpoint": str(snapshot),
                "sha256": digest,
                "evaluations": evaluations,
            }
            changed = True
            _write_json(progress_path, progress)
            print(
                json.dumps(
                    {
                        "model": key,
                        "actual_step": actual_step,
                        "crossed": crossed,
                        "conditions": list(conditions),
                    }
                ),
                flush=True,
            )
            del stable_payload
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if all_complete:
            progress["terminal"] = True
            progress["completed_at"] = time.time()
            _write_json(progress_path, progress)
            return progress
        if not changed:
            time.sleep(float(poll_seconds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=20)
    args = parser.parse_args()
    result = monitor(
        args.task,
        args.training_root,
        args.output_root,
        args.seed_base,
        args.episodes,
        args.poll_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
