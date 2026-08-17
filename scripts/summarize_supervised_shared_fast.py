#!/usr/bin/env python3
"""Audit learned shared-fast rollout results and emit exact label metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit_result(result: dict) -> dict:
    labels = ["fast", *sorted(result["controller"]["protected_labels"])]
    label_index = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    exact = []
    protected_exact = []
    protected_recall = []
    speed_choice = []
    speed_map = {
        "fast": float(result["controller"]["fast_speed"]),
        **{
            label: float(speed)
            for label, speed in result["controller"]["protected_speed_map"].items()
        },
    }
    per_seed = []
    for rollout in result["candidate"]:
        seed_truth = []
        seed_prediction = []
        for frame in rollout["trace"]:
            truth = frame["oracle_label"]
            prediction = frame["prediction"]
            confusion[label_index[truth], label_index[prediction]] += 1
            is_exact = truth == prediction
            is_protected = truth != "fast"
            exact.append(is_exact)
            if is_protected:
                protected_exact.append(is_exact)
                protected_recall.append(prediction != "fast")
            speed_choice.append(np.isclose(float(frame["speed"]), speed_map[truth]))
            seed_truth.append(truth)
            seed_prediction.append(prediction)
        seed_protected = [
            prediction != "fast"
            for truth, prediction in zip(seed_truth, seed_prediction)
            if truth != "fast"
        ]
        per_seed.append(
            {
                "seed": rollout["seed"],
                "success": rollout["success"],
                "decision_frames": len(rollout["trace"]),
                "false_fast_rate": (
                    None
                    if not seed_protected
                    else 1.0 - float(np.mean(seed_protected))
                ),
            }
        )
    return {
        **result["summary"],
        "labels": labels,
        "confusion_rows_truth_columns_prediction": confusion.tolist(),
        "exact_label_accuracy": float(np.mean(exact)),
        "protected_segment_exact_accuracy": float(np.mean(protected_exact)),
        "protected_recall": float(np.mean(protected_recall)),
        "false_fast_rate": 1.0 - float(np.mean(protected_recall)),
        "speed_choice_accuracy": float(np.mean(speed_choice)),
        "candidate_success_seeds": [
            item["seed"] for item in result["candidate"] if item["success"]
        ],
        "candidate_failure_seeds": [
            item["seed"] for item in result["candidate"] if not item["success"]
        ],
        "per_seed": per_seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [json.loads(path.read_text()) for path in args.result]
    tasks = {result["task"]: audit_result(result) for result in results}
    if len(tasks) != len(results):
        raise ValueError("task names must be unique")
    summary = {
        "schema": "speedtuning-supervised-shared-fast-audit-v1",
        "tasks": tasks,
        "all_native_success": all(
            rollout["success"]
            for result in results
            for rollout in result["native_1x"]
        ),
        "all_candidate_success": all(
            rollout["success"]
            for result in results
            for rollout in result["candidate"]
        ),
        "new_rollouts": sum(result["summary"]["new_rollouts"] for result in results),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
