#!/usr/bin/env python3
"""Select a deterministic maximin panel from outcome-blind simulator resets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from one_reset_phase_schedule import sample_object_pose
from scripts.run_act_strider_frontier_v4 import write_json


TASK_NAMES = {
    "pick": "pick_and_place",
    "tea": "tea_bag",
    "insertion": "insertion",
}


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def select_maximin(
    seeds: list[int], poses: np.ndarray, count: int
) -> tuple[list[int], dict]:
    if poses.ndim != 2 or poses.shape[0] != len(seeds):
        raise ValueError("poses must be a row per seed")
    if not 1 <= count <= len(seeds):
        raise ValueError("invalid panel size")
    spans = np.ptp(poses, axis=0)
    variable = np.flatnonzero(spans > 1e-8)
    if variable.size == 0:
        raise ValueError("candidate reset pool has no varying pose coordinates")
    minimum = poses[:, variable].min(axis=0)
    maximum = poses[:, variable].max(axis=0)
    normalized = (poses[:, variable] - minimum) / (maximum - minimum)
    centroid = normalized.mean(axis=0)

    first = max(
        range(len(seeds)),
        key=lambda index: (
            float(np.linalg.norm(normalized[index] - centroid)),
            -int(seeds[index]),
        ),
    )
    selected = [first]
    while len(selected) < count:
        remaining = [index for index in range(len(seeds)) if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (
                min(
                    float(np.linalg.norm(normalized[index] - normalized[prior]))
                    for prior in selected
                ),
                -int(seeds[index]),
            ),
        )
        selected.append(next_index)

    distances = [
        float(np.linalg.norm(normalized[left] - normalized[right]))
        for offset, left in enumerate(selected)
        for right in selected[offset + 1 :]
    ]
    return [int(seeds[index]) for index in selected], {
        "variable_pose_indices": variable.tolist(),
        "variable_pose_minimum": minimum.tolist(),
        "variable_pose_maximum": maximum.tolist(),
        "normalization": "per-coordinate min-max over the frozen candidate pool",
        "initial_point": "farthest from normalized candidate-pool centroid",
        "subsequent_points": "maximize distance to the nearest selected point",
        "tie_breaker": "lowest integer seed",
        "selected_minimum_pairwise_distance": min(distances) if distances else None,
        "selected_pairwise_distances": distances,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-label", choices=tuple(TASK_NAMES), required=True)
    parser.add_argument("--candidate-start", type=int, required=True)
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--panel-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = list(range(args.candidate_start, args.candidate_start + args.candidate_count))
    poses = np.asarray(
        [sample_object_pose(TASK_NAMES[args.task_label], seed) for seed in seeds],
        dtype=np.float64,
    )
    selected, metric = select_maximin(seeds, poses, args.panel_size)
    pose_by_seed = {seed: poses[index].tolist() for index, seed in enumerate(seeds)}
    receipt = {
        "schema": "act-diverse-reset-panel-v1",
        "task_label": args.task_label,
        "task_name": TASK_NAMES[args.task_label],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "candidate_pool": seeds,
        "candidate_pool_sha256": canonical_sha256(seeds),
        "pose_vectors_sha256": canonical_sha256(poses.tolist()),
        "selection_uses_policy_outcomes": False,
        "selection_uses_only_initial_reset_pose": True,
        "panel_size": args.panel_size,
        "selected_seeds": selected,
        "selected_pose_vectors": [pose_by_seed[seed] for seed in selected],
        "selected_pose_vectors_sha256": canonical_sha256(
            [pose_by_seed[seed] for seed in selected]
        ),
        "metric": metric,
    }
    write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
