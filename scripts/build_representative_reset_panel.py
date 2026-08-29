#!/usr/bin/env python3
"""Build deterministic, outcome-blind reset panels from declared uniform ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
PANEL_IDS = {
    "pick": list(range(264100000, 264100008)),
    "tea": list(range(264100100, 264100108)),
    "insertion": list(range(264100200, 264100208)),
}
POSITION_BOUNDS = {
    "pick": [[0.0, 0.2], [0.4, 0.6]],
    "tea": [[0.0, 0.2], [0.4, 0.6]],
    "insertion": [[0.1, 0.2], [0.4, 0.6], [-0.2, -0.1], [0.4, 0.6]],
}


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def quadrature_2d_nested_eight() -> np.ndarray:
    """Return a four-point Gaussian prefix plus a moment-balanced extension."""

    gaussian_offset = 1.0 / (2.0 * math.sqrt(3.0))
    low, high = 0.5 - gaussian_offset, 0.5 + gaussian_offset
    first = np.asarray(
        [[low, low], [low, high], [high, low], [high, high]], dtype=np.float64
    )

    # These four axial points separately match the uniform distribution's mean,
    # marginal variance, and zero cross-covariance.  Appending them preserves
    # those moments while adding coverage that is not a second copy of the 2x2 grid.
    axial_offset = 1.0 / math.sqrt(6.0)
    second = np.asarray(
        [
            [0.5 - axial_offset, 0.5],
            [0.5 + axial_offset, 0.5],
            [0.5, 0.5 - axial_offset],
            [0.5, 0.5 + axial_offset],
        ],
        dtype=np.float64,
    )
    return np.vstack([first, second])


def latin_hypercube_4d_nested_eight() -> np.ndarray:
    """Return two deterministic four-point LHS blocks for the 4-D task."""

    # Every four-row block uses each quartile midpoint exactly once in every
    # coordinate.  The first block minimizes the maximum absolute inter-column
    # correlation among deterministic permutations (0.6 is unavoidable with
    # four samples and four varying coordinates).  The extension minimizes the
    # combined eight-row correlation, whose maximum absolute value is 0.2.
    return np.asarray(
        [
            [0.125, 0.125, 0.375, 0.375],
            [0.375, 0.625, 0.875, 0.875],
            [0.625, 0.875, 0.125, 0.625],
            [0.875, 0.375, 0.625, 0.125],
            [0.125, 0.375, 0.375, 0.625],
            [0.375, 0.875, 0.875, 0.125],
            [0.625, 0.625, 0.125, 0.375],
            [0.875, 0.125, 0.625, 0.875],
        ],
        dtype=np.float64,
    )


def map_unit_to_bounds(unit: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    if unit.ndim != 2 or bounds.shape != (unit.shape[1], 2):
        raise ValueError("unit design and bounds have incompatible shapes")
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


def build_pose_vectors(
    task_label: str, positions: np.ndarray, *, tea_template: np.ndarray | None = None
) -> list[list[float]]:
    poses: list[list[float]] = []
    if task_label in {"pick", "tea"} and positions.shape[1] != 2:
        raise ValueError("Pick and Tea require two position coordinates")
    if task_label == "insertion" and positions.shape[1] != 4:
        raise ValueError("Insertion requires four position coordinates")

    for row in positions:
        if task_label == "pick":
            poses.append([float(row[0]), float(row[1]), 0.05, 1.0, 0.0, 0.0, 0.0])
        elif task_label == "tea":
            if tea_template is None or tea_template.shape != (39,):
                raise ValueError("Tea requires a 39-value frozen qpos template")
            pose = tea_template.copy()
            pose[:7] = [float(row[0]), float(row[1]), 0.05, 1.0, 0.0, 0.0, 0.0]
            poses.append(pose.tolist())
        elif task_label == "insertion":
            poses.append(
                [
                    float(row[0]),
                    float(row[1]),
                    0.05,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    float(row[2]),
                    float(row[3]),
                    0.05,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        else:
            raise ValueError(f"unknown task label: {task_label}")
    return poses


def build_receipt(task_label: str, tea_template: np.ndarray | None = None) -> dict:
    if task_label not in TASK_NAMES:
        raise ValueError(f"unknown task label: {task_label}")
    unit = (
        quadrature_2d_nested_eight()
        if task_label in {"pick", "tea"}
        else latin_hypercube_4d_nested_eight()
    )
    bounds = np.asarray(POSITION_BOUNDS[task_label], dtype=np.float64)
    positions = map_unit_to_bounds(unit, bounds)
    poses = build_pose_vectors(task_label, positions, tea_template=tea_template)
    corr4 = np.corrcoef(unit[:4], rowvar=False)
    corr8 = np.corrcoef(unit, rowvar=False)
    return {
        "schema": "act-representative-reset-panel-v1",
        "task_label": task_label,
        "task_name": TASK_NAMES[task_label],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "assumed_reset_prior": "independent_uniform_over_declared_position_bounds",
        "declared_position_bounds": bounds.tolist(),
        "selection_uses_policy_outcomes": False,
        "selection_uses_trajectory_or_reward": False,
        "panel_ids": PANEL_IDS[task_label],
        "stage_prefix_sizes": [4, 8],
        "normalized_design": unit.tolist(),
        "position_vectors": positions.tolist(),
        "object_pose_vectors": poses,
        "object_pose_vectors_sha256": canonical_sha256(poses),
        "design": {
            "first_four": (
                "two-dimensional tensor Gauss-Legendre quadrature"
                if task_label in {"pick", "tea"}
                else "four-dimensional quartile-midpoint Latin hypercube"
            ),
            "next_four": (
                "axis-aligned moment-balanced extension"
                if task_label in {"pick", "tea"}
                else "second quartile-midpoint Latin hypercube minimizing combined correlation"
            ),
            "first_four_max_abs_coordinate_correlation": float(
                np.max(np.abs(corr4 - np.eye(corr4.shape[0])))
            ),
            "eight_max_abs_coordinate_correlation": float(
                np.max(np.abs(corr8 - np.eye(corr8.shape[0])))
            ),
        },
        "tea_fixed_suffix_template_sha256": (
            None if tea_template is None else canonical_sha256(tea_template.tolist())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-label", choices=tuple(TASK_NAMES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tea-template-seed", type=int, default=264099999)
    args = parser.parse_args()

    tea_template = None
    if args.task_label == "tea":
        tea_template = np.asarray(
            sample_object_pose("tea_bag", args.tea_template_seed), dtype=np.float64
        )
    receipt = build_receipt(args.task_label, tea_template=tea_template)
    write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
