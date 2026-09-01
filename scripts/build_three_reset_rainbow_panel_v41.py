#!/usr/bin/env python3
"""Build the frozen outcome-blind three-reset panel for v41."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from one_reset_phase_schedule import sample_object_pose  # noqa: E402
from scripts import build_representative_reset_panel as base  # noqa: E402


def stratified_three(task_label: str, design_seed: int) -> np.ndarray:
    dimensions = 2 if task_label in {"pick", "tea"} else 4
    rng = np.random.default_rng(int(design_seed))
    design = np.empty((3, dimensions), dtype=np.float64)
    for column in range(dimensions):
        strata = rng.permutation(3)
        design[:, column] = (strata + rng.random(3)) / 3.0
    return design


def build(task_label: str, design_seed: int) -> dict:
    unit = stratified_three(task_label, design_seed)
    bounds = np.asarray(base.POSITION_BOUNDS[task_label], dtype=np.float64)
    positions = base.map_unit_to_bounds(unit, bounds)
    template = None
    if task_label == "tea":
        template = np.asarray(
            sample_object_pose("tea_bag", int(design_seed) + 999), dtype=np.float64
        )
    poses = base.build_pose_vectors(task_label, positions, tea_template=template)
    return {
        "schema": "act-three-reset-rainbow-panel-v41",
        "task_label": task_label,
        "task_name": base.TASK_NAMES[task_label],
        "design": "fresh three-point Latin hypercube over declared position bounds",
        "design_seed": int(design_seed),
        "assumed_reset_prior": "independent_uniform_over_declared_position_bounds",
        "declared_position_bounds": bounds.tolist(),
        "selection_uses_policy_outcomes": False,
        "selection_uses_trajectory_or_reward": False,
        "training_pose_order": [0, 1, 2] * 6,
        "training_visits_per_pose": [6, 6, 6],
        "normalized_design": unit.tolist(),
        "position_vectors": positions.tolist(),
        "object_pose_vectors": poses,
        "object_pose_vectors_sha256": base.canonical_sha256(poses),
        "tea_fixed_suffix_template_sha256": (
            None if template is None else base.canonical_sha256(template.tolist())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--design-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.task_label, args.design_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text()) != value:
            raise RuntimeError("existing v41 reset panel differs")
    else:
        from scripts.run_act_speed_benchmark_cell import immutable_json

        immutable_json(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
