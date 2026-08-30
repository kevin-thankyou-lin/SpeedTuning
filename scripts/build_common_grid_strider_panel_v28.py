#!/usr/bin/env python3
"""Build fresh outcome-blind 3+5 representative panels for STRIDER v28."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from one_reset_phase_schedule import sample_object_pose
from scripts import build_representative_reset_panel as base
from scripts.run_act_speed_benchmark_cell import immutable_json

SEEDS = {"pick": 28701, "tea": 28702, "insertion": 28703}
PANEL_IDS = {
    "pick": list(range(284000000, 284000008)),
    "tea": list(range(284000100, 284000108)),
    "insertion": list(range(284000200, 284000208)),
}


def stratified(task: str) -> np.ndarray:
    dimensions = 2 if task in {"pick", "tea"} else 4
    rng = np.random.default_rng(SEEDS[task])
    blocks = []
    for size in (3, 5):
        block = np.empty((size, dimensions), dtype=np.float64)
        for column in range(dimensions):
            strata = rng.permutation(size)
            block[:, column] = (strata + rng.random(size)) / size
        blocks.append(block)
    return np.vstack(blocks)


def build(task: str) -> dict:
    unit = stratified(task)
    bounds = np.asarray(base.POSITION_BOUNDS[task], dtype=np.float64)
    positions = base.map_unit_to_bounds(unit, bounds)
    template = None
    if task == "tea":
        template = np.asarray(sample_object_pose("tea_bag", 284099999), dtype=np.float64)
    poses = base.build_pose_vectors(task, positions, tea_template=template)
    return {
        "schema": "act-common-grid-strider-panel-v28",
        "task_label": task,
        "assumed_reset_prior": "independent_uniform_over_declared_position_bounds",
        "design": "fresh randomized three-point and five-point Latin-hypercube blocks",
        "design_seed": SEEDS[task],
        "selection_uses_policy_outcomes": False,
        "selection_uses_trajectory_or_reward": False,
        "stage_prefix_sizes": [3, 8],
        "panel_ids": PANEL_IDS[task],
        "normalized_design": unit.tolist(),
        "object_pose_vectors": poses,
        "object_pose_vectors_sha256": base.canonical_sha256(poses),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.task_label)
    if args.output.exists():
        if json.loads(args.output.read_text()) != value:
            raise RuntimeError("existing panel differs")
    else:
        immutable_json(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
