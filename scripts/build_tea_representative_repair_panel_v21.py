#!/usr/bin/env python3
"""Build the fresh outcome-blind nested Tea panel for the v21 repair."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from one_reset_phase_schedule import sample_object_pose
from scripts import build_representative_reset_panel as base
from scripts.run_act_strider_frontier_v4 import write_json


PANEL_IDS = list(range(266100100, 266100116))
STAGE_PREFIXES = [4, 8, 16]
TEA_TEMPLATE_SEED = 266099999


def normalized_nested_sixteen() -> np.ndarray:
    """Return nested 4/8/16 prefixes from a centered 4x4 uniform grid."""

    levels = np.asarray([0.125, 0.375, 0.625, 0.875], dtype=np.float64)
    # Each four-row block is a Latin hypercube.  The first two blocks have zero
    # cross-correlation; all four blocks contain the complete 4x4 grid exactly.
    permutations = (
        (1, 3, 0, 2),
        (2, 0, 3, 1),
        (0, 2, 1, 3),
        (3, 1, 2, 0),
    )
    return np.asarray(
        [(levels[x], levels[y]) for permutation in permutations for x, y in enumerate(permutation)],
        dtype=np.float64,
    )


def build_receipt() -> dict:
    unit = normalized_nested_sixteen()
    bounds = np.asarray(base.POSITION_BOUNDS["tea"], dtype=np.float64)
    positions = base.map_unit_to_bounds(unit, bounds)
    template = np.asarray(sample_object_pose("tea_bag", TEA_TEMPLATE_SEED), dtype=np.float64)
    poses = base.build_pose_vectors("tea", positions, tea_template=template)
    return {
        "schema": "act-tea-representative-repair-panel-v21",
        "task_label": "tea",
        "task_name": "tea_bag",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "assumed_reset_prior": "independent_uniform_over_declared_position_bounds",
        "declared_position_bounds": bounds.tolist(),
        "selection_uses_policy_outcomes": False,
        "selection_uses_trajectory_or_reward": False,
        "panel_ids": PANEL_IDS,
        "stage_prefix_sizes": STAGE_PREFIXES,
        "normalized_design": unit.tolist(),
        "position_vectors": positions.tolist(),
        "object_pose_vectors": poses,
        "object_pose_vectors_sha256": base.canonical_sha256(poses),
        "tea_fixed_suffix_template_seed": TEA_TEMPLATE_SEED,
        "tea_fixed_suffix_template_sha256": base.canonical_sha256(template.tolist()),
        "design": {
            "first_four": "centered four-stratum Latin hypercube with zero coordinate correlation",
            "first_eight": "two disjoint centered Latin hypercubes with zero combined coordinate correlation",
            "sixteen": "complete centered 4x4 product grid",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt()
    write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
