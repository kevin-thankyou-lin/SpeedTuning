#!/usr/bin/env python3
"""Run Tea STRIDER with tea-bag-center-inside-cup success."""

from __future__ import annotations

import numpy as np

from scripts import run_act_strider_tea_volume_v5 as implementation
from sim_tasks import _oriented_boxes_overlap, _point_in_oriented_box


def run_center_semantic_regression(_runtime, root):
    """Prove that center-inside passes while overlap-only placement fails."""

    cup_center = np.array([-0.1, 0.6, 0.0425], dtype=np.float64)
    cup_rotation = np.eye(3, dtype=np.float64)
    cup_half_extents = np.array([0.04, 0.04, 0.0375], dtype=np.float64)
    bag_rotation = np.eye(3, dtype=np.float64)
    bag_half_extents = np.array([0.02, 0.02, 0.02], dtype=np.float64)
    cases = []
    for name, bag_center, expected_center_inside, expected_overlap in (
        ("center_inside", [-0.1, 0.6, 0.04], True, True),
        ("rim_overlap_only", [-0.1, 0.6, 0.0999], False, True),
        ("side_overlap_only", [-0.1, 0.5401, 0.06], False, True),
        ("fully_separated", [-0.1, 0.6, 0.1001], False, False),
    ):
        bag_center = np.asarray(bag_center, dtype=np.float64)
        center_inside = _point_in_oriented_box(
            bag_center, cup_center, cup_rotation, cup_half_extents
        )
        overlaps = _oriented_boxes_overlap(
            bag_center,
            bag_rotation,
            bag_half_extents,
            cup_center,
            cup_rotation,
            cup_half_extents,
        )
        if center_inside is not expected_center_inside or overlaps is not expected_overlap:
            raise RuntimeError(f"center-success semantic regression failed: {name}")
        cases.append(
            {
                "name": name,
                "bag_center": bag_center.tolist(),
                "center_inside": center_inside,
                "oriented_boxes_overlap": overlaps,
            }
        )
    report = {
        "schema": implementation.METRIC_REGRESSION_SCHEMA,
        "criterion": "tea_bag_geom_center_inside_cup_success_volume",
        "episodes": 0,
        "new_rollouts_this_invocation": 0,
        "excluded_from_search_and_final": True,
        "cases": cases,
    }
    implementation.write_json(root / "semantic_regression" / "RECEIPT.json", report)
    return report


def main() -> int:
    implementation.VERSION = 7
    implementation.SUCCESS_CRITERION_SCHEMA = "tea-cup-center-success-v1"
    implementation.METRIC_REGRESSION_SCHEMA = "tea-cup-center-semantic-regression-v1"
    implementation.METRIC_REGRESSION_SEEDS = ()
    implementation.run_metric_regression = run_center_semantic_regression
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
