import numpy as np

from scripts.select_diverse_reset_panel import select_maximin


def test_maximin_selects_spread_corners_deterministically():
    seeds = [10, 11, 12, 13, 14]
    poses = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 1.0],
        ]
    )

    selected, receipt = select_maximin(seeds, poses, 4)

    assert selected == [10, 13, 11, 12]
    assert receipt["variable_pose_indices"] == [0, 1]
    assert receipt["selected_minimum_pairwise_distance"] == 1.0


def test_maximin_rejects_an_invariant_pool():
    try:
        select_maximin([1, 2], np.ones((2, 3)), 2)
    except ValueError as exc:
        assert "no varying pose coordinates" in str(exc)
    else:
        raise AssertionError("invariant reset pool must be rejected")
