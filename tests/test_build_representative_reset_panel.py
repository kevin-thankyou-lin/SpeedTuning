import math

import numpy as np

from scripts import build_representative_reset_panel as panel


def test_quadrature_prefix_uses_interior_gaussian_nodes():
    design = panel.quadrature_2d_nested_eight()
    expected = {0.5 - 1.0 / (2.0 * math.sqrt(3.0)), 0.5 + 1.0 / (2.0 * math.sqrt(3.0))}
    assert design.shape == (8, 2)
    assert set(design[:4, 0]) == expected
    assert set(design[:4, 1]) == expected
    assert np.all((design > 0.0) & (design < 1.0))


def test_quadrature_prefix_and_extension_match_uniform_first_two_moments():
    design = panel.quadrature_2d_nested_eight()
    for block in (design[:4], design[4:], design):
        np.testing.assert_allclose(block.mean(axis=0), [0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(block.var(axis=0), [1.0 / 12.0] * 2, atol=1e-12)
        np.testing.assert_allclose(np.cov(block, rowvar=False, bias=True)[0, 1], 0.0, atol=1e-12)


def test_insertion_design_is_nested_latin_hypercube():
    design = panel.latin_hypercube_4d_nested_eight()
    levels = [0.125, 0.375, 0.625, 0.875]
    assert design.shape == (8, 4)
    for block in (design[:4], design[4:]):
        for column in range(4):
            assert sorted(block[:, column]) == levels
    corr8 = np.corrcoef(design, rowvar=False)
    off_diagonal = corr8 - np.eye(4)
    assert np.max(np.abs(off_diagonal)) <= 0.2 + 1e-12


def test_pose_shapes_and_declared_bounds():
    tea_template = np.arange(39, dtype=np.float64)
    for task, expected_size in (("pick", 7), ("tea", 39), ("insertion", 14)):
        receipt = panel.build_receipt(
            task, tea_template=tea_template if task == "tea" else None
        )
        assert receipt["selection_uses_policy_outcomes"] is False
        assert receipt["stage_prefix_sizes"] == [4, 8]
        assert len(receipt["object_pose_vectors"]) == 8
        assert all(len(pose) == expected_size for pose in receipt["object_pose_vectors"])
        positions = np.asarray(receipt["position_vectors"])
        bounds = np.asarray(receipt["declared_position_bounds"])
        assert np.all(positions >= bounds[:, 0])
        assert np.all(positions <= bounds[:, 1])


def test_tea_changes_only_teabag_free_joint_from_frozen_template():
    template = np.arange(39, dtype=np.float64)
    poses = panel.build_receipt("tea", tea_template=template)["object_pose_vectors"]
    assert all(pose[7:] == template[7:].tolist() for pose in poses)
    assert all(pose[2:7] == [0.05, 1.0, 0.0, 0.0, 0.0] for pose in poses)
