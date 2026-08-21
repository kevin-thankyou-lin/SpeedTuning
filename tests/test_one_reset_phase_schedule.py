import numpy as np
import pytest

from one_reset_phase_schedule import (
    run_phase_schedule,
    sample_object_pose,
    validate_schedule,
)


def test_schedule_requires_four_allowed_speeds():
    assert validate_schedule([1, 1.5, 2, 4]) == (1.0, 1.5, 2.0, 4.0)
    with pytest.raises(ValueError, match="one speed per oracle phase"):
        validate_schedule([1, 2])
    with pytest.raises(ValueError, match="selected from"):
        validate_schedule([1, 1, 1, 4.5])


def test_fixed_pose_reproduces_the_same_learning_scene():
    pose = sample_object_pose("pick_and_place", 90317)
    first = run_phase_schedule("pick_and_place", [1, 1, 1, 1], 90317, object_pose=pose)
    second = run_phase_schedule("pick_and_place", [1, 1, 1, 1], 12345, object_pose=pose)
    assert first["success"] and second["success"]
    assert first["physics_steps"] == second["physics_steps"]
    assert first["phase_decisions"] == second["phase_decisions"]
    assert len(pose) == 7
    assert np.isfinite(pose).all()
