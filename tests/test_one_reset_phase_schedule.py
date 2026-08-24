import numpy as np
import pytest

from one_reset_phase_schedule import (
    estimate_phase_workload,
    run_phase_schedule,
    sample_object_pose,
    score_schedule_change,
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


def test_sampled_pose_contains_only_task_objects():
    assert len(sample_object_pose("pick_and_place", 90317)) == 7
    assert len(sample_object_pose("tea_bag", 90317)) == 7
    assert len(sample_object_pose("insertion", 90317)) == 14


def test_expected_time_score_prioritizes_absolute_steps_saved():
    anchor = {
        "schedule": [1, 1, 1, 1],
        "physics_steps": 250,
        "phase_decisions": [
            {"phase": "pre_grasp", "physics_step": 0, "speed": 1},
            {"phase": "grasp_lift", "physics_step": 150, "speed": 1},
            {"phase": "transport", "physics_step": 200, "speed": 1},
            {"phase": "interaction", "physics_step": 230, "speed": 1},
        ],
    }
    assert estimate_phase_workload(anchor) == {
        "pre_grasp": 150.0,
        "grasp_lift": 50.0,
        "transport": 30.0,
        "interaction": 20.0,
    }
    long_phase = score_schedule_change(
        anchor, [1.5, 1, 1, 1], safe_success_probability=0.8
    )
    short_phase = score_schedule_change(
        anchor, [1, 1, 1, 4], safe_success_probability=1.0
    )
    assert long_phase["predicted_absolute_steps_saved"] == pytest.approx(50.0)
    assert long_phase["expected_absolute_steps_saved"] == pytest.approx(40.0)
    assert short_phase["predicted_absolute_steps_saved"] == pytest.approx(15.0)
    assert long_phase["expected_absolute_steps_saved"] > short_phase["expected_absolute_steps_saved"]


def test_expected_time_score_aggregates_repeated_detector_phases():
    anchor = {
        "schedule": [2, 1, 1, 1],
        "physics_steps": 40,
        "phase_decisions": [
            {"phase": "pre_grasp", "physics_step": 0, "speed": 2},
            {"phase": "grasp_lift", "physics_step": 10, "speed": 1},
            {"phase": "pre_grasp", "physics_step": 20, "speed": 2},
            {"phase": "grasp_lift", "physics_step": 25, "speed": 1},
        ],
    }
    assert estimate_phase_workload(anchor) == {
        "pre_grasp": 30.0,
        "grasp_lift": 25.0,
        "transport": 0.0,
        "interaction": 0.0,
    }
    score = score_schedule_change(anchor, [3, 1, 1, 1])
    assert score["predicted_anchor_steps"] == pytest.approx(40.0)
    assert score["predicted_candidate_steps"] == pytest.approx(35.0)


def test_phase_workload_stops_at_first_success_for_full_horizon_rollout():
    anchor = {
        "schedule": [2, 1, 1, 1],
        "physics_steps": 400,
        "first_success_step": 40,
        "phase_decisions": [
            {"phase": "pre_grasp", "physics_step": 0, "speed": 2},
            {"phase": "grasp_lift", "physics_step": 10, "speed": 1},
            {"phase": "transport", "physics_step": 30, "speed": 1},
            {"phase": "interaction", "physics_step": 35, "speed": 1},
            {"phase": "transport", "physics_step": 80, "speed": 1},
        ],
    }

    assert estimate_phase_workload(anchor) == {
        "pre_grasp": 20.0,
        "grasp_lift": 20.0,
        "transport": 5.0,
        "interaction": 5.0,
    }
