import inspect

import numpy as np
import pytest

from behavior_speed_observation import SpeedObservation, behavior_metrics


def pose(xyz):
    return [*xyz, 1.0, 0.0, 0.0, 0.0]


def test_speed_observation_contains_only_external_behavior():
    fields = set(SpeedObservation.__dataclass_fields__)
    assert fields == {"effector_poses", "object_poses", "gripper_positions"}
    forbidden = {"policy", "reward", "phase", "replan", "step_count"}
    assert not fields.intersection(forbidden)


def test_behavior_metrics_use_paired_object_effector_motion():
    initial = SpeedObservation(
        effector_poses=[pose([0, 0, 0]), pose([1, 0, 0])],
        object_poses=[pose([0, 0, -0.05]), pose([1, 0, -0.05])],
        gripper_positions=[0, 0],
    )
    current = SpeedObservation(
        effector_poses=[pose([0, 0, 0.1]), pose([1, 0, 0.1])],
        object_poses=[pose([0, 0, 0.05]), pose([1, 0, 0.05])],
        gripper_positions=[0, 0],
    )
    metrics = behavior_metrics(current, initial, initial)

    np.testing.assert_allclose(metrics["object_lift_m"], [0.1, 0.1])
    np.testing.assert_allclose(
        metrics["object_effector_translation_delta_m"], [0, 0]
    )
    assert metrics["object_pair_distance_m"] == pytest.approx(1.0)


def test_observation_contract_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        SpeedObservation(
            effector_poses=np.zeros((2, 7)),
            object_poses=np.zeros((1, 7)),
            gripper_positions=np.zeros(2),
        )


def test_metric_api_has_no_policy_or_reward_parameter():
    parameters = set(inspect.signature(behavior_metrics).parameters)
    assert parameters == {"current", "initial", "previous"}
