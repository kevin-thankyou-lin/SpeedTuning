import inspect
from pathlib import Path

from behavior_speed_observation import SpeedObservation
from observable_behavior_speed_selector import (
    BehaviorRegionConfig,
    ObservableBehaviorRegionSelector,
)


def pose(x, z):
    return [x, 0.5, z, 1, 0, 0, 0]


def observation(object_distance, lift, relative_shift=0.0):
    return SpeedObservation(
        effector_poses=[pose(0.0, 0.1), pose(object_distance, 0.1)],
        object_poses=[
            pose(relative_shift, 0.05 + lift),
            pose(object_distance + relative_shift, 0.05 + lift),
        ],
        gripper_positions=[0.1, 0.1],
    )


def config():
    return BehaviorRegionConfig(
        protected_speed=2,
        fast_speed=3,
        min_object_lift_m=0.005,
        max_relative_translation_delta_m=0.005,
        max_object_rotation_delta_deg=1,
        max_closed_gripper_position=0.9,
        terminal_object_distance_m=0.13,
        stable_observations=2,
    )


def test_selector_api_accepts_only_external_observation():
    parameters = list(
        inspect.signature(ObservableBehaviorRegionSelector.select_speed).parameters
    )
    assert parameters == ["self", "observation"]


def test_selector_source_has_no_internal_state_or_reward_access():
    source = Path(inspect.getsourcefile(ObservableBehaviorRegionSelector)).read_text()
    for forbidden in ("replan_count", "replan_event", ".reward", ".step_count"):
        assert forbidden not in source


def test_stable_lifted_attachment_enters_then_terminal_proximity_exits():
    selector = ObservableBehaviorRegionSelector(config())
    assert selector.select_speed(observation(0.4, 0.0)) == 2
    assert selector.select_speed(observation(0.3, 0.02)) == 2
    assert selector.select_speed(observation(0.29, 0.02)) == 2
    assert selector.select_speed(observation(0.28, 0.02)) == 3
    assert selector.select_speed(observation(0.12, 0.02)) == 2
    assert selector.terminal_region_latched
    assert selector.entry_events[0]["reason"] == "stable_lifted_attachment"
    assert selector.exit_events[0]["reason"] == "terminal_proximity"


def test_attachment_instability_immediately_downshifts():
    selector = ObservableBehaviorRegionSelector(config())
    selector.select_speed(observation(0.4, 0.0))
    selector.select_speed(observation(0.3, 0.02))
    selector.select_speed(observation(0.29, 0.02))
    assert selector.select_speed(observation(0.28, 0.02)) == 3
    assert selector.select_speed(observation(0.27, 0.02, relative_shift=0.02)) == 2
    assert selector.exit_events[-1]["reason"] == "attachment_instability"
