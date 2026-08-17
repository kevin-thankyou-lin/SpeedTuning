from types import SimpleNamespace

import pytest

from postgrasp_speed_schedule import (
    PostgraspLatchedSpeedSchedule,
    PostgraspScheduleConfig,
)


def timestep(reward):
    return SimpleNamespace(reward=reward)


def policy(replan_count, step_count=230):
    return SimpleNamespace(replan_count=replan_count, step_count=step_count)


def test_replan_tick_and_pre_replan_ticks_stay_protected():
    schedule = PostgraspLatchedSpeedSchedule(PostgraspScheduleConfig())

    assert schedule.select_speed(timestep(2), policy(0), 10) == 2.0
    assert schedule.release_events == []


def test_one_stable_post_replan_observation_releases_ceiling():
    schedule = PostgraspLatchedSpeedSchedule(PostgraspScheduleConfig())

    assert schedule.select_speed(timestep(2), policy(1), 11) == 4.0
    assert schedule.release_events == [
        {
            "physics_step": 11,
            "policy_time": 230.0,
            "observed_reward": 2,
            "stable_observations": 1,
        }
    ]


def test_two_observation_gate_releases_only_on_second_stable_tick():
    schedule = PostgraspLatchedSpeedSchedule(
        PostgraspScheduleConfig(release_stability=2)
    )

    assert schedule.select_speed(timestep(2), policy(1), 11) == 2.0
    assert schedule.select_speed(timestep(2), policy(1, 232), 12) == 4.0


def test_reward_envelope_violation_immediately_downshifts_and_resets():
    schedule = PostgraspLatchedSpeedSchedule(PostgraspScheduleConfig())
    assert schedule.select_speed(timestep(2), policy(1), 11) == 4.0

    assert schedule.select_speed(timestep(0), policy(1, 234), 12) == 2.0
    assert schedule.released is False
    assert schedule.stable_observations == 0
    assert schedule.downshift_events[0]["physics_step"] == 12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pre_replan_speed": 0},
        {"post_replan_speed": 1, "pre_replan_speed": 2},
        {"release_stability": 0},
    ],
)
def test_invalid_schedule_contracts_fail_closed(kwargs):
    with pytest.raises(ValueError):
        PostgraspScheduleConfig(**kwargs)
