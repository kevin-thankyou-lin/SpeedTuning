import numpy as np

from reference_schedule import (
    CausalTemporalPool,
    EventController,
    expand_protected_speed_map,
    select_aligned_speed,
)


def state(step, distance, reward=0):
    return {
        "policy_time": float(step),
        "physics_steps": step,
        "task_reward": reward,
        "success": False,
        "contacts": [],
        "env_state": np.array([0.0, 0.0, 0.0]),
        "mocap_right": np.array([distance, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    }


def test_event_controller_waits_protects_and_releases():
    controller = EventController(
        {
            "ceiling": 4.5,
            "segments": [
                {
                    "entry": {"feature": "distance:right:0", "op": "le", "value": 0.08},
                    "exit": {"feature": "task_reward", "op": "ge", "value": 1},
                    "speed": 1.5,
                    "release_stability": 1,
                }
            ],
        }
    )
    assert controller.select(state(0, 0.2))[0] == 4.5
    assert controller.select(state(1, 0.05))[0] == 1.5
    assert controller.select(state(2, 0.05, reward=1))[0] == 4.5


def test_expansion_is_conservative_at_overlaps():
    speeds = np.array([4.5, 4.5, 1.5, 4.5, 1.0, 4.5])
    expanded = expand_protected_speed_map(speeds, ceiling=4.5, margin_indices=1)
    np.testing.assert_allclose(expanded, [4.5, 1.5, 1.5, 1.0, 1.0, 1.0])


def test_confidence_fallback_overrides_lookup():
    speed_map = np.array([4.5, 2.0, 4.5])
    assert select_aligned_speed(speed_map, 1, 0.8, confidence_threshold=0.55) == (2.0, False)
    assert select_aligned_speed(speed_map, 1, 0.2, confidence_threshold=0.55) == (1.0, True)


def test_temporal_pool_is_causal_and_normalized():
    pool = CausalTemporalPool(frames=3)
    first = pool.update(np.array([1.0, 0.0]))
    second = pool.update(np.array([0.0, 1.0]))
    assert first.shape == (6,)
    assert np.isclose(np.linalg.norm(second), 1.0)
    assert not np.allclose(first, second)
