from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from act_speed_benchmark import (
    METHODS,
    PHASE_METHODS,
    SailInspiredAdaptivePolicy,
    build_offline_artifact,
    candidate_for_episode,
    preregistration,
    select_candidate,
)


def test_every_frozen_method_has_exact_preregistered_budget_and_dependencies():
    assert len(METHODS) == 6
    for method in METHODS:
        value = preregistration(method)
        assert value["search_rollouts"] == 50
        assert value["final_rollouts"] == 50
        assert value["phase_detector_required"] is (method in PHASE_METHODS)
        assert len(value["preregistration_sha256"]) == 64
        if "candidates" in value:
            assert len(value["candidates"]) == 5
            assert [candidate_for_episode(value, index)["id"] for index in range(50)] == [
                candidate["id"] for candidate in value["candidates"] for _ in range(10)
            ]


def _write_episode(path, offset):
    qpos = np.zeros((20, 14), dtype=np.float64)
    qpos[:, 0] = np.linspace(0, 1, 20) ** 2 + offset
    qpos[10:, 6] = 1.0
    action = qpos + 0.1
    with h5py.File(path, "w") as root:
        observations = root.create_group("observations")
        observations.create_dataset("qpos", data=qpos)
        root.create_dataset("action", data=action)


@pytest.mark.parametrize("method", ["awe_offline_proxy", "sail_inspired_adaptive"])
def test_offline_artifact_is_hash_pinned_and_does_not_use_detector(tmp_path, method):
    _write_episode(tmp_path / "episode_0.hdf5", 0.0)
    _write_episode(tmp_path / "episode_1.hdf5", 0.2)

    artifact = build_offline_artifact(tmp_path, method)

    assert artifact["method"] == method
    assert artifact["episode_count"] == 2
    assert artifact["paper_faithful_sail"] is False
    assert len(artifact["dataset_array_sha256"]) == 64
    assert len(artifact["candidates"]) == 5
    assert all(len(candidate["profile"]) == 20 for candidate in artifact["candidates"])
    assert all(1.0 <= value <= 2.0 for candidate in artifact["candidates"] for value in candidate["profile"])


def test_sail_inspired_policy_slows_for_gripper_transition():
    candidate = {
        "profile": [2.0] * 20,
        "maximum_speed": 2.0,
        "online_motion_gain": 1.0,
        "gripper_delta_threshold": 0.01,
    }
    policy = SailInspiredAdaptivePolicy(candidate, np.ones(14))
    context = SimpleNamespace(policy_time=0.0, episode_len=100)
    first = np.zeros(28)
    second = first.copy()
    second[6] = 0.02

    assert policy.select_speed(first, context) == 2.0
    assert policy.select_speed(second, context) == 1.0


def test_candidate_selection_enforces_nine_of_ten_and_preserves_best_effort():
    prereg = preregistration("uniform_sweep")
    records = []
    successes = (10, 9, 8, 10, 10)
    steps = (300, 250, 200, 240, 260)
    for candidate, success_count, first_step in zip(
        prereg["candidates"], successes, steps
    ):
        for index in range(10):
            success = index < success_count
            records.append(
                {
                    "candidate_id": candidate["id"],
                    "success": success,
                    "first_success_step": first_step if success else None,
                    "safety_violation": None,
                }
            )

    value = select_candidate(prereg, records)

    assert value["selected"]["candidate"]["id"] == "uniform_1.75x"
    assert value["fastest_observed_best_effort"]["candidate"]["id"] == "uniform_1.5x"


def test_candidate_selection_rejects_safety_even_with_ten_successes():
    prereg = preregistration("uniform_sweep")
    records = []
    for candidate_index, candidate in enumerate(prereg["candidates"]):
        for episode in range(10):
            records.append(
                {
                    "candidate_id": candidate["id"],
                    "success": candidate_index == 0,
                    "first_success_step": 300 if candidate_index == 0 else None,
                    "safety_violation": "unsafe" if candidate_index == 0 and episode == 0 else None,
                }
            )
    with pytest.raises(RuntimeError, match="no preregistered candidate"):
        select_candidate(prereg, records)
