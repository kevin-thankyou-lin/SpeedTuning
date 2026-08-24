from types import SimpleNamespace

import numpy as np
import pytest
import torch

from act_integration import OriginalACTSpeedAdapter, _resolve_stats
from chunked_policy import ChunkPredictorAdapter
from policy_speed_env import ChunkedActionSource, create_speed_env


CAMERAS = ("angle", "left_wrist", "right_wrist")


def test_resolved_stats_preserve_serialized_float64_dtype():
    stats = {
        key: np.linspace(0.01, 0.2, 14, dtype=np.float64)
        for key in ("qpos_mean", "qpos_std", "action_mean", "action_std")
    }

    resolved = _resolve_stats({"stats": stats}, stats_path=None)

    assert all(value.dtype == np.float64 for value in resolved.values())
    for key in stats:
        np.testing.assert_array_equal(resolved[key], stats[key])


class RecordingACT(torch.nn.Module):
    def __init__(self, chunks):
        super().__init__()
        self.chunks = [torch.as_tensor(item, dtype=torch.float32) for item in chunks]
        self.inputs = []

    def forward(self, qpos, images):
        self.inputs.append((qpos.detach().cpu().clone(), images.detach().cpu().clone()))
        return self.chunks[len(self.inputs) - 1][None].to(qpos.device)


def _observation(qpos_value=3.0):
    return {
        "qpos": np.full(14, qpos_value, dtype=np.float32),
        "images": {
            "angle": np.full((3, 4, 3), 51, dtype=np.uint8),
            "left_wrist": np.full((3, 4, 3), 102, dtype=np.uint8),
            "right_wrist": np.full((3, 4, 3), 153, dtype=np.uint8),
        },
    }


def _adapter(chunks, *, episode_len=5, qpos_mean=1.0, qpos_std=2.0):
    model = RecordingACT(chunks)
    adapter = OriginalACTSpeedAdapter(
        model,
        camera_names=CAMERAS,
        qpos_mean=np.full(14, qpos_mean),
        qpos_std=np.full(14, qpos_std),
        action_mean=np.full(14, 10.0),
        action_std=np.full(14, 2.0),
        episode_len=episode_len,
        num_queries=len(chunks[0]),
        temporal_ensemble_m=0.01,
        device="cpu",
    )
    return adapter, model


def test_multiview_act_uses_progress_three_cameras_and_paper_ensemble():
    first = np.stack([np.full(14, value) for value in (1.0, 2.0, 3.0, 4.0)])
    second = np.stack([np.full(14, value) for value in (10.0, 20.0, 30.0, 40.0)])
    adapter, model = _adapter([first, second])

    action0 = adapter.action(_observation(), speed=1.0)
    action1 = adapter.action(_observation(), speed=1.0)

    np.testing.assert_allclose(action0, np.full(14, 12.0))
    weights = np.exp(-0.01 * np.arange(2))
    expected_normalized = np.dot(weights / weights.sum(), [2.0, 10.0])
    np.testing.assert_allclose(
        action1, np.full(14, expected_normalized * 2.0 + 10.0), rtol=1e-6
    )
    assert len(model.inputs) == 2
    assert model.inputs[0][0].shape == (1, 15)
    np.testing.assert_allclose(model.inputs[0][0][0, :14], np.ones(14))
    assert model.inputs[0][0][0, 14].item() == pytest.approx(-1.0)
    assert model.inputs[1][0][0, 14].item() == pytest.approx(-0.5)
    camera_means = model.inputs[0][1].mean(dim=(0, 2, 3, 4)).numpy()
    np.testing.assert_allclose(camera_means, [0.2, 0.4, 0.6], atol=1e-6)


def test_uniform_one_matches_100_step_reference_temporal_ensemble():
    chunks = []
    for query in range(3):
        chunks.append(
            np.stack(
                [np.full(14, 1.0 + query * 100 + offset) for offset in range(100)]
            )
        )
    adapter, model = _adapter(chunks, episode_len=500, qpos_mean=0.0, qpos_std=1.0)

    actual = [adapter.action(_observation(), speed=1.0) for _ in range(3)]
    all_time_actions = torch.zeros((3, 103, 14))
    expected = []
    for step, chunk in enumerate(chunks):
        all_time_actions[step, step : step + 100] = torch.as_tensor(chunk)
        candidates = all_time_actions[:, step]
        candidates = candidates[torch.all(candidates != 0, dim=1)]
        weights = np.exp(-0.01 * np.arange(len(candidates)))
        weights = torch.as_tensor(
            weights / weights.sum(), dtype=torch.float32
        )[:, None]
        normalized = (candidates * weights).sum(dim=0).numpy()
        expected.append(normalized * 2.0 + 10.0)

    np.testing.assert_array_equal(np.stack(actual), np.stack(expected))
    # Match the retained evaluator's float32 ensemble multiplied by its
    # serialized float64 normalization arrays.
    assert all(action.dtype == np.float64 for action in actual)
    assert len(model.inputs) == 3


def test_uniform_one_uses_retained_dense_ledger_without_interpolation(monkeypatch):
    chunks = [
        np.stack([np.full(14, query + offset + 1.0) for offset in range(4)])
        for query in range(2)
    ]
    adapter, _ = _adapter(chunks, episode_len=5)

    def interpolation_is_not_retained(*args, **kwargs):
        del args, kwargs
        raise AssertionError("uniform 1x must use the retained dense ledger")

    monkeypatch.setattr(adapter, "_sample_chunk", interpolation_is_not_retained)

    adapter.action(_observation(), speed=1.0)
    adapter.action(_observation(), speed=1.0)

    assert adapter._uniform_one is True
    assert adapter._predictions == []
    assert adapter._all_time_actions.shape == (5, 9, 14)


def test_acceleration_reconstructs_live_history_from_uniform_dense_ledger():
    first = np.stack([np.full(14, value) for value in (1.0, 2.0, 3.0, 4.0)])
    second = np.stack([np.full(14, value) for value in (10.0, 20.0, 30.0, 40.0)])
    adapter, _ = _adapter([first, second])

    adapter.action(_observation(), speed=1.0)
    action = adapter.action(_observation(), speed=2.0)

    weights = np.exp(-0.01 * np.arange(2))
    expected_normalized = np.dot(weights / weights.sum(), [2.0, 10.0])
    np.testing.assert_allclose(
        action, np.full(14, expected_normalized * 2.0 + 10.0), rtol=1e-6
    )
    assert adapter._uniform_one is False
    assert [origin for origin, _ in adapter._predictions] == [0.0, 1.0]


def test_qpos_normalization_matches_frozen_evaluator_dtype_order():
    chunk = np.ones((4, 14), dtype=np.float32)
    mean = np.linspace(-0.7, 0.9, 14, dtype=np.float32)
    std = np.linspace(0.03, 1.1, 14, dtype=np.float32)
    qpos = np.linspace(-0.91, 0.83, 14, dtype=np.float64)
    model = RecordingACT([chunk])
    adapter = OriginalACTSpeedAdapter(
        model,
        camera_names=CAMERAS,
        qpos_mean=mean,
        qpos_std=std,
        action_mean=np.zeros(14),
        action_std=np.ones(14),
        episode_len=5,
        num_queries=4,
        device="cpu",
    )
    observation = _observation()
    observation["qpos"] = qpos

    adapter.action(observation, speed=1.0)

    reference = torch.as_tensor((qpos - mean) / std, dtype=torch.float32)
    torch.testing.assert_close(model.inputs[0][0][0, :14], reference, rtol=0, atol=0)


def test_float64_stats_and_action_denormalization_match_frozen_evaluator():
    chunk = np.stack(
        [np.linspace(-0.3 + index, 0.8 + index, 14) for index in range(4)]
    ).astype(np.float32)
    qpos_mean = np.linspace(-0.2, 0.2, 14, dtype=np.float64)
    qpos_std = np.linspace(0.01, 0.4, 14, dtype=np.float64)
    action_mean = np.linspace(-0.1, 0.1, 14, dtype=np.float64)
    action_std = np.linspace(0.01, 0.2, 14, dtype=np.float64)
    model = RecordingACT([chunk])
    adapter = OriginalACTSpeedAdapter(
        model,
        camera_names=CAMERAS,
        qpos_mean=qpos_mean,
        qpos_std=qpos_std,
        action_mean=action_mean,
        action_std=action_std,
        episode_len=5,
        num_queries=4,
        device="cpu",
    )
    observation = _observation()
    observation["qpos"] = np.linspace(-0.9, 0.9, 14, dtype=np.float64)

    action = adapter.action(observation, speed=1.0)

    expected_qpos = torch.as_tensor(
        (observation["qpos"] - qpos_mean) / qpos_std, dtype=torch.float32
    )
    expected_action = chunk[0] * action_std + action_mean
    torch.testing.assert_close(model.inputs[0][0][0, :14], expected_qpos, rtol=0, atol=0)
    np.testing.assert_array_equal(action, expected_action)
    assert action.dtype == np.float64


def test_multiview_act_progress_tracks_nominal_policy_time_at_acceleration():
    chunk = np.ones((4, 14), dtype=np.float32)
    adapter, model = _adapter([chunk, chunk], episode_len=9)

    adapter.action(_observation(), speed=2.0)
    adapter.action(_observation(), speed=1.5)

    assert model.inputs[0][0][0, 14].item() == pytest.approx(-1.0)
    assert model.inputs[1][0][0, 14].item() == pytest.approx(-0.5)
    assert adapter.policy_time == pytest.approx(3.5)


def test_specialized_adapter_survives_public_chunk_wrapper_and_decision_boundaries():
    chunk = np.ones((4, 14), dtype=np.float32)
    adapter, model = _adapter([chunk])
    source = ChunkedActionSource(ChunkPredictorAdapter(adapter))
    timestep = SimpleNamespace(observation=_observation())

    source.reset()
    source.begin_decision(timestep, speed=1.0)
    action = source.action(timestep, speed=1.0)

    assert action.shape == (14,)
    assert len(model.inputs) == 1


def test_policy_and_phase_detector_camera_requirements_are_unioned(monkeypatch):
    captured = {}

    class FakePredictor:
        per_physics_step_action = True
        render_camera_names = CAMERAS

        def reset(self):
            return None

        def action(self, observation, speed):
            del observation, speed
            return np.zeros(14)

    class AngleEncoder:
        render_camera_names = ("angle",)

        @staticmethod
        def output_dim(env_state_dim):
            return env_state_dim + 28

        def __call__(self, observation):
            del observation
            return np.zeros(28)

    class FakeEnv:
        physics = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(16)))
        task = SimpleNamespace(max_reward=1)

    def fake_make_sim_env(*args, **kwargs):
        del args
        captured["render_camera_names"] = kwargs["render_camera_names"]
        return FakeEnv()

    monkeypatch.setattr("policy_speed_env.make_sim_env", fake_make_sim_env)
    create_speed_env(
        "pick_and_place",
        chunk_predictor=FakePredictor(),
        observation_encoder=AngleEncoder(),
    )

    assert captured["render_camera_names"] == CAMERAS
