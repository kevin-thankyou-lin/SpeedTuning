import numpy as np
import h5py

from imitation_data import (
    decode_relative_chunk,
    fit_normalization,
    relative_chunk,
    robust_denormalize,
    robust_normalize,
    speed_condition_from_schedule,
)
from relative_imitation import RelativeJointDataset, split_episodes


def test_relative_chunk_uses_one_observation_anchor():
    first = np.arange(14, dtype=np.float32)
    targets = np.stack((first + 1, first + 3, first - 2))
    delta = relative_chunk(targets, first)
    np.testing.assert_allclose(delta[:, 0], [1, 3, -2])
    np.testing.assert_allclose(decode_relative_chunk(delta, first), targets)


def test_normalization_is_per_chunk_step_and_joint(tmp_path):
    paths = []
    for episode, scale in enumerate((1.0, 2.0)):
        path = tmp_path / f"episode_{episode}.hdf5"
        qpos = np.stack((np.zeros(14), np.ones(14), np.full(14, 2))) * scale
        target = qpos + np.arange(1, 4)[:, None]
        with h5py.File(path, "w") as root:
            root.attrs["speed_condition"] = episode % 2
            root.create_group("observations").create_dataset("qpos", data=qpos)
            root.create_dataset("target_qpos", data=target)
        paths.append(path)
    stats = fit_normalization(paths, chunk_size=3)
    assert stats["qpos_q01"].shape == (14,)
    assert stats["qpos_q99"].shape == (14,)
    assert stats["delta_q01"].shape == (3, 14)
    assert stats["delta_q99"].shape == (3, 14)
    assert stats["delta_low"].shape == (3, 14)
    assert np.all(stats["delta_high"] - stats["delta_low"] >= 1e-3 - 1e-7)


def test_q01_q99_scaling_clips_and_round_trips_inside_bounds():
    low = np.array([-2.0, 0.0], dtype=np.float32)
    high = np.array([2.0, 4.0], dtype=np.float32)
    value = np.array([-3.0, 2.0], dtype=np.float32)
    normalized = robust_normalize(value, low, high)
    np.testing.assert_allclose(normalized, [-1.0, 0.0])
    np.testing.assert_allclose(
        robust_denormalize(normalized, low, high), [-2.0, 2.0]
    )
    np.testing.assert_allclose(
        robust_denormalize([-4.0, 4.0], low, high), low + [0.0, 4.0]
    )


def test_speed_condition_is_zero_only_for_uniform_native():
    assert speed_condition_from_schedule([1, 1, 1, 1]) == 0
    assert speed_condition_from_schedule([1, 1, 1.5, 1]) == 1


def test_dataset_appends_raw_condition_without_changing_action_dim(tmp_path):
    path = tmp_path / "episode_0000.hdf5"
    qpos = np.zeros((2, 14), dtype=np.float32)
    target = np.ones((2, 14), dtype=np.float32)
    image = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    with h5py.File(path, "w") as root:
        root.attrs["speed_condition"] = 1
        observations = root.create_group("observations")
        observations.create_dataset("qpos", data=qpos)
        observations.create_group("images").create_dataset("angle", data=image)
        root.create_dataset("target_qpos", data=target)
    stats = {
        "qpos_low": np.full(14, -1.0),
        "qpos_high": np.full(14, 1.0),
        "delta_low": np.full((2, 14), -2.0),
        "delta_high": np.full((2, 14), 2.0),
    }
    _, conditioned_qpos, action, _ = RelativeJointDataset([path], stats, 2)[0]
    assert conditioned_qpos.shape == (15,)
    assert conditioned_qpos[-1].item() == 1.0
    assert action.shape == (2, 14)


def test_dataset_can_oversample_the_episode_start(tmp_path):
    path = tmp_path / "episode_0000.hdf5"
    qpos = np.stack((np.zeros(14), np.ones(14)), dtype=np.float32)
    target = qpos + 0.25
    image = np.stack(
        (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8, 3), 255, dtype=np.uint8),
        )
    )
    with h5py.File(path, "w") as root:
        root.attrs["speed_condition"] = 0
        observations = root.create_group("observations")
        observations.create_dataset("qpos", data=qpos)
        observations.create_group("images").create_dataset("angle", data=image)
        root.create_dataset("target_qpos", data=target)
    stats = {
        "qpos_low": np.full(14, -1.0),
        "qpos_high": np.full(14, 1.0),
        "delta_low": np.full((2, 14), -1.0),
        "delta_high": np.full((2, 14), 1.0),
    }
    dataset = RelativeJointDataset(
        [path], stats, chunk_size=2, episode_start_probability=1.0
    )

    image_tensor, conditioned_qpos, _, _ = dataset[0]

    assert image_tensor.max().item() == 0.0
    np.testing.assert_allclose(conditioned_qpos[:14], 0.0)


def test_dataset_rejects_invalid_episode_start_probability(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "must be in"):
        RelativeJointDataset([], {}, 2, episode_start_probability=1.1)


def test_split_is_stratified_by_condition(tmp_path):
    for index in range(20):
        path = tmp_path / f"episode_{index:04d}.hdf5"
        with h5py.File(path, "w") as root:
            root.attrs["speed_condition"] = int(index >= 10)
    train, validation = split_episodes(tmp_path, validation_fraction=0.2, seed=4)
    assert len(train) == 16
    assert len(validation) == 4
    with h5py.File(validation[0], "r") as root:
        labels = [int(root.attrs["speed_condition"])]
    for path in validation[1:]:
        with h5py.File(path, "r") as root:
            labels.append(int(root.attrs["speed_condition"]))
    assert labels.count(0) == labels.count(1) == 2
