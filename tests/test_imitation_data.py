import numpy as np
import h5py

from imitation_data import decode_relative_chunk, fit_normalization, relative_chunk


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
            root.create_group("observations").create_dataset("qpos", data=qpos)
            root.create_dataset("target_qpos", data=target)
        paths.append(path)
    stats = fit_normalization(paths, chunk_size=3)
    assert stats["qpos_mean"].shape == (14,)
    assert stats["delta_mean"].shape == (3, 14)
    assert stats["delta_std"].shape == (3, 14)
    assert np.all(stats["delta_std"] >= 1e-3)
