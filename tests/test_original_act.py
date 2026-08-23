import h5py
import numpy as np
import pytest
import torch

from original_act import (
    ORIGINAL_ACT_CONFIG,
    OriginalACTDataset,
    fit_original_act_stats,
    normalized_episode_progress,
    split_original_act_episodes,
)
from scripts.collect_original_act_data import _existing_records, _summary


def _episode(path, offset):
    with h5py.File(path, "w") as root:
        root.attrs["sim"] = True
        observations = root.create_group("observations")
        observations.create_dataset("qpos", data=np.full((4, 14), offset, dtype=np.float32))
        images = observations.create_group("images")
        images.create_dataset("top", data=np.zeros((4, 480, 640, 3), dtype=np.uint8))
        root.create_dataset("action", data=np.full((4, 14), offset + 1, dtype=np.float32))


def test_original_contract_and_dataset(tmp_path, monkeypatch):
    paths = []
    for index in range(10):
        path = tmp_path / f"episode_{index}.hdf5"
        _episode(path, index)
        paths.append(path)
    stats = fit_original_act_stats(paths)
    train, validation = split_original_act_episodes(paths)
    assert len(train) == 8 and len(validation) == 2
    monkeypatch.setattr(np.random, "choice", lambda _: 2)
    image, qpos, action, is_pad = OriginalACTDataset(paths, stats)[0]
    assert image.shape == (1, 3, 480, 640)
    assert qpos.shape == (14,)
    assert action.shape == (4, 14)
    assert is_pad.tolist() == [False, False, True, True]
    assert ORIGINAL_ACT_CONFIG == {
        "camera_names": ["top"], "num_queries": 100,
        "hidden_dim": 512, "dim_feedforward": 3200,
        "enc_layers": 4, "dec_layers": 7, "nheads": 8,
        "backbone": "resnet18", "pretrained_backbone": True,
        "kl_weight": 10.0, "lr": 1e-5, "lr_backbone": 1e-5,
        "weight_decay": 1e-4, "qpos_dim": 14, "action_dim": 14,
        "original_loss_reduction": True, "deterministic_training": False,
    }


def test_transposed_camera_frame_can_be_materialized_by_torch():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    channels_first = image.transpose(2, 0, 1).copy()
    assert channels_first.flags.c_contiguous
    assert torch.as_tensor(channels_first).shape == (3, 480, 640)


def test_original_split_accepts_exact_validation_count(tmp_path):
    paths = [tmp_path / f"episode_{index}.hdf5" for index in range(270)]
    first = split_original_act_episodes(paths, seed=1, validation_count=20)
    second = split_original_act_episodes(paths, seed=1, validation_count=20)
    train, validation = first
    assert len(train) == 250
    assert len(validation) == 20
    assert set(train).isdisjoint(validation)
    assert first == second


def test_original_split_rejects_invalid_validation_count(tmp_path):
    paths = [tmp_path / "episode_0.hdf5", tmp_path / "episode_1.hdf5"]
    for count in (0, 2, 3):
        with pytest.raises(ValueError, match="validation_count"):
            split_original_act_episodes(paths, validation_count=count)


def test_original_collection_reconstructs_resumable_prefix(tmp_path):
    for index in range(2):
        path = tmp_path / f"episode_{index}.hdf5"
        with h5py.File(path, "w") as root:
            root.attrs.update(
                task="pick_and_place",
                seed=100 + index,
                source_success=True,
                replay_success=index == 0,
            )
            root.create_dataset("action", data=np.zeros((4, 14), dtype=np.float32))
    records = _existing_records("pick_and_place", tmp_path, seed_base=100)
    assert [record["seed"] for record in records] == [100, 101]
    assert [record["replay_success"] for record in records] == [True, False]
    summary = _summary("pick_and_place", records)
    assert summary["attempted_episodes"] == 2
    assert summary["source_successes"] == 2
    assert summary["replay_successes"] == 1


def test_original_collection_rejects_noncontiguous_resume(tmp_path):
    path = tmp_path / "episode_1.hdf5"
    with h5py.File(path, "w") as root:
        root.attrs.update(
            task="pick_and_place", seed=101, source_success=True, replay_success=True
        )
        root.create_dataset("action", data=np.zeros((4, 14), dtype=np.float32))
    with pytest.raises(ValueError, match="contiguous"):
        _existing_records("pick_and_place", tmp_path, seed_base=100)


def test_normalized_episode_progress_has_stable_endpoints():
    assert normalized_episode_progress(0, 500) == np.float32(-1.0)
    assert normalized_episode_progress(499, 500) == np.float32(1.0)
    with pytest.raises(ValueError):
        normalized_episode_progress(500, 500)


def test_original_dataset_can_append_progress_feature(tmp_path, monkeypatch):
    path = tmp_path / "episode_0.hdf5"
    _episode(path, 0.0)
    stats = fit_original_act_stats([path])
    monkeypatch.setattr(np.random, "choice", lambda _: 2)
    sample = OriginalACTDataset([path], stats, include_progress=True)[0]
    assert sample[1].shape == (15,)
    assert sample[1][-1].item() == pytest.approx(normalized_episode_progress(2, 4))
