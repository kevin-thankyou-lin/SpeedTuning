import h5py
import numpy as np

from original_act import (
    ORIGINAL_ACT_CONFIG,
    OriginalACTDataset,
    fit_original_act_stats,
    split_original_act_episodes,
)


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
