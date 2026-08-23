"""Faithful simulated-ACT dataset, model, and inference utilities.

This module intentionally preserves the public ACT repository's choices:
absolute target joint positions, full-episode padding, all-episode mean/std
statistics, an 80/20 episode split, and the published model dimensions.
"""

from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


ORIGINAL_ACT_CONFIG = {
    "camera_names": ["top"],
    "num_queries": 100,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "backbone": "resnet18",
    "pretrained_backbone": True,
    "kl_weight": 10.0,
    "lr": 1e-5,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "qpos_dim": 14,
    "action_dim": 14,
    "original_loss_reduction": True,
    "deterministic_training": False,
}


def episode_paths(dataset_dir):
    paths = sorted(Path(dataset_dir).glob("episode_*.hdf5"))
    if not paths:
        raise ValueError(f"no ACT episodes in {dataset_dir}")
    return paths


def fit_original_act_stats(paths):
    """Fit the upstream per-joint mean/std over every saved episode."""

    all_qpos, all_actions = [], []
    for path in map(Path, paths):
        with h5py.File(path, "r") as root:
            all_qpos.append(torch.from_numpy(root["observations/qpos"][()]))
            all_actions.append(torch.from_numpy(root["action"][()]))
    qpos = torch.stack(all_qpos)
    actions = torch.stack(all_actions)
    return {
        "qpos_mean": qpos.mean(dim=(0, 1)).numpy(),
        "qpos_std": torch.clamp(qpos.std(dim=(0, 1)), min=1e-2).numpy(),
        "action_mean": actions.mean(dim=(0, 1)).numpy(),
        "action_std": torch.clamp(actions.std(dim=(0, 1)), min=1e-2).numpy(),
    }


def split_original_act_episodes(paths, seed=1, validation_count=None):
    """Match ACT's 80/20 episode split after NumPy seed 1."""

    paths = list(map(Path, paths))
    if len(paths) < 2:
        raise ValueError("ACT needs at least two episodes")
    if validation_count is not None and not 1 <= validation_count < len(paths):
        raise ValueError("validation_count must leave at least one training episode")
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(paths))
    boundary = (
        int(0.8 * len(paths))
        if validation_count is None
        else len(paths) - validation_count
    )
    return ([paths[i] for i in indices[:boundary]], [paths[i] for i in indices[boundary:]])


def normalized_episode_progress(step, episode_len):
    """Map an observation index to a stable [-1, 1] episode-progress feature."""

    if episode_len < 2:
        raise ValueError("episode_len must be at least two")
    if not 0 <= step < episode_len:
        raise ValueError("step must be inside the episode")
    return np.float32(2.0 * step / (episode_len - 1) - 1.0)


class OriginalACTDataset(Dataset):
    """One random observation and its padded absolute-action suffix per episode."""

    def __init__(self, paths, stats, camera_names=("top",), include_progress=False):
        self.paths = tuple(map(Path, paths))
        self.stats = stats
        self.camera_names = tuple(camera_names)
        self.include_progress = bool(include_progress)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with h5py.File(self.paths[index], "r") as root:
            episode_len, action_dim = root["action"].shape
            start = np.random.choice(episode_len)
            qpos = np.asarray(root["observations/qpos"][start], dtype=np.float32)
            images = np.stack(
                [root[f"observations/images/{name}"][start] for name in self.camera_names]
            )
            actions = np.asarray(root["action"][start:], dtype=np.float32)
        padded = np.zeros((episode_len, action_dim), dtype=np.float32)
        padded[: len(actions)] = actions
        is_pad = np.zeros(episode_len, dtype=bool)
        is_pad[len(actions) :] = True
        qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        if self.include_progress:
            qpos = np.concatenate(
                [qpos, np.asarray([normalized_episode_progress(start, episode_len)])]
            )
        padded = (padded - self.stats["action_mean"]) / self.stats["action_std"]
        return (
            torch.from_numpy(images.transpose(0, 3, 1, 2)).float() / 255.0,
            torch.from_numpy(qpos).float(),
            torch.from_numpy(padded).float(),
            torch.from_numpy(is_pad),
        )


def create_original_act_policy(device=None, qpos_dim=None, camera_names=None):
    from policy import ACTPolicy

    config = dict(ORIGINAL_ACT_CONFIG)
    if qpos_dim is not None:
        config["qpos_dim"] = int(qpos_dim)
    if camera_names is not None:
        config["camera_names"] = list(camera_names)
    config["device"] = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return ACTPolicy(config), config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
