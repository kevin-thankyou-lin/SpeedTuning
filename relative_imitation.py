"""Matched ACT and Diffusion training on observation-anchored joint deltas."""

from __future__ import annotations

import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from imitation_data import fit_normalization, save_normalization


class RelativeJointDataset(Dataset):
    def __init__(self, paths, stats, chunk_size):
        self.paths = tuple(map(Path, paths))
        self.chunk_size = int(chunk_size)
        self.qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
        self.qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
        self.delta_mean = np.asarray(stats["delta_mean"], dtype=np.float32)
        self.delta_std = np.asarray(stats["delta_std"], dtype=np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with h5py.File(self.paths[index], "r") as root:
            length = len(root["target_qpos"])
            start = random.randrange(length)
            first_qpos = np.asarray(root["observations/qpos"][start], dtype=np.float32)
            image = np.asarray(root["observations/images/angle"][start], dtype=np.uint8)
            targets = np.asarray(
                root["target_qpos"][start : start + self.chunk_size], dtype=np.float32
            )
        valid = len(targets)
        padded = np.repeat(targets[-1:], self.chunk_size, axis=0)
        padded[:valid] = targets
        delta = padded - first_qpos[None]
        delta = (delta - self.delta_mean) / self.delta_std
        is_pad = np.arange(self.chunk_size) >= valid
        return (
            torch.from_numpy(image.transpose(2, 0, 1)[None]).float() / 255.0,
            torch.from_numpy((first_qpos - self.qpos_mean) / self.qpos_std),
            torch.from_numpy(delta),
            torch.from_numpy(is_pad),
        )


def split_episodes(dataset_dir, validation_fraction=0.1, seed=0):
    paths = sorted(Path(dataset_dir).glob("episode_*.hdf5"))
    if len(paths) < 2:
        raise ValueError("at least two episodes are required")
    random.Random(seed).shuffle(paths)
    validation_count = max(1, round(len(paths) * validation_fraction))
    return paths[validation_count:], paths[:validation_count]


def prepare_datasets(dataset_dir, output_dir, chunk_size, split_seed=0):
    train_paths, validation_paths = split_episodes(dataset_dir, seed=split_seed)
    stats = fit_normalization(train_paths, chunk_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_normalization(stats, output_dir / "normalization.npz")
    (output_dir / "split.json").write_text(
        json.dumps(
            {
                "train": [str(path) for path in train_paths],
                "validation": [str(path) for path in validation_paths],
                "normalization_fit": "train split only",
            },
            indent=2,
        )
        + "\n"
    )
    return (
        RelativeJointDataset(train_paths, stats, chunk_size),
        RelativeJointDataset(validation_paths, stats, chunk_size),
        stats,
    )


class DiffusionJointPolicy(nn.Module):
    """Image-conditioned denoising policy with an ACT-compatible interface."""

    def __init__(self, config):
        super().__init__()
        from diffusers import DDIMScheduler
        from torchvision.models import ResNet18_Weights, resnet18
        from diffusion_unet import ConditionalUnet1D

        self.num_queries = int(config["num_queries"])
        self.num_inference_steps = int(config.get("num_inference_steps", 10))
        self.image_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_encoder.fc = nn.Identity()
        self.qpos_encoder = nn.Sequential(nn.Linear(14, 128), nn.Mish(), nn.Linear(128, 128))
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=14,
            global_cond_dim=640,
            down_dims=[128, 256, 512],
        )
        self.scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            prediction_type="epsilon",
        )
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=float(config["lr"]), weight_decay=1e-6
        )

    def _condition(self, qpos, image):
        from torchvision.transforms.functional import normalize

        batch, cameras = image.shape[:2]
        image = image.flatten(0, 1)
        image = F.interpolate(image, size=(240, 320), mode="bilinear", align_corners=False)
        image = normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        visual = self.image_encoder(image).reshape(batch, cameras, -1).mean(1)
        return torch.cat((visual, self.qpos_encoder(qpos)), dim=-1)

    def forward(self, qpos, image, actions=None, is_pad=None):
        condition = self._condition(qpos, image)
        if actions is not None:
            actions = actions[:, : self.num_queries]
            valid = ~is_pad[:, : self.num_queries]
            noise = torch.randn_like(actions)
            timestep = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (len(actions),),
                device=actions.device,
            )
            noisy = self.scheduler.add_noise(actions, noise, timestep)
            predicted = self.noise_pred_net(noisy, timestep, global_cond=condition)
            loss = ((predicted - noise).square() * valid.unsqueeze(-1)).sum()
            loss = loss / valid.sum().clamp_min(1) / actions.shape[-1]
            return {"loss": loss, "l2": loss}
        sample = torch.randn(
            len(qpos), self.num_queries, 14, device=qpos.device, dtype=qpos.dtype
        )
        self.scheduler.set_timesteps(self.num_inference_steps, device=qpos.device)
        for timestep in self.scheduler.timesteps:
            predicted = self.noise_pred_net(sample, timestep, global_cond=condition)
            sample = self.scheduler.step(predicted, timestep, sample).prev_sample
        return sample

    def configure_optimizers(self):
        return self.optimizer


def create_policy(kind, chunk_size, device, lr=1e-4):
    config = {
        "camera_names": ["angle"],
        "num_queries": int(chunk_size),
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "enc_layers": 4,
        "dec_layers": 6,
        "nheads": 8,
        "backbone": "resnet18",
        "pretrained_backbone": True,
        "kl_weight": 10.0,
        "lr": float(lr),
        "lr_backbone": float(lr) / 10,
        "device": str(device),
    }
    if kind == "act":
        from policy import ACTPolicy

        return ACTPolicy(config), config
    if kind == "diffusion":
        return DiffusionJointPolicy(config).to(device), config
    raise ValueError("kind must be act or diffusion")


class RelativeChunkPredictor:
    """Decode normalized relative chunks back to absolute joint commands."""

    def __init__(self, checkpoint, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        config = payload["policy_config"]
        self.model, _ = create_policy(
            payload["kind"], config["num_queries"], self.device, config["lr"]
        )
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        stats = payload["stats"]
        self.qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
        self.qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
        self.delta_mean = np.asarray(stats["delta_mean"], dtype=np.float32)
        self.delta_std = np.asarray(stats["delta_std"], dtype=np.float32)

    def __call__(self, observation):
        first_qpos = np.asarray(observation["qpos"], dtype=np.float32)
        normalized_qpos = (first_qpos - self.qpos_mean) / self.qpos_std
        image = np.asarray(observation["images"]["angle"], dtype=np.uint8)
        image = torch.from_numpy(image.transpose(2, 0, 1)[None, None]).float()
        qpos = torch.from_numpy(normalized_qpos[None]).float()
        with torch.inference_mode():
            normalized_delta = self.model(
                qpos.to(self.device), image.to(self.device) / 255.0
            )[0].cpu().numpy()
        delta = normalized_delta * self.delta_std + self.delta_mean
        return first_qpos[None] + delta
