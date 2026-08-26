"""Matched multiview Diffusion Policy utilities for the original ACT data.

This module deliberately consumes the exact HDF5 contract produced by
``scripts.collect_original_act_data``: absolute 14-D joint targets, three named
RGB cameras, and measured joint positions.  It does not depend on the older
relative-action or slow/fast-conditioned imitation path.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from detr.models.backbone import FrozenBatchNorm2d
from diffusion_unet import ConditionalUnet1D
from original_act import normalized_episode_progress


ORIGINAL_DIFFUSION_CONFIG = {
    "camera_names": ["angle", "left_wrist", "right_wrist"],
    "prediction_horizon": 16,
    "observation_horizon": 2,
    "action_horizon": 8,
    "action_dim": 14,
    "qpos_dim": 15,
    "num_train_timesteps": 100,
    "num_inference_steps": 100,
    "down_dims": [256, 512, 1024],
    "image_size": [120, 160],
    "pretrained_backbone": True,
}


def _safe_range(low, high, minimum_span=1e-3):
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    center = (low + high) / 2
    span = np.maximum(high - low, np.float32(minimum_span))
    return center - span / 2, center + span / 2


@dataclass(frozen=True)
class JointRangeNormalizer:
    """One global range per joint; never one range per horizon offset."""

    qpos_low: np.ndarray
    qpos_high: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray

    @classmethod
    def fit(cls, paths):
        qpos_low = np.full(14, np.inf, dtype=np.float32)
        qpos_high = np.full(14, -np.inf, dtype=np.float32)
        action_low = np.full(14, np.inf, dtype=np.float32)
        action_high = np.full(14, -np.inf, dtype=np.float32)
        for path in map(Path, paths):
            with h5py.File(path, "r") as root:
                qpos = np.asarray(root["observations/qpos"], dtype=np.float32)
                action = np.asarray(root["action"], dtype=np.float32)
            qpos_low = np.minimum(qpos_low, qpos.min(axis=0))
            qpos_high = np.maximum(qpos_high, qpos.max(axis=0))
            action_low = np.minimum(action_low, action.min(axis=0))
            action_high = np.maximum(action_high, action.max(axis=0))
        qpos_low, qpos_high = _safe_range(qpos_low, qpos_high)
        action_low, action_high = _safe_range(action_low, action_high)
        return cls(qpos_low, qpos_high, action_low, action_high)

    @staticmethod
    def _normalize(value, low, high):
        value = np.asarray(value, dtype=np.float32)
        return np.clip(2 * (value - low) / (high - low) - 1, -1, 1).astype(np.float32)

    @staticmethod
    def _denormalize(value, low, high):
        value = np.asarray(value, dtype=np.float32)
        return ((np.clip(value, -1, 1) + 1) * 0.5 * (high - low) + low).astype(np.float32)

    def normalize_qpos(self, value):
        return self._normalize(value, self.qpos_low, self.qpos_high)

    def normalize_action(self, value):
        return self._normalize(value, self.action_low, self.action_high)

    def denormalize_action(self, value):
        return self._denormalize(value, self.action_low, self.action_high)

    def state_dict(self):
        return {
            "schema": "original-diffusion-global-joint-range-v1",
            "qpos_low": self.qpos_low,
            "qpos_high": self.qpos_high,
            "action_low": self.action_low,
            "action_high": self.action_high,
        }

    @classmethod
    def from_state_dict(cls, value):
        if value.get("schema") != "original-diffusion-global-joint-range-v1":
            raise ValueError("unsupported diffusion normalizer schema")
        return cls(
            *(
                np.asarray(value[key], dtype=np.float32)
                for key in ("qpos_low", "qpos_high", "action_low", "action_high")
            )
        )


class OriginalDiffusionDataset(Dataset):
    """Aligned HDF5 timesteps as standard 16/2/8 DP sequences."""

    def __init__(
        self,
        paths,
        normalizer,
        camera_names=("angle", "left_wrist", "right_wrist"),
        prediction_horizon=16,
        observation_horizon=2,
        image_size=(120, 160),
        sample_stride=1,
    ):
        self.paths = tuple(map(Path, paths))
        self.normalizer = normalizer
        self.camera_names = tuple(camera_names)
        self.prediction_horizon = int(prediction_horizon)
        self.observation_horizon = int(observation_horizon)
        self.image_size = tuple(map(int, image_size))
        self.sample_stride = int(sample_stride)
        if self.observation_horizon != 2:
            raise ValueError("the matched baseline requires observation_horizon=2")
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        self.indices = []
        for path_index, path in enumerate(self.paths):
            with h5py.File(path, "r") as root:
                length = int(root["action"].shape[0])
                if set(root["observations/images"]) != set(self.camera_names):
                    raise ValueError(f"camera contract mismatch in {path}")
            self.indices.extend(
                (path_index, step) for step in range(0, length, self.sample_stride)
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        path_index, step = self.indices[index]
        with h5py.File(self.paths[path_index], "r") as root:
            length = int(root["action"].shape[0])
            observation_indices = np.clip(
                np.arange(step - self.observation_horizon + 1, step + 1), 0, length - 1
            )
            action_indices_raw = np.arange(
                step - self.observation_horizon + 1,
                step - self.observation_horizon + 1 + self.prediction_horizon,
            )
            action_indices = np.clip(action_indices_raw, 0, length - 1)
            images = np.stack(
                [
                    np.stack(
                        [root[f"observations/images/{name}"][obs_index] for name in self.camera_names]
                    )
                    for obs_index in observation_indices
                ]
            )
            # NumPy indexing permits the repeated boundary indices used for
            # sequence padding; h5py fancy indexing requires strictly
            # increasing unique indices.
            qpos = np.asarray(root["observations/qpos"][()], dtype=np.float32)[observation_indices]
            actions = np.asarray(root["action"][()], dtype=np.float32)[action_indices]
        progress = np.asarray(
            [normalized_episode_progress(int(i), length) for i in observation_indices],
            dtype=np.float32,
        )[:, None]
        qpos = np.concatenate((self.normalizer.normalize_qpos(qpos), progress), axis=-1)
        actions = self.normalizer.normalize_action(actions)
        is_pad = action_indices_raw >= length
        images = torch.from_numpy(images.transpose(0, 1, 4, 2, 3)).float() / 255
        original_shape = images.shape
        images = F.interpolate(
            images.flatten(0, 1),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).reshape(*original_shape[:3], *self.image_size)
        return (
            images,
            torch.from_numpy(qpos).float(),
            torch.from_numpy(actions).float(),
            torch.from_numpy(is_pad),
        )


class MultiviewObservationEncoder(nn.Module):
    """Shared visual weights with order-preserving camera feature concatenation."""

    def __init__(self, qpos_dim=15, camera_count=3, image_size=(120, 160), pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.qpos_encoder = nn.Sequential(nn.Linear(qpos_dim, 128), nn.Mish(), nn.Linear(128, 128))
        self.camera_count = int(camera_count)
        self.image_size = tuple(image_size)
        self.output_per_observation = self.camera_count * 512 + 128

    def forward(self, image, qpos):
        batch, observations, cameras = image.shape[:3]
        if cameras != self.camera_count:
            raise ValueError(f"expected {self.camera_count} cameras, got {cameras}")
        value = image.flatten(0, 2)
        value = F.interpolate(value, size=self.image_size, mode="bilinear", align_corners=False)
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=value.device, dtype=value.dtype)[None, :, None, None]
        std = torch.as_tensor([0.229, 0.224, 0.225], device=value.device, dtype=value.dtype)[None, :, None, None]
        value = (value - mean) / std
        visual = self.image_encoder(value).flatten(1).reshape(batch, observations, cameras * 512)
        proprio = self.qpos_encoder(qpos)
        return torch.cat((visual, proprio), dim=-1).flatten(1)


class OriginalDiffusionPolicy(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        from diffusers import DDPMScheduler

        self.config = {**ORIGINAL_DIFFUSION_CONFIG, **(config or {})}
        self.prediction_horizon = int(self.config["prediction_horizon"])
        self.observation_horizon = int(self.config["observation_horizon"])
        self.action_horizon = int(self.config["action_horizon"])
        self.num_inference_steps = int(self.config["num_inference_steps"])
        training_stride = int(
            self.config.get("training_sample_stride", self.action_horizon)
        )
        if training_stride != self.action_horizon:
            raise ValueError(
                "training_sample_stride must match action_horizon for executable-state training"
            )
        self.observation_encoder = MultiviewObservationEncoder(
            qpos_dim=int(self.config["qpos_dim"]),
            camera_count=len(self.config["camera_names"]),
            image_size=self.config["image_size"],
            pretrained=bool(self.config["pretrained_backbone"]),
        )
        condition_dim = self.observation_horizon * self.observation_encoder.output_per_observation
        self.noise_predictor = ConditionalUnet1D(
            input_dim=int(self.config["action_dim"]),
            global_cond_dim=condition_dim,
            down_dims=tuple(self.config["down_dims"]),
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=int(self.config["num_train_timesteps"]),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def compute_loss(self, qpos, image, actions, is_pad, generator=None):
        condition = self.observation_encoder(image, qpos)
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (len(actions),),
            device=actions.device,
            generator=generator,
        ).long()
        noisy = self.scheduler.add_noise(actions, noise, timesteps)
        prediction = self.noise_predictor(noisy, timesteps, global_cond=condition)
        valid = (~is_pad).unsqueeze(-1).expand_as(prediction)
        loss = ((prediction - noise).square() * valid).sum() / valid.sum().clamp_min(1)
        return loss

    def sample(self, qpos, image, generator=None):
        condition = self.observation_encoder(image, qpos)
        sample = torch.randn(
            (len(qpos), self.prediction_horizon, int(self.config["action_dim"])),
            device=qpos.device,
            dtype=qpos.dtype,
            generator=generator,
        )
        self.scheduler.set_timesteps(self.num_inference_steps, device=qpos.device)
        for timestep in self.scheduler.timesteps:
            prediction = self.noise_predictor(sample, timestep, global_cond=condition)
            sample = self.scheduler.step(prediction, timestep, sample, generator=generator).prev_sample
        return sample

    def executed_slice(self, prediction):
        start = self.observation_horizon - 1
        return prediction[..., start : start + self.action_horizon, :]


class ModelEMA:
    """Warm-started EMA matching the public Diffusion Policy schedule."""

    def __init__(self, model, power=0.75, maximum=0.9999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.power = float(power)
        self.maximum = float(maximum)
        self.optimization_step = 0

    def decay(self):
        step = max(0, self.optimization_step - 1)
        return min(self.maximum, 1 - (1 + step) ** (-self.power)) if step else 0.0

    @torch.no_grad()
    def update(self, source):
        decay = self.decay()
        source_state = source.state_dict()
        for name, target in self.model.state_dict().items():
            source_value = source_state[name].to(device=target.device, dtype=target.dtype)
            if torch.is_floating_point(target):
                target.mul_(decay).add_(source_value, alpha=1 - decay)
            else:
                target.copy_(source_value)
        self.optimization_step += 1

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "power": self.power,
            "maximum": self.maximum,
            "optimization_step": self.optimization_step,
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.power = float(state["power"])
        self.maximum = float(state["maximum"])
        self.optimization_step = int(state["optimization_step"])
