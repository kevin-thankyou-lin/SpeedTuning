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

from imitation_data import (
    fit_normalization,
    normalization_clipping_report,
    robust_denormalize,
    robust_normalize,
    save_normalization,
)


class RelativeJointDataset(Dataset):
    def __init__(self, paths, stats, chunk_size):
        self.paths = tuple(map(Path, paths))
        self.chunk_size = int(chunk_size)
        self.qpos_low = np.asarray(stats["qpos_low"], dtype=np.float32)
        self.qpos_high = np.asarray(stats["qpos_high"], dtype=np.float32)
        self.delta_low = np.asarray(stats["delta_low"], dtype=np.float32)
        self.delta_high = np.asarray(stats["delta_high"], dtype=np.float32)

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
            speed_condition = float(root.attrs["speed_condition"])
        valid = len(targets)
        padded = np.repeat(targets[-1:], self.chunk_size, axis=0)
        padded[:valid] = targets
        delta = padded - first_qpos[None]
        delta = robust_normalize(delta, self.delta_low, self.delta_high)
        is_pad = np.arange(self.chunk_size) >= valid
        conditioned_qpos = np.concatenate(
            (robust_normalize(first_qpos, self.qpos_low, self.qpos_high), [speed_condition])
        ).astype(np.float32)
        return (
            torch.from_numpy(image.transpose(2, 0, 1)[None]).float() / 255.0,
            torch.from_numpy(conditioned_qpos),
            torch.from_numpy(delta),
            torch.from_numpy(is_pad),
        )


def split_episodes(dataset_dir, validation_fraction=0.1, seed=0):
    paths = sorted(Path(dataset_dir).glob("episode_*.hdf5"))
    if len(paths) < 2:
        raise ValueError("at least two episodes are required")
    by_condition = {0: [], 1: []}
    for path in paths:
        with h5py.File(path, "r") as root:
            condition = int(root.attrs["speed_condition"])
        if condition not in by_condition:
            raise ValueError(f"invalid speed_condition={condition} in {path}")
        by_condition[condition].append(path)
    train, validation = [], []
    rng = random.Random(seed)
    for group in by_condition.values():
        if not group:
            continue
        rng.shuffle(group)
        validation_count = max(1, round(len(group) * validation_fraction))
        if validation_count >= len(group):
            raise ValueError("each represented condition needs at least two episodes")
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def prepare_datasets(dataset_dir, output_dir, chunk_size, split_seed=0):
    train_paths, validation_paths = split_episodes(dataset_dir, seed=split_seed)
    stats = fit_normalization(train_paths, chunk_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_normalization(stats, output_dir / "normalization.npz")
    clipping = {
        "normalization": stats["normalization"],
        "train": normalization_clipping_report(train_paths, stats, chunk_size),
        "validation": normalization_clipping_report(
            validation_paths, stats, chunk_size
        ),
    }
    (output_dir / "normalization_audit.json").write_text(
        json.dumps(clipping, indent=2) + "\n"
    )
    def condition_counts(paths):
        counts = {"slow": 0, "fast": 0}
        for path in paths:
            with h5py.File(path, "r") as root:
                key = "fast" if int(root.attrs["speed_condition"]) else "slow"
            counts[key] += 1
        return counts

    (output_dir / "split.json").write_text(
        json.dumps(
            {
                "train": [str(path) for path in train_paths],
                "validation": [str(path) for path in validation_paths],
                "normalization_fit": "train split only",
                "normalization": stats["normalization"],
                "normalization_audit": str(output_dir / "normalization_audit.json"),
                "train_condition_counts": condition_counts(train_paths),
                "validation_condition_counts": condition_counts(validation_paths),
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
        self.qpos_encoder = nn.Sequential(nn.Linear(15, 128), nn.Mish(), nn.Linear(128, 128))
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
        "qpos_dim": 15,
        "action_dim": 14,
    }
    if kind == "act":
        from policy import ACTPolicy

        return ACTPolicy(config), config
    if kind == "diffusion":
        return DiffusionJointPolicy(config).to(device), config
    raise ValueError("kind must be act or diffusion")


class RelativeChunkPredictor:
    """Decode normalized relative chunks back to absolute joint commands."""

    def __init__(self, checkpoint, speed_condition=0, device=None):
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
        self.qpos_low = np.asarray(stats["qpos_low"], dtype=np.float32)
        self.qpos_high = np.asarray(stats["qpos_high"], dtype=np.float32)
        self.delta_low = np.asarray(stats["delta_low"], dtype=np.float32)
        self.delta_high = np.asarray(stats["delta_high"], dtype=np.float32)
        self._clip_counts = {
            "qpos_below": 0,
            "qpos_above": 0,
            "qpos_elements": 0,
            "delta_below": 0,
            "delta_above": 0,
            "delta_elements": 0,
        }
        self.set_speed_condition(speed_condition)

    def set_speed_condition(self, speed_condition):
        """Set the externally controlled mode used at the next policy query."""

        self.speed_condition = float(speed_condition)
        if self.speed_condition not in (0.0, 1.0):
            raise ValueError("speed_condition must be 0 (slow) or 1 (fast)")
        return self

    def __call__(self, observation, speed_condition=None):
        if speed_condition is not None:
            self.set_speed_condition(speed_condition)
        first_qpos = np.asarray(observation["qpos"], dtype=np.float32)
        self._clip_counts["qpos_below"] += int(
            np.count_nonzero(first_qpos < self.qpos_low)
        )
        self._clip_counts["qpos_above"] += int(
            np.count_nonzero(first_qpos > self.qpos_high)
        )
        self._clip_counts["qpos_elements"] += int(first_qpos.size)
        normalized_qpos = np.concatenate(
            (
                robust_normalize(first_qpos, self.qpos_low, self.qpos_high),
                [self.speed_condition],
            )
        ).astype(np.float32)
        image = np.asarray(observation["images"]["angle"], dtype=np.uint8)
        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()[None, None]
        ).float()
        qpos = torch.from_numpy(normalized_qpos[None]).float()
        with torch.inference_mode():
            normalized_delta = self.model(
                qpos.to(self.device), image.to(self.device) / 255.0
            )[0].cpu().numpy()
        self._clip_counts["delta_below"] += int(
            np.count_nonzero(normalized_delta < -1.0)
        )
        self._clip_counts["delta_above"] += int(
            np.count_nonzero(normalized_delta > 1.0)
        )
        self._clip_counts["delta_elements"] += int(normalized_delta.size)
        delta = robust_denormalize(
            normalized_delta, self.delta_low, self.delta_high
        )
        return first_qpos[None] + delta

    def clipping_metrics(self):
        def summarize(prefix):
            below = self._clip_counts[f"{prefix}_below"]
            above = self._clip_counts[f"{prefix}_above"]
            total = self._clip_counts[f"{prefix}_elements"]
            denominator = max(total, 1)
            return {
                "elements": total,
                "below": below,
                "above": above,
                "below_fraction": below / denominator,
                "above_fraction": above / denominator,
                "clipped_fraction": (below + above) / denominator,
            }

        return {
            "qpos_input": summarize("qpos"),
            "delta_output": summarize("delta"),
        }
