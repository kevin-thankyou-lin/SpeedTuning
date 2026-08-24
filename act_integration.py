"""Load retained ACT checkpoints through the public chunk-policy interface."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np

from chunked_policy import TorchChunkPredictor
from original_act import normalized_episode_progress
from sim_tasks import get_task_spec, normalize_task_name


REQUIRED_STATS = ("qpos_mean", "qpos_std", "action_mean", "action_std")
DETERMINISTIC_CUBLAS_CONFIGS = (":4096:8", ":16:8")


def configure_act_inference_determinism(device="cuda"):
    """Enable and describe the frozen deterministic ACT inference runtime."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("ACT integration requires: uv sync --extra learned") from exc
    resolved = torch.device(device)
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if resolved.type == "cuda" and workspace not in DETERMINISTIC_CUBLAS_CONFIGS:
        raise RuntimeError(
            "deterministic CUDA ACT inference requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before Python starts"
        )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return {
        "enabled": True,
        "scope": "torch_strict",
        "cublas_workspace_config": workspace,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "dense_position_grid": "one_indexed_arange_equivalent_to_cumsum_of_ones",
    }


def _load_mapping(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as values:
            return {key: values[key] for key in values.files}
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as stream:
            return pickle.load(stream)
    raise ValueError("ACT stats must use .npz, .json, .pkl, or .pickle")


def _checkpoint_parts(checkpoint, device):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("ACT integration requires: uv sync --extra learned") from exc
    if checkpoint is None:
        raise ValueError("An ACT checkpoint path is required")
    # ACT checkpoints may include NumPy normalization arrays and legacy config
    # objects. They therefore require pickle loading and must come from a
    # trusted source (normally the user's own task-policy training run).
    payload = torch.load(
        Path(checkpoint), map_location=device, weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError("ACT checkpoint must contain a state dictionary or payload")
    for key in ("model_state_dict", "policy_state_dict", "state_dict"):
        if key in payload:
            return payload, payload[key]
    # A raw torch state dictionary maps names to tensors.
    if payload and all(isinstance(key, str) for key in payload):
        return {}, payload
    raise ValueError("ACT checkpoint does not contain recognizable model weights")


def _resolve_config(payload, policy_config, camera_names, device):
    config = dict(payload.get("policy_config", {}))
    config.update(policy_config or {})
    if camera_names is not None:
        config["camera_names"] = list(camera_names)
    if not config.get("camera_names"):
        raise ValueError("ACT policy_config must provide camera_names")
    required = ("num_queries", "hidden_dim", "dim_feedforward", "enc_layers", "dec_layers", "nheads")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"ACT policy_config is missing: {', '.join(missing)}")
    config.setdefault("lr", 1e-4)
    config.setdefault("lr_backbone", 0.0)
    config.setdefault("kl_weight", 10.0)
    config.setdefault("backbone", "resnet18")
    config.setdefault("pretrained_backbone", False)
    config["device"] = device
    return config


def _resolve_stats(payload, stats_path):
    stats = dict(payload.get("stats", {}))
    if stats_path is not None:
        stats.update(_load_mapping(stats_path))
    missing = [key for key in REQUIRED_STATS if key not in stats]
    if missing:
        raise ValueError(
            "ACT normalization stats are missing: " + ", ".join(missing)
        )
    # The retained ACT evaluator uses the serialized NumPy arrays without a
    # dtype conversion.  The accepted multiview banks store float64 stats, so
    # downcasting here changes both normalized qpos and denormalized actions.
    return {key: np.asarray(stats[key]) for key in REQUIRED_STATS}


def load_act_policy(
    checkpoint,
    device="cpu",
    stats_path=None,
    policy_config=None,
    camera_names=None,
    strict=True,
):
    """Return the ACT module, resolved configuration, and normalization stats."""

    from policy import ACTPolicy

    payload, state_dict = _checkpoint_parts(checkpoint, device)
    config = _resolve_config(payload, policy_config, camera_names, device)
    stats = _resolve_stats(payload, stats_path)
    model = ACTPolicy(config)
    incompatible = model.load_state_dict(state_dict, strict=bool(strict))
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        # Keep the information available to callers without printing during imports.
        model.checkpoint_incompatibilities = {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
    model.eval()
    return model, config, stats


def build_act_chunk_predictor(
    task_name,
    checkpoint,
    device="cpu",
    stats_path=None,
    policy_config=None,
    camera_names=None,
    strict=True,
):
    """Factory usable as ``act_integration:build_act_chunk_predictor``."""

    del task_name
    model, config, stats = load_act_policy(
        checkpoint=checkpoint,
        device=device,
        stats_path=stats_path,
        policy_config=policy_config,
        camera_names=camera_names,
        strict=strict,
    )
    return TorchChunkPredictor(
        model=model,
        camera_names=config["camera_names"],
        qpos_mean=stats["qpos_mean"],
        qpos_std=stats["qpos_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
        device=device,
    )


class OriginalACTSpeedAdapter:
    """Retimed original-ACT inference with paper temporal aggregation.

    A fresh 100-step prediction is produced at every physics step.  Predictions
    are stored on the nominal policy-time axis, so uniform ``1x`` is exactly the
    original ACT inference rule while larger speeds skip nominal targets without
    changing the task policy or its observation contract.
    """

    per_physics_step_action = True

    def __init__(
        self,
        model,
        *,
        camera_names,
        qpos_mean,
        qpos_std,
        action_mean,
        action_std,
        episode_len,
        num_queries=100,
        temporal_ensemble_m=0.01,
        device=None,
    ):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("ACT integration requires: uv sync --extra learned") from exc

        self.torch = torch
        self.model = model
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.render_camera_names = tuple(camera_names)
        if self.render_camera_names != ("angle", "left_wrist", "right_wrist"):
            raise ValueError(
                "Frozen multiview ACT requires angle, left_wrist, and right_wrist cameras"
            )
        self.qpos_mean = np.asarray(qpos_mean)
        self.qpos_std = np.maximum(np.asarray(qpos_std), 1e-6)
        self.action_mean = np.asarray(action_mean)
        self.action_std = np.asarray(action_std)
        for value, name in (
            (self.qpos_mean, "qpos_mean"),
            (self.qpos_std, "qpos_std"),
            (self.action_mean, "action_mean"),
            (self.action_std, "action_std"),
        ):
            if value.shape != (14,):
                raise ValueError(f"{name} must have shape (14,)")
        self.episode_len = int(episode_len)
        self.num_queries = int(num_queries)
        self.temporal_ensemble_m = float(temporal_ensemble_m)
        if self.episode_len < 2:
            raise ValueError("episode_len must be at least two")
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if self.temporal_ensemble_m < 0:
            raise ValueError("temporal_ensemble_m must be non-negative")
        self.model.to(self.device).eval()
        self.reset()

    @property
    def policy_time(self):
        return self._policy_time

    def reset(self):
        self._policy_time = 0.0
        self._predictions = []

    def begin_decision(self, observation, speed):
        """Speed decisions must not change ACT's per-physics-step query rate."""

        del observation, speed

    def _model_chunk(self, observation):
        torch = self.torch
        if "images" not in observation:
            raise ValueError("The environment must render the three ACT cameras")
        missing = [
            name for name in self.render_camera_names
            if name not in observation["images"]
        ]
        if missing:
            raise ValueError("Missing ACT camera images: " + ", ".join(missing))

        # Preserve the frozen evaluator's numerical path exactly: simulator
        # qpos is normalized in its native dtype and only converted to float32
        # by torch.as_tensor below.  Casting qpos before normalization can move
        # the ACT input by a few ulps and invalidate strict uniform-1x parity.
        qpos = (np.asarray(observation["qpos"]) - self.qpos_mean) / self.qpos_std
        if qpos.shape != (14,):
            raise ValueError("ACT qpos observation must have shape (14,)")
        progress = normalized_episode_progress(self._policy_time, self.episode_len)
        model_qpos = np.concatenate([qpos, np.asarray([progress], dtype=np.float32)])
        images = np.stack(
            [observation["images"][name] for name in self.render_camera_names], axis=0
        ).transpose(0, 3, 1, 2).copy()
        qpos_tensor = torch.as_tensor(
            model_qpos, dtype=torch.float32, device=self.device
        )[None]
        image_tensor = torch.as_tensor(
            images, dtype=torch.float32, device=self.device
        )[None] / 255.0
        with torch.inference_mode():
            output = self.model(qpos_tensor, image_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        chunk = output
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.shape != (self.num_queries, 14):
            raise ValueError(
                f"ACT output must have shape ({self.num_queries}, 14), got {chunk.shape}"
            )
        if not bool(torch.all(torch.isfinite(chunk))):
            raise ValueError("ACT produced a non-finite action chunk")
        return chunk

    def predict_chunk(self, observation):
        """Expose the public chunk protocol without advancing nominal time."""

        normalized = self._model_chunk(observation).detach().cpu().numpy()
        return normalized * self.action_std + self.action_mean

    @staticmethod
    def _sample_chunk(chunk, offset):
        lower = int(np.floor(offset))
        upper = min(lower + 1, len(chunk) - 1)
        fraction = float(offset - lower)
        return chunk[lower] + fraction * (chunk[upper] - chunk[lower])

    def action(self, observation, speed):
        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")

        chunk = self._model_chunk(observation)
        self._predictions.append((self._policy_time, chunk))
        oldest_allowed = self._policy_time - (self.num_queries - 1)
        self._predictions = [
            item for item in self._predictions if item[0] >= oldest_allowed
        ]

        candidates = []
        for origin, prediction in self._predictions:
            offset = self._policy_time - origin
            if 0 <= offset <= self.num_queries - 1:
                candidate = self._sample_chunk(prediction, offset)
                # Match the original evaluator's populated-row test.
                if bool(self.torch.all(candidate != 0)):
                    candidates.append(candidate)
        if not candidates:
            raise RuntimeError("ACT temporal ensemble has no populated candidates")
        candidates = self.torch.stack(candidates)
        weights = np.exp(
            -self.temporal_ensemble_m * np.arange(len(candidates), dtype=np.float64)
        )
        weights /= weights.sum()
        weights = self.torch.as_tensor(
            weights, dtype=self.torch.float32, device=self.device
        )[:, None]
        normalized_action = (
            candidates * weights
        ).sum(dim=0).detach().cpu().numpy()
        action = normalized_action * self.action_std + self.action_mean
        self._policy_time += speed
        return np.asarray(action)


def build_original_act_speed_adapter(
    task_name,
    checkpoint,
    device="cpu",
    stats_path=None,
    policy_config=None,
    policy_config_path=None,
    strict=True,
    temporal_ensemble_m=0.01,
):
    """Load the frozen 3PV+wrist, progress-conditioned ACT speed adapter."""

    if policy_config_path is not None:
        file_config = _load_mapping(policy_config_path)
        file_config.update(policy_config or {})
        policy_config = file_config
    model, config, stats = load_act_policy(
        checkpoint=checkpoint,
        device=device,
        stats_path=stats_path,
        policy_config=policy_config,
        camera_names=("angle", "left_wrist", "right_wrist"),
        strict=strict,
    )
    if int(config.get("qpos_dim", 0)) != 15:
        raise ValueError("Frozen multiview ACT policy_config must set qpos_dim=15")
    if int(config["num_queries"]) != 100:
        raise ValueError("Frozen multiview ACT policy_config must set num_queries=100")
    task_name = normalize_task_name(task_name)
    return OriginalACTSpeedAdapter(
        model,
        camera_names=config["camera_names"],
        qpos_mean=stats["qpos_mean"],
        qpos_std=stats["qpos_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
        episode_len=get_task_spec(task_name).episode_len,
        num_queries=config["num_queries"],
        temporal_ensemble_m=temporal_ensemble_m,
        device=device,
    )


class ACTBackboneObservationEncoder:
    """Use a supplied ACT task policy's ResNet backbone for speed features."""

    requires_images = True

    def __init__(self, model, camera_names, include_qvel=True, device="cpu"):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.camera_names = tuple(camera_names)
        self.include_qvel = bool(include_qvel)
        self.backbone = model.model.backbones[0]
        self.backbone.to(self.device).eval()
        self.feature_dim = int(self.backbone.num_channels)
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

    def reset(self):
        return None

    def __call__(self, observation):
        torch = self.torch
        if "images" not in observation:
            raise ValueError("ACT-backbone speed observations require images")
        images = np.stack(
            [observation["images"][name] for name in self.camera_names]
        ).transpose(0, 3, 1, 2)
        tensor = torch.as_tensor(images, dtype=torch.float32, device=self.device) / 255.0
        tensor = (tensor - self.mean) / self.std
        features = []
        with torch.inference_mode():
            for image in tensor:
                backbone_features, _ = self.backbone(image.unsqueeze(0))
                feature_map = backbone_features[-1]
                features.append(feature_map.mean(dim=(2, 3)).squeeze(0))
        proprioception = [np.asarray(observation["qpos"], dtype=np.float32)]
        if self.include_qvel:
            proprioception.append(np.asarray(observation["qvel"], dtype=np.float32))
        return np.concatenate(
            proprioception
            + [torch.cat(features).detach().cpu().numpy().astype(np.float32)]
        )

    def output_dim(self, env_state_dim):
        del env_state_dim
        return 14 + (14 if self.include_qvel else 0) + len(self.camera_names) * self.feature_dim

    def spec(self):
        return {
            "type": "act_backbone",
            "camera_names": list(self.camera_names),
            "include_qpos": True,
            "include_qvel": self.include_qvel,
            "include_env_state": False,
            "feature_dim": self.feature_dim,
        }

    def state_dict(self):
        return {
            key: value.detach().cpu()
            for key, value in self.backbone.state_dict().items()
        }

    def load_state_dict(self, state_dict):
        self.backbone.load_state_dict(state_dict)


def build_act_observation_encoder(
    task_name,
    checkpoint,
    device="cpu",
    stats_path=None,
    policy_config=None,
    camera_names=None,
    include_qvel=True,
    strict=True,
):
    """Factory for the task-policy image-encoder ablation."""

    del task_name
    model, config, _ = load_act_policy(
        checkpoint=checkpoint,
        device=device,
        stats_path=stats_path,
        policy_config=policy_config,
        camera_names=camera_names,
        strict=strict,
    )
    return ACTBackboneObservationEncoder(
        model,
        config["camera_names"],
        include_qvel=include_qvel,
        device=device,
    )
