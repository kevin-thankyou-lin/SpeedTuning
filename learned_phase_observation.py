"""Frozen RGB+proprio phase observation for deployable speed control."""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np


PHASES = ("pre_grasp", "grasp_lift", "transport", "interaction")
PROPRIO_FEATURES = (
    "effector.left.delta", "effector.left.x", "effector.left.y", "effector.left.z",
    "effector.right.delta", "effector.right.x", "effector.right.y", "effector.right.z",
    "gripper.left", "gripper.right",
)
_PREDICTORS = {}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LearnedPhaseEncoder:
    """Return raw phase argmax from the sealed deployable detector."""

    requires_images = True
    output_dim = 4

    def __init__(
        self,
        checkpoint_path,
        source_root,
        checkpoint_sha256,
        inference_sha256,
        model_source_sha256,
        device="auto",
        history_stride=5,
        predictor=None,
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.source_root = Path(source_root).resolve()
        self.hashes = {
            "checkpoint": str(checkpoint_sha256),
            "inference": str(inference_sha256),
            "model_source": str(model_source_sha256),
        }
        self.history_stride = int(history_stride)
        if self.history_stride <= 0:
            raise ValueError("history_stride must be positive")
        if predictor is None:
            actual = {
                "checkpoint": sha256(self.checkpoint_path),
                "inference": sha256(self.source_root / "phase_detector/rgb_inference.py"),
                "model_source": sha256(self.source_root / "phase_detector/rgb_proprio.py"),
            }
            if actual != self.hashes:
                raise ValueError("learned phase detector hash mismatch")
            if str(self.source_root) not in sys.path:
                sys.path.insert(0, str(self.source_root))
            key = (str(self.source_root), str(self.checkpoint_path), str(device))
            if key not in _PREDICTORS:
                module = importlib.import_module("phase_detector.rgb_inference")
                _PREDICTORS[key] = (
                    module.RGBPhasePredictor(self.checkpoint_path, device=device), module
                )
            predictor, module = _PREDICTORS[key]
            self.causal_history_windows = module.causal_history_windows
        else:
            self.causal_history_windows = predictor.causal_history_windows
        if tuple(predictor.phases) != PHASES:
            raise ValueError("phase order differs from the speed-controller contract")
        if tuple(predictor.proprio_feature_names) != PROPRIO_FEATURES:
            raise ValueError("proprio feature order differs from the detector contract")
        self.predictor = predictor
        self.reset()

    def reset(self):
        self.raw_proprio = []
        self.previous_effectors = {}
        self.phase_index = 0

    def _proprio(self, observation):
        effectors = {
            side: np.asarray(observation[f"mocap_pose_{side}"], dtype=np.float64)[:3]
            for side in ("left", "right")
        }
        qpos = np.asarray(observation["qpos"], dtype=np.float64)
        runtime = {}
        for side, gripper_index in (("left", 6), ("right", 13)):
            previous = self.previous_effectors.get(side)
            runtime[f"effector.{side}.delta"] = (
                0.0 if previous is None else float(np.linalg.norm(effectors[side] - previous))
            )
            self.previous_effectors[side] = effectors[side].copy()
            for axis, axis_name in enumerate(("x", "y", "z")):
                runtime[f"effector.{side}.{axis_name}"] = float(effectors[side][axis])
            runtime[f"gripper.{side}"] = float(qpos[gripper_index])
        return [runtime[name] for name in PROPRIO_FEATURES]

    def __call__(self, observation):
        self.raw_proprio.append(self._proprio(observation))
        history = self.causal_history_windows(
            np.asarray(self.raw_proprio, dtype=np.float32),
            history=self.predictor.history,
            stride=self.history_stride,
        )[-1]
        prediction = self.predictor.predict(
            np.asarray(observation["images"]["angle"], dtype=np.uint8), history
        )
        self.phase_index = int(prediction["phase_index"])
        value = np.zeros(len(PHASES), dtype=np.float32)
        value[self.phase_index] = 1.0
        return value

    def decision_token(self):
        return self.phase_index

    def spec(self):
        return {
            "type": "sealed_rgb_proprio_phase_one_hot",
            "phases": list(PHASES),
            "checkpoint_sha256": self.hashes["checkpoint"],
            "inference_sha256": self.hashes["inference"],
            "model_source_sha256": self.hashes["model_source"],
            "inputs": "current angle RGB plus causal robot-only proprioception",
            "history_stride": self.history_stride,
            "temporal_postprocessing": "none_raw_argmax",
        }


def create_learned_phase_encoder(**kwargs):
    kwargs.pop("task_name", None)
    kwargs.pop("checkpoint", None)
    return LearnedPhaseEncoder(**kwargs)
