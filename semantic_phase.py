"""Future semantic-phase prediction and conservative chunk-stride selection.

The model contract is deliberately small: given causal features from the
current observation and proposed action chunk, predict one semantic phase ID
for each configured future chunk offset.  The runtime controller then maps
those IDs to speeds while refusing to skip a slower phase or a gripper
transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def parse_future_offsets(value: str | Sequence[int]) -> tuple[int, ...]:
    """Return unique, increasing, non-negative offsets that include zero."""

    if isinstance(value, str):
        offsets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        offsets = tuple(int(item) for item in value)
    if not offsets or offsets[0] != 0:
        raise ValueError("future offsets must begin at zero")
    if offsets != tuple(sorted(set(offsets))) or any(offset < 0 for offset in offsets):
        raise ValueError("future offsets must be unique, increasing, and non-negative")
    return offsets


def future_phase_targets(
    records: Sequence[Mapping[str, Any]],
    phase_ids: Sequence[str],
    offsets: str | Sequence[int],
) -> np.ndarray:
    """Align future phase IDs to nominal policy-time offsets.

    Labels are treated as piecewise constant between recorded samples.  Targets
    beyond the end of a trajectory retain the final phase.  This creates
    offline supervision only; no future observation or label is required by
    the deployed predictor.
    """

    offsets = parse_future_offsets(offsets)
    labels = np.asarray(phase_ids, dtype=str)
    if len(records) != len(labels) or len(labels) == 0:
        raise ValueError("records and phase_ids must have the same non-zero length")
    times = np.asarray(
        [float(record.get("policy_time", index)) for index, record in enumerate(records)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("record policy_time values must be finite and non-decreasing")

    targets = np.empty((len(labels), len(offsets)), dtype=labels.dtype)
    for column, offset in enumerate(offsets):
        target_times = times + float(offset)
        indices = np.searchsorted(times, target_times, side="right") - 1
        indices = np.clip(indices, 0, len(labels) - 1)
        targets[:, column] = labels[indices]
    return targets


@dataclass(frozen=True)
class FuturePhasePrediction:
    """Semantic phase IDs predicted at configured nominal chunk offsets."""

    offsets: tuple[int, ...]
    phase_ids: tuple[str, ...]

    def __post_init__(self):
        offsets = parse_future_offsets(self.offsets)
        phase_ids = tuple(str(value) for value in self.phase_ids)
        if len(offsets) != len(phase_ids):
            raise ValueError("offsets and phase_ids must have equal length")
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "phase_ids", phase_ids)

    def as_dict(self) -> dict[int, str]:
        return dict(zip(self.offsets, self.phase_ids))


class FuturePhaseSequencePredictor:
    """Serializable bundle of one categorical phase head per future offset."""

    def __init__(self, offsets: Sequence[int], models: Sequence[Any]):
        self.offsets = parse_future_offsets(offsets)
        self.models = tuple(models)
        if len(self.offsets) != len(self.models):
            raise ValueError("one phase model is required for every future offset")
        if not all(callable(getattr(model, "predict", None)) for model in self.models):
            raise TypeError("every future phase model must define predict(features)")

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features)
        if features.ndim == 1:
            features = features[None, :]
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("features must have shape [batch, feature]")
        columns = [np.asarray(model.predict(features), dtype=str) for model in self.models]
        if any(column.shape != (len(features),) for column in columns):
            raise ValueError("phase heads must return one label per feature row")
        return np.stack(columns, axis=1)

    def predict_one(self, features: np.ndarray) -> FuturePhasePrediction:
        values = self.predict(np.asarray(features))
        if len(values) != 1:
            raise ValueError("predict_one expects exactly one feature row")
        return FuturePhasePrediction(self.offsets, tuple(values[0]))


class SemanticPhaseStrideController:
    """Map predicted semantic phases to a safe action-chunk stride.

    A candidate stride is reduced before it crosses any future phase with a
    lower configured speed.  Missing dense predictions, unknown phases, and
    gripper transitions fail closed to ``fallback_speed``.
    """

    def __init__(
        self,
        phase_speeds: Mapping[str, float],
        *,
        fallback_speed: float = 1.0,
        gripper_indices: Sequence[int] = (6, 13),
        gripper_transition_threshold: float = 0.25,
    ):
        self.phase_speeds = {
            str(phase): self._validate_speed(speed)
            for phase, speed in phase_speeds.items()
        }
        if not self.phase_speeds:
            raise ValueError("phase_speeds must not be empty")
        self.fallback_speed = self._validate_speed(fallback_speed)
        self.gripper_indices = tuple(int(index) for index in gripper_indices)
        if any(index < 0 for index in self.gripper_indices):
            raise ValueError("gripper indices must be non-negative")
        self.gripper_transition_threshold = float(gripper_transition_threshold)
        if not np.isfinite(self.gripper_transition_threshold) or self.gripper_transition_threshold <= 0:
            raise ValueError("gripper transition threshold must be positive")

    @staticmethod
    def _validate_speed(speed: float) -> float:
        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("phase speeds must be finite and positive")
        return speed

    def choose_stride(
        self,
        actions: np.ndarray,
        prediction: FuturePhasePrediction,
    ) -> float:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or len(actions) == 0:
            raise ValueError("actions must have shape [time, action]")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions must be finite")

        phase_by_offset = prediction.as_dict()
        current_phase = phase_by_offset[0]
        candidate = self.phase_speeds.get(current_phase, self.fallback_speed)
        required_offsets = range(1, int(np.ceil(candidate)) + 1)
        if any(offset not in phase_by_offset for offset in required_offsets):
            return self.fallback_speed
        for offset in required_offsets:
            candidate = min(
                candidate,
                self.phase_speeds.get(phase_by_offset[offset], self.fallback_speed),
            )

        scan_steps = min(int(np.ceil(candidate)), len(actions) - 1)
        for index in self.gripper_indices:
            if index >= actions.shape[1]:
                raise ValueError(f"gripper index {index} is outside action dimension")
            deltas = np.abs(np.diff(actions[: scan_steps + 1, index]))
            if np.any(deltas >= self.gripper_transition_threshold):
                return self.fallback_speed
        return candidate
