"""Causal runtime helpers for the supervised protected-segment controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class CausalTemporalFeatureBuffer:
    """Match ``temporal_features`` one observation at a time."""

    def __init__(self) -> None:
        self._first: np.ndarray | None = None
        self._history: list[np.ndarray] = []

    def update(self, value: np.ndarray) -> np.ndarray:
        current = np.asarray(value, dtype=np.float32).reshape(-1)
        if self._first is None:
            self._first = current.copy()
        previous_1 = self._history[-1] if self._history else self._first
        previous_3 = self._history[-3] if len(self._history) >= 3 else self._first
        self._history.append(current.copy())
        if len(self._history) > 3:
            self._history.pop(0)
        return np.concatenate(
            [current, current - previous_1, current - previous_3]
        ).astype(np.float32, copy=False)


@dataclass
class ConservativeBinaryDecoder:
    """Stateful form of the validation-tuned conservative decoder."""

    classes: np.ndarray
    risk_threshold: float
    exit_threshold: float
    exit_stability: int

    def __post_init__(self) -> None:
        self.classes = np.asarray(self.classes)
        fast = np.flatnonzero(self.classes == "fast")
        if len(fast) != 1:
            raise ValueError("classes must contain exactly one fast label")
        self._fast_index = int(fast[0])
        self._risk_indices = np.flatnonzero(self.classes != "fast")
        if not len(self._risk_indices):
            raise ValueError("classes must contain at least one protected label")
        if self.exit_stability <= 0:
            raise ValueError("exit_stability must be positive")
        self._active: int | None = None
        self._fast_streak = 0

    def update(self, probabilities: np.ndarray) -> str:
        row = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if row.shape != self.classes.shape:
            raise ValueError("probabilities must match classes")
        if not np.all(np.isfinite(row)):
            raise ValueError("probabilities must be finite")

        risk_probability = float(row[self._risk_indices].sum())
        best_risk = int(
            self._risk_indices[np.argmax(row[self._risk_indices])]
        )
        if self._active is None:
            if risk_probability >= self.risk_threshold:
                self._active = best_risk
                self._fast_streak = 0
        else:
            if row[self._fast_index] >= self.exit_threshold:
                self._fast_streak += 1
            else:
                self._fast_streak = 0
                if risk_probability >= self.risk_threshold:
                    self._active = best_risk
            if self._fast_streak >= self.exit_stability:
                self._active = None
                self._fast_streak = 0
        return "fast" if self._active is None else str(self.classes[self._active])

    def state(self) -> dict[str, object]:
        return {
            "active_label": (
                None if self._active is None else str(self.classes[self._active])
            ),
            "fast_streak": self._fast_streak,
        }


def shared_fast_speed(
    label: str, *, fast_speed: float, protected_speed: float
) -> float:
    """Map every protected label to one speed and ``fast`` to one ceiling."""

    speed = fast_speed if label == "fast" else protected_speed
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speeds must be finite and positive")
    return float(speed)


def compose_online_features(
    method: str,
    visual: np.ndarray,
    action: np.ndarray,
    visual_buffer: CausalTemporalFeatureBuffer,
    action_buffer: CausalTemporalFeatureBuffer,
) -> np.ndarray:
    if method == "visual":
        return visual_buffer.update(visual)
    if method == "action":
        return action_buffer.update(action)
    if method == "fused":
        visual_temporal = visual_buffer.update(visual)
        action_temporal = action_buffer.update(action)
        return np.concatenate([visual_temporal, action_temporal])
    raise ValueError(f"unknown feature method: {method}")
