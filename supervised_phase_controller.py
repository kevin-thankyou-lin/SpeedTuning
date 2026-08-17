"""Causal runtime helpers for the supervised protected-segment controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def resolve_conservative_decoder_config(
    base: dict[str, object],
    *,
    risk_threshold: float | None = None,
    exit_threshold: float | None = None,
    exit_stability: int | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Apply explicit runtime safety margins to a frozen decoder config."""
    resolved = dict(base)
    overrides: dict[str, object] = {}
    requested = {
        "risk_threshold": risk_threshold,
        "exit_threshold": exit_threshold,
        "exit_stability": exit_stability,
    }
    for key, value in requested.items():
        if value is not None:
            resolved[key] = value
            overrides[key] = value

    for key in ("risk_threshold", "exit_threshold"):
        value = float(resolved[key])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be finite and within [0, 1]")
        resolved[key] = value
    stability = int(resolved["exit_stability"])
    if stability <= 0:
        raise ValueError("exit_stability must be positive")
    resolved["exit_stability"] = stability
    return resolved, overrides


@dataclass
class PortableStandardizedLogisticRegression:
    """NumPy inference for a fitted StandardScaler + LogisticRegression."""

    classes: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray

    @classmethod
    def load(cls, path) -> "PortableStandardizedLogisticRegression":
        with np.load(path, allow_pickle=False) as payload:
            return cls(
                classes=payload["classes"],
                mean=payload["mean"],
                scale=payload["scale"],
                coef=payload["coef"],
                intercept=payload["intercept"],
            )

    def __post_init__(self) -> None:
        self.classes = np.asarray(self.classes).astype(str)
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        self.scale = np.asarray(self.scale, dtype=np.float64).reshape(-1)
        self.coef = np.asarray(self.coef, dtype=np.float64)
        self.intercept = np.asarray(self.intercept, dtype=np.float64).reshape(-1)
        if np.any(self.scale <= 0) or not np.all(np.isfinite(self.scale)):
            raise ValueError("scale must be finite and positive")
        if self.coef.shape[1] != len(self.mean):
            raise ValueError("coefficient and scaler dimensions do not match")
        if self.coef.shape[0] != len(self.intercept):
            raise ValueError("coefficient and intercept dimensions do not match")
        expected_rows = 1 if len(self.classes) == 2 else len(self.classes)
        if self.coef.shape[0] != expected_rows:
            raise ValueError("coefficient and class dimensions do not match")

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        standardized = (x - self.mean) / self.scale
        logits = standardized @ self.coef.T + self.intercept
        if len(self.classes) == 2:
            positive = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            return np.stack([1.0 - positive, positive], axis=1)
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    @property
    def classes_(self) -> np.ndarray:
        """Expose the sklearn-compatible fitted class attribute."""

        return self.classes


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


def mapped_protected_speed(
    label: str,
    *,
    fast_speed: float,
    default_protected_speed: float,
    protected_speed_map: dict[str, float],
) -> float:
    """Use one fast ceiling while allowing protected labels to differ."""

    speed = (
        fast_speed
        if label == "fast"
        else protected_speed_map.get(label, default_protected_speed)
    )
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
