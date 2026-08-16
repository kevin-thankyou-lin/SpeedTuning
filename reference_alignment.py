"""Causal alignment of live clip embeddings to one reference trajectory.

The aligner deliberately knows nothing about semantic phases or playback
speeds.  It maintains a probability distribution over reference positions and
returns a continuous coordinate that downstream metadata can interpret.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("reference embeddings must have shape [time, features]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return -np.inf
    return maximum + float(np.log(np.exp(values - maximum).sum()))


@dataclass(frozen=True)
class AlignmentResult:
    """One causal alignment estimate."""

    reference_position: float
    reference_index: int
    confidence: float
    local_progress_rate: float
    raw_reference_position: float
    raw_reference_index: int
    likely_region_width: float
    posterior: np.ndarray = field(repr=False)
    confidence_components: dict[str, float] = field(default_factory=dict)


class OnlineReferenceAligner:
    """Causal HMM-style filter over positions in one reference execution.

    ``reference_video`` may either be a ``[T, D]`` embedding array or a frame
    sequence when ``clip_encoder`` is supplied.  A clip encoder is any callable
    that accepts the latest causal frame window and returns one feature vector.
    No future query frames are accessed.
    """

    def __init__(
        self,
        reference_video: np.ndarray | Sequence[np.ndarray],
        *,
        clip_encoder: Callable[[Sequence[np.ndarray]], np.ndarray] | None = None,
        clip_frames: int = 1,
        max_advance: int = 5,
        max_backtrack: int = 1,
        emission_temperature: float = 0.07,
        expected_advance: float = 1.0,
        transition_scale: float = 1.5,
        backward_penalty: float = 2.0,
        initialization_fraction: float = 0.12,
        updates_per_second: float = 1.0,
        rate_history: int = 7,
    ):
        if clip_frames <= 0:
            raise ValueError("clip_frames must be positive")
        if max_advance < 0 or max_backtrack < 0:
            raise ValueError("transition bounds must be non-negative")
        if emission_temperature <= 0 or transition_scale <= 0:
            raise ValueError("temperatures and scales must be positive")
        if updates_per_second <= 0:
            raise ValueError("updates_per_second must be positive")

        self.clip_encoder = clip_encoder
        self.clip_frames = int(clip_frames)
        self.max_advance = int(max_advance)
        self.max_backtrack = int(max_backtrack)
        self.emission_temperature = float(emission_temperature)
        self.expected_advance = float(expected_advance)
        self.transition_scale = float(transition_scale)
        self.backward_penalty = float(backward_penalty)
        self.initialization_fraction = float(initialization_fraction)
        self.updates_per_second = float(updates_per_second)
        self._frames: deque[np.ndarray] = deque(maxlen=self.clip_frames)
        self._positions: deque[float] = deque(maxlen=max(2, int(rate_history)))

        if clip_encoder is None:
            reference_embeddings = np.asarray(reference_video, dtype=np.float32)
        else:
            frames = list(reference_video)
            if len(frames) < 2:
                raise ValueError("reference video must contain at least two frames")
            reference_embeddings = np.stack(
                [
                    np.asarray(
                        clip_encoder(frames[max(0, index - self.clip_frames + 1) : index + 1]),
                        dtype=np.float32,
                    )
                    for index in range(len(frames))
                ]
            )
        self.reference_embeddings = _normalize_rows(reference_embeddings)
        self.reference_length = self.reference_embeddings.shape[0]
        self._log_posterior: np.ndarray | None = None
        self._last_result: AlignmentResult | None = None

    def reset(self) -> None:
        self._frames.clear()
        self._positions.clear()
        self._log_posterior = None
        self._last_result = None

    def update(self, live_frames: np.ndarray | Sequence[np.ndarray]) -> AlignmentResult:
        """Consume new frame(s) and return the estimate at the newest frame."""

        if self.clip_encoder is None:
            values = np.asarray(live_frames, dtype=np.float32)
            if values.ndim == 1:
                return self.update_embedding(values)
            if values.ndim != 2:
                raise ValueError("embedding updates must have shape [D] or [T, D]")
            result = None
            for value in values:
                result = self.update_embedding(value)
            assert result is not None
            return result

        frames = np.asarray(live_frames)
        if frames.ndim == 3:
            frames = frames[None, ...]
        if frames.ndim != 4:
            raise ValueError("frame updates must have shape [H, W, C] or [T, H, W, C]")
        result = None
        for frame in frames:
            self._frames.append(frame)
            embedding = np.asarray(self.clip_encoder(tuple(self._frames)), dtype=np.float32)
            result = self.update_embedding(embedding)
        assert result is not None
        return result

    def update_embedding(self, embedding: np.ndarray) -> AlignmentResult:
        """Consume one already-computed causal clip embedding."""

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding.shape[0] != self.reference_embeddings.shape[1]:
            raise ValueError(
                f"embedding dimension {embedding.shape[0]} does not match reference "
                f"dimension {self.reference_embeddings.shape[1]}"
            )
        embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
        similarities = self.reference_embeddings @ embedding
        raw_index = int(np.argmax(similarities))
        emission = similarities.astype(np.float64) / self.emission_temperature

        if self._log_posterior is None:
            allowed = max(2, int(np.ceil(self.reference_length * self.initialization_fraction)))
            log_posterior = np.full(self.reference_length, -np.inf, dtype=np.float64)
            log_posterior[:allowed] = emission[:allowed]
        else:
            log_posterior = np.full(self.reference_length, -np.inf, dtype=np.float64)
            for destination in range(self.reference_length):
                source_start = max(0, destination - self.max_advance)
                source_stop = min(
                    self.reference_length,
                    destination + self.max_backtrack + 1,
                )
                sources = np.arange(source_start, source_stop)
                deltas = destination - sources
                transition = -0.5 * (
                    (deltas - self.expected_advance) / self.transition_scale
                ) ** 2
                transition -= self.backward_penalty * np.maximum(-deltas, 0)
                log_posterior[destination] = emission[destination] + _logsumexp(
                    self._log_posterior[sources] + transition
                )

        normalization = _logsumexp(log_posterior)
        if not np.isfinite(normalization):
            raise RuntimeError("alignment posterior collapsed")
        self._log_posterior = log_posterior - normalization
        posterior = np.exp(self._log_posterior)
        reference_indices = np.arange(self.reference_length, dtype=np.float64)
        map_index = int(np.argmax(posterior))
        mean_index = float(posterior @ reference_indices)
        denominator = max(1, self.reference_length - 1)
        position = mean_index / denominator
        raw_position = raw_index / denominator

        centered = reference_indices - mean_index
        std_index = float(np.sqrt(posterior @ np.square(centered)))
        likely_width = min(1.0, 4.0 * std_index / denominator)
        entropy = float(-np.sum(posterior * np.log(np.maximum(posterior, 1e-12))))
        entropy_confidence = 1.0 - entropy / np.log(self.reference_length)
        top_two = np.partition(similarities, -2)[-2:]
        margin = float(top_two.max() - top_two.min())
        margin_confidence = 1.0 - np.exp(-margin / 0.03)
        width_confidence = 1.0 - likely_width

        self._positions.append(position)
        if len(self._positions) < 2:
            local_rate = 0.0
            continuity_confidence = 1.0
        else:
            deltas = np.diff(np.asarray(self._positions))
            local_rate = float(np.median(deltas) * self.updates_per_second)
            continuity_confidence = float(
                np.exp(-np.std(deltas) * denominator / max(1.0, self.max_advance))
            )
        confidence = float(
            np.clip(
                0.35 * entropy_confidence
                + 0.20 * margin_confidence
                + 0.30 * width_confidence
                + 0.15 * continuity_confidence,
                0.0,
                1.0,
            )
        )
        result = AlignmentResult(
            reference_position=float(np.clip(position, 0.0, 1.0)),
            reference_index=map_index,
            confidence=confidence,
            local_progress_rate=local_rate,
            raw_reference_position=raw_position,
            raw_reference_index=raw_index,
            likely_region_width=likely_width,
            posterior=posterior.copy(),
            confidence_components={
                "entropy": entropy_confidence,
                "similarity_margin": margin_confidence,
                "posterior_width": width_confidence,
                "temporal_consistency": continuity_confidence,
            },
        )
        self._last_result = result
        return result


class ReferencePositionSpeedPolicy:
    """Separate metadata lookup demonstrating zero-retraining segment edits."""

    def __init__(self, segments: Sequence[tuple[float, float, float]], default: float):
        self.segments = tuple((float(a), float(b), float(speed)) for a, b, speed in segments)
        self.default = float(default)
        for start, stop, speed in self.segments:
            if not 0.0 <= start < stop <= 1.0 or speed <= 0:
                raise ValueError("segments must be valid normalized intervals with positive speeds")

    def __call__(self, reference_position: float) -> float:
        position = float(reference_position)
        for start, stop, speed in self.segments:
            if start <= position < stop:
                return speed
        return self.default
