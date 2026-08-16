"""Utilities for evaluating reference-aligned speed schedules.

This module deliberately keeps semantic segment metadata outside the visual
aligner.  The exact event controller mirrors the frozen R3 implementation so
the privileged and reference-derived schedules share identical source bytes.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Any

import numpy as np


def scalar_feature(state: dict[str, Any], name: str) -> float | bool:
    if name in ("policy_time", "physics_steps", "task_reward", "success"):
        return state[name]
    if name.startswith("contact:"):
        expected = "|".join(sorted(name.split(":", 1)[1].split("|")))
        return expected in state["contacts"]
    if name.startswith("distance:"):
        _, mocap_name, object_index = name.split(":")
        mocap = state[f"mocap_{mocap_name}"][:3]
        start = int(object_index) * 7
        target = state["env_state"][start : start + 3]
        return float(np.linalg.norm(np.asarray(mocap) - np.asarray(target)))
    if name.startswith("object_distance:"):
        _, first_index, second_index = name.split(":")
        first_start = int(first_index) * 7
        second_start = int(second_index) * 7
        first = np.asarray(state["env_state"][first_start : first_start + 3])
        second = np.asarray(state["env_state"][second_start : second_start + 3])
        if first.shape != (3,) or second.shape != (3,):
            raise ValueError(f"object distance index out of range: {name}")
        return float(np.linalg.norm(first - second))
    field, index = name.rsplit(".", 1)
    return float(state[field][int(index)])


def predicate_matches(spec: dict[str, Any] | None, state: dict[str, Any]) -> bool:
    if spec is None:
        return True
    if "all" in spec:
        return all(predicate_matches(item, state) for item in spec["all"])
    if "any" in spec:
        return any(predicate_matches(item, state) for item in spec["any"])
    if "not" in spec:
        return not predicate_matches(spec["not"], state)
    actual = scalar_feature(state, spec["feature"])
    expected = spec.get("value", True)
    op = spec.get("op", "eq")
    operations = {
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
        "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b,
    }
    if op not in operations:
        raise ValueError(f"unsupported predicate operation: {op}")
    return bool(operations[op](actual, expected))


@dataclasses.dataclass
class SegmentState:
    status: str = "pending"
    stable_observations: int = 0
    entered_step: int | None = None
    released_step: int | None = None
    latched_speed_rule: int | None = None
    speed_rule_selection_locked: bool = False


class EventController:
    """Control-semantic port of the frozen R3 event controller."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.ceiling = float(config["ceiling"])
        self.segments = list(config.get("segments", []))
        self.states = [SegmentState() for _ in self.segments]
        self.events: list[dict[str, Any]] = []

    def select(self, state: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        for index, (segment, segment_state) in enumerate(zip(self.segments, self.states)):
            if segment_state.status == "released":
                continue
            if segment_state.status == "pending":
                if predicate_matches(segment["entry"], state):
                    segment_state.status = "active"
                    segment_state.entered_step = int(state["physics_steps"])
                    self.events.append({"event": "segment_entry", "segment": index})
                else:
                    return self.ceiling, "ceiling_waiting", self.latch_state()
            if segment_state.status == "active":
                if predicate_matches(segment["exit"], state):
                    segment_state.stable_observations += 1
                else:
                    segment_state.stable_observations = 0
                required = int(segment.get("release_stability", 1))
                if segment_state.stable_observations >= required:
                    segment_state.status = "released"
                    segment_state.released_step = int(state["physics_steps"])
                    self.events.append({"event": "segment_release", "segment": index})
                    continue
                rules = segment.get("speed_rules", [])
                if rules and not segment_state.speed_rule_selection_locked:
                    for rule_index, rule in enumerate(rules):
                        if predicate_matches(rule["predicate"], state):
                            if rule.get("latch", True):
                                segment_state.latched_speed_rule = rule_index
                                segment_state.speed_rule_selection_locked = True
                            return (
                                float(rule["speed"]),
                                f"protected_segment_{index}_speed_rule_{rule_index}",
                                self.latch_state(),
                            )
                    segment_state.speed_rule_selection_locked = True
                if (
                    segment_state.latched_speed_rule is not None
                    and segment_state.latched_speed_rule < len(rules)
                ):
                    rule_index = segment_state.latched_speed_rule
                    return (
                        float(rules[rule_index]["speed"]),
                        f"protected_segment_{index}_latched_speed_rule_{rule_index}",
                        self.latch_state(),
                    )
                return (
                    float(segment.get("speed", 1.0)),
                    f"protected_segment_{index}",
                    self.latch_state(),
                )
        safety = self.config.get("immediate_safety")
        if safety and predicate_matches(safety["predicate"], state):
            return float(safety.get("speed", 1.0)), "immediate_safety", self.latch_state()
        return self.ceiling, "ceiling_released", self.latch_state()

    def latch_state(self) -> dict[str, Any]:
        return {
            str(index): dataclasses.asdict(state)
            for index, state in enumerate(self.states)
        }


class CausalTemporalPool:
    """Build the same 1-second RN18 descriptor used by the V0 benchmark."""

    def __init__(self, frames: int = 10):
        if frames <= 0:
            raise ValueError("frames must be positive")
        self._values: deque[np.ndarray] = deque(maxlen=frames)

    def update(self, frame_embedding: np.ndarray) -> np.ndarray:
        value = np.asarray(frame_embedding, dtype=np.float32).reshape(-1)
        self._values.append(value)
        values = np.stack(tuple(self._values))
        descriptor = np.concatenate([values.mean(axis=0), values[-1], values[-1] - values[0]])
        return descriptor / max(float(np.linalg.norm(descriptor)), 1e-8)


def expand_protected_speed_map(
    speeds: np.ndarray,
    *,
    ceiling: float,
    margin_indices: int,
) -> np.ndarray:
    """Dilate protected regions and use the safest overlapping speed."""

    source = np.asarray(speeds, dtype=np.float32).reshape(-1)
    if margin_indices < 0:
        raise ValueError("margin_indices must be non-negative")
    expanded = np.full_like(source, float(ceiling))
    for index, speed in enumerate(source):
        if speed >= ceiling:
            continue
        start = max(0, index - margin_indices)
        stop = min(len(source), index + margin_indices + 1)
        expanded[start:stop] = np.minimum(expanded[start:stop], speed)
    return expanded


def select_aligned_speed(
    speed_map: np.ndarray,
    reference_index: int,
    confidence: float,
    *,
    confidence_threshold: float | None,
    fallback_speed: float = 1.0,
) -> tuple[float, bool]:
    if confidence_threshold is not None and confidence < confidence_threshold:
        return float(fallback_speed), True
    index = int(np.clip(reference_index, 0, len(speed_map) - 1))
    return float(speed_map[index]), False
