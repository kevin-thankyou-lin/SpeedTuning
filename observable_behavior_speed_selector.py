"""Speed selection from external object/effector behavior only."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from behavior_speed_observation import SpeedObservation, behavior_metrics


@dataclass(frozen=True)
class BehaviorRegionConfig:
    protected_speed: float
    fast_speed: float
    min_object_lift_m: float
    max_relative_translation_delta_m: float
    max_object_rotation_delta_deg: float
    max_closed_gripper_position: float
    terminal_object_distance_m: float
    stable_observations: int

    def __post_init__(self):
        if self.protected_speed <= 0 or self.fast_speed < self.protected_speed:
            raise ValueError("speed bounds are invalid")
        if self.min_object_lift_m < 0:
            raise ValueError("minimum lift must be nonnegative")
        if self.max_relative_translation_delta_m <= 0:
            raise ValueError("translation stability threshold must be positive")
        if self.max_object_rotation_delta_deg <= 0:
            raise ValueError("rotation stability threshold must be positive")
        if self.terminal_object_distance_m <= 0:
            raise ValueError("terminal distance must be positive")
        if self.stable_observations < 1:
            raise ValueError("stable observation count must be positive")

    def payload(self) -> dict:
        return {
            "kind": "observable_behavior_region",
            **asdict(self),
            "runtime_inputs": [
                "paired_effector_poses",
                "paired_object_poses",
                "gripper_positions",
                "observation_history",
            ],
            "runtime_forbidden_inputs": [
                "base_policy_object",
                "base_policy_time",
                "base_policy_phase",
                "replan_state",
                "reward",
                "success_flag",
            ],
        }


class ObservableBehaviorRegionSelector:
    """Latch a measured fast region and slow before terminal proximity."""

    def __init__(self, config: BehaviorRegionConfig):
        self.config = config
        self.initial = None
        self.previous = None
        self.observation_index = 0
        self.stable_count = 0
        self.fast_region_active = False
        self.terminal_region_latched = False
        self.entry_events = []
        self.exit_events = []

    def _attached_and_stable(self, metrics: dict) -> bool:
        return bool(
            np.min(metrics["object_lift_m"]) >= self.config.min_object_lift_m
            and np.max(metrics["gripper_positions"])
            <= self.config.max_closed_gripper_position
            and np.max(metrics["object_effector_translation_delta_m"])
            <= self.config.max_relative_translation_delta_m
            and np.max(metrics["object_rotation_delta_deg"])
            <= self.config.max_object_rotation_delta_deg
        )

    @staticmethod
    def _event(metrics: dict, observation_index: int, reason: str) -> dict:
        return {
            "observation_index": observation_index,
            "reason": reason,
            "object_pair_distance_m": metrics["object_pair_distance_m"],
            "min_object_lift_m": float(np.min(metrics["object_lift_m"])),
            "max_relative_translation_delta_m": float(
                np.max(metrics["object_effector_translation_delta_m"])
            ),
            "max_object_rotation_delta_deg": float(
                np.max(metrics["object_rotation_delta_deg"])
            ),
            "max_gripper_position": float(np.max(metrics["gripper_positions"])),
        }

    def select_speed(self, observation: SpeedObservation) -> float:
        self.observation_index += 1
        if self.initial is None:
            self.initial = observation
        metrics = behavior_metrics(observation, self.initial, self.previous)
        self.previous = observation

        if self.terminal_region_latched:
            return self.config.protected_speed

        if metrics["object_pair_distance_m"] <= self.config.terminal_object_distance_m:
            if self.fast_region_active:
                self.exit_events.append(
                    self._event(metrics, self.observation_index, "terminal_proximity")
                )
            self.fast_region_active = False
            self.terminal_region_latched = True
            self.stable_count = 0
            return self.config.protected_speed

        if not self._attached_and_stable(metrics):
            if self.fast_region_active:
                self.exit_events.append(
                    self._event(metrics, self.observation_index, "attachment_instability")
                )
            self.fast_region_active = False
            self.stable_count = 0
            return self.config.protected_speed

        self.stable_count += 1
        if not self.fast_region_active and self.stable_count >= self.config.stable_observations:
            self.fast_region_active = True
            self.entry_events.append(
                self._event(metrics, self.observation_index, "stable_lifted_attachment")
            )
        if self.fast_region_active:
            return self.config.fast_speed
        return self.config.protected_speed


class FixedBehaviorSpeedSelector:
    def __init__(self, speed: float):
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = float(speed)
        self.entry_events = []
        self.exit_events = []

    def select_speed(self, observation: SpeedObservation) -> float:
        del observation
        return self.speed
