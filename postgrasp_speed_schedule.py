"""Causal speed schedules for the post-grasp Insertion base policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PostgraspScheduleConfig:
    """Hold a safe speed through replanning, then release to a ceiling."""

    pre_replan_speed: float = 2.0
    post_replan_speed: float = 4.0
    release_stability: int = 1
    protected_reward: int = 2

    def __post_init__(self):
        if self.pre_replan_speed <= 0 or self.post_replan_speed <= 0:
            raise ValueError("speeds must be positive")
        if self.post_replan_speed < self.pre_replan_speed:
            raise ValueError("post-replan speed must not be slower than the protected speed")
        if self.release_stability < 1:
            raise ValueError("release stability must be at least one observation")

    def payload(self) -> dict:
        return {
            "kind": "postgrasp_latched_release",
            **asdict(self),
            "release_event": "replan_latched_and_reward_envelope_stable",
            "risk_downshift": "reward_below_protected_reward",
        }


class PostgraspLatchedSpeedSchedule:
    """Evaluate release and risk every physics tick.

    The tick that performs the one-shot spatial replan remains protected.  A
    faster speed is first allowed on a later tick after the requested number
    of stable post-replan observations.  Falling below the grasp/lift reward
    envelope immediately cancels the release and returns to the protected
    speed.
    """

    def __init__(self, config: PostgraspScheduleConfig):
        self.config = config
        self.stable_observations = 0
        self.released = False
        self.release_events = []
        self.downshift_events = []

    def select_speed(self, timestep, policy, physics_step: int) -> float:
        reward = int(timestep.reward or 0)
        if not policy.replan_count:
            return self.config.pre_replan_speed

        if reward < self.config.protected_reward:
            if self.released:
                self.downshift_events.append(
                    {
                        "physics_step": physics_step,
                        "policy_time": float(policy.step_count),
                        "observed_reward": reward,
                    }
                )
            self.released = False
            self.stable_observations = 0
            return self.config.pre_replan_speed

        if not self.released:
            self.stable_observations += 1
            if self.stable_observations >= self.config.release_stability:
                self.released = True
                self.release_events.append(
                    {
                        "physics_step": physics_step,
                        "policy_time": float(policy.step_count),
                        "observed_reward": reward,
                        "stable_observations": self.stable_observations,
                    }
                )

        if self.released:
            return self.config.post_replan_speed
        return self.config.pre_replan_speed


class FixedSpeedSchedule:
    """Fixed-speed control used for matched native references and fallback."""

    def __init__(self, speed: float):
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = float(speed)
        self.release_events = []
        self.downshift_events = []

    def select_speed(self, timestep, policy, physics_step: int) -> float:
        del timestep, policy, physics_step
        return self.speed

    def payload(self) -> dict:
        return {"kind": "fixed", "speed": self.speed}
