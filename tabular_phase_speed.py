"""Tiny phase-entry speed controller for low-budget oracle experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TabularTrainingConfig:
    episodes: int = 50
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05

    def validate(self):
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")


def phase_index(observation):
    """Decode the supplied oracle one-hot phase observation."""

    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("tabular phase policy requires a finite four-value phase vector")
    return int(np.argmax(values))


class TabularPhaseSpeedPolicy:
    """Greedy phase-to-speed lookup learned by tabular Monte Carlo control."""

    frame_skip = 10

    def __init__(self, q_values, speed_values):
        values = np.asarray(q_values, dtype=np.float64)
        speeds = tuple(float(value) for value in speed_values)
        if values.shape != (4, len(speeds)):
            raise ValueError("q_values must have shape (4, number of speeds)")
        self.q_values = values
        self.speed_values = speeds

    @property
    def schedule(self):
        return tuple(
            self.speed_values[int(np.argmax(row))] for row in self.q_values
        )

    def select_speed(self, observation, context):
        del context
        return self.schedule[phase_index(observation)]


def train_tabular_phase_speed_policy(env, checkpoint_path, config=None, seed=0):
    """Learn 4 x |speed_values| values from complete phase-entry episodes."""

    config = config or TabularTrainingConfig()
    config.validate()
    rng = np.random.default_rng(seed)
    q_values = np.zeros((4, env.action_space), dtype=np.float64)
    visits = np.zeros_like(q_values, dtype=np.int64)
    episodes = []

    for episode_index in range(config.episodes):
        fraction = episode_index / max(config.episodes - 1, 1)
        epsilon = config.epsilon_start + fraction * (
            config.epsilon_end - config.epsilon_start
        )
        observation = env.reset()
        trajectory = []
        done = False
        info = {"success": False}

        while not done:
            phase = phase_index(observation)
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space))
            else:
                maxima = np.flatnonzero(q_values[phase] == q_values[phase].max())
                action = int(rng.choice(maxima))
            next_observation, reward, done, info = env.step_decision(action)
            trajectory.append((phase, action, float(reward)))
            observation = next_observation

        episode_return = 0.0
        returns = []
        for phase, action, reward in reversed(trajectory):
            episode_return = reward + config.gamma * episode_return
            returns.append((phase, action, episode_return))
        for phase, action, value in reversed(returns):
            visits[phase, action] += 1
            q_values[phase, action] += (
                value - q_values[phase, action]
            ) / visits[phase, action]

        episodes.append(
            {
                "episode": episode_index + 1,
                "epsilon": epsilon,
                "success": bool(info["success"]),
                "physics_steps": int(info["physics_steps"]),
                "return": float(sum(item[2] for item in trajectory)),
                "phase_actions": [
                    {
                        "phase": phase,
                        "action": action,
                        "speed": env.speed_values[action],
                    }
                    for phase, action, _ in trajectory
                ],
            }
        )

    checkpoint = {
        "format_version": 1,
        "algorithm": "tabular_monte_carlo_phase_speed",
        "speed_values": list(env.speed_values),
        "q_values": q_values.tolist(),
        "visits": visits.tolist(),
        "schedule": list(TabularPhaseSpeedPolicy(q_values, env.speed_values).schedule),
        "seed": int(seed),
        "training_config": asdict(config),
        "observation_spec": env.observation_spec(),
        "environment_spec": env.environment_spec(),
    }
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    return {
        "checkpoint": str(checkpoint_path),
        "episodes": config.episodes,
        "decisions": sum(len(item["phase_actions"]) for item in episodes),
        "successes": sum(int(item["success"]) for item in episodes),
        "schedule": checkpoint["schedule"],
        "q_values": checkpoint["q_values"],
        "visits": checkpoint["visits"],
        "episode_history": episodes,
    }


def load_tabular_phase_speed_policy(checkpoint, **_kwargs):
    payload = json.loads(Path(checkpoint).read_text())
    if payload.get("algorithm") != "tabular_monte_carlo_phase_speed":
        raise ValueError("not a tabular phase-speed checkpoint")
    return TabularPhaseSpeedPolicy(payload["q_values"], payload["speed_values"])
