"""Oracle-phase schedules for fixed-scene learning and randomized evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from oracle_phase_observation import OraclePhaseEncoder, PHASES
from policy_speed_env import create_speed_env
from sim_tasks import normalize_task_name
from speed_policy import SpeedContext


ALLOWED_SPEEDS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
TASK_OBJECTS = {"pick_and_place": 1, "tea_bag": 1, "insertion": 2}


def validate_schedule(values) -> tuple[float, ...]:
    schedule = tuple(float(value) for value in values)
    if len(schedule) != len(PHASES):
        raise ValueError("schedule must contain one speed per oracle phase")
    if any(value not in ALLOWED_SPEEDS for value in schedule):
        raise ValueError(f"speeds must be selected from {ALLOWED_SPEEDS}")
    return schedule


@dataclass
class PhaseSchedulePolicy:
    schedule: tuple[float, ...]
    decisions: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.schedule = validate_schedule(self.schedule)

    def reset(self):
        self.decisions.clear()

    def select_speed(self, observation, context: SpeedContext):
        phase = int(np.argmax(np.asarray(observation, dtype=np.float64)))
        speed = self.schedule[phase]
        self.decisions.append(
            {
                "phase": PHASES[phase],
                "physics_step": int(context.physics_steps),
                "speed": speed,
            }
        )
        return speed


def sample_object_pose(task: str, seed: int) -> tuple[float, ...]:
    task = normalize_task_name(task)
    env = create_speed_env(
        task_name=task,
        seed=int(seed),
        randomize_object_pose=True,
        observation_encoder=OraclePhaseEncoder(task),
        terminate_on_success=True,
    )
    try:
        env.reset()
        return tuple(
            float(value) for value in env.cur_ts.observation["env_state"]
        )
    finally:
        env.close()


def workspace_violation(task: str, observation) -> str | None:
    task = normalize_task_name(task)
    state = np.asarray(observation["env_state"], dtype=np.float64)
    for index in range(TASK_OBJECTS[task]):
        x, y, z = state[index * 7:index * 7 + 3]
        if abs(x) > 1.0 or not -0.2 <= y <= 1.2 or not -0.1 <= z <= 1.2:
            return f"object_{index}_outside_preregistered_workspace"
    return None


def run_phase_schedule(
    task: str,
    schedule,
    seed: int,
    *,
    object_pose=None,
    video_path: Path | None = None,
) -> dict:
    """Run one schedule, choosing only at reset and oracle phase entries."""

    task = normalize_task_name(task)
    schedule = validate_schedule(schedule)
    policy = PhaseSchedulePolicy(schedule)
    env = create_speed_env(
        task_name=task,
        seed=int(seed),
        object_pose=object_pose,
        randomize_object_pose=object_pose is None,
        observation_encoder=OraclePhaseEncoder(task),
        decision_mode="phase_entry",
        decision_frame_skip=1,
        terminate_on_success=True,
        save_video=video_path is not None,
        video_path="output_video.mp4" if video_path is None else video_path,
    )
    safety = None
    info = {"success": False, "physics_steps": 0, "policy_time": 0.0}
    try:
        policy.reset()
        observation = env.reset()
        done = False
        while not done:
            phase = int(np.argmax(np.asarray(observation, dtype=np.float64)))
            context = SpeedContext(
                policy_time=env.policy_time,
                physics_steps=env.physics_steps,
                episode_len=env.episode_len,
                speed_values=env.speed_values,
            )
            speed = policy.select_speed(observation, context)
            while not done:
                observation, _, done, info = env.step(speed, quantized=False)
                safety = safety or workspace_violation(task, env.cur_ts.observation)
                if int(np.argmax(np.asarray(observation, dtype=np.float64))) != phase:
                    break
        steps = int(info["physics_steps"])
        return {
            "task": task,
            "seed": int(seed),
            "schedule": list(schedule),
            "success": bool(info["success"]) and safety is None,
            "raw_task_success": bool(info["success"]),
            "physics_steps": steps,
            "success_only_acceleration": (
                float(env.episode_len / max(steps, 1))
                if bool(info["success"]) and safety is None
                else None
            ),
            "safety_violation": safety,
            "phase_decisions": policy.decisions,
            "video_path": None if video_path is None else str(video_path),
        }
    finally:
        env.close()
