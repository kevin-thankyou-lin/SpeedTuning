"""Phase schedules for fixed-scene learning and randomized evaluation."""

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


def rollout_metric_steps(result: dict) -> int:
    """Return first-success steps when available, otherwise terminal steps."""

    first_success = result.get("first_success_step")
    return int(result["physics_steps"] if first_success is None else first_success)


def estimate_phase_workload(result: dict) -> dict[str, float]:
    """Estimate native policy work per phase from one completed rollout.

    A segment that took ``n`` physics steps at speed ``s`` represents roughly
    ``n * s`` native-speed policy steps. Repeated detector segments are summed.
    """

    workloads = {phase: 0.0 for phase in PHASES}
    decisions = list(result.get("phase_decisions", ()))
    final_step = rollout_metric_steps(result)
    for index, decision in enumerate(decisions):
        start = min(int(decision["physics_step"]), final_step)
        end = min(
            int(decisions[index + 1]["physics_step"])
            if index + 1 < len(decisions)
            else final_step,
            final_step,
        )
        phase = str(decision["phase"])
        if phase not in workloads:
            raise ValueError(f"unknown phase in decision trace: {phase}")
        workloads[phase] += max(end - start, 0) * float(decision["speed"])
    return workloads


def score_schedule_change(
    anchor_result: dict,
    candidate_schedule,
    *,
    safe_success_probability: float = 1.0,
) -> dict:
    """Predict absolute time saved by a candidate relative to an anchor.

    The VLM supplies a conservative probability that the untested candidate
    will remain safe and successful. This is an acquisition score, not rollout
    evidence and not a reliability claim.
    """

    anchor_schedule = validate_schedule(anchor_result["schedule"])
    candidate_schedule = validate_schedule(candidate_schedule)
    probability = float(safe_success_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("safe_success_probability must be between 0 and 1")
    workloads = estimate_phase_workload(anchor_result)
    anchor_steps = sum(
        workloads[phase] / speed
        for phase, speed in zip(PHASES, anchor_schedule)
    )
    candidate_steps = sum(
        workloads[phase] / speed
        for phase, speed in zip(PHASES, candidate_schedule)
    )
    contributions = {
        phase: workloads[phase] * (1.0 / old - 1.0 / new)
        for phase, old, new in zip(PHASES, anchor_schedule, candidate_schedule)
    }
    saved = anchor_steps - candidate_steps
    return {
        "anchor_schedule": list(anchor_schedule),
        "candidate_schedule": list(candidate_schedule),
        "phase_workload_steps": workloads,
        "phase_predicted_steps_saved": contributions,
        "predicted_anchor_steps": anchor_steps,
        "predicted_candidate_steps": candidate_steps,
        "predicted_absolute_steps_saved": saved,
        "predicted_relative_speedup": (
            anchor_steps / candidate_steps if candidate_steps > 0 else None
        ),
        "safe_success_probability": probability,
        "expected_absolute_steps_saved": probability * saved,
        "warning": "acquisition estimate only; testing supplies reliability evidence",
    }


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
        # Some scenes include additional free-joint props after the task
        # object. Only the controlled task objects belong in the frozen pose.
        pose_values = TASK_OBJECTS[task] * 7
        return tuple(
            float(value)
            for value in env.cur_ts.observation["env_state"][:pose_values]
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
    observation_encoder=None,
    chunk_predictor=None,
    terminate_on_success=True,
) -> dict:
    """Run one schedule, choosing at reset and supplied phase entries."""

    task = normalize_task_name(task)
    schedule = validate_schedule(schedule)
    policy = PhaseSchedulePolicy(schedule)
    env = create_speed_env(
        task_name=task,
        seed=int(seed),
        object_pose=object_pose,
        randomize_object_pose=object_pose is None,
        chunk_predictor=chunk_predictor,
        speed_values=ALLOWED_SPEEDS,
        observation_encoder=(
            OraclePhaseEncoder(task)
            if observation_encoder is None
            else observation_encoder
        ),
        decision_mode="phase_entry",
        decision_frame_skip=1,
        terminate_on_success=bool(terminate_on_success),
        save_video=video_path is not None,
        video_path="output_video.mp4" if video_path is None else video_path,
    )
    if chunk_predictor is not None:
        # Learned phase proprioception was sealed against FK-derived end-effector
        # positions for joint-control ACT evaluation.  Keep that exact causal
        # interface when the generic phase-schedule runner uses an ACT chunk
        # predictor instead of the retained scripted waypoint controller.
        from act_speed_benchmark import JointEffectorObservationWrapper

        env.env = JointEffectorObservationWrapper(env.env)
        env._environment_metadata["learned_phase_effector_source"] = (
            "joint_fk_body_xpos"
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
        first_success_step = info.get("first_success_step")
        metric_steps = steps if first_success_step is None else int(first_success_step)
        result = {
            "task": task,
            "seed": int(seed),
            "schedule": list(schedule),
            "success": bool(info["success"]) and safety is None,
            "raw_task_success": bool(info["success"]),
            "physics_steps": steps,
            "first_success_step": first_success_step,
            "success_only_acceleration": (
                float(env.episode_len / max(metric_steps, 1))
                if bool(info["success"]) and safety is None
                else None
            ),
            "safety_violation": safety,
            "phase_decisions": policy.decisions,
            "video_path": None if video_path is None else str(video_path),
        }
        if info.get("physics_error") is not None:
            result["physics_error"] = str(info["physics_error"])
        return result
    finally:
        env.close()
