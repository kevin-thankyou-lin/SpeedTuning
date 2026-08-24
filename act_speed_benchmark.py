"""Frozen method definitions for the multiview ACT speed benchmark.

This module contains only preregistered method contracts, offline artifact
construction, policy reconstruction, and selection logic. Rollout accounting
and immutable evidence live in ``scripts/run_act_speed_benchmark_cell.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from learned_phase_observation import PHASES
from one_reset_phase_schedule import PhaseSchedulePolicy
from speed_observation import StateObservationEncoder
from speed_policy import FixedSpeedPolicy, SpeedProfilePolicy


METHODS = (
    "uniform_sweep",
    "learned_phase_subtask",
    "learned_phase_tabular_rl",
    "learned_phase_rainbow_rl",
    "awe_offline_proxy",
    "sail_inspired_adaptive",
)
PHASE_METHODS = frozenset(
    {
        "learned_phase_subtask",
        "learned_phase_tabular_rl",
        "learned_phase_rainbow_rl",
    }
)
SWEEP_METHODS = frozenset(
    {
        "uniform_sweep",
        "learned_phase_subtask",
        "awe_offline_proxy",
        "sail_inspired_adaptive",
    }
)
SPEED_VALUES = (1.0, 1.25, 1.5, 1.75, 2.0)
PROFILE_BINS = 20


class JointEffectorObservationWrapper:
    """Add causal FK effector positions to joint-control observations."""

    BODY_NAMES = {
        "left": "vx300s_left/gripper_link",
        "right": "vx300s_right/gripper_link",
    }

    def __init__(self, environment):
        self.environment = environment

    def _augment(self, timestep):
        observation = dict(timestep.observation)
        for side, body in self.BODY_NAMES.items():
            observation[f"effector_position_{side}"] = np.asarray(
                self.environment.physics.named.data.xpos[body], dtype=np.float64
            ).copy()
        return timestep._replace(observation=observation)

    def reset(self):
        return self._augment(self.environment.reset())

    def step(self, action):
        return self._augment(self.environment.step(action))

    def close(self):
        return self.environment.close()

    def __getattr__(self, name):
        return getattr(self.environment, name)


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preregistration(method: str) -> dict:
    """Return the frozen, task-independent 50-rollout allocation."""

    if method not in METHODS:
        raise ValueError(f"unknown ACT speed method: {method}")
    common = {
        "schema": "act-speed-method-preregistration-v1",
        "method": method,
        "search_rollouts": 50,
        "final_rollouts": 50,
        "speed_values": list(SPEED_VALUES),
        "decision_frame_skip": 10,
        "selection": {
            "minimum_candidate_successes": 9,
            "candidate_rollouts": 10,
            "reject_any_safety_violation": True,
            "primary": "successes_descending",
            "secondary": "successful_mean_first_success_steps_ascending",
            "preserve_fastest_best_effort": True,
        },
        "phase_detector_required": method in PHASE_METHODS,
    }
    if method == "uniform_sweep":
        common.update(
            allocation="five_candidates_contiguous_ten_seed_blocks",
            candidates=[
                {"id": f"uniform_{speed:g}x", "speed": speed}
                for speed in SPEED_VALUES
            ],
        )
    elif method == "learned_phase_subtask":
        schedules = (
            (1.0, 1.0, 1.0, 1.0),
            (1.5, 1.0, 1.5, 1.0),
            (1.75, 1.0, 1.75, 1.0),
            (2.0, 1.0, 2.0, 1.0),
            (2.0, 1.0, 1.5, 1.0),
        )
        common.update(
            allocation="five_candidates_contiguous_ten_seed_blocks",
            phase_order=list(PHASES),
            decision_mode="fixed_or_phase_entry",
            candidates=[
                {"id": f"phase_schedule_{index}", "schedule": list(schedule)}
                for index, schedule in enumerate(schedules)
            ],
        )
    elif method == "learned_phase_tabular_rl":
        common.update(
            allocation="fifty_sequential_online_training_episodes",
            terminal_artifact_only=True,
            phase_order=list(PHASES),
            decision_mode="fixed_or_phase_entry",
            training={
                "algorithm": "first_visit_monte_carlo_control",
                "gamma": 0.97,
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
                "seed": 1701,
                "reward": {
                    "success_bonus": 100.0,
                    "speed_weight": 0.01,
                    "speed_power": 2.0,
                },
            },
        )
    elif method == "learned_phase_rainbow_rl":
        common.update(
            allocation="fifty_sequential_online_training_episodes",
            terminal_artifact_only=True,
            phase_order=list(PHASES),
            decision_mode="fixed_or_phase_entry",
            training={
                "algorithm": "categorical_double_dueling_noisynet_dqn_per_nstep",
                "seed": 2701,
                "memory_size": 4096,
                "batch_size": 32,
                "learning_starts": 32,
                "gradient_steps": 1,
                "target_update": 50,
                "learning_rate": 0.0001,
                "gamma": 0.97,
                "tau": 0.5,
                "epsilon_start": 1.0,
                "epsilon_end": 0.1,
                "exploration_decisions": 1000,
                "per_alpha": 0.2,
                "per_beta": 0.6,
                "atom_size": 121,
                "v_min": 0.0,
                "v_max": 120.0,
                "n_step": 3,
                "hidden_dim": 128,
                "reward": {
                    "success_bonus": 100.0,
                    "speed_weight": 0.01,
                    "speed_power": 2.0,
                },
            },
        )
    elif method == "awe_offline_proxy":
        candidates = (
            (0.0, 1.0),
            (0.02, 1.25),
            (0.05, 1.5),
            (0.10, 1.75),
            (0.20, 2.0),
        )
        common.update(
            allocation="five_candidates_contiguous_ten_seed_blocks",
            paper_faithful_sail=False,
            proxy_scope=(
                "offline RDP joint-trajectory waypoint density converted to a "
                "nominal-time speed profile; the frozen ACT is not retrained"
            ),
            profile_bins=PROFILE_BINS,
            candidates=[
                {
                    "id": f"awe_proxy_{index}",
                    "rdp_tolerance": tolerance,
                    "maximum_speed": maximum_speed,
                }
                for index, (tolerance, maximum_speed) in enumerate(candidates)
            ],
        )
    else:
        common.update(
            allocation="five_candidates_contiguous_ten_seed_blocks",
            paper_faithful_sail=False,
            provenance_label="sail_inspired_adaptive_v1",
            limitations=(
                "offline motion-complexity target plus online proprioceptive "
                "slowdown; no EAG, controller-invariant target training, or "
                "paper action scheduler"
            ),
            observation="causal_robot_qpos_qvel_only",
            profile_bins=PROFILE_BINS,
            candidates=[
                {
                    "id": f"sail_inspired_{index}",
                    "maximum_speed": speed,
                    "online_motion_gain": gain,
                    "gripper_delta_threshold": 0.01,
                }
                for index, (speed, gain) in enumerate(
                    zip(SPEED_VALUES, (0.0, 1.0, 2.0, 4.0, 8.0))
                )
            ],
        )
    common["preregistration_sha256"] = canonical_sha256(common)
    return common


def candidate_for_episode(prereg: dict, episode_index: int) -> dict:
    if prereg["method"] not in SWEEP_METHODS:
        raise ValueError("online RL methods do not allocate search candidates")
    if not 0 <= episode_index < 50:
        raise ValueError("search episode index must be in [0, 49]")
    return prereg["candidates"][episode_index // 10]


def _rdp_indices(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Return deterministic Ramer-Douglas-Peucker waypoint indices."""

    count = len(points)
    if count < 3 or tolerance <= 0:
        return np.arange(count, dtype=np.int64)
    keep = {0, count - 1}
    stack = [(0, count - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        fraction = np.arange(1, end - start, dtype=np.float64) / (end - start)
        line = points[start][None] + fraction[:, None] * (
            points[end] - points[start]
        )[None]
        errors = np.linalg.norm(points[start + 1 : end] - line, axis=1)
        offset = int(np.argmax(errors))
        if float(errors[offset]) > tolerance:
            split = start + 1 + offset
            keep.add(split)
            stack.append((split, end))
            stack.append((start, split))
    return np.asarray(sorted(keep), dtype=np.int64)


def _bin_indices(indices: np.ndarray, length: int) -> np.ndarray:
    return np.minimum(indices * PROFILE_BINS // max(length, 1), PROFILE_BINS - 1)


def build_offline_artifact(dataset_dir: Path, method: str) -> dict:
    """Build a hash-pinned offline proxy artifact from ACT demonstration arrays."""

    if method not in {"awe_offline_proxy", "sail_inspired_adaptive"}:
        raise ValueError("offline artifacts are only defined for AWE/SAIL-inspired")
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("offline ACT artifacts require h5py") from exc

    paths = sorted(Path(dataset_dir).glob("episode_*.hdf5"))
    if not paths:
        raise ValueError(f"no ACT demonstrations found in {dataset_dir}")
    trajectories = []
    dataset_digest = hashlib.sha256()
    for path in paths:
        with h5py.File(path, "r") as root:
            qpos = np.asarray(root["/observations/qpos"], dtype=np.float64)
            action = np.asarray(root["/action"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != 14 or action.shape[1] != 14:
            raise ValueError(f"unexpected ACT demonstration shape in {path}")
        dataset_digest.update(path.name.encode())
        for name, value in (("qpos", qpos), ("action", action)):
            dataset_digest.update(name.encode())
            dataset_digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            dataset_digest.update(value.tobytes())
        trajectories.append(qpos)

    all_qpos = np.concatenate(trajectories, axis=0)
    scale = np.maximum(np.std(all_qpos, axis=0), 1e-6)
    gripper_counts = np.zeros(PROFILE_BINS, dtype=np.float64)
    complexity = np.zeros(PROFILE_BINS, dtype=np.float64)
    bin_samples = np.zeros(PROFILE_BINS, dtype=np.float64)
    for qpos in trajectories:
        normalized = (qpos - np.mean(qpos, axis=0)) / scale
        velocity = np.diff(normalized, axis=0, prepend=normalized[:1])
        curvature = np.linalg.norm(
            np.diff(velocity, axis=0, prepend=velocity[:1]), axis=1
        )
        bins = _bin_indices(np.arange(len(qpos)), len(qpos))
        for bin_index in range(PROFILE_BINS):
            mask = bins == bin_index
            complexity[bin_index] += float(np.sum(curvature[mask]))
            bin_samples[bin_index] += int(np.sum(mask))
        gripper_delta = np.max(
            np.abs(np.diff(qpos[:, (6, 13)], axis=0, prepend=qpos[:1, (6, 13)])),
            axis=1,
        )
        event_bins = np.unique(bins[gripper_delta > 0.01])
        gripper_counts[event_bins] += 1
    complexity /= np.maximum(bin_samples, 1)
    if float(np.max(complexity)) > 0:
        complexity /= float(np.max(complexity))
    gripper_frequency = gripper_counts / len(trajectories)

    prereg = preregistration(method)
    candidates = []
    for candidate in prereg["candidates"]:
        maximum_speed = float(candidate["maximum_speed"])
        if method == "awe_offline_proxy":
            waypoint_counts = np.zeros(PROFILE_BINS, dtype=np.float64)
            for qpos in trajectories:
                normalized = (qpos - np.mean(qpos, axis=0)) / scale
                indices = _rdp_indices(normalized, float(candidate["rdp_tolerance"]))
                waypoint_counts[np.unique(_bin_indices(indices, len(qpos)))] += 1
            importance = waypoint_counts / len(trajectories)
        else:
            importance = complexity.copy()
        importance = np.maximum(importance, np.minimum(gripper_frequency * 4.0, 1.0))
        profile = 1.0 + (maximum_speed - 1.0) * (1.0 - importance)
        profile[gripper_frequency >= 0.10] = 1.0
        payload = dict(candidate)
        payload.update(
            profile=[float(value) for value in profile],
            importance=[float(value) for value in importance],
        )
        candidates.append(payload)

    artifact = {
        "schema": "act-speed-offline-proxy-artifact-v1",
        "method": method,
        "paper_faithful_sail": False,
        "dataset_directory": str(Path(dataset_dir).resolve()),
        "episode_count": len(paths),
        "dataset_array_sha256": dataset_digest.hexdigest(),
        "qpos_scale": [float(value) for value in scale],
        "profile_bins": PROFILE_BINS,
        "gripper_event_frequency": [float(value) for value in gripper_frequency],
        "normalized_motion_complexity": [float(value) for value in complexity],
        "candidates": candidates,
        "preregistration_sha256": prereg["preregistration_sha256"],
    }
    artifact["artifact_payload_sha256"] = canonical_sha256(artifact)
    return artifact


class SailInspiredAdaptivePolicy:
    """Causal proprioceptive modulation; explicitly not paper-faithful SAIL."""

    frame_skip = 10

    def __init__(self, candidate: dict, qpos_scale):
        self.profile = tuple(float(value) for value in candidate["profile"])
        self.maximum_speed = float(candidate["maximum_speed"])
        self.motion_gain = float(candidate["online_motion_gain"])
        self.gripper_delta_threshold = float(candidate["gripper_delta_threshold"])
        self.qpos_scale = np.maximum(np.asarray(qpos_scale, dtype=np.float64), 1e-6)
        if self.qpos_scale.shape != (14,):
            raise ValueError("SAIL-inspired qpos_scale must have shape (14,)")
        self.reset()

    def reset(self):
        self.previous_qpos = None

    def select_speed(self, observation, context):
        value = np.asarray(observation, dtype=np.float64)
        if value.size != 28:
            raise ValueError("SAIL-inspired policy requires qpos+qvel only")
        qpos = value[:14]
        fraction = min(max(context.policy_time / context.episode_len, 0.0), 1.0)
        index = min(int(fraction * len(self.profile)), len(self.profile) - 1)
        base = self.profile[index]
        if self.previous_qpos is None:
            speed = base
        else:
            delta = np.abs(qpos - self.previous_qpos)
            gripper_event = bool(
                np.max(delta[[6, 13]]) >= self.gripper_delta_threshold
            )
            motion = float(np.linalg.norm(delta / self.qpos_scale) / np.sqrt(14))
            speed = 1.0 if gripper_event else base / (1.0 + self.motion_gain * motion)
        self.previous_qpos = qpos.copy()
        return float(np.clip(speed, 1.0, self.maximum_speed))


def nonphase_observation_encoder(method: str):
    if method == "sail_inspired_adaptive":
        return StateObservationEncoder(
            include_qpos=True, include_qvel=True, include_env_state=False
        )
    return StateObservationEncoder(
        include_qpos=True, include_qvel=False, include_env_state=False
    )


def policy_from_candidate(method: str, candidate: dict, offline_artifact=None):
    if method == "uniform_sweep":
        return FixedSpeedPolicy(candidate["speed"])
    if method == "learned_phase_subtask":
        return PhaseSchedulePolicy(tuple(candidate["schedule"]))
    if method == "awe_offline_proxy":
        return SpeedProfilePolicy(candidate["profile"])
    if method == "sail_inspired_adaptive":
        if offline_artifact is None:
            raise ValueError("SAIL-inspired policy requires its offline artifact")
        return SailInspiredAdaptivePolicy(candidate, offline_artifact["qpos_scale"])
    raise ValueError(f"candidate policies are not defined for {method}")


def summarize_candidate(records: list[dict], candidate: dict) -> dict:
    successes = [item for item in records if item["success"]]
    first_steps = [item["first_success_step"] for item in successes]
    return {
        "candidate": candidate,
        "episodes": len(records),
        "successes": len(successes),
        "success_rate": len(successes) / len(records),
        "successful_mean_first_success_steps": (
            None if not first_steps else float(np.mean(first_steps))
        ),
        "safety_violations": sum(
            item.get("safety_violation") is not None for item in records
        ),
        "physics_errors": sum("physics_error" in item for item in records),
    }


def select_candidate(prereg: dict, records: list[dict], offline_artifact=None) -> dict:
    if len(records) != 50:
        raise ValueError("candidate selection requires exactly 50 search records")
    reports = []
    for index, candidate in enumerate(prereg["candidates"]):
        assigned = records[index * 10 : (index + 1) * 10]
        expected_id = candidate["id"]
        if any(item.get("candidate_id") != expected_id for item in assigned):
            raise ValueError("search receipt candidate allocation mismatch")
        report_candidate = candidate
        if offline_artifact is not None:
            report_candidate = offline_artifact["candidates"][index]
        reports.append(summarize_candidate(assigned, report_candidate))
    eligible = [
        item
        for item in reports
        if item["successes"] >= 9 and item["safety_violations"] == 0
    ]
    if not eligible:
        raise RuntimeError("no preregistered candidate passed the 9/10 safety gate")

    def rank(item):
        first = item["successful_mean_first_success_steps"]
        return (-item["successes"], float("inf") if first is None else first)

    selected = min(eligible, key=rank)
    best_effort = min(
        (item for item in reports if item["successful_mean_first_success_steps"] is not None),
        key=lambda item: item["successful_mean_first_success_steps"],
    )
    return {
        "selected": selected,
        "fastest_observed_best_effort": best_effort,
        "candidate_reports": reports,
    }
