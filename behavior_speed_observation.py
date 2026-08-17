"""Policy-agnostic observations for outer speed selectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _poses(value, name: str) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"{name} must have shape [objects, 7]")
    if not np.all(np.isfinite(poses)):
        raise ValueError(f"{name} must be finite")
    return poses.copy()


@dataclass(frozen=True)
class SpeedObservation:
    """Only externally observable behavior available to a speed selector.

    Rows in ``effector_poses`` and ``object_poses`` are explicitly paired by
    the environment adapter. No base-policy object, phase, reward, future
    action, or policy clock is part of this contract.
    """

    effector_poses: np.ndarray
    object_poses: np.ndarray
    gripper_positions: np.ndarray

    def __post_init__(self):
        effectors = _poses(self.effector_poses, "effector_poses")
        objects = _poses(self.object_poses, "object_poses")
        grippers = np.asarray(self.gripper_positions, dtype=np.float64)
        if effectors.shape != objects.shape:
            raise ValueError("effector and object poses must have matching shapes")
        if grippers.shape != (effectors.shape[0],):
            raise ValueError("gripper_positions must have one value per pair")
        if not np.all(np.isfinite(grippers)):
            raise ValueError("gripper_positions must be finite")
        object.__setattr__(self, "effector_poses", effectors)
        object.__setattr__(self, "object_poses", objects)
        object.__setattr__(self, "gripper_positions", grippers.copy())


def insertion_speed_observation(observation: dict) -> SpeedObservation:
    """Map Insertion environment state into the generic paired contract.

    Pair order is right-effector/peg followed by left-effector/socket. This is
    environment knowledge, not knowledge of the scripted policy or its phase.
    """

    env_state = np.asarray(observation["env_state"], dtype=np.float64)
    qpos = np.asarray(observation["qpos"], dtype=np.float64)
    if env_state.shape != (14,) or qpos.shape != (14,):
        raise ValueError("unexpected Insertion observation shape")
    return SpeedObservation(
        effector_poses=np.stack(
            [observation["mocap_pose_right"], observation["mocap_pose_left"]]
        ),
        object_poses=env_state.reshape(2, 7),
        gripper_positions=np.asarray([qpos[13], qpos[6]]),
    )


def quaternion_angle_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return sign-invariant angular distance for paired wxyz quaternions."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / np.linalg.norm(left, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    cosine = np.clip(np.abs(np.sum(left * right, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cosine))


def behavior_metrics(
    current: SpeedObservation,
    initial: SpeedObservation,
    previous: SpeedObservation | None,
) -> dict:
    """Compute task-policy-independent attachment and proximity features."""

    relative_xyz = current.object_poses[:, :3] - current.effector_poses[:, :3]
    if previous is None:
        translation_delta = np.full(relative_xyz.shape[0], np.inf)
        rotation_delta = np.full(relative_xyz.shape[0], np.inf)
    else:
        previous_relative_xyz = (
            previous.object_poses[:, :3] - previous.effector_poses[:, :3]
        )
        translation_delta = np.linalg.norm(
            relative_xyz - previous_relative_xyz, axis=1
        )
        rotation_delta = quaternion_angle_degrees(
            current.object_poses[:, 3:], previous.object_poses[:, 3:]
        )
    return {
        "object_lift_m": current.object_poses[:, 2] - initial.object_poses[:, 2],
        "object_effector_translation_m": np.linalg.norm(relative_xyz, axis=1),
        "object_effector_translation_delta_m": translation_delta,
        "object_rotation_delta_deg": rotation_delta,
        "object_pair_distance_m": float(
            np.linalg.norm(current.object_poses[0, :3] - current.object_poses[1, :3])
        ),
        "gripper_positions": current.gripper_positions.copy(),
    }
