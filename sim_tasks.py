"""Shared definitions for the reconstructed SpeedTuning simulation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    """Metadata shared by the end-effector and joint-control environments."""

    name: str
    legacy_name: str
    episode_len: int
    ee_xml: str
    joint_xml: str


TASK_SPECS = {
    "pick_and_place": TaskSpec(
        name="pick_and_place",
        legacy_name="sim_transfer_cube_scripted",
        episode_len=400,
        ee_xml="bimanual_viperx_ee_transfer_cube.xml",
        joint_xml="bimanual_viperx_transfer_cube.xml",
    ),
    "insertion": TaskSpec(
        name="insertion",
        legacy_name="sim_insertion_scripted",
        episode_len=400,
        ee_xml="bimanual_viperx_ee_insertion.xml",
        joint_xml="bimanual_viperx_insertion.xml",
    ),
    "tea_bag": TaskSpec(
        name="tea_bag",
        legacy_name="sim_transfer_tea_bag_scripted",
        episode_len=500,
        ee_xml="bimanual_viperx_ee_transfer_tea_bag.xml",
        joint_xml="bimanual_viperx_transfer_tea_bag.xml",
    ),
}

_ALIASES = {
    "pick_and_place": "pick_and_place",
    "pick_place": "pick_and_place",
    "transfer_cube": "pick_and_place",
    "sim_transfer_cube": "pick_and_place",
    "sim_transfer_cube_scripted": "pick_and_place",
    "sim_transfer_cube_human": "pick_and_place",
    "insertion": "insertion",
    "sim_insertion": "insertion",
    "sim_insertion_scripted": "insertion",
    "sim_insertion_human": "insertion",
    "tea_bag": "tea_bag",
    "teabag": "tea_bag",
    "transfer_tea_bag": "tea_bag",
    "sim_transfer_tea_bag": "tea_bag",
    "sim_transfer_tea_bag_scripted": "tea_bag",
}


def normalize_task_name(task_name: str) -> str:
    """Return the public task name while accepting names used by the old code."""

    normalized = task_name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(TASK_SPECS)
        raise ValueError(f"Unknown task {task_name!r}. Supported tasks: {supported}") from exc


def get_task_spec(task_name: str) -> TaskSpec:
    return TASK_SPECS[normalize_task_name(task_name)]


def sample_box_pose(random_state=None) -> np.ndarray:
    """Sample the cube pose from the range used by the original ACT simulator."""

    rng = np.random if random_state is None else random_state
    position = rng.uniform([0.0, 0.4, 0.05], [0.2, 0.6, 0.05])
    return np.concatenate([position, [1.0, 0.0, 0.0, 0.0]])


def sample_insertion_pose(random_state=None) -> tuple[np.ndarray, np.ndarray]:
    """Sample peg and socket poses from the original simulator ranges."""

    rng = np.random if random_state is None else random_state
    peg_position = rng.uniform([0.1, 0.4, 0.05], [0.2, 0.6, 0.05])
    socket_position = rng.uniform([-0.2, 0.4, 0.05], [-0.1, 0.6, 0.05])
    identity_quaternion = [1.0, 0.0, 0.0, 0.0]
    return (
        np.concatenate([peg_position, identity_quaternion]),
        np.concatenate([socket_position, identity_quaternion]),
    )


def contact_pairs(physics) -> set[frozenset[str]]:
    """Collect active contacts without depending on MuJoCo's geom ordering."""

    pairs = set()
    for contact in physics.data.contact[: physics.data.ncon]:
        geom_1 = physics.model.id2name(contact.geom1, "geom")
        geom_2 = physics.model.id2name(contact.geom2, "geom")
        if geom_1 is not None and geom_2 is not None:
            pairs.add(frozenset((geom_1, geom_2)))
    return pairs


def _touching(pairs: set[frozenset[str]], first: str, second: str) -> bool:
    return frozenset((first, second)) in pairs


def _gripper_touch(
    pairs: set[frozenset[str]], object_geom: str, side: str
) -> bool:
    return any(
        _touching(pairs, object_geom, f"vx300s_{side}/10_{finger}_gripper_finger")
        for finger in ("left", "right")
    )


def _point_in_oriented_box(
    point: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether a world-space point lies inside an oriented box.

    MuJoCo exposes a box site's world rotation as a local-to-world matrix, so
    its transpose maps the point offset back into the site's local frame.
    Boundaries are inclusive to avoid contact-scale numerical flicker.
    """

    point = np.asarray(point, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    half_extents = np.asarray(half_extents, dtype=np.float64)
    local_point = rotation.T @ (point - center)
    return bool(np.all(np.abs(local_point) <= half_extents + tolerance))


def _oriented_boxes_overlap(
    center_a: np.ndarray,
    rotation_a: np.ndarray,
    half_extents_a: np.ndarray,
    center_b: np.ndarray,
    rotation_b: np.ndarray,
    half_extents_b: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether two oriented boxes overlap, including boundary contact."""

    center_a = np.asarray(center_a, dtype=np.float64)
    center_b = np.asarray(center_b, dtype=np.float64)
    rotation_a = np.asarray(rotation_a, dtype=np.float64).reshape(3, 3)
    rotation_b = np.asarray(rotation_b, dtype=np.float64).reshape(3, 3)
    half_extents_a = np.asarray(half_extents_a, dtype=np.float64)
    half_extents_b = np.asarray(half_extents_b, dtype=np.float64)

    relative_rotation = rotation_a.T @ rotation_b
    absolute_rotation = np.abs(relative_rotation) + tolerance
    translation = rotation_a.T @ (center_b - center_a)

    for axis_a in range(3):
        radius_a = half_extents_a[axis_a]
        radius_b = float(half_extents_b @ absolute_rotation[axis_a, :])
        if abs(translation[axis_a]) > radius_a + radius_b:
            return False

    for axis_b in range(3):
        radius_a = float(half_extents_a @ absolute_rotation[:, axis_b])
        radius_b = half_extents_b[axis_b]
        projected = abs(float(translation @ relative_rotation[:, axis_b]))
        if projected > radius_a + radius_b:
            return False

    for axis_a in range(3):
        next_a = (axis_a + 1) % 3
        last_a = (axis_a + 2) % 3
        for axis_b in range(3):
            radius_a = (
                half_extents_a[next_a] * absolute_rotation[last_a, axis_b]
                + half_extents_a[last_a] * absolute_rotation[next_a, axis_b]
            )
            radius_b = (
                half_extents_b[(axis_b + 1) % 3]
                * absolute_rotation[axis_a, (axis_b + 2) % 3]
                + half_extents_b[(axis_b + 2) % 3]
                * absolute_rotation[axis_a, (axis_b + 1) % 3]
            )
            projected = abs(
                translation[last_a] * relative_rotation[next_a, axis_b]
                - translation[next_a] * relative_rotation[last_a, axis_b]
            )
            if projected > radius_a + radius_b:
                return False
    return True


def tea_bag_overlaps_cup_volume(physics) -> bool:
    """Return whether any tea-bag volume lies inside the cup interior."""

    return _oriented_boxes_overlap(
        physics.named.data.geom_xpos["tea_bag"],
        physics.named.data.geom_xmat["tea_bag"],
        physics.named.model.geom_size["tea_bag"],
        physics.named.data.site_xpos["cup_success_volume"],
        physics.named.data.site_xmat["cup_success_volume"],
        physics.named.model.site_size["cup_success_volume"],
    )


def tea_bag_center_inside_cup_volume(physics) -> bool:
    """Return whether the tea-bag geom center lies inside the cup interior."""

    return _point_in_oriented_box(
        physics.named.data.geom_xpos["tea_bag"],
        physics.named.data.site_xpos["cup_success_volume"],
        physics.named.data.site_xmat["cup_success_volume"],
        physics.named.model.site_size["cup_success_volume"],
    )


def transfer_cube_reward(physics) -> int:
    pairs = contact_pairs(physics)
    left = _gripper_touch(pairs, "red_box", "left")
    right = _gripper_touch(pairs, "red_box", "right")
    table = _touching(pairs, "red_box", "table")

    if left and not table:
        return 4
    if left:
        return 3
    if right and not table:
        return 2
    if right:
        return 1
    return 0


def insertion_reward(physics) -> int:
    pairs = contact_pairs(physics)
    socket_geoms: Iterable[str] = ("socket-1", "socket-2", "socket-3", "socket-4")
    right = _gripper_touch(pairs, "red_peg", "right")
    left = any(
        _gripper_touch(pairs, socket, "left")
        for socket in socket_geoms
    )
    peg_on_table = _touching(pairs, "red_peg", "table")
    socket_on_table = any(_touching(pairs, socket, "table") for socket in socket_geoms)
    peg_in_socket = any(_touching(pairs, "red_peg", socket) for socket in socket_geoms)
    pin_touched = _touching(pairs, "red_peg", "pin")

    if pin_touched:
        return 4
    if peg_in_socket and not peg_on_table and not socket_on_table:
        return 3
    if left and right and not peg_on_table and not socket_on_table:
        return 2
    if left and right:
        return 1
    return 0


def tea_bag_reward(physics) -> int:
    pairs = contact_pairs(physics)
    right = _gripper_touch(pairs, "red_box", "right")
    on_table = _touching(pairs, "tea_bag", "table") or _touching(
        pairs, "red_box", "table"
    )
    in_cup = tea_bag_center_inside_cup_volume(physics)

    if in_cup:
        return 3
    if right and not on_table:
        return 2
    if right:
        return 1
    return 0
