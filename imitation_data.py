"""Absolute-command demonstrations and observation-anchored relative chunks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from ee_sim_env import make_ee_sim_env
from scripted_policy import make_scripted_policy
from sim_env import make_sim_env
from sim_tasks import get_task_spec, normalize_task_name
from oracle_phase_observation import OraclePhaseEncoder, PHASES


def relative_chunk(target_qpos, first_qpos):
    """Encode every future absolute command relative to one observation qpos."""

    targets = np.asarray(target_qpos, dtype=np.float32)
    anchor = np.asarray(first_qpos, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 14 or anchor.shape != (14,):
        raise ValueError("target_qpos must be [time, 14] and first_qpos must be [14]")
    return targets - anchor[None]


def decode_relative_chunk(delta_qpos, first_qpos):
    return np.asarray(delta_qpos, dtype=np.float32) + np.asarray(
        first_qpos, dtype=np.float32
    )[None]


def speed_condition_from_schedule(schedule, atol=1e-8):
    """Map an episode schedule to the registered slow/fast condition bit."""

    values = np.asarray(schedule, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("schedule must be a non-empty finite one-dimensional array")
    if np.any(values < 1.0 - atol):
        raise ValueError("dataset schedules may not run below native 1x")
    return int(not np.allclose(values, 1.0, atol=atol, rtol=0.0))


def _joint_target(timestep):
    target = np.asarray(timestep.observation["qpos"], dtype=np.float32).copy()
    gripper = timestep.observation["gripper_ctrl"]
    target[6] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(gripper[0])
    target[13] = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(gripper[2])
    return target


def _trace_speeds(phase_decisions, steps):
    decisions = sorted(phase_decisions, key=lambda item: int(item["physics_step"]))
    if not decisions or int(decisions[0]["physics_step"]) != 0:
        raise ValueError("phase decision trace must begin at physics step zero")
    speeds = []
    cursor = 0
    for step in range(int(steps)):
        while cursor + 1 < len(decisions) and int(
            decisions[cursor + 1]["physics_step"]
        ) <= step:
            cursor += 1
        speeds.append(float(decisions[cursor]["speed"]))
    return speeds


def collect_absolute_targets(task, seed, phase_decisions, physics_steps):
    """Recreate a frozen accelerated EE rollout as absolute joint commands."""

    task = normalize_task_name(task)
    env = make_ee_sim_env(
        task,
        render_images=False,
        seed=int(seed),
        randomize_object_pose=True,
    )
    try:
        timestep = env.reset()
        object_pose = np.asarray(timestep.observation["env_state"], dtype=np.float64)
        policy = make_scripted_policy(task)
        targets = []
        rewards = []
        speeds = _trace_speeds(phase_decisions, physics_steps)
        for speed in speeds:
            timestep = env.step(policy(timestep, step_inc=speed))
            targets.append(_joint_target(timestep))
            rewards.append(int(timestep.reward or 0))
        # The EE benchmark stops on first success. Joint-position replay has
        # actuator lag, so retain the scripted controller's remaining commands
        # through its nominal horizon using the already-frozen final speed.
        while policy.step_count < get_task_spec(task).episode_len:
            timestep = env.step(policy(timestep, step_inc=speeds[-1]))
            targets.append(_joint_target(timestep))
            rewards.append(int(timestep.reward or 0))
        return {
            "target_qpos": np.asarray(targets, dtype=np.float32),
            "object_pose": object_pose,
            "success": max(rewards, default=0) == env.task.max_reward,
        }
    finally:
        env.close()


def collect_scheduled_absolute_targets(task, seed, schedule):
    """Generate absolute targets online while an oracle-phase schedule executes."""

    task = normalize_task_name(task)
    schedule = tuple(float(value) for value in schedule)
    if len(schedule) != len(PHASES):
        raise ValueError(f"schedule must contain {len(PHASES)} phase speeds")
    condition = speed_condition_from_schedule(schedule)
    env = make_ee_sim_env(
        task,
        render_images=False,
        seed=int(seed),
        randomize_object_pose=True,
    )
    encoder = OraclePhaseEncoder(task)
    try:
        timestep = env.reset()
        encoder.reset()
        object_pose = np.asarray(timestep.observation["env_state"], dtype=np.float64)
        policy = make_scripted_policy(task)
        targets, rewards, decisions = [], [], []
        prior_phase = None
        while policy.step_count < get_task_spec(task).episode_len:
            phase = int(np.argmax(encoder(timestep.observation)))
            speed = schedule[phase]
            if phase != prior_phase:
                decisions.append(
                    {
                        "phase": PHASES[phase],
                        "physics_step": len(targets),
                        "speed": speed,
                    }
                )
                prior_phase = phase
            timestep = env.step(policy(timestep, step_inc=speed))
            targets.append(_joint_target(timestep))
            rewards.append(int(timestep.reward or 0))
        return {
            "target_qpos": np.asarray(targets, dtype=np.float32),
            "object_pose": object_pose,
            "source_success": max(rewards, default=0) == env.task.max_reward,
            "phase_decisions": decisions,
            "schedule": schedule,
            "speed_condition": condition,
        }
    finally:
        env.close()


def _run_joint_commands(task, seed, object_pose, commands, *, render_images):
    env = make_sim_env(
        task,
        render_images=render_images,
        render_camera_names=("angle",) if render_images else (),
        seed=int(seed),
        object_pose=object_pose,
    )
    qpos, images, rewards = [], [], []
    try:
        timestep = env.reset()
        for command in np.asarray(commands, dtype=np.float32):
            qpos.append(np.asarray(timestep.observation["qpos"], dtype=np.float32))
            if render_images:
                images.append(
                    np.asarray(timestep.observation["images"]["angle"], dtype=np.uint8)
                )
            timestep = env.step(command)
            rewards.append(int(timestep.reward or 0))
        return {
            "qpos": np.asarray(qpos, dtype=np.float32),
            "images": np.asarray(images, dtype=np.uint8),
            "success": max(rewards, default=0) == env.task.max_reward,
            "max_reward": max(rewards, default=0),
        }
    finally:
        env.close()


def validate_command_replays(task, seed, object_pose, qpos, commands, chunk_size=48):
    """Validate absolute and relative representations in fresh joint environments."""

    qpos = np.asarray(qpos, dtype=np.float32)
    commands = np.asarray(commands, dtype=np.float32)
    if qpos.shape != commands.shape or qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError("qpos and commands must have matching shape [time, 14]")

    absolute = _run_joint_commands(
        task, seed, object_pose, commands, render_images=False
    )
    if not absolute["success"]:
        raise RuntimeError("stored absolute joint-command replay did not preserve success")
    np.testing.assert_allclose(absolute["qpos"], qpos, rtol=0.0, atol=1e-6)

    decoded = []
    for start in range(0, len(commands), int(chunk_size)):
        stop = min(start + int(chunk_size), len(commands))
        delta = relative_chunk(commands[start:stop], qpos[start])
        decoded.append(decode_relative_chunk(delta, qpos[start]))
    decoded = np.concatenate(decoded)
    np.testing.assert_allclose(decoded, commands, rtol=0.0, atol=1e-6)
    relative = _run_joint_commands(
        task, seed, object_pose, decoded, render_images=False
    )
    if not relative["success"]:
        raise RuntimeError("decoded relative joint-command replay did not preserve success")
    return {
        "absolute_success": True,
        "relative_success": True,
        "max_absolute_command_error": float(np.max(np.abs(decoded - commands))),
        "max_relative_replay_qpos_deviation": float(
            np.max(np.abs(relative["qpos"] - qpos))
        ),
        "qpos_replay_atol": 1e-6,
        "chunk_size": int(chunk_size),
    }


def record_episode(
    task,
    seed,
    phase_decisions,
    physics_steps,
    output_path,
    *,
    replay_chunk_size=48,
):
    """Record actual joint states/commands from one successful joint rollout."""

    import h5py

    task = normalize_task_name(task)
    source = collect_absolute_targets(task, seed, phase_decisions, physics_steps)
    if not source["success"]:
        raise RuntimeError("source EE trace did not reproduce its recorded success")
    commands = np.asarray(source["target_qpos"], dtype=np.float32)
    rollout = _run_joint_commands(
        task, seed, source["object_pose"], commands, render_images=True
    )
    if not rollout["success"]:
        raise RuntimeError("absolute joint-command replay did not preserve task success")

    speeds = _trace_speeds(phase_decisions, physics_steps)
    condition = speed_condition_from_schedule(speeds)
    replay = validate_command_replays(
        task,
        seed,
        source["object_pose"],
        rollout["qpos"],
        commands,
        chunk_size=replay_chunk_size,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as root:
        root.attrs.update(
            task=task,
            seed=int(seed),
            source_physics_steps=int(physics_steps),
            relative_action="target_qpos[t+k] - observations/qpos[t]",
            speed_condition=condition,
            speed_label="fast" if condition else "slow",
            schedule=json.dumps([float(value) for value in sorted(set(speeds))]),
            replay_validation=json.dumps(replay, sort_keys=True),
            success=True,
        )
        observations = root.create_group("observations")
        observations.create_dataset("qpos", data=rollout["qpos"], compression="gzip")
        image_group = observations.create_group("images")
        image_group.create_dataset(
            "angle",
            data=rollout["images"],
            compression="gzip",
            chunks=(1, *rollout["images"][0].shape),
        )
        root.create_dataset("target_qpos", data=commands, compression="gzip")
        root.create_dataset("object_pose", data=source["object_pose"])
    return {
        "path": str(output_path),
        "steps": len(commands),
        "success": True,
        "speed_condition": condition,
        "replay_validation": replay,
    }


def record_scheduled_joint_episode(
    task, seed, schedule, output_path, *, replay_chunk_size=48
):
    """Retain one schedule only if joint execution and both replays succeed."""

    import h5py

    task = normalize_task_name(task)
    source = collect_scheduled_absolute_targets(task, seed, schedule)
    if not source["source_success"]:
        raise RuntimeError("scheduled source EE rollout did not succeed")
    commands = source["target_qpos"]
    rollout = _run_joint_commands(
        task, seed, source["object_pose"], commands, render_images=True
    )
    if not rollout["success"]:
        raise RuntimeError("scheduled joint-control rollout did not succeed")
    replay = validate_command_replays(
        task,
        seed,
        source["object_pose"],
        rollout["qpos"],
        commands,
        chunk_size=replay_chunk_size,
    )
    condition = int(source["speed_condition"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as root:
        root.attrs.update(
            task=task,
            seed=int(seed),
            relative_action="target_qpos[t+k] - observations/qpos[t]",
            speed_condition=condition,
            speed_label="fast" if condition else "slow",
            schedule=json.dumps(list(source["schedule"])),
            phase_decisions=json.dumps(source["phase_decisions"], sort_keys=True),
            replay_validation=json.dumps(replay, sort_keys=True),
            success=True,
        )
        observations = root.create_group("observations")
        observations.create_dataset("qpos", data=rollout["qpos"], compression="gzip")
        image_group = observations.create_group("images")
        image_group.create_dataset(
            "angle",
            data=rollout["images"],
            compression="gzip",
            chunks=(1, *rollout["images"][0].shape),
        )
        root.create_dataset("target_qpos", data=commands, compression="gzip")
        root.create_dataset("object_pose", data=source["object_pose"])
    return {
        "path": str(output_path),
        "seed": int(seed),
        "schedule": list(source["schedule"]),
        "speed_condition": condition,
        "steps": len(commands),
        "success": True,
        "replay_validation": replay,
    }


def fit_normalization(episode_paths, chunk_size):
    """Fit train-only qpos and per-chunk-step delta statistics."""

    import h5py

    qpos_values = []
    delta_values = [[] for _ in range(int(chunk_size))]
    for path in map(Path, episode_paths):
        with h5py.File(path, "r") as root:
            qpos = np.asarray(root["observations/qpos"], dtype=np.float32)
            targets = np.asarray(root["target_qpos"], dtype=np.float32)
        qpos_values.append(qpos)
        for start, anchor in enumerate(qpos):
            end = min(start + chunk_size, len(targets))
            for offset, delta in enumerate(targets[start:end] - anchor[None]):
                delta_values[offset].append(delta)
    qpos = np.concatenate(qpos_values)
    delta_mean, delta_std = [], []
    for values in delta_values:
        array = np.asarray(values, dtype=np.float32)
        delta_mean.append(array.mean(axis=0))
        delta_std.append(np.maximum(array.std(axis=0), 1e-3))
    return {
        "qpos_mean": qpos.mean(axis=0),
        "qpos_std": np.maximum(qpos.std(axis=0), 1e-3),
        "delta_mean": np.asarray(delta_mean),
        "delta_std": np.asarray(delta_std),
        "chunk_size": int(chunk_size),
        "relative_action": "target_qpos[t+k] - observations/qpos[t]",
    }


def save_normalization(stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)


def load_manifest(path):
    return json.loads(Path(path).read_text())
