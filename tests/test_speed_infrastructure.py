from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dm_control.rl import control

from policy_speed_env import create_recorded_chunk_speed_env, create_speed_env
from speed_policy import (
    FixedSpeedPolicy,
    SpeedContext,
    SpeedPolicyAdapter,
    SpeedProfilePolicy,
    rollout_speed_policy,
)
from scripts.policy_cli import build_speed_env


def test_speed_policy_adapter_validates_external_output():
    context = SpeedContext(0.0, 0, 100, (1.0, 1.5))
    policy = SpeedPolicyAdapter(lambda observation, metadata: 1.5)
    assert policy(np.zeros(3), context) == 1.5

    invalid = SpeedPolicyAdapter(lambda observation, metadata: 0.0)
    with pytest.raises(ValueError, match="positive"):
        invalid(np.zeros(3), context)


def test_profile_policy_segments_nominal_time():
    policy = SpeedProfilePolicy([1.0, 2.0, 3.0])
    assert policy.select_speed(None, SpeedContext(0, 0, 90, (1.0,))) == 1.0
    assert policy.select_speed(None, SpeedContext(40, 0, 90, (1.0,))) == 2.0
    assert policy.select_speed(None, SpeedContext(89, 0, 90, (1.0,))) == 3.0


def test_fixed_speed_policy_pairs_with_scripted_environment():
    env = create_speed_env("tea_bag", seed=0)
    result = rollout_speed_policy(env, FixedSpeedPolicy(1.5))
    assert result["success"]
    assert result["physics_steps"] < env.episode_len


def test_physics_instability_becomes_failed_terminal_transition(monkeypatch):
    env = create_speed_env("tea_bag", seed=0)
    env.reset()

    def unstable_step(action):
        del action
        raise control.PhysicsError("invalid simulated state")

    monkeypatch.setattr(env.env, "step", unstable_step)
    observation, reward, done, info = env.step(3.0, quantized=False)

    assert done
    assert not info["success"]
    assert info["first_success_step"] is None
    assert "invalid simulated state" in info["physics_error"]
    assert observation.shape == (env.obs_space,)
    assert np.isfinite(reward)


def test_safety_monitor_is_checked_each_physics_tick_and_latched():
    calls = []

    def monitor(observation):
        calls.append(np.asarray(observation["qpos"]).copy())
        return "test_workspace_violation" if len(calls) == 2 else None

    env = create_speed_env(
        "tea_bag", seed=0, safety_monitor=monitor, decision_frame_skip=5
    )
    try:
        env.reset()
        _, _, done, info = env.step_decision(1.0, quantized=False)
    finally:
        env.close()

    assert not done
    assert len(calls) == 5
    assert info["safety_violation"] == "test_workspace_violation"


def test_recorded_chunk_policy_pairs_with_speed_environment():
    env = create_recorded_chunk_speed_env("tea_bag", chunk_size=25, seed=0)
    result = rollout_speed_policy(env, FixedSpeedPolicy(1.0))
    assert result["success"]
    assert result["mean_speed"] == 1.0


def test_cli_environment_can_terminate_on_success():
    args = SimpleNamespace(
        task="pick_and_place",
        seed=0,
        speed_values=(1.0,),
        frame_stack=1,
        frame_skip=10,
        randomize_object_pose=False,
        terminate_on_success=True,
        speed_decision_mode="fixed",
        base_policy="scripted",
        speed_observation="state",
        include_qpos=True,
        include_qvel=True,
        include_env_state=True,
        device="cpu",
    )
    env = build_speed_env(args)
    try:
        result = rollout_speed_policy(env, FixedSpeedPolicy(1.0))
    finally:
        env.close()
    assert result["success"]
    assert result["physics_steps"] < 400
    assert result["first_success_step"] == result["physics_steps"]


def test_cli_environment_reuses_explicit_object_pose():
    pose = (0.03, 0.51, 0.05, 1.0, 0.0, 0.0, 0.0)
    args = SimpleNamespace(
        task="pick_and_place",
        seed=17,
        speed_values=(1.0,),
        frame_stack=1,
        frame_skip=10,
        randomize_object_pose=False,
        object_pose=pose,
        terminate_on_success=True,
        speed_decision_mode="fixed",
        base_policy="scripted",
        speed_observation="state",
        include_qpos=True,
        include_qvel=True,
        include_env_state=True,
        device="cpu",
    )
    env = build_speed_env(args)
    try:
        first = env.reset()
        first_state = env.cur_ts.observation["env_state"].copy()
        second = env.reset()
        second_state = env.cur_ts.observation["env_state"].copy()
    finally:
        env.close()
    assert first.shape == second.shape
    np.testing.assert_allclose(first_state, pose)
    np.testing.assert_allclose(second_state, pose)


def test_phase_entry_mode_makes_one_decision_per_phase():
    from oracle_phase_observation import OraclePhaseEncoder

    env = create_speed_env(
        "pick_and_place",
        speed_values=(1.0,),
        observation_encoder=OraclePhaseEncoder("pick_and_place"),
        decision_mode="phase_entry",
        terminate_on_success=True,
        seed=0,
    )
    try:
        result = rollout_speed_policy(env, FixedSpeedPolicy(1.0))
    finally:
        env.close()
    assert result["success"]
    assert result["decisions"] == 4


def test_tabular_phase_policy_checkpoint_round_trip(tmp_path: Path):
    from oracle_phase_observation import OraclePhaseEncoder
    from tabular_phase_speed import (
        TabularTrainingConfig,
        load_tabular_phase_speed_policy,
        train_tabular_phase_speed_policy,
    )

    env = create_speed_env(
        "pick_and_place",
        speed_values=(1.0, 2.0),
        observation_encoder=OraclePhaseEncoder("pick_and_place"),
        decision_mode="phase_entry",
        terminate_on_success=True,
        seed=7,
    )
    checkpoint = tmp_path / "table.json"
    try:
        result = train_tabular_phase_speed_policy(
            env,
            checkpoint,
            config=TabularTrainingConfig(episodes=2),
            seed=7,
        )
        policy = load_tabular_phase_speed_policy(checkpoint)
        rollout = rollout_speed_policy(env, policy)
    finally:
        env.close()

    assert result["episodes"] == 2
    assert len(result["schedule"]) == 4
    assert checkpoint.exists()
    assert rollout["decisions"] <= 4


@pytest.mark.rl
def test_prioritized_replay_samples_newest_transition():
    pytest.importorskip("torch")
    from rl.rainbowDQN.replayBuffer import PrioritizedReplayBuffer

    replay = PrioritizedReplayBuffer(
        obs_dim=1,
        size=8,
        batch_size=4,
        alpha=1.0,
    )
    for value in range(4):
        replay.store(
            np.array([value], dtype=np.float32),
            value,
            0.0,
            np.array([value + 1], dtype=np.float32),
            False,
        )
    replay.update_priorities(
        np.arange(4),
        np.array([1e-6, 1e-6, 1e-6, 100.0]),
    )

    assert 3 in replay.sample_batch(beta=0.4)["indices"]


@pytest.mark.rl
def test_public_rainbow_training_checkpoint_round_trip(tmp_path: Path):
    pytest.importorskip("torch")
    from speed_policy import RainbowSpeedPolicy
    from speed_training import (
        RainbowTrainingConfig,
        evaluate_rainbow_speed_policy,
        train_rainbow_speed_policy,
    )

    env = create_speed_env("tea_bag", speed_values=(1.0,), seed=0)
    config = RainbowTrainingConfig(
        decisions=12,
        memory_size=64,
        batch_size=4,
        learning_starts=4,
        frame_skip=50,
        gradient_steps=1,
        train_interval=1,
        target_update=2,
        norm_update_interval=2,
        exploration_steps=20,
        atom_size=11,
        n_step=3,
        hidden_dim=32,
        update_schedule="episode",
        checkpoint_interval=6,
    )
    checkpoint = tmp_path / "speed.pt"
    result = train_rainbow_speed_policy(
        env,
        checkpoint,
        config=config,
        seed=0,
        device="cpu",
        progress=False,
    )
    assert checkpoint.exists()
    assert len(result["numbered_checkpoints"]) == 2
    assert all(Path(path).exists() for path in result["numbered_checkpoints"])
    assert result["updates"] > 0
    assert result["losses_finite"]

    observation = env.reset()
    policy = RainbowSpeedPolicy.load(checkpoint)
    assert policy.frame_skip == 50
    assert policy.observation_spec == env.observation_spec()
    assert policy.environment_spec == env.environment_spec()
    assert policy.select_speed(
        observation,
        SpeedContext(0, 0, env.episode_len, env.speed_values),
    ) in env.speed_values
    evaluation = evaluate_rainbow_speed_policy(
        env, checkpoint, episodes=1, device="cpu"
    )
    assert evaluation["episodes"] == 1
    assert "mean_acceleration" in evaluation

    randomized_env = create_speed_env(
        "tea_bag",
        speed_values=(1.0,),
        randomize_object_pose=True,
        seed=0,
    )
    with pytest.raises(ValueError, match="Checkpoint environment"):
        rollout_speed_policy(randomized_env, policy)


@pytest.mark.rl
def test_rainbow_training_stops_after_completed_episode(tmp_path: Path):
    pytest.importorskip("torch")
    from speed_training import RainbowTrainingConfig, train_rainbow_speed_policy

    env = create_speed_env(
        "pick_and_place", speed_values=(2.0,), seed=4, terminate_on_success=True
    )
    try:
        result = train_rainbow_speed_policy(
            env,
            tmp_path / "episode-limited.pt",
            config=RainbowTrainingConfig(
                decisions=100,
                max_episodes=1,
                memory_size=16,
                batch_size=4,
                learning_starts=4,
                frame_skip=50,
                gradient_steps=1,
                hidden_dim=16,
                atom_size=11,
                n_step=1,
            ),
            seed=4,
            device="cpu",
            progress=False,
        )
    finally:
        env.close()
    assert result["episodes"] == 1
    assert result["decisions"] < 100
