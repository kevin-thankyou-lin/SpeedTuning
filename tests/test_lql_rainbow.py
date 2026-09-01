from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.rainbowDQN.dqnAgent import DQNAgent
from scripts.run_act_speed_benchmark_cell import rainbow_snapshot, restore_rainbow


class DummyEnv:
    obs_space = 1
    action_space = 2


class StateQ(torch.nn.Module):
    """Return the state scalar as action 0 and zero as action 1."""

    def forward(self, states):
        value = states[:, :1]
        return torch.cat((value, torch.zeros_like(value)), dim=1)

    def reset_noise(self):
        pass


def make_agent(**kwargs):
    return DQNAgent(
        DummyEnv(),
        memory_size=32,
        batch_size=2,
        target_update=10,
        seed=3,
        hidden_dim=8,
        atom_size=5,
        v_min=0.0,
        v_max=10.0,
        n_step=1,
        lql_trajectory_length=4,
        **kwargs,
    )


def transition(state, action, reward, next_state, done=False):
    return (
        np.asarray([state], dtype=np.float32),
        action,
        reward,
        np.asarray([next_state], dtype=np.float32),
        done,
    )


def test_lql_uses_trajectory_bounds_not_success_or_speed_order():
    agent = make_agent()
    agent.dqn = StateQ()
    agent.dqn_target = StateQ()
    # No success label and no ordinal interpretation of action ids is present.
    agent.lql_trajectories.append(
        (
            transition(0.0, 1, 4.0, 1.0),
            transition(1.0, 0, 4.0, 2.0),
            transition(2.0, 1, 4.0, 3.0, True),
        )
    )

    lower, upper, stats = agent._compute_lql_loss()

    assert lower.item() > 0
    assert upper.item() > 0
    assert stats["lql_lb_active_fraction"] > 0
    assert stats["lql_ub_active_fraction"] > 0


def test_lql_zero_when_no_complete_trajectory_exists():
    agent = make_agent()
    lower, upper, stats = agent._compute_lql_loss()
    assert lower.item() == 0
    assert upper.item() == 0
    assert stats == {
        "lql_lb_active_fraction": 0.0,
        "lql_ub_active_fraction": 0.0,
    }


def test_lql_configuration_and_trajectory_replay_round_trip():
    agent = make_agent(lql_lambda_lb=0.5, lql_lambda_ub=0.25)
    agent.lql_trajectories.append(
        (transition(0.0, 0, 1.0, 1.0), transition(1.0, 1, 0.0, 2.0, True))
    )
    snapshot = rainbow_snapshot(agent, decision=2, update_count=0, history=[])

    restored = make_agent(lql_lambda_lb=0.5, lql_lambda_ub=0.25)
    decision, updates, _ = restore_rainbow(restored, snapshot)

    assert (decision, updates) == (2, 0)
    assert len(restored.lql_trajectories) == 1
    assert restored.lql_trajectory_length == 4
    assert restored.lql_lambda_lb == 0.5
    assert restored.lql_lambda_ub == 0.25


def test_lql_resume_fails_closed_on_configuration_change():
    agent = make_agent()
    snapshot = rainbow_snapshot(agent, decision=0, update_count=0, history=[])
    changed = make_agent(lql_lambda_lb=0.5)
    with pytest.raises(RuntimeError, match="configuration differs"):
        restore_rainbow(changed, snapshot)
