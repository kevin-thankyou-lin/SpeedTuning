from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.rainbowDQN.dqnAgent import DQNAgent
from scripts.run_act_speed_benchmark_cell import rainbow_snapshot, restore_rainbow


class DummyEnv:
    obs_space = 1
    action_space = 3
    speed_values = (1.0, 1.5, 2.0)


class StepEnv(DummyEnv):
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def step_decision(self, action, frame_skip):
        task_reward, done, success = next(self.outcomes)
        return (
            np.asarray([task_reward], dtype=np.float32),
            0.0,
            done,
            {
                "task_reward": task_reward,
                "success": success,
                "safety_violation": None,
            },
        )


class FixedQ(torch.nn.Module):
    def forward(self, states):
        row = torch.tensor([2.0, 1.0, 3.0], device=states.device)
        return row.unsqueeze(0).repeat(states.shape[0], 1)

    def reset_noise(self):
        pass


def make_agent(**kwargs):
    return DQNAgent(
        DummyEnv(),
        memory_size=32,
        batch_size=2,
        target_update=10,
        seed=7,
        hidden_dim=8,
        atom_size=5,
        v_min=0.0,
        v_max=10.0,
        n_step=1,
        adjacent_success_trajectory_length=8,
        **kwargs,
    )


def transition(action):
    state = np.asarray([0.0], dtype=np.float32)
    return (state, action, 0.0, state.copy(), False)


def test_adjacent_success_hinge_is_zero_margin_l1():
    agent = make_agent()
    agent.dqn = FixedQ()
    agent.adjacent_success_trajectories.append((transition(1), transition(2)))

    loss, stats = agent._compute_adjacent_success_loss()

    # Action 1 violates by 1 / support-width 10; action 2 does not violate.
    # Mean L1 hinge is therefore 0.05 (a squared hinge would be 0.005).
    assert loss.item() == pytest.approx(0.05)
    assert stats == {
        "adjacent_success_active_fraction": 0.5,
        "adjacent_success_comparisons": 2,
    }


def test_slowest_action_creates_no_adjacent_comparison():
    agent = make_agent()
    agent.adjacent_success_trajectories.append((transition(0),))
    loss, stats = agent._compute_adjacent_success_loss()
    assert loss.item() == 0.0
    assert stats == {
        "adjacent_success_active_fraction": 0.0,
        "adjacent_success_comparisons": 0,
    }


def test_adjacent_success_resume_round_trip_and_config_gate():
    agent = make_agent(adjacent_success_lambda=0.75)
    agent.adjacent_success_trajectories.append((transition(1),))
    snapshot = rainbow_snapshot(agent, decision=4, update_count=2, history=[])

    restored = make_agent(adjacent_success_lambda=0.75)
    decision, updates, _ = restore_rainbow(restored, snapshot)
    assert (decision, updates) == (4, 2)
    assert len(restored.adjacent_success_trajectories) == 1

    changed = make_agent(adjacent_success_lambda=0.5)
    with pytest.raises(RuntimeError, match="adjacent-success.*configuration"):
        restore_rainbow(changed, snapshot)


def test_adjacent_success_requires_ordered_speed_grid():
    env = SimpleNamespace(
        obs_space=1, action_space=3, speed_values=(1.0, 2.0, 1.5)
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        DQNAgent(
            env,
            memory_size=32,
            batch_size=2,
            target_update=10,
            seed=7,
            hidden_dim=8,
            atom_size=5,
            v_min=0.0,
            v_max=10.0,
            n_step=1,
            adjacent_success_trajectory_length=8,
        )


@pytest.mark.parametrize(
    ("outcomes", "accepted", "rejected"),
    [
        (((0.0, False, False), (1.0, True, True)), 1, 0),
        (((1.0, False, False), (0.0, True, True)), 0, 1),
        (((0.0, False, False), (1.0, True, False)), 0, 0),
    ],
)
def test_only_safe_successful_nonregressive_episodes_are_eligible(
    outcomes, accepted, rejected
):
    agent = DQNAgent(
        StepEnv(outcomes),
        memory_size=32,
        batch_size=2,
        target_update=10,
        seed=7,
        hidden_dim=8,
        atom_size=5,
        v_min=0.0,
        v_max=10.0,
        n_step=1,
        adjacent_success_trajectory_length=8,
    )
    state = np.asarray([0.0], dtype=np.float32)
    for index in range(2):
        agent.transition = [state.copy(), 1]
        state, _, done, _ = agent.step(1, 10)
        if done:
            break
    assert agent.adjacent_success_accepted_episodes == accepted
    assert agent.adjacent_success_rejected_regression == rejected
    assert len(agent.adjacent_success_trajectories) == accepted
