from types import SimpleNamespace

import numpy as np
import torch

from rl.rainbowDQN.dqnAgent import DQNAgent
from scripts.run_act_speed_benchmark_cell import (
    load_contiguous_states,
    rainbow_snapshot,
    restore_rainbow,
)


class TinyEnv:
    obs_space = 4
    action_space = 5


def _agent(seed):
    return DQNAgent(
        TinyEnv(), memory_size=64, batch_size=4, target_update=5, seed=seed,
        atom_size=11, v_min=0, v_max=10, hidden_dim=16, device="cpu",
    )


def test_rainbow_episode_snapshot_restores_optimizer_replay_and_rng(tmp_path):
    agent = _agent(7)
    state = np.arange(4, dtype=np.float32)
    agent.memory.store(state, 2, 1.0, state + 1, False)
    agent.epsilon = 0.37
    agent.beta = 0.81
    snapshot = rainbow_snapshot(agent, 12, 3, [state])
    path = tmp_path / "resume.pt"
    torch.save(snapshot, path)

    restored = _agent(99)
    decision, updates, history = restore_rainbow(
        restored, torch.load(path, weights_only=False)
    )

    assert (decision, updates) == (12, 3)
    assert restored.epsilon == 0.37
    assert restored.beta == 0.81
    assert len(restored.memory) == 1
    np.testing.assert_array_equal(history[0], state)
    for key, value in agent.dqn.state_dict().items():
        torch.testing.assert_close(restored.dqn.state_dict()[key], value)


def test_contiguous_resume_rejects_gap(tmp_path):
    states = tmp_path / "states"
    states.mkdir()
    (states / "10.json").write_text('{"seed": 10, "identity_sha256": "x"}\n')
    (states / "12.json").write_text('{"seed": 12, "identity_sha256": "x"}\n')

    try:
        load_contiguous_states(states, [10, 11, 12], "x")
    except RuntimeError as exc:
        assert "non-contiguous" in str(exc)
    else:
        raise AssertionError("resume gap was accepted")
