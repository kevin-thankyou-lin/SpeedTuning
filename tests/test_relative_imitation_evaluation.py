import json
from types import SimpleNamespace

import numpy as np
from dm_control.rl import control

from scripts import evaluate_relative_imitation as evaluation


class _FakeEnvironment:
    def __init__(self):
        self.task = SimpleNamespace(max_reward=4)
        self.closed = False

    def reset(self):
        return SimpleNamespace(observation={"qpos": np.zeros(14)})

    def step(self, action):
        del action
        raise control.PhysicsError("unstable test state")

    def close(self):
        self.closed = True


def test_rollout_records_physics_instability_as_failure(monkeypatch):
    environment = _FakeEnvironment()
    monkeypatch.setattr(evaluation, "make_sim_env", lambda *args, **kwargs: environment)
    monkeypatch.setattr(
        evaluation,
        "get_task_spec",
        lambda task: SimpleNamespace(episode_len=5),
    )

    result = evaluation.rollout(
        "tea_bag", lambda observation: np.zeros((8, 14)), seed=17, replan_interval=8
    )

    assert result["success"] is False
    assert result["failure_reason"] == "physics_error"
    assert "unstable test state" in result["physics_error"]
    assert environment.closed


def test_atomic_partial_report_is_valid_json(tmp_path):
    destination = tmp_path / "partial.json"
    evaluation._write_json_atomic(destination, {"rollouts": [{"seed": 1}]})

    assert json.loads(destination.read_text()) == {"rollouts": [{"seed": 1}]}
