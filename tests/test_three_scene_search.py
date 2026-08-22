import json

import pytest

from scripts import three_scene_server as module
from scripts.train_tabular_phase_speed import CyclingSpeedEnv


def fake_rollout(task, schedule, seed, *, object_pose=None, video_path=None, observation_encoder=None):
    del object_pose, observation_encoder
    speed = float(schedule[0])
    success = not (speed == 2.0 and seed == 201)
    steps = int(300 / speed)
    return {
        "task": task,
        "seed": int(seed),
        "schedule": list(schedule),
        "success": success,
        "raw_task_success": success,
        "physics_steps": steps,
        "success_only_acceleration": None if not success else 400 / steps,
        "safety_violation": None,
        "phase_decisions": [
            {"phase": "pre_grasp", "physics_step": 0, "speed": speed},
            {"phase": "grasp_lift", "physics_step": steps // 4, "speed": schedule[1]},
            {"phase": "transport", "physics_step": steps // 2, "speed": schedule[2]},
            {"phase": "interaction", "physics_step": 3 * steps // 4, "speed": schedule[3]},
        ],
        "video_path": None if video_path is None else str(video_path),
    }


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "sample_object_pose", lambda task, seed: (float(seed),) * 7)
    monkeypatch.setattr(module, "run_phase_schedule", fake_rollout)
    return module.ThreeSceneServer(
        tmp_path,
        "pick_and_place",
        [101, 102, 103],
        list(range(200, 210)),
        50,
        None,
    )


def test_three_scene_rank_uses_measured_shared_bank(server):
    first = server.probe([2, 1, 1, 1])
    second = server.probe([3, 1, 1, 1])
    assert server.screen(first["schedule_hash"])["discovery_successes"] == 3
    assert server.screen(second["schedule_hash"])["discovery_successes"] == 3

    result = server.rank([first["schedule_hash"], second["schedule_hash"]])

    assert result["budget_used"] == 29
    assert result["selected_schedule"] == [3.0, 1.0, 1.0, 1.0]
    assert [value["successes"] for value in result["finalists"]] == [9, 10]
    selection = json.loads((server.root / "public" / "SELECTION.json").read_text())
    assert selection["schedule"] == [3.0, 1.0, 1.0, 1.0]
    assert server.rank([first["schedule_hash"], second["schedule_hash"]])["cache_hit"]


def test_native_baseline_is_eligible_without_duplicate_rollouts(server):
    candidate = server.probe([2, 1, 1, 1])
    server.screen(candidate["schedule_hash"])
    native_hash = module.schedule_hash((1, 1, 1, 1))

    result = server.rank([candidate["schedule_hash"], native_hash])

    assert result["budget_used"] == 26
    assert server.state["episodes_used"] == 26


def test_screen_is_fail_fast_and_ineligible_candidate_cannot_rank(server):
    failing = server.probe([2, 1, 1, 1])
    # Make scene B fail for this schedule.
    server.discovery_seeds[1] = 201
    screened = server.screen(failing["schedule_hash"])
    assert screened["discovery_completed"] == 2
    assert screened["discovery_successes"] == 1
    with pytest.raises(ValueError, match="three safe discovery successes"):
        server.rank([failing["schedule_hash"], "missing"])


def test_refinement_pool_is_bounded(server):
    for speed in (1.5, 2.0, 2.5):
        server.refine([speed, 1, 1, 1])
    assert server.state["refinement_episodes"] == 8


class DummyEnv:
    action_space = 2
    speed_values = (1.0, 2.0)

    def __init__(self, name):
        self.name = name
        self.closed = False

    def reset(self):
        return self.name

    def step_decision(self, value):
        return self.name, value

    def observation_spec(self):
        return {"name": self.name}

    def environment_spec(self):
        return {"name": self.name}

    def close(self):
        self.closed = True


def test_cycling_speed_env_reuses_exact_three_scenes():
    envs = [DummyEnv(value) for value in "abc"]
    cycle = CyclingSpeedEnv(envs)
    assert [cycle.reset() for _ in range(5)] == ["a", "b", "c", "a", "b"]
    assert cycle.step_decision(7) == ("b", 7)
    assert cycle.environment_spec()["fixed_scene_cycle"] == 3
    cycle.close()
    assert all(env.closed for env in envs)
