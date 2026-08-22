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
    base = server.probe([3, 1, 1, 1])
    assert server.screen(base["schedule_hash"])["discovery_successes"] == 3
    ladder = server.backoff(base["schedule_hash"])
    finalists = [value["schedule_hash"] for value in ladder["variants"]]

    result = server.rank(finalists)

    assert result["budget_used"] == 32
    assert result["accelerated_qualified"]
    assert result["qualified_schedule"] == [2.5, 1.0, 1.0, 1.0]
    assert [value["successes"] for value in result["finalists"]] == [10, 9]
    selection = json.loads((server.root / "public" / "SELECTION.json").read_text())
    assert selection["deployment_schedule"] == [2.5, 1.0, 1.0, 1.0]
    assert server.rank(finalists)["cache_hit"]


def test_native_baseline_is_external_fallback_not_finalist(server):
    candidate = server.probe([2, 1, 1, 1])
    server.screen(candidate["schedule_hash"])
    native_hash = module.schedule_hash((1, 1, 1, 1))

    with pytest.raises(ValueError, match="must be accelerated"):
        server.rank([candidate["schedule_hash"], native_hash])


def test_backoff_ladder_uses_protected_reserve_and_creates_accelerated_finalists(server):
    base = server.probe([3.5, 2.5, 4, 2])
    server.screen(base["schedule_hash"])
    server.state["episodes_used"] = 24
    server._persist()

    with pytest.raises(ValueError, match="use backoff"):
        server.probe([4, 3, 4, 2.5])

    result = server.backoff(base["schedule_hash"])

    assert [value["schedule"] for value in result["variants"]] == [
        [3.0, 2.0, 3.5, 1.5],
        [2.5, 1.5, 3.0, 1.0],
    ]
    assert all(value["discovery_successes"] == 3 for value in result["variants"])
    assert len(result["accelerated_finalist_hashes"]) == 3
    assert server.state["episodes_used"] == 30
    cached = server.backoff(base["schedule_hash"])
    assert cached["cache_hit"]
    assert server.state["episodes_used"] == 30

    ranked = server.rank([value["schedule_hash"] for value in result["variants"]])
    assert ranked["budget_used"] == 50
    assert ranked["qualified_schedule"] == [3.0, 2.0, 3.5, 1.5]


def test_rank_requires_mandatory_ladder(server):
    first = server.probe([2.5, 1, 1, 1])
    second = server.probe([3, 1, 1, 1])
    server.screen(first["schedule_hash"])
    server.screen(second["schedule_hash"])
    with pytest.raises(ValueError, match="mandatory backoff"):
        server.rank([first["schedule_hash"], second["schedule_hash"]])


def test_minimal_acceleration_can_rank_as_sole_degenerate_finalist(server):
    base = server.probe([1, 1, 1.5, 1])
    server.screen(base["schedule_hash"])
    ladder = server.backoff(base["schedule_hash"])

    assert ladder["variants"] == []
    ranked = server.rank([base["schedule_hash"]])

    assert ranked["budget_used"] == 16
    assert ranked["accelerated_qualified"]
    assert ranked["qualified_schedule"] == [1.0, 1.0, 1.5, 1.0]


def test_subthreshold_ranking_keeps_native_deployment_and_accelerated_benchmark(
    server, monkeypatch
):
    base = server.probe([3, 1, 1, 1])
    server.screen(base["schedule_hash"])
    ladder = server.backoff(base["schedule_hash"])
    original = module.run_phase_schedule

    def unreliable(*args, **kwargs):
        result = original(*args, **kwargs)
        seed = int(args[2])
        if seed in (200, 201):
            result["success"] = False
            result["raw_task_success"] = False
        return result

    monkeypatch.setattr(module, "run_phase_schedule", unreliable)
    result = server.rank([value["schedule_hash"] for value in ladder["variants"]])

    assert not result["accelerated_qualified"]
    assert result["qualified_schedule"] is None
    assert result["deployment_schedule"] == [1.0, 1.0, 1.0, 1.0]
    assert result["benchmark_schedule"] == [2.5, 1.0, 1.0, 1.0]
    selection = json.loads((server.root / "public" / "SELECTION.json").read_text())
    assert selection["schedule"] == result["benchmark_schedule"]
    assert selection["deployment_schedule"] == [1.0, 1.0, 1.0, 1.0]


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
