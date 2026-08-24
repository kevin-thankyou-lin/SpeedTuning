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
        "first_success_step": None if not success else steps,
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
    server.probe([2, 2, 2, 2])
    base = server.probe([3, 1, 1, 1])
    assert base["discovery_successes"] == 3
    ladder = server.backoff(
        base["schedule_hash"], "pre_grasp", "approach overshoots on the three previews"
    )
    finalists = [base["schedule_hash"], ladder["variants"][0]["schedule_hash"]]

    result = server.rank(finalists)

    assert result["budget_used"] == 35
    assert result["accelerated_qualified"]
    assert result["qualified_schedule"] == [3.0, 1.0, 1.0, 1.0]
    assert [value["successes"] for value in result["finalists"]] == [10, 10]
    selection = json.loads((server.root / "public" / "SELECTION.json").read_text())
    assert selection["deployment_schedule"] == [3.0, 1.0, 1.0, 1.0]
    assert server.rank(finalists)["cache_hit"]


def test_frontier_score_averages_marginal_saved_steps(server):
    anchor = server.probe([2, 2, 2, 2])

    score = server.score(
        anchor["schedule_hash"], [3, 2, 2, 2], safe_success_probability=0.8
    )

    assert score["candidate_schedule"] == [3.0, 2.0, 2.0, 2.0]
    assert score["mean_predicted_absolute_steps_saved"] > 0
    assert score["mean_expected_absolute_steps_saved"] == pytest.approx(
        0.8 * score["mean_predicted_absolute_steps_saved"]
    )
    assert score["mean_phase_predicted_steps_saved"]["pre_grasp"] > 0
    assert score["mean_phase_predicted_steps_saved"]["grasp_lift"] == 0


def test_native_baseline_is_external_fallback_not_finalist(server):
    candidate = server.probe([2, 2, 2, 2])
    native_hash = module.schedule_hash((1, 1, 1, 1))

    with pytest.raises(ValueError, match="must be accelerated"):
        server.rank([candidate["schedule_hash"], native_hash])


def test_backoff_ladder_uses_protected_reserve_and_creates_accelerated_finalists(server):
    server.probe([2, 2, 2, 2])
    base = server.probe([3.5, 2.5, 4, 2])
    server.state["episodes_used"] = 24
    server._persist()

    with pytest.raises(ValueError, match="use backoff"):
        server.probe([4, 3, 4, 2.5])

    result = server.backoff(
        base["schedule_hash"], "grasp_lift", "grasp is visibly unsettled before lift"
    )

    assert [value["schedule"] for value in result["variants"]] == [
        [3.5, 2.0, 4.0, 2.0],
        [3.5, 1.5, 4.0, 2.0],
    ]
    assert result["attributed_phase"] == "grasp_lift"
    assert result["causal_evidence"] == "grasp is visibly unsettled before lift"
    assert all(value["discovery_successes"] == 3 for value in result["variants"])
    assert len(result["accelerated_finalist_hashes"]) == 4
    assert server.state["episodes_used"] == 30
    cached = server.backoff(
        base["schedule_hash"], "grasp_lift", "grasp is visibly unsettled before lift"
    )
    assert cached["cache_hit"]
    assert server.state["episodes_used"] == 30

    ranked = server.rank([base["schedule_hash"], result["variants"][0]["schedule_hash"]])
    assert ranked["budget_used"] == 50
    assert ranked["qualified_schedule"] == [3.5, 2.0, 4.0, 2.0]


def test_rank_requires_mandatory_ladder(server):
    server.probe([2, 2, 2, 2])
    first = server.probe([2.5, 1, 1, 1])
    second = server.probe([3, 1, 1, 1])
    with pytest.raises(ValueError, match="mandatory backoff"):
        server.rank([first["schedule_hash"], second["schedule_hash"]])


def test_minimal_acceleration_can_rank_as_sole_degenerate_finalist(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "sample_object_pose", lambda task, seed: (float(seed),) * 7)
    monkeypatch.setattr(module, "run_phase_schedule", fake_rollout)
    custom = module.ThreeSceneServer(
        tmp_path, "pick_and_place", [101, 201, 103], list(range(300, 310)), 50, None
    )
    custom.probe([2, 2, 2, 2])
    base = custom.probe([1, 1, 1.5, 1])
    ladder = custom.backoff(
        base["schedule_hash"], "transport", "transport is the only accelerated phase"
    )

    assert ladder["variants"] == []
    ranked = custom.rank([base["schedule_hash"]])

    assert ranked["budget_used"] == 19
    assert ranked["accelerated_qualified"]
    assert ranked["qualified_schedule"] == [1.0, 1.0, 1.5, 1.0]


def test_subthreshold_ranking_keeps_native_deployment_and_accelerated_benchmark(
    server, monkeypatch
):
    server.probe([2, 2, 2, 2])
    base = server.probe([3, 1, 1, 1])
    ladder = server.backoff(base["schedule_hash"], "pre_grasp", "approach has least margin")
    original = module.run_phase_schedule

    def unreliable(*args, **kwargs):
        result = original(*args, **kwargs)
        seed = int(args[2])
        if seed in (200, 201):
            result["success"] = False
            result["raw_task_success"] = False
        return result

    monkeypatch.setattr(module, "run_phase_schedule", unreliable)
    result = server.rank([base["schedule_hash"], ladder["variants"][0]["schedule_hash"]])

    assert not result["accelerated_qualified"]
    assert result["qualified_schedule"] is None
    assert result["deployment_schedule"] == [1.0, 1.0, 1.0, 1.0]
    assert result["benchmark_schedule"] == [3.0, 1.0, 1.0, 1.0]
    selection = json.loads((server.root / "public" / "SELECTION.json").read_text())
    assert selection["schedule"] == result["benchmark_schedule"]
    assert selection["deployment_schedule"] == [1.0, 1.0, 1.0, 1.0]


def test_probe_always_completes_all_three_distinct_poses(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "sample_object_pose", lambda task, seed: (float(seed),) * 7)
    monkeypatch.setattr(module, "run_phase_schedule", fake_rollout)
    custom = module.ThreeSceneServer(
        tmp_path, "pick_and_place", [101, 201, 103], list(range(300, 310)), 50, None
    )

    failing = custom.probe([2, 2, 2, 2])

    assert failing["discovery_completed"] == 3
    assert failing["discovery_successes"] == 2
    assert custom.state["episodes_used"] == 6


def test_probe_pool_is_five_complete_three_pose_candidates(server):
    anchor = server.probe([2, 2, 2, 2])
    assert anchor["discovery_completed"] == 3
    for speed in (1.5, 2.5, 3.0, 3.5):
        result = server.probe([speed, 1, 1, 1])
        assert result["discovery_completed"] == 3
    assert server.state["episodes_used"] == 18
    with pytest.raises(ValueError, match="five-schedule"):
        server.probe([4, 1, 1, 1])


def test_backoff_rejects_native_attributed_phase(server):
    server.probe([2, 2, 2, 2])
    base = server.probe([3, 1, 2, 1])
    with pytest.raises(ValueError, match="already at native"):
        server.backoff(base["schedule_hash"], "grasp_lift", "grasp looks risky")


def test_imperfect_three_pose_base_can_enter_causal_ladder(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "sample_object_pose", lambda task, seed: (float(seed),) * 7)
    monkeypatch.setattr(module, "run_phase_schedule", fake_rollout)
    custom = module.ThreeSceneServer(
        tmp_path, "pick_and_place", [101, 201, 103], list(range(300, 310)), 50, None
    )
    base = custom.probe([2, 2, 2, 2])
    assert base["discovery_successes"] == 2
    assert custom.info()["preferred_backoff_base_hash"] == base["schedule_hash"]

    ladder = custom.backoff(
        base["schedule_hash"], "pre_grasp", "scene B overshoots during accelerated approach"
    )

    assert ladder["variants"][0]["schedule"] == [1.5, 2.0, 2.0, 2.0]
    assert ladder["variants"][0]["discovery_successes"] == 3
    required = custom._required_ranking_hashes()
    assert required == [base["schedule_hash"], ladder["variants"][0]["schedule_hash"]]


def test_imperfect_base_ranks_alone_when_causal_backoffs_are_not_safe(server, monkeypatch):
    original = module.run_phase_schedule

    def always_fails(*args, **kwargs):
        result = original(*args, **kwargs)
        result["success"] = False
        result["raw_task_success"] = False
        return result

    monkeypatch.setattr(module, "run_phase_schedule", always_fails)
    base = server.probe([2, 2, 2, 2])
    ladder = server.backoff(base["schedule_hash"], "pre_grasp", "all poses miss the approach")

    assert all(value["discovery_successes"] == 0 for value in ladder["variants"])
    ranked = server.rank([base["schedule_hash"]])
    assert not ranked["accelerated_qualified"]
    assert ranked["deployment_schedule"] == [1.0, 1.0, 1.0, 1.0]


def test_first_accelerated_challenger_must_be_uniform_two(server):
    with pytest.raises(ValueError, match="first accelerated challenger"):
        server.probe([4, 4, 4, 4])
    result = server.probe([2, 2, 2, 2])
    assert result["schedule"] == [2.0, 2.0, 2.0, 2.0]


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
