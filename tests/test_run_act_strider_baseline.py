import json

from scripts import run_act_strider_baseline as module


def test_episode_metric_steps_charges_failures_to_terminal_horizon():
    assert module.episode_metric_steps({"success": True, "first_success_step": 12, "physics_steps": 100}) == 12
    assert module.episode_metric_steps({"success": False, "first_success_step": None, "physics_steps": 100}) == 100


def test_earliest_failed_phase_uses_last_reached_phase():
    value = {
        "success": False,
        "safety_violation": None,
        "phase_decisions": [
            {"phase": "pre_grasp"},
            {"phase": "grasp_lift"},
        ],
    }
    assert module.earliest_failed_phase(value) == "grasp_lift"


def test_proposals_are_five_unique_valid_schedules():
    path = module.REPO_ROOT / "experiments/act_strider_baseline_v1/proposals.json"
    receipt = json.loads(path.read_text())
    for task in ("pick", "tea", "insertion"):
        schedules = receipt["tasks"][task]["candidate_schedules"]
        assert len(schedules) == len({tuple(value) for value in schedules}) == 5
        assert schedules[0] == [2.0, 2.0, 2.0, 2.0]
        for schedule in schedules:
            assert module.validate_schedule(schedule) == tuple(schedule)

