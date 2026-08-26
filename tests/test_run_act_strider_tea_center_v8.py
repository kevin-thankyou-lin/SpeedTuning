import json
from pathlib import Path

from scripts import run_act_strider_tea_center_v8 as module
from scripts import run_act_strider_tea_volume_v5 as implementation


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "act_strider_tea_center_v8"


def _seeds(bank):
    return set(range(bank["start"], bank["start"] + bank["count"]))


def _summary(successes, throughput, mean_steps):
    return {
        "episodes": 50,
        "successes": successes,
        "success_rate": successes / 50,
        "successful_mean_first_success_steps": mean_steps,
        "total_episode_metric_steps": 10_000,
        "achieved_throughput_per_step": throughput,
        "safety_violations": 0,
        "physics_errors": 0,
    }


def _result(schedule, successes, throughput, mean_steps):
    return {
        "schedule": schedule,
        "schedule_sha256": module.v4.schedule_sha256(schedule),
        "summary": _summary(successes, throughput, mean_steps),
    }


def test_v8_center_criterion_hashes_are_frozen_and_current():
    old_schema = implementation.SUCCESS_CRITERION_SCHEMA
    try:
        implementation.SUCCESS_CRITERION_SCHEMA = "tea-cup-center-success-v1"
        criterion = implementation.checked_success_criterion(
            EXPERIMENT_ROOT / "SUCCESS_CRITERION.json"
        )
    finally:
        implementation.SUCCESS_CRITERION_SCHEMA = old_schema
    assert criterion["center_inside_required"] is True
    assert criterion["overlap_only_is_success"] is False


def test_v8_banks_are_fresh_and_disjoint_from_all_prior_strider_banks():
    current = json.loads((EXPERIMENT_ROOT / "BANKS.json").read_text())["tasks"][
        "tea"
    ]
    current_search = _seeds(current["search"])
    current_final = _seeds(current["final"])
    assert not current_search & current_final

    for path in sorted((REPO_ROOT / "experiments").glob("act_strider_*")):
        if path == EXPERIMENT_ROOT or not (path / "BANKS.json").exists():
            continue
        banks = json.loads((path / "BANKS.json").read_text()).get("tasks", {})
        if "tea" not in banks:
            continue
        prior = banks["tea"]
        for split in ("search", "final"):
            if split not in prior:
                continue
            prior_seeds = _seeds(prior[split])
            assert not current_search & prior_seeds, (path.name, split)
            assert not current_final & prior_seeds, (path.name, split)


def test_parallel_final_assembly_matches_sequential_result_shape():
    schedules = [[speed] * 4 for speed in (1.0, 1.5, 2.0, 2.5, 3.0)]
    selected = [2.5, 2.0, 2.5, 2.5]
    results = {
        module.v4.schedule_sha256(schedule): _result(
            schedule,
            50 if schedule == [1.0] * 4 else 49,
            0.005 + index * 0.001,
            200 - index * 20,
        )
        for index, schedule in enumerate(schedules + [selected])
    }

    final = module._assemble_final(
        results,
        {"selected_schedule": selected},
    )

    assert final["unique_controllers_evaluated"] == 6
    assert final["new_final_rollouts"] == 300
    assert final["parallel_final_workers"] == 4
    assert final["methods"]["strider_selected"]["schedule"] == selected
    assert final["methods"]["strider_selected"]["selected_by_strider"] is True
    assert "strider_selected" in final["empirical_frontier"]


def test_parallel_final_assembly_aliases_selected_uniform_without_rerun():
    schedules = [[speed] * 4 for speed in (1.0, 1.5, 2.0, 2.5, 3.0)]
    results = {
        module.v4.schedule_sha256(schedule): _result(
            schedule, 50, 0.005 + index * 0.001, 200 - index * 20
        )
        for index, schedule in enumerate(schedules)
    }

    final = module._assemble_final(
        results,
        {"selected_schedule": [2.0] * 4},
    )

    assert final["unique_controllers_evaluated"] == 5
    assert final["new_final_rollouts"] == 250
    assert final["methods"]["strider_selected"]["alias_of"] == "uniform_2x"
    assert final["selected_empirical_frontier_name"] == "uniform_2x"
