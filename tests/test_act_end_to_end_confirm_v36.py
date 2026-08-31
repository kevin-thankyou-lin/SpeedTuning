import json

from scripts import run_act_end_to_end_confirm_v36 as module


def record(seed, schedule, *, success=True, steps=120, divergent_phase=None):
    telemetry = []
    for index, phase in enumerate(module.v32.PHASES):
        position = [0.01 * index, 0.5, 0.05]
        if divergent_phase is not None and index >= module.v32.PHASES.index(divergent_phase):
            position[0] += 0.08
        telemetry.append(
            {
                "physics_step": 10 * (index + 1),
                "policy_time": 20.0 * (index + 1),
                "observed_phase": phase,
                "task_reward": float(index),
                "object_positions": [position],
            }
        )
    return {
        "seed": seed,
        "schedule": list(schedule),
        "success": success,
        "first_success_step": steps if success else None,
        "physics_steps": 300,
        "safety_violation": None,
        "physics_error": None,
        "phase_decisions": [
            {"phase": phase, "physics_step": 10 * index, "speed": schedule[index]}
            for index, phase in enumerate(module.v32.PHASES)
        ],
        "attribution_telemetry": telemetry,
    }


def prior():
    return {
        "schedule": [2.0, 1.5, 3.0, 1.5],
        "phase_importance": [0.4, 1.0, 0.1, 1.0],
        "offline_training_rollouts": 60,
    }


def test_search_confirms_complete_schedules_in_exactly_25_episodes(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            return record(seed, schedule, steps=200 if list(schedule) == [1.0] * 4 else 120)

    ledger = module.v32.Ledger(Runtime(), tmp_path, [0, 1, 2], [10, 11, 12, 13, 14])
    selection = module.run_search(ledger, "pick", prior())
    assert ledger.used() == selection["search_scientific_rollouts"] == 25
    assert selection["phase_dp_estimator_used"] is False
    assert selection["runtime_risk_gate_used"] is False
    assert selection["candidate_execution"] == "complete_schedule_end_to_end_without_runtime_gate"
    assert len(selection["discovery_reports"]) == 5
    assert all(item["summary"]["episodes"] == 3 for item in selection["discovery_reports"])
    assert len(selection["finalists"]) == 2
    assert all(item["summary"]["episodes"] == 8 for item in selection["finalists"])
    assert selection["selected_schedule_sha256"] in {
        item["schedule_sha256"] for item in selection["finalists"]
    }
    assert selection["final_bank_opened"] is False


def test_failed_prior_causes_one_phase_repairs_without_using_dp(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            schedule = list(schedule)
            native = schedule == [1.0] * 4
            return record(
                seed,
                schedule,
                success=native or schedule[0] <= 1.5,
                divergent_phase=None if native or schedule[0] <= 1.5 else "pre_grasp",
            )

    ledger = module.v32.Ledger(Runtime(), tmp_path, [0, 1, 2], [10, 11, 12, 13, 14])
    selection = module.run_search(ledger, "tea", prior())
    schedules = [item["schedule"] for item in selection["discovery_reports"]]
    assert len(schedules) == len({tuple(item) for item in schedules}) == 5
    assert any(
        item.get("operation") == "one_rung_causal_backoff"
        for item in selection["update_receipts"]
    )
    assert selection["search_scientific_rollouts"] == 25


def test_banks_are_fresh_exact_and_disjoint():
    banks = json.loads(
        (module.REPO_ROOT / "experiments/act_end_to_end_confirm_v36/BANKS.json").read_text()
    )
    all_seeds = []
    for spec in banks["tasks"].values():
        assert len(spec["discovery"]) == 3
        assert len(spec["confirmation"]) == 5
        assert len(spec["final"]) == 50
        task_seeds = spec["discovery"] + spec["confirmation"] + spec["final"]
        assert len(task_seeds) == len(set(task_seeds)) == 58
        assert min(task_seeds) >= 360000000
        all_seeds.extend(task_seeds)
    assert len(all_seeds) == len(set(all_seeds)) == 174


def test_unqualified_selection_falls_back_to_native(tmp_path):
    selection_dir = tmp_path / "search" / "tea"
    selection_dir.mkdir(parents=True)
    selection = {
        "selected_schedule": None,
        "selected_schedule_sha256": None,
    }
    (selection_dir / "SELECTION.json").write_text(json.dumps(selection))
    schedule, _ = module.method_schedule(tmp_path, "tea", "confirmed_phase_schedule")
    assert schedule == [1.0, 1.0, 1.0, 1.0]
