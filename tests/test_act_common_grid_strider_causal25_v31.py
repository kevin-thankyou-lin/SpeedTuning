import json

from scripts import run_act_common_grid_strider_causal25_v31 as module


def record(seed, schedule, success, divergent_phase=None, steps=120):
    telemetry = []
    for index, phase in enumerate(module.PHASES):
        position = [0.01 * index, 0.5, 0.05]
        if divergent_phase is not None and index >= module.PHASES.index(divergent_phase):
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
            for index, phase in enumerate(module.PHASES)
        ],
        "attribution_telemetry": telemetry,
    }


def test_exact_25_budget_includes_native_and_confirmation(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            schedule = list(schedule)
            if schedule == [2.0] * 4 and seed % 10 == 0:
                return record(seed, schedule, False, "pre_grasp", steps=150)
            return record(seed, schedule, True, steps=240 if schedule == [1.0] * 4 else 130)

    ledger = module.Ledger(Runtime(), tmp_path, [0, 1, 2], [10, 11, 12, 13, 14])
    selection = module.run_search(ledger, "tea")
    schedules = [item["schedule"] for item in selection["discovery_reports"]]
    assert schedules[:3] == [[1.0] * 4, [2.0] * 4, [1.5, 2.0, 2.0, 2.0]]
    assert selection["update_receipts"][0]["operation"] == "one_rung_causal_backoff"
    assert selection["update_receipts"][0]["phase"] == "pre_grasp"
    assert len({tuple(item) for item in schedules}) == 5
    assert len(selection["finalists"]) == 2
    assert selection["search_scientific_rollouts"] == ledger.used() == 25
    assert selection["final_bank_opened"] is False


def test_aggressive_ceiling_then_two_distinct_causal_backoffs(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            schedule = list(schedule)
            if schedule == [3.0] * 4:
                return record(seed, schedule, False, "grasp_lift", steps=100)
            if schedule == [3.0, 2.5, 3.0, 3.0]:
                return record(seed, schedule, False, "transport", steps=110)
            return record(seed, schedule, True, steps=140)

    ledger = module.Ledger(Runtime(), tmp_path, [0, 1, 2], [10, 11, 12, 13, 14])
    selection = module.run_search(ledger, "pick")
    schedules = [item["schedule"] for item in selection["discovery_reports"]]
    assert schedules == [
        [1.0] * 4,
        [2.0] * 4,
        [3.0] * 4,
        [3.0, 2.5, 3.0, 3.0],
        [3.0, 2.5, 2.5, 3.0],
    ]
    repairs = [
        item for item in selection["update_receipts"]
        if item["operation"] == "one_rung_causal_backoff"
    ]
    assert [item["phase"] for item in repairs] == ["grasp_lift", "transport"]
    assert selection["search_scientific_rollouts"] == 25


def test_banks_are_fresh_disjoint_and_final_unopened():
    banks = json.loads(
        (module.REPO_ROOT / "experiments/act_common_grid_strider_causal25_v31/BANKS.json").read_text()
    )
    seeds = []
    for task in banks["tasks"].values():
        for name in ("discovery", "confirmation", "final"):
            spec = task[name]
            seeds.extend(range(spec["start"], spec["start"] + spec["count"]))
    assert len(seeds) == len(set(seeds)) == 174
    assert banks["final_banks_registered_but_unopened"] is True
    assert module.SEARCH_BUDGET == 25
