import json

from scripts import run_act_common_grid_strider_causal_v30 as module


def record(seed, schedule, success, *, divergent_phase=None, steps=120):
    telemetry = []
    for index, phase in enumerate(module.PHASES):
        position = [0.01 * index, 0.5, 0.05 + 0.02 * index]
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


def test_common_grid_and_registered_gate():
    assert module.GRID == (1.0, 1.5, 2.0, 2.5, 3.0)
    assert module.CANDIDATE_BUDGET == 100
    assert module.gate_decision(module.summarize([record(i, [2.0] * 4, i < 18) for i in range(20)])) == "qualified"
    assert module.gate_decision(module.summarize([record(i, [2.0] * 4, i < 8) for i in range(10)])) == "reject_reliability"
    try:
        module.validate_schedule([3.5] * 4)
    except ValueError:
        pass
    else:
        raise AssertionError("3.5x escaped the common grid")


def test_rejected_aggressive_uniform_is_repaired_in_attributed_phase(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "CANDIDATE_BUDGET", 50)
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            schedule = list(schedule)
            index = seed % 20
            if schedule == [1.0] * 4:
                return record(seed, schedule, True, steps=240)
            if schedule == [2.0] * 4:
                return record(seed, schedule, True, steps=150)
            if schedule == [2.5] * 4:
                return record(seed, schedule, index < 8, divergent_phase=None if index < 8 else "grasp_lift", steps=120)
            if schedule == [2.5, 2.0, 2.5, 2.5]:
                return record(seed, schedule, True, steps=125)
            raise AssertionError(f"unexpected schedule {schedule}")

    ledger = module.Ledger(
        Runtime(), tmp_path, list(range(20)), list(range(100, 150)), record_search_telemetry=True
    )
    selection = module.run_search(ledger, "pick")
    assert selection["chronology"][:2] == [
        module.schedule_sha256([2.0] * 4),
        module.schedule_sha256([2.5] * 4),
    ]
    repair = selection["attribution_receipts"][0]
    assert repair["operation"] == "one_rung_causal_backoff"
    assert repair["phase"] == "grasp_lift"
    assert repair["proposed_schedule"] == [2.5, 2.0, 2.5, 2.5]
    assert repair["evidence"]["method"] == "same_seed_phase_exit_physical_divergence"
    assert selection["selected_schedule"] == [2.5, 2.0, 2.5, 2.5]
    assert selection["candidate_rollouts"] == 50
    assert selection["native_reference_rollouts"] == 20
    assert selection["final_bank_opened"] is False


def test_aggressive_three_x_can_receive_two_causal_repairs(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            schedule = list(schedule)
            index = seed % 20
            if schedule == [1.0] * 4:
                return record(seed, schedule, True, steps=240)
            if schedule == [2.0] * 4:
                return record(seed, schedule, True, steps=150)
            if schedule == [2.5] * 4:
                return record(seed, schedule, True, steps=130)
            if schedule == [3.0] * 4:
                return record(
                    seed,
                    schedule,
                    index < 8,
                    divergent_phase=None if index < 8 else "grasp_lift",
                    steps=110,
                )
            if schedule == [3.0, 2.5, 3.0, 3.0]:
                return record(
                    seed,
                    schedule,
                    index < 8,
                    divergent_phase=None if index < 8 else "transport",
                    steps=115,
                )
            if schedule == [3.0, 2.5, 2.5, 3.0]:
                return record(seed, schedule, True, steps=120)
            raise AssertionError(f"unexpected schedule {schedule}")

    ledger = module.Ledger(
        Runtime(), tmp_path, list(range(20)), list(range(100, 150)), record_search_telemetry=True
    )
    selection = module.run_search(ledger, "pick")
    repairs = [
        item for item in selection["attribution_receipts"]
        if item["operation"] == "one_rung_causal_backoff"
    ]
    assert [item["phase"] for item in repairs] == ["grasp_lift", "transport"]
    assert selection["selected_schedule"] == [3.0, 2.5, 2.5, 3.0]
    assert selection["frozen_repaired_phases"] == ["grasp_lift", "transport"]
    assert selection["candidate_rollouts"] == 80


def test_banks_are_pairwise_disjoint_and_final_unopened():
    banks = json.loads(
        (module.REPO_ROOT / "experiments/act_common_grid_strider_causal_v30/BANKS.json").read_text()
    )
    values = []
    for task in banks["tasks"].values():
        for name in ("search", "final"):
            spec = task[name]
            values.extend(range(spec["start"], spec["start"] + spec["count"]))
    assert len(values) == len(set(values)) == 210
    assert banks["final_banks_registered_but_unopened"] is True


def test_missing_same_seed_reference_stops_without_semantic_repair(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            schedule = list(schedule)
            index = seed % 20
            if schedule == [1.0] * 4:
                return record(seed, schedule, index >= 2, steps=240)
            if schedule == [2.0] * 4:
                return record(seed, schedule, index >= 2, divergent_phase="transport", steps=150)
            if schedule == [1.5] * 4:
                result = record(seed, schedule, True, steps=180)
                if index < 2:
                    result["attribution_telemetry"] = []
                return result
            raise AssertionError(f"unexpected schedule {schedule}")

    ledger = module.Ledger(
        Runtime(), tmp_path, list(range(20)), list(range(100, 150)), record_search_telemetry=True
    )
    selection = module.run_search(ledger, "insertion")
    receipt = selection["attribution_receipts"][0]
    assert receipt["operation"] == "causal_attribution_unavailable"
    assert receipt["proposed_schedule"] is None
    assert selection["selected_schedule"] == [1.5] * 4


def test_matched_terminal_failure_without_physical_divergence_stops(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            schedule = list(schedule)
            index = seed % 20
            if schedule == [1.0] * 4:
                return record(seed, schedule, True, steps=240)
            if schedule == [2.0] * 4:
                return record(seed, schedule, index < 8, steps=150)
            if schedule == [1.5] * 4:
                return record(seed, schedule, True, steps=180)
            raise AssertionError(f"unexpected schedule {schedule}")

    ledger = module.Ledger(
        Runtime(), tmp_path, list(range(20)), list(range(100, 150)), record_search_telemetry=True
    )
    selection = module.run_search(ledger, "tea")
    receipt = selection["attribution_receipts"][0]
    assert receipt["operation"] == "causal_attribution_unavailable"
    assert receipt["evidence"]["method"] == "matched_telemetry_without_observable_divergence"
    assert receipt["proposed_schedule"] is None
