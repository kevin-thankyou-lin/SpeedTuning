from scripts import run_act_strider_frontier_v2 as module


def report(schedule, successes, throughput, qualified=True):
    return {
        "role": "test",
        "schedule": schedule,
        "schedule_sha256": module.schedule_sha256(schedule),
        "qualified": qualified,
        "summary": {
            "successes": successes,
            "success_rate": successes / 15,
            "achieved_throughput_per_step": throughput,
        },
    }


def test_gate_is_successive_and_reliability_first():
    assert module.gate_decision({"episodes": 5, "successes": 4, "safety_violations": 0, "physics_errors": 0}) == "continue"
    assert module.gate_decision({"episodes": 5, "successes": 3, "safety_violations": 0, "physics_errors": 0}) == "reject_reliability"
    assert module.gate_decision({"episodes": 10, "successes": 9, "safety_violations": 0, "physics_errors": 0}) == "continue"
    assert module.gate_decision({"episodes": 15, "successes": 14, "safety_violations": 0, "physics_errors": 0}) == "qualified"


def test_incidents_override_success_count():
    assert module.gate_decision({"episodes": 15, "successes": 15, "safety_violations": 1, "physics_errors": 0}) == "reject_safety"
    assert module.gate_decision({"episodes": 5, "successes": 5, "safety_violations": 0, "physics_errors": 1}) == "halt_physics_error"


def test_failures_are_charged_to_terminal_horizon():
    records = [
        {"success": True, "first_success_step": 10, "physics_steps": 100, "safety_violation": None, "physics_error": None},
        {"success": False, "first_success_step": None, "physics_steps": 100, "safety_violation": None, "physics_error": None},
    ]
    value = module.summarize(records)
    assert value["total_episode_metric_steps"] == 110
    assert value["achieved_throughput_per_step"] == 1 / 110


def test_backoff_changes_only_implicated_phase_by_one_rung():
    assert module.make_backoff([2.5, 2.5, 2.5, 2.5], "grasp_lift") == [2.5, 2.0, 2.5, 2.5]


def test_promotion_changes_only_high_workload_phase_by_one_rung():
    assert module.make_promotion([3.0, 3.0, 3.0, 3.0], "transport") == [3.0, 3.0, 3.5, 3.0]


def test_phase_workload_uses_decision_boundaries_and_first_success():
    record = {
        "success": True,
        "first_success_step": 40,
        "physics_steps": 100,
        "safety_violation": None,
        "physics_error": None,
        "phase_decisions": [
            {"phase": "pre_grasp", "physics_step": 0, "speed": 2.0},
            {"phase": "grasp_lift", "physics_step": 10, "speed": 1.0},
            {"phase": "transport", "physics_step": 30, "speed": 1.0},
            {"phase": "interaction", "physics_step": 35, "speed": 1.0},
            {"phase": "transport", "physics_step": 80, "speed": 1.0},
        ],
    }
    assert module.phase_workloads(record) == {
        "pre_grasp": 20.0,
        "grasp_lift": 20.0,
        "transport": 5.0,
        "interaction": 5.0,
    }


def test_bend_cannot_replace_more_reliable_uniform():
    incumbent = report([2.0] * 4, 15, 0.010)
    bend = report([2.5, 1.5, 2.5, 2.5], 14, 0.020)
    assert not module.bend_replaces_uniform(bend, incumbent)


def test_bend_cannot_replace_faster_uniform():
    incumbent = report([2.0] * 4, 14, 0.020)
    bend = report([2.5, 1.5, 2.5, 2.5], 15, 0.019)
    assert not module.bend_replaces_uniform(bend, incumbent)


def test_bend_replaces_only_when_reliability_and_throughput_are_no_worse():
    incumbent = report([2.0] * 4, 14, 0.020)
    bend = report([2.5, 1.5, 2.5, 2.5], 14, 0.021)
    assert module.bend_replaces_uniform(bend, incumbent)


def test_uniform_incumbent_is_highest_failure_aware_throughput():
    slower = report([2.0] * 4, 15, 0.015)
    faster = report([2.5] * 4, 14, 0.017)
    rejected = report([3.0] * 4, 13, 0.020, qualified=False)
    assert module.choose_uniform_incumbent([slower, faster, rejected]) is faster


def test_pareto_names_keeps_reliability_speed_tradeoff():
    summaries = {
        "reliable": {"success_rate": 1.0, "throughput_delta_percent_vs_native": 50.0},
        "fast": {"success_rate": 0.9, "throughput_delta_percent_vs_native": 100.0},
        "dominated": {"success_rate": 0.8, "throughput_delta_percent_vs_native": 40.0},
    }
    assert module.pareto_names(summaries) == ["fast", "reliable"]


def test_full_search_preserves_uniform_and_uses_exact_budget(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed):
            mean_speed = sum(schedule) / len(schedule)
            return {
                "seed": seed,
                "schedule": list(schedule),
                "success": True,
                "first_success_step": int(400 / mean_speed),
                "physics_steps": 400,
                "safety_violation": None,
                "physics_error": None,
                "phase_decisions": [
                    {"phase": "pre_grasp", "physics_step": 0, "speed": schedule[0]},
                    {"phase": "grasp_lift", "physics_step": 80, "speed": schedule[1]},
                    {"phase": "transport", "physics_step": 100, "speed": schedule[2]},
                    {"phase": "interaction", "physics_step": 120, "speed": schedule[3]},
                ],
            }

    ledger = module.RolloutLedger(Runtime(), tmp_path, list(range(15)), list(range(100, 150)))
    selection = module.run_search(ledger, "pick")

    assert selection["search_rollouts"] == 60
    assert selection["uniform_incumbent_sha256"] == module.schedule_sha256([3.0] * 4)
    assert selection["bend_replaced_uniform"]
    assert selection["selected_schedule"] == [3.5, 3.0, 3.0, 3.0]
    assert selection["selected_schedule_sha256"] in selection["search_frontier_sha256"]


def test_final_bank_deduplicates_selected_uniform_controller():
    class Ledger:
        final_seeds = list(range(50))

        def __init__(self):
            self.calls = []

        def evaluate_final(self, schedule):
            self.calls.append(tuple(schedule))
            speed = sum(schedule) / 4
            result = {
                "schedule": list(schedule),
                "schedule_sha256": module.schedule_sha256(schedule),
                "summary": {
                    "episodes": 50,
                    "successes": 50,
                    "success_rate": 1.0,
                    "successful_mean_first_success_steps": 300 / speed,
                    "total_episode_metric_steps": 50 * 300 / speed,
                    "achieved_throughput_per_step": speed / 300,
                    "safety_violations": 0,
                    "physics_errors": 0,
                },
            }
            return result, []

    ledger = Ledger()
    result = module.run_final(ledger, {"selected_schedule": [2.5] * 4})

    assert len(ledger.calls) == 5
    assert result["unique_controllers_evaluated"] == 5
    assert result["new_final_rollouts"] == 250
    assert result["methods"]["strider_selected"]["alias_of"] == "uniform_2p5x"
