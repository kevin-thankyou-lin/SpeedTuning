from scripts import aggregate_act_strider_table as module


def test_throughput_charges_failed_episode_horizon():
    native = [
        {"success": True, "first_success_step": 100, "physics_steps": 500},
        {"success": True, "first_success_step": 100, "physics_steps": 500},
    ]
    candidate = [
        {"success": True, "first_success_step": 50, "physics_steps": 500},
        {"success": False, "first_success_step": None, "physics_steps": 500},
    ]
    value = module.summarize(candidate, native)
    assert value["successful_rollout_speedup"] == 2.0
    assert value["achieved_throughput_per_step"] == 1 / 550
    assert value["throughput_delta_percent_vs_native"] < 0


def test_strider_reliability_rejection_reports_native_fallback():
    native = [
        {"success": True, "first_success_step": 100, "physics_steps": 500},
        {"success": True, "first_success_step": 100, "physics_steps": 500},
    ]
    accelerated = module.summarize([
        {"success": True, "first_success_step": 50, "physics_steps": 500},
        {"success": False, "first_success_step": None, "physics_steps": 500},
    ], native)
    result = {
        "accelerated_qualified_in_search": False,
        "final": accelerated,
        "search": {"episodes_used": 34},
        "selected_schedule": [2.0, 2.0, 2.0, 2.0],
    }
    selection = {"deployment_schedule": [1.0, 1.0, 1.0, 1.0]}

    deployed = module.strider_deployment(result, selection, native)

    assert deployed["native_fallback"]
    assert deployed["successes"] == 2
    assert deployed["successful_rollout_speedup"] == 1.0
    assert deployed["throughput_delta_percent_vs_native"] == 0.0
    assert deployed["exploratory_best_effort"]["final"] == accelerated
