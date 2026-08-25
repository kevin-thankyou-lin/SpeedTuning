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

