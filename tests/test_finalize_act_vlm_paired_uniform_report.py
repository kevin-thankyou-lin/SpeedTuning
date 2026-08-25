from scripts import finalize_act_vlm_paired_uniform_report as module


def item(schedule, completed, successes, verdict, speedup):
    return {
        "schedule": schedule,
        "completed": completed,
        "successes": successes,
        "verdict": verdict,
        "matched_native_speedup": speedup,
        "safety_violations": 0,
        "physics_errors": 0,
    }


def test_no_qualified_uniform_is_reported_as_gate_win_not_strict_pair():
    items = [
        item([2.0] * 4, 20, 17, "rejected_at_20", 1.85),
        item([2.5] * 4, 10, 7, "rejected_at_10", 2.31),
        item(module.BALANCED, 20, 18, "qualified", 2.20),
        item([3.0, 1.5, 4.0, 4.0], 20, 17, "rejected_at_20", 2.38),
    ]
    result = module.repaired_comparison(items)
    assert result["selected"] == items[2]
    assert result["best_uniform"] is None
    assert result["balanced_strictly_beats_best_uniform"] is None
    assert result["balanced_wins_registered_gate_over_all_uniforms"]
    assert result["pareto_frontier"] == [items[2]]

