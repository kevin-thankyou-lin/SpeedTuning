from scripts import run_act_vlm_paired_uniform_pick as module


def item(schedule, successes, speedup, verdict="qualified"):
    return {
        "schedule": schedule,
        "completed": 20,
        "successes": successes,
        "verdict": verdict,
        "matched_native_speedup": speedup,
        "safety_violations": 0,
        "physics_errors": 0,
    }


def test_reliability_first_selection_can_keep_uniform():
    items = [
        item([2.0] * 4, 20, 1.8),
        item([2.5] * 4, 19, 2.0),
        item(list(module.BALANCED), 18, 2.2),
        item(list(module.AGGRESSIVE), 17, 2.4, "rejected_at_20"),
    ]
    result = module.compare(items)
    assert result["selected"]["schedule"] == [2.0] * 4
    assert not result["balanced_strictly_beats_best_uniform"]
    assert len(result["pareto_frontier"]) == 3


def test_balanced_can_strictly_dominate_uniform():
    items = [
        item([2.0] * 4, 18, 1.8),
        item([2.5] * 4, 18, 2.0),
        item(list(module.BALANCED), 19, 2.2),
        item(list(module.AGGRESSIVE), 17, 2.4, "rejected_at_20"),
    ]
    result = module.compare(items)
    assert result["selected"]["schedule"] == list(module.BALANCED)
    assert result["balanced_strictly_beats_best_uniform"]
    assert result["pareto_frontier"] == [items[2]]


def test_incomplete_schedule_is_not_eligible():
    balanced = item(list(module.BALANCED), 18, 2.2)
    uniform = item([2.0] * 4, 9, 1.8, "rejected_at_10")
    uniform["completed"] = 10
    items = [uniform, item([2.5] * 4, 17, 2.0, "rejected_at_20"), balanced, item(list(module.AGGRESSIVE), 17, 2.4, "rejected_at_20")]
    assert module.compare(items)["selected"] == balanced

