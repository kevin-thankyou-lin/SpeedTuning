from scripts import run_act_vlm_targeted_pick_screen as module
from scripts.three_scene_server import schedule_hash


def result(seed, schedule, *, success=True, steps=100):
    return {
        "seed": seed,
        "schedule": list(schedule),
        "success": success,
        "physics_steps": steps,
        "first_success_step": steps if success else None,
        "safety_violation": None,
        "physics_error": None,
    }


def state(overrides=None):
    overrides = overrides or {}
    candidates = {}
    for definition in module.CANDIDATES:
        schedule = definition["schedule"]
        steps, successes = overrides.get(definition["id"], (100, 3))
        candidates[schedule_hash(schedule)] = {
            "schedule": schedule,
            "discovery": [
                result(seed, schedule, success=index < successes, steps=steps)
                for index, seed in enumerate((1, 2, 3))
            ],
        }
    return {"candidates": candidates}


def test_candidates_include_uniforms_coarse_labels_and_disclosed_hypothesis():
    assert [item["id"] for item in module.CANDIDATES] == [
        "uniform_2x",
        "uniform_2p5x",
        "prior_incumbent",
        "coarse_fast_protected_fast_fast",
        "user_hypothesis_3_1p5_4_4",
    ]
    disclosed = module.CANDIDATES[-1]
    assert disclosed["schedule"] == [3.0, 1.5, 4.0, 4.0]
    assert "not_independent_vlm_output" in disclosed["origin"]


def test_selection_uses_only_clean_three_of_three_then_speed():
    selected = module.select_discovery_candidate(state({
        "uniform_2x": (90, 3),
        "uniform_2p5x": (70, 2),
        "prior_incumbent": (75, 3),
        "coarse_fast_protected_fast_fast": (65, 3),
        "user_hypothesis_3_1p5_4_4": (55, 2),
    }))
    assert selected["id"] == "coarse_fast_protected_fast_fast"


def test_selection_tie_prefers_simpler_then_registered_order():
    selected = module.select_discovery_candidate(state({
        "uniform_2x": (60, 3),
        "uniform_2p5x": (60, 3),
        "prior_incumbent": (60, 3),
        "coarse_fast_protected_fast_fast": (60, 3),
        "user_hypothesis_3_1p5_4_4": (60, 3),
    }))
    assert selected["id"] == "uniform_2x"

