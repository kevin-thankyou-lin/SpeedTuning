import json
from pathlib import Path

from scripts import run_act_strider_codex_v16_budget48 as v16
from scripts import run_act_strider_vlm_v10 as base


def summary(successes, throughput=1.0, safety=0, physics=0):
    return {
        "successes": successes,
        "achieved_throughput_per_step": throughput,
        "safety_violations": safety,
        "physics_errors": physics,
    }


def test_eight_pose_gate_requires_no_reliability_deficit():
    assert v16.confirmation_decision(
        8, {"uniform": summary(8), "adaptive": summary(7, 2.0)}
    ) == "reject_paired_reliability_regression"
    assert v16.confirmation_decision(
        8, {"uniform": summary(7), "adaptive": summary(7, 2.0)}
    ) == "continue"


def test_sixteen_pose_gate_is_reliability_first_then_throughput():
    assert v16.confirmation_decision(
        16, {"uniform": summary(15, 1.0), "adaptive": summary(15, 1.02)}
    ) == "reject_throughput"
    assert v16.confirmation_decision(
        16, {"uniform": summary(15, 1.0), "adaptive": summary(15, 1.03)}
    ) == "select_adaptive"
    assert v16.confirmation_decision(
        16, {"uniform": summary(16, 1.0), "adaptive": summary(15, 2.0)}
    ) == "reject_paired_reliability_regression"


def test_safety_or_physics_always_rejects():
    assert v16.confirmation_decision(
        8, {"uniform": summary(8), "adaptive": summary(8, 2.0, safety=1)}
    ) == "reject_safety_or_physics"
    assert v16.confirmation_decision(
        16, {"uniform": summary(16), "adaptive": summary(16, 2.0, physics=1)}
    ) == "reject_safety_or_physics"


def test_maximum_budget_is_exactly_48():
    assert v16.DISCOVERY_ROLLOUTS + 2 * (16 - 4) == v16.MAX_SEARCH_ROLLOUTS == 48


def test_no_incumbent_uses_native_as_deployment_fallback():
    reports = [
        (
            {
                "role": "uniform_fallback",
                "schedule": [1.5] * 4,
                "schedule_sha256": "uniform",
                "qualified": False,
                "summary": summary(3, 0.003),
            },
            Path("uniform"),
        ),
        (
            {
                "role": "one_phase_promotion",
                "schedule": [2.0, 1.5, 2.0, 2.0],
                "schedule_sha256": "adaptive",
                "qualified": False,
                "summary": summary(3, 0.004),
            },
            Path("adaptive"),
        ),
    ]
    finalists = v16.choose_finalists({"uniform_incumbent": None}, reports)
    assert finalists["native_deployment_fallback"] is True
    assert finalists["adaptive"]["report"]["schedule"] == [2.0, 1.5, 2.0, 2.0]


def test_registered_panels_are_outcome_blind_and_match_banks():
    root = Path("experiments/act_strider_codex_v16_budget48")
    banks = json.loads((root / "BANKS.json").read_text())
    observed = set()
    for task, specs in banks["tasks"].items():
        panel = json.loads((root / "panels" / f"{task}.json").read_text())
        assert panel["panel_size"] == 16
        assert panel["selection_uses_policy_outcomes"] is False
        assert panel["selected_seeds"] == specs["search_primary"]["seeds"]
        for name in ("search_primary", "search_reserve", "final_primary", "final_reserve"):
            seeds = set(base._range(specs[name]))
            assert observed.isdisjoint(seeds)
            observed |= seeds
