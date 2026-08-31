import json

import numpy as np
import pytest

from scripts import run_act_transport3_boundary_v38 as module


def record(seed, controller, *, success=True, steps=120, events=0):
    return {
        "seed": int(seed),
        "schedule": list(controller["schedule"]),
        "controller_sha256": controller["controller_sha256"],
        "success": bool(success),
        "first_success_step": int(steps) if success else None,
        "physics_steps": 300,
        "safety_violation": None,
        "physics_error": None,
        "phase_decisions": [],
        "attribution_telemetry": [],
        "terminal_approach_event_count": int(events),
    }


def controllers():
    champion = module.static_controller([1.5, 2.5, 2.0, 2.0])
    challenger = module.validate_controller(
        {
            "type": "transport3_terminal_approach_gate",
            "task_label": "tea",
            "phase_order": list(module.PHASES),
            "schedule": [1.5, 2.5, 3.0, 2.0],
            "gate": {
                "phase": "transport",
                "entry_metric": "right_effector_to_fixed_target_xy",
                "entry_threshold_lte": 0.12,
                "protected_speed": 2.0,
                "latch_until_phase_exit": True,
                "current_observation_only": True,
                "reward_visible": False,
                "object_state_visible": False,
                "future_or_terminal_signal_visible": False,
                "fixed_target_xy": [-0.1, 0.6],
            },
        }
    )
    return champion, challenger


def bank():
    return {
        "challenger_discovery": [0, 1, 2, 3, 4],
        "paired": list(range(10, 20)),
        "final": list(range(100, 150)),
    }


def test_exact25_promotes_reliable_faster_transport_challenger(tmp_path):
    champion, challenger = controllers()

    class Runtime:
        def rollout(self, controller, seed, *, record_attribution_telemetry=False):
            del record_attribution_telemetry
            steps = 90 if controller["controller_sha256"] == challenger["controller_sha256"] else 120
            return record(seed, controller, steps=steps, events=1)

    ledger = module.SearchLedger(Runtime(), tmp_path)
    selection = module.run_search(ledger, "tea", bank(), champion, challenger)
    assert ledger.used() == selection["search_scientific_rollouts"] == 25
    assert selection["challenger_discovery"]["summary"]["episodes"] == 5
    assert selection["champion_paired"]["summary"]["episodes"] == 10
    assert selection["challenger_paired"]["summary"]["episodes"] == 10
    assert selection["selection_status"] == "transport3_boundary_promoted"
    assert selection["selected_controller_sha256"] == challenger["controller_sha256"]
    assert selection["final_bank_opened"] is False


def test_ambiguous_result_retains_accelerated_champion(tmp_path):
    champion, challenger = controllers()

    class Runtime:
        def rollout(self, controller, seed, *, record_attribution_telemetry=False):
            del record_attribution_telemetry
            steps = 117 if controller["controller_sha256"] == challenger["controller_sha256"] else 120
            return record(seed, controller, steps=steps)

    selection = module.run_search(
        module.SearchLedger(Runtime(), tmp_path), "tea", bank(), champion, challenger
    )
    assert selection["challenger_qualified"] is True
    assert selection["challenger_preferred"] is False
    assert selection["selection_status"] == "accelerated_champion_retained"
    assert selection["selected_controller_sha256"] == champion["controller_sha256"]


def test_clear_failures_are_required_before_native_fallback(tmp_path):
    champion, challenger = controllers()

    class Runtime:
        def rollout(self, controller, seed, *, record_attribution_telemetry=False):
            del record_attribution_telemetry
            success = int(seed) not in {10, 11}
            if controller["controller_sha256"] == challenger["controller_sha256"]:
                success = success and int(seed) % 3 != 0
            return record(seed, controller, success=success, steps=100)

    selection = module.run_search(
        module.SearchLedger(Runtime(), tmp_path), "tea", bank(), champion, challenger
    )
    assert selection["champion_paired"]["summary"]["successes"] == 8
    assert selection["challenger_qualified"] is False
    assert selection["selection_status"] == "accelerated_champion_retained"


def test_terminal_gate_is_current_observation_latched_and_transport_only():
    _, challenger = controllers()
    gate = module.TerminalApproachGate(challenger["gate"])
    far = {
        "effector_position_left": np.array([-0.3, 0.5, 0.3]),
        "effector_position_right": np.array([0.2, 0.5, 0.3]),
    }
    near = {
        "effector_position_left": np.array([-0.3, 0.5, 0.3]),
        "effector_position_right": np.array([-0.08, 0.58, 0.3]),
    }
    assert gate.choose(3.0, module.TRANSPORT_INDEX, far, 10) == 3.0
    assert gate.choose(3.0, module.TRANSPORT_INDEX, near, 11) == 2.0
    assert gate.latched is True
    assert gate.choose(3.0, module.TRANSPORT_INDEX, far, 12) == 2.0
    assert gate.choose(2.0, module.PHASES.index("interaction"), far, 13) == 2.0
    assert gate.latched is False
    assert gate.events == [
        {
            "physics_step": 11,
            "phase": "transport",
            "event": "terminal_approach_entry",
            "metric": "right_effector_to_fixed_target_xy",
            "observed_value": pytest.approx(np.sqrt(0.0008)),
            "threshold_lte": 0.12,
            "base_speed": 3.0,
            "effective_speed": 2.0,
        }
    ]


def test_registered_controllers_use_3x_transport_without_outcome_leakage():
    path = module.REPO_ROOT / "experiments/act_transport3_boundary_v38/CONTROLLERS.json"
    for task in module.TASKS:
        champion, challenger, _ = module.load_controllers(path, task)
        assert champion["type"] == "static_phase_schedule"
        assert challenger["schedule"][module.TRANSPORT_INDEX] == 3.0
        assert challenger["gate"]["current_observation_only"] is True
        assert challenger["gate"]["reward_visible"] is False
        assert challenger["gate"]["object_state_visible"] is False
        assert challenger["gate"]["future_or_terminal_signal_visible"] is False


def test_banks_are_fresh_exact_and_disjoint():
    path = module.REPO_ROOT / "experiments/act_transport3_boundary_v38/BANKS.json"
    banks = json.loads(path.read_text())
    module.validate_banks(banks)
    all_seeds = []
    for spec in banks["tasks"].values():
        assert len(spec["challenger_discovery"]) == 5
        assert len(spec["paired"]) == 10
        assert len(spec["final"]) == 50
        task_seeds = spec["challenger_discovery"] + spec["paired"] + spec["final"]
        assert len(task_seeds) == len(set(task_seeds)) == 65
        assert min(task_seeds) >= 380000000
        all_seeds.extend(task_seeds)
    assert len(all_seeds) == len(set(all_seeds)) == 195
