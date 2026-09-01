import json

from scripts import run_act_phase_bandit25_v40 as module


def record(seed, controller, *, success=True, steps=120, boundaries=None):
    boundaries = boundaries or [0, 10, 30, 90]
    decisions = [
        {"phase": phase, "physics_step": step, "speed": controller["schedule"][index]}
        for index, (phase, step) in enumerate(zip(module.PHASES, boundaries))
    ]
    return {
        "seed": int(seed),
        "schedule": list(controller["schedule"]),
        "controller_sha256": controller["controller_sha256"],
        "success": bool(success),
        "first_success_step": int(steps) if success else None,
        "physics_steps": 300,
        "safety_violation": None,
        "physics_error": None,
        "phase_decisions": decisions,
        "attribution_telemetry": [],
    }


def bank():
    return {
        "diagnostic": [0, 1, 2, 3, 4],
        "paired": list(range(10, 20)),
        "final": list(range(100, 150)),
    }


def test_workload_bandit_selects_largest_predicted_saving():
    uniform = module.static_controller(module.UNIFORM_SCHEDULE)
    diagnostics = [record(seed, uniform, boundaries=[0, 10, 30, 90]) for seed in range(5)]
    challenger, proposal = module.choose_phase_bump(diagnostics, uniform)
    assert proposal["selected_phase"] == "transport"
    assert challenger["schedule"] == [2.0, 2.0, 3.0, 2.0]
    assert proposal["predicted_steps_saved"]["transport"] > proposal["predicted_steps_saved"]["interaction"]
    assert proposal["diagnostic_successful_incident_free_episodes"] == 5


def test_exact25_promotes_reliable_phase_bump(tmp_path):
    uniform = module.static_controller(module.UNIFORM_SCHEDULE)

    class Runtime:
        def rollout(self, controller, seed, *, record_attribution_telemetry=False):
            del record_attribution_telemetry
            bumped = 3.0 in controller["schedule"]
            return record(seed, controller, steps=100 if bumped else 120)

    ledger = module.SearchLedger(Runtime(), tmp_path / "ledger")
    selection = module.run_search(
        ledger, "pick", bank(), uniform, tmp_path / "PROPOSAL.json"
    )
    assert ledger.used() == selection["search_scientific_rollouts"] == 25
    assert selection["uniform_2x_diagnostic"]["summary"]["episodes"] == 5
    assert selection["uniform_2x_paired"]["summary"]["episodes"] == 10
    assert selection["phase_bump_3x_paired"]["summary"]["episodes"] == 10
    assert selection["selection_status"] == "phase_bump_3x_promoted"
    assert selection["selected_controller"]["schedule"] == [2.0, 2.0, 3.0, 2.0]
    assert selection["proposal_sha256"] == module.file_sha256(tmp_path / "PROPOSAL.json")
    assert selection["historical_speed_outcomes_used_for_initialization"] is False
    assert selection["final_bank_opened"] is False


def test_ambiguous_result_retains_uniform(tmp_path):
    uniform = module.static_controller(module.UNIFORM_SCHEDULE)

    class Runtime:
        def rollout(self, controller, seed, *, record_attribution_telemetry=False):
            del record_attribution_telemetry
            bumped = 3.0 in controller["schedule"]
            return record(seed, controller, steps=117 if bumped else 120)

    selection = module.run_search(
        module.SearchLedger(Runtime(), tmp_path / "ledger"),
        "tea", bank(), uniform, tmp_path / "PROPOSAL.json",
    )
    assert selection["challenger_qualified"] is True
    assert selection["challenger_preferred"] is False
    assert selection["selection_status"] == "uniform_2x_retained"
    assert selection["selected_controller_sha256"] == uniform["controller_sha256"]


def test_no_success_uses_preregistered_phase_tie_break():
    uniform = module.static_controller(module.UNIFORM_SCHEDULE)
    diagnostics = [record(seed, uniform, success=False) for seed in range(5)]
    challenger, proposal = module.choose_phase_bump(diagnostics, uniform)
    assert proposal["diagnostic_successful_incident_free_episodes"] == 0
    assert proposal["selected_phase"] == module.PHASES[0]
    assert challenger["schedule"] == [3.0, 2.0, 2.0, 2.0]


def test_no_historical_initialization_and_detector_only_controller():
    path = module.REPO_ROOT / "experiments/act_phase_bandit25_v40/CONTROLLERS.json"
    payload = json.loads(path.read_text())
    uniform, _ = module.load_controllers(path)
    assert payload["historical_speed_outcomes_used_for_initialization"] is False
    assert payload["secondary_speed_override"] is None
    assert payload["phase_speed_selector"] == "learned_phase_detector_argmax"
    assert payload["proposal_rule"] == "max_mean_predicted_steps_saved"
    assert uniform["schedule"] == [2.0, 2.0, 2.0, 2.0]
    assert uniform["type"] == "static_phase_schedule"
    assert "gate" not in uniform


def test_banks_are_fresh_exact_and_disjoint():
    path = module.REPO_ROOT / "experiments/act_phase_bandit25_v40/BANKS.json"
    banks = json.loads(path.read_text())
    module.validate_banks(banks)
    all_seeds = []
    for spec in banks["tasks"].values():
        assert len(spec["diagnostic"]) == 5
        assert len(spec["paired"]) == 10
        assert len(spec["final"]) == 50
        task_seeds = spec["diagnostic"] + spec["paired"] + spec["final"]
        assert len(task_seeds) == len(set(task_seeds)) == 65
        assert min(task_seeds) >= 400000000
        all_seeds.extend(task_seeds)
    assert len(all_seeds) == len(set(all_seeds)) == 195


def test_final_methods_include_proposed_and_cacheable_selection():
    assert module.FINAL_METHODS == (
        "native_1x", "uniform_2x", "phase_bump_3x", "selected"
    )
