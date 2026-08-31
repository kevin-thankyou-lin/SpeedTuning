import json

from scripts import run_act_fresh_transport25_v39 as module


def record(seed, controller, *, success=True, steps=120):
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
    }


def bank():
    return {
        "challenger_discovery": [0, 1, 2, 3, 4],
        "paired": list(range(10, 20)),
        "final": list(range(100, 150)),
    }


def controllers():
    return (
        module.static_controller(module.UNIFORM_SCHEDULE),
        module.static_controller(module.TRANSPORT_SCHEDULE),
    )


def test_exact25_promotes_reliable_transport_challenger(tmp_path):
    uniform, challenger = controllers()

    class Runtime:
        def rollout(self, controller, seed, *, record_attribution_telemetry=False):
            del record_attribution_telemetry
            steps = 105 if controller["controller_sha256"] == challenger["controller_sha256"] else 120
            return record(seed, controller, steps=steps)

    ledger = module.SearchLedger(Runtime(), tmp_path)
    selection = module.run_search(ledger, "tea", bank(), uniform, challenger)
    assert ledger.used() == selection["search_scientific_rollouts"] == 25
    assert selection["transport_2p5_discovery"]["summary"]["episodes"] == 5
    assert selection["uniform_2x_paired"]["summary"]["episodes"] == 10
    assert selection["transport_2p5_paired"]["summary"]["episodes"] == 10
    assert selection["selection_status"] == "transport_2p5_promoted"
    assert selection["selected_controller_sha256"] == challenger["controller_sha256"]
    assert selection["historical_speed_outcomes_used_for_initialization"] is False
    assert selection["final_bank_opened"] is False


def test_ambiguous_result_retains_uniform_anchor(tmp_path):
    uniform, challenger = controllers()

    class Runtime:
        def rollout(self, controller, seed, *, record_attribution_telemetry=False):
            del record_attribution_telemetry
            steps = 117 if controller["controller_sha256"] == challenger["controller_sha256"] else 120
            return record(seed, controller, steps=steps)

    selection = module.run_search(
        module.SearchLedger(Runtime(), tmp_path), "pick", bank(), uniform, challenger
    )
    assert selection["challenger_qualified"] is True
    assert selection["challenger_preferred"] is False
    assert selection["selection_status"] == "uniform_2x_retained"
    assert selection["selected_controller_sha256"] == uniform["controller_sha256"]


def test_qualified_challenger_replaces_clearly_failed_uniform(tmp_path):
    uniform, challenger = controllers()

    class Runtime:
        def rollout(self, controller, seed, *, record_attribution_telemetry=False):
            del record_attribution_telemetry
            success = True
            if controller["controller_sha256"] == uniform["controller_sha256"]:
                success = int(seed) not in {10, 11, 12}
            return record(seed, controller, success=success, steps=110)

    selection = module.run_search(
        module.SearchLedger(Runtime(), tmp_path), "insertion", bank(), uniform, challenger
    )
    assert selection["uniform_2x_paired"]["summary"]["successes"] == 7
    assert selection["challenger_qualified"] is True
    assert selection["selection_status"] == "transport_2p5_promoted"


def test_no_historical_initialization_and_detector_only_controllers():
    path = module.REPO_ROOT / "experiments/act_fresh_transport25_v39/CONTROLLERS.json"
    payload = json.loads(path.read_text())
    uniform, challenger, _ = module.load_controllers(path)
    assert payload["historical_speed_outcomes_used_for_initialization"] is False
    assert payload["secondary_speed_override"] is None
    assert payload["phase_speed_selector"] == "learned_phase_detector_argmax"
    assert uniform["schedule"] == [2.0, 2.0, 2.0, 2.0]
    assert challenger["schedule"] == [2.0, 2.0, 2.5, 2.0]
    assert uniform["type"] == challenger["type"] == "static_phase_schedule"
    assert "gate" not in uniform and "gate" not in challenger


def test_banks_are_fresh_exact_and_disjoint():
    path = module.REPO_ROOT / "experiments/act_fresh_transport25_v39/BANKS.json"
    banks = json.loads(path.read_text())
    module.validate_banks(banks)
    all_seeds = []
    for spec in banks["tasks"].values():
        assert len(spec["challenger_discovery"]) == 5
        assert len(spec["paired"]) == 10
        assert len(spec["final"]) == 50
        task_seeds = spec["challenger_discovery"] + spec["paired"] + spec["final"]
        assert len(task_seeds) == len(set(task_seeds)) == 65
        assert min(task_seeds) >= 390000000
        all_seeds.extend(task_seeds)
    assert len(all_seeds) == len(set(all_seeds)) == 195


def test_final_methods_include_both_candidates_and_cacheable_selection():
    assert module.FINAL_METHODS == (
        "native_1x",
        "uniform_2x",
        "transport_2p5",
        "selected",
    )
