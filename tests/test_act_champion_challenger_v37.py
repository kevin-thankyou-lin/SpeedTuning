import json

from scripts import run_act_champion_challenger_v37 as module

CHAMPION = [1.5, 2.5, 2.0, 2.0]
CHALLENGER = [1.5, 2.5, 2.5, 2.0]


def record(seed, schedule, *, success=True, steps=120):
    return {
        "seed": int(seed),
        "schedule": list(map(float, schedule)),
        "success": bool(success),
        "first_success_step": int(steps) if success else None,
        "physics_steps": 300,
        "safety_violation": None,
        "physics_error": None,
        "phase_decisions": [
            {"phase": phase, "physics_step": 10 * index, "speed": schedule[index]}
            for index, phase in enumerate(module.v32.PHASES)
        ],
        "attribution_telemetry": [],
    }


def incumbent():
    return {
        "champion_schedule": CHAMPION,
        "champion_schedule_sha256": module.schedule_sha256(CHAMPION),
        "challenger_schedule": CHALLENGER,
        "challenger_schedule_sha256": module.schedule_sha256(CHALLENGER),
        "proposal_phase": "transport",
    }


def bank():
    return {
        "challenger_discovery": [0, 1, 2, 3, 4],
        "paired": list(range(10, 20)),
        "final": list(range(100, 150)),
    }


def test_exact25_promotes_reliable_ten_percent_faster_challenger(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            steps = 90 if list(schedule) == CHALLENGER else 120
            return record(seed, schedule, steps=steps)

    ledger = module.SearchLedger(Runtime(), tmp_path)
    selection = module.run_search(ledger, "tea", bank(), incumbent())
    assert ledger.used() == selection["search_scientific_rollouts"] == 25
    assert selection["challenger_discovery"]["summary"]["episodes"] == 5
    assert selection["champion_paired"]["summary"]["episodes"] == 10
    assert selection["challenger_paired"]["summary"]["episodes"] == 10
    assert selection["selection_status"] == "challenger_promoted"
    assert selection["selected_schedule"] == CHALLENGER
    assert selection["final_bank_opened"] is False


def test_ambiguous_challenger_retains_accelerated_champion(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            return record(
                seed, schedule, steps=116 if list(schedule) == CHALLENGER else 120
            )

    selection = module.run_search(
        module.SearchLedger(Runtime(), tmp_path), "tea", bank(), incumbent()
    )
    assert selection["challenger_qualified"] is True
    assert selection["challenger_dominates"] is False
    assert selection["selection_status"] == "champion_retained"
    assert selection["selected_schedule"] == CHAMPION


def test_challenger_failure_does_not_force_native_when_champion_qualifies(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            is_challenger = list(schedule) == CHALLENGER
            success = not is_challenger or int(seed) % 3 != 0
            return record(
                seed, schedule, success=success, steps=90 if is_challenger else 120
            )

    selection = module.run_search(
        module.SearchLedger(Runtime(), tmp_path), "tea", bank(), incumbent()
    )
    assert selection["challenger_qualified"] is False
    assert selection["champion_qualified"] is True
    assert selection["selection_status"] == "champion_retained"
    assert selection["selected_schedule"] != [1.0] * 4


def test_neither_qualified_fails_closed_to_native(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            return record(seed, schedule, success=int(seed) % 4 != 0)

    selection = module.run_search(
        module.SearchLedger(Runtime(), tmp_path), "tea", bank(), incumbent()
    )
    assert selection["champion_qualified"] is False
    assert selection["challenger_qualified"] is False
    assert selection["selection_status"] == "native_fallback"
    assert selection["selected_schedule"] == [1.0] * 4


def test_registered_champions_are_adjacent_and_hash_pinned():
    path = module.REPO_ROOT / "experiments/act_champion_challenger_v37/CHAMPIONS.json"
    for task in module.TASKS:
        value = module.load_champion(path, task)
        assert value["champion_schedule"] != [1.0] * 4
        assert (
            sum(
                left != right
                for left, right in zip(
                    value["champion_schedule"], value["challenger_schedule"]
                )
            )
            == 1
        )


def test_banks_are_fresh_exact_and_disjoint():
    path = module.REPO_ROOT / "experiments/act_champion_challenger_v37/BANKS.json"
    banks = json.loads(path.read_text())
    module.validate_banks(banks)
    all_seeds = []
    for spec in banks["tasks"].values():
        assert len(spec["challenger_discovery"]) == 5
        assert len(spec["paired"]) == 10
        assert len(spec["final"]) == 50
        task_seeds = spec["challenger_discovery"] + spec["paired"] + spec["final"]
        assert len(task_seeds) == len(set(task_seeds)) == 65
        assert min(task_seeds) >= 370000000
        all_seeds.extend(task_seeds)
    assert len(all_seeds) == len(set(all_seeds)) == 195
