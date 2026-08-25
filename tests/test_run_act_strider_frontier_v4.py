from scripts import run_act_strider_frontier_v4 as module


def record(seed, schedule, success, steps=100):
    return {
        "seed": seed,
        "schedule": list(schedule),
        "success": success,
        "first_success_step": steps if success else None,
        "physics_steps": 200,
        "safety_violation": None,
        "physics_error": None,
        "phase_decisions": [
            {"phase": phase, "physics_step": 10 * index, "speed": schedule[index]}
            for index, phase in enumerate(module.PHASES)
        ],
        "attribution_telemetry": [],
    }


def report(schedule, successes, throughput, qualified=True):
    return {
        "schedule": list(schedule),
        "qualified": qualified,
        "summary": {
            "successes": successes,
            "achieved_throughput_per_step": throughput,
            "safety_violations": 0,
            "physics_errors": 0,
        },
    }


def test_strict_gate_rejects_eighteen_of_twenty():
    ledger = object.__new__(module.RolloutLedger)
    assert ledger.gate_decision(
        {
            "episodes": 20,
            "successes": 18,
            "safety_violations": 0,
            "physics_errors": 0,
        }
    ) == "reject_reliability"
    assert ledger.gate_decision(
        {
            "episodes": 20,
            "successes": 19,
            "safety_violations": 0,
            "physics_errors": 0,
        }
    ) == "qualified"


def test_adaptive_requires_qualified_uniform_lower_bound():
    candidate = report([2.5, 2.0, 2.5, 2.5], 20, 0.007)
    assert not module.adaptive_replaces_uniform(candidate, None)


def test_slower_bend_requires_reliability_lift_and_material_throughput_gain():
    incumbent = report([2.0] * 4, 19, 0.004)
    same_reliability = report([2.0, 2.0, 2.0, 1.5], 19, 0.005)
    lifted = report([2.0, 2.0, 2.0, 1.5], 20, 0.0042)
    assert not module.adaptive_replaces_uniform(same_reliability, incumbent)
    assert module.adaptive_replaces_uniform(lifted, incumbent)


def test_search_falls_back_to_native_without_qualified_uniform(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            schedule = list(schedule)
            index = seed % 20
            if schedule == [2.0] * 4:
                return record(seed, schedule, index < 18, 150)
            if schedule == [1.5] * 4:
                return record(seed, schedule, index < 18, 170)
            if schedule == [2.0, 2.0, 2.0, 1.5]:
                return record(seed, schedule, True, 140)
            raise AssertionError(f"unexpected schedule: {schedule}")

    ledger = module.RolloutLedger(
        Runtime(),
        tmp_path,
        list(range(20)),
        list(range(100, 150)),
        record_search_telemetry=True,
    )
    selection = module.run_search(ledger, "insertion")
    assert selection["selected_role"] == "native_fallback"
    assert selection["selected_schedule"] == [1.0] * 4
    assert selection["uniform_incumbent_sha256"] is None
    assert selection["search_rollouts"] == 60
