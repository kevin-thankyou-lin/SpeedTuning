from scripts import run_act_strider_frontier_v3 as module


def telemetry(seed, divergent_phase=None):
    rows = []
    for index, phase in enumerate(module.PHASES):
        position = [0.01 * index, 0.5, 0.05 + 0.02 * index]
        if divergent_phase is not None and index >= module.PHASES.index(
            divergent_phase
        ):
            position[0] += 0.08
        rows.append(
            {
                "physics_step": 10 * (index + 1),
                "policy_time": 20.0 * (index + 1),
                "observed_phase": phase,
                "task_reward": float(index),
                "object_positions": [position],
            }
        )
    return rows


def record(seed, schedule, *, success, divergent_phase=None):
    return {
        "seed": seed,
        "schedule": list(schedule),
        "success": success,
        "first_success_step": 100 if success else None,
        "physics_steps": 200,
        "safety_violation": None,
        "physics_error": None,
        "phase_decisions": [
            {"phase": phase, "physics_step": 10 * index, "speed": schedule[index]}
            for index, phase in enumerate(module.PHASES)
        ],
        "attribution_telemetry": telemetry(seed, divergent_phase),
    }


def test_paired_attribution_finds_earliest_physical_divergence():
    reference = record(7, [2.0] * 4, success=True)
    candidate = record(
        7,
        [2.5] * 4,
        success=False,
        divergent_phase="grasp_lift",
    )

    phase, evidence = module.paired_failure_phase(
        [candidate], [reference], "interaction"
    )

    assert phase == "grasp_lift"
    assert evidence["method"] == "same_seed_phase_exit_physical_divergence"
    assert evidence["paired_counterexamples"][0]["cause"] == (
        "object_position_left_matched_reference_envelope"
    )


def test_attribution_fails_closed_without_successful_matched_reference():
    candidate = record(7, [2.5] * 4, success=False, divergent_phase="grasp_lift")
    failed_reference = record(7, [2.0] * 4, success=False)

    phase, evidence = module.paired_failure_phase(
        [candidate], [failed_reference], "grasp_lift"
    )

    assert phase == "grasp_lift"
    assert evidence["method"] == "preregistered_semantic_fallback"
    assert evidence["unmatched_failed_seeds"] == [7]


def test_search_uses_fourth_block_for_second_causal_repair(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            schedule = list(schedule)
            index = seed % 15
            if schedule == [2.0] * 4:
                success = index < 14
                divergence = None
                steps = 150
            elif schedule == [2.5] * 4:
                success = index < 12
                divergence = None if success else "grasp_lift"
                steps = 120
            elif schedule == [2.5, 2.0, 2.5, 2.5]:
                success = index < 13
                divergence = None if success else "interaction"
                steps = 135
            elif schedule == [2.5, 2.0, 2.5, 2.0]:
                success = index < 14
                divergence = None
                steps = 130
            else:
                raise AssertionError(f"unexpected schedule: {schedule}")
            value = record(seed, schedule, success=success, divergent_phase=divergence)
            value["first_success_step"] = steps if success else None
            return value

    ledger = module.RolloutLedger(
        Runtime(),
        tmp_path,
        list(range(15)),
        list(range(100, 150)),
        record_search_telemetry=True,
    )

    selection = module.run_search(ledger, "pick")

    assert selection["search_rollouts"] == 60
    assert selection["unused_budget"] == 0
    assert selection["selected_schedule"] == [2.5, 2.0, 2.5, 2.0]
    assert [item["phase"] for item in selection["attribution_receipts"]] == [
        "grasp_lift",
        "interaction",
    ]
    assert selection["uniform_incumbent_sha256"] == module.schedule_sha256([2.0] * 4)


def test_bang_for_buck_freezes_repaired_phases():
    schedule = [2.5, 2.0, 2.5, 2.0]
    values = [record(seed, schedule, success=True) for seed in range(3)]

    phase, evidence = module.highest_bang_for_buck_phase(
        values, schedule, {"grasp_lift", "interaction"}
    )

    assert phase in {"pre_grasp", "transport"}
    assert phase not in {"grasp_lift", "interaction"}
    assert evidence["method"] == "preregistered_bang_for_buck"
