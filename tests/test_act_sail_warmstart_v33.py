import json

import numpy as np

from scripts import run_act_sail_warmstart_v33 as module


def record(seed, schedule, success=True, divergent_phase=None, steps=120):
    telemetry = []
    for index, phase in enumerate(module.PHASES):
        position = [0.01 * index, 0.5, 0.05]
        if divergent_phase is not None and index >= module.PHASES.index(divergent_phase):
            position[0] += 0.08
        telemetry.append(
            {
                "physics_step": 10 * (index + 1),
                "policy_time": 20.0 * (index + 1),
                "observed_phase": phase,
                "task_reward": float(index),
                "object_positions": [position],
            }
        )
    return {
        "seed": seed,
        "schedule": list(schedule),
        "success": success,
        "first_success_step": steps if success else None,
        "physics_steps": 300,
        "safety_violation": None,
        "physics_error": None,
        "phase_decisions": [
            {"phase": phase, "physics_step": 10 * index, "speed": schedule[index]}
            for index, phase in enumerate(module.PHASES)
        ],
        "attribution_telemetry": telemetry,
    }


def test_sail_prior_maps_to_accelerated_common_grid():
    artifact = {
        "artifact_payload_sha256": "a" * 64,
        "candidates": [
            {
                "id": "max",
                "maximum_speed": 3.0,
                "profile": [1.2, 1.6, 2.1, 2.4, 2.6, 2.9, 1.0, 1.1],
                "importance": [0.9, 0.8, 0.5, 0.4, 0.2, 0.1, 1.0, 0.9],
            }
        ],
    }
    prior = module.sail_phase_prior(artifact)
    assert prior["schedule"] == [1.5, 2.0, 2.5, 1.5]
    assert min(prior["schedule"]) == 1.5
    assert prior["paper_faithful_sail"] is False


def test_causal_arm_uses_exactly_25_and_one_phase_updates(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            schedule = list(schedule)
            if schedule[1] == 3.0:
                return record(seed, schedule, False, "grasp_lift")
            return record(seed, schedule, True, steps=200 if schedule == [1.0] * 4 else 120)

    prior = {
        "schedule": [2.0, 2.5, 2.0, 2.0],
        "phase_importance": [0.2, 0.3, 0.8, 0.9],
    }
    ledger = module.v32.Ledger(Runtime(), tmp_path, [0, 1, 2], [10, 11, 12, 13, 14])
    selection = module.run_causal_search(ledger, "pick", prior)
    schedules = [item["schedule"] for item in selection["discovery_reports"]]
    assert schedules[0] == [1.0] * 4
    assert schedules[1] == prior["schedule"]
    assert len(schedules) == len({tuple(item) for item in schedules}) == 5
    assert all(sum(a != b for a, b in zip(left, right)) <= 1 for left, right in zip(schedules[1:], schedules[2:]))
    assert selection["search_scientific_rollouts"] == ledger.used() == 25
    assert selection["final_bank_opened"] is False


def test_tabular_q_prior_prefers_sail_schedule_before_updates():
    schedule = [1.5, 2.0, 2.5, 3.0]
    q_values, visits = module.tabular_rebuild([], schedule)
    assert module.greedy_schedule(q_values) == schedule
    assert np.count_nonzero(visits) == 0


def test_v33_search_banks_are_exact_and_disjoint():
    banks = json.loads((module.REPO_ROOT / "experiments/act_sail_warmstart_v33/BANKS.json").read_text())
    seeds = []
    for task in banks["tasks"].values():
        causal = task["sail_causal"]
        assert len(causal["discovery"]) == 3
        assert len(causal["confirmation"]) == 5
        assert len(task["sail_tabular"]) == len(set(task["sail_tabular"])) == 25
        seeds.extend(causal["discovery"] + causal["confirmation"] + task["sail_tabular"])
    assert len(seeds) == len(set(seeds)) == 99
