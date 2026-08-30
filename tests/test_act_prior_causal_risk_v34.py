import json

import numpy as np

from scripts import run_act_prior_causal_risk_v34 as module


def record(seed, schedule, *, success=True, divergent_phase=None, steps=120):
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


def test_risk_gate_is_causal_and_releases_after_one_stable_observation():
    spec = {
        "protected_phases": ["grasp_lift", "interaction"],
        "gripper_indices": [6, 13],
        "gripper_delta_threshold": 0.01,
        "override_speed": 1.0,
    }
    gate = module.CausalRiskGate(spec)
    qpos = np.zeros(14)
    speed, _ = gate.choose(2.5, 1, qpos, 0)
    assert speed == 1.0
    speed, _ = gate.choose(2.5, 1, qpos, 1)
    assert speed == 2.5
    qpos[6] = 0.02
    speed, reasons = gate.choose(2.5, 1, qpos, 2)
    assert speed == 1.0
    assert reasons == ["observed_gripper_transition"]
    speed, _ = gate.choose(2.5, 1, qpos, 3)
    assert speed == 2.5
    assert len(gate.events) == 2


def test_combined_prior_uses_semantic_schedule_and_pinned_sail_ranking():
    prior = module.combined_prior(
        module.REPO_ROOT / "experiments/act_sail_warmstart_v33/OFFLINE_PRIORS.json",
        "pick",
    )
    assert prior["schedule"] == [2.0, 1.5, 3.0, 1.5]
    assert prior["offline_training_rollouts"] == 60
    assert prior["online_rollouts"] == 0
    assert prior["historical_speed_outcomes_used_by_runtime"] is False
    assert prior["study_design_informed_by_v33_results"] is True
    assert module.risk_gate_spec(prior)["future_action_or_terminal_signal_visible"] is False


def test_search_uses_exactly_25_and_only_one_phase_updates(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            return record(seed, list(schedule), steps=200 if list(schedule) == [1.0] * 4 else 120)

    prior = {
        "schedule": [2.0, 1.5, 3.0, 1.5],
        "phase_importance": [0.4, 1.0, 0.1, 1.0],
    }
    gate = {"controller_sha256": "a" * 64}
    ledger = module.v32.Ledger(Runtime(), tmp_path, [0, 1, 2], [10, 11, 12, 13, 14])
    selection = module.run_search(ledger, "pick", prior, gate)
    schedules = [item["schedule"] for item in selection["discovery_reports"]]
    assert schedules[:2] == [[1.0] * 4, prior["schedule"]]
    assert len(schedules) == len({tuple(item) for item in schedules}) == 5
    assert all(
        sum(left != right for left, right in zip(a, b)) <= 1
        for a, b in zip(schedules[1:], schedules[2:])
    )
    assert selection["search_scientific_rollouts"] == ledger.used() == 25
    assert selection["selected_schedule"] is not None
    assert selection["final_bank_opened"] is False


def test_exhausted_causal_phase_repairs_another_accelerated_dimension(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, record_attribution_telemetry=False):
            schedule = list(schedule)
            native = schedule == [1.0] * 4
            return record(
                seed,
                schedule,
                success=native,
                divergent_phase=None if native else "pre_grasp",
            )

    prior = {
        "schedule": [2.0, 1.5, 3.0, 1.5],
        "phase_importance": [0.4, 1.0, 0.1, 1.0],
    }
    ledger = module.v32.Ledger(Runtime(), tmp_path, [0, 1, 2], [10, 11, 12, 13, 14])
    selection = module.run_search(ledger, "tea", prior, {"controller_sha256": "a" * 64})
    schedules = [item["schedule"] for item in selection["discovery_reports"]]
    assert len(schedules) == len({tuple(item) for item in schedules}) == 5
    assert schedules[3] == [1.0, 1.5, 3.0, 1.5]
    assert sum(left != right for left, right in zip(schedules[3], schedules[4])) == 1
    assert selection["search_scientific_rollouts"] == ledger.used() == 25
    assert selection["selected_schedule"] is None


def test_banks_are_exact_fresh_and_disjoint():
    banks = json.loads(
        (module.REPO_ROOT / "experiments/act_prior_causal_risk_v34/BANKS.json").read_text()
    )
    all_seeds = []
    for task in banks["tasks"].values():
        assert len(task["discovery"]) == 3
        assert len(task["confirmation"]) == 5
        assert len(task["final"]) == 50
        task_seeds = task["discovery"] + task["confirmation"] + task["final"]
        assert len(task_seeds) == len(set(task_seeds)) == 58
        assert min(task_seeds) >= 340000000
        all_seeds.extend(task_seeds)
    assert len(all_seeds) == len(set(all_seeds)) == 174
