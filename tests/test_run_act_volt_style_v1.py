import json

import pytest

from scripts import run_act_volt_style_v1 as module


def test_binary_schedule_has_one_shared_fast_and_slow_value():
    assert module.binary_schedule(fast_speed=3.0, slow_speed=1.5) == [3.0, 1.5, 3.0, 1.5]
    with pytest.raises(ValueError, match="slow speed cannot exceed"):
        module.binary_schedule(fast_speed=1.5, slow_speed=2.0)


def test_full_search_promotes_only_the_fast_phase_class(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed):
            mean_speed = sum(schedule) / 4
            return {
                "seed": seed,
                "schedule": list(schedule),
                "success": True,
                "first_success_step": int(400 / mean_speed),
                "physics_steps": 400,
                "safety_violation": None,
                "physics_error": None,
                "phase_decisions": [
                    {"phase": "pre_grasp", "physics_step": 0, "speed": schedule[0]},
                    {"phase": "grasp_lift", "physics_step": 80, "speed": schedule[1]},
                    {"phase": "transport", "physics_step": 100, "speed": schedule[2]},
                    {"phase": "interaction", "physics_step": 120, "speed": schedule[3]},
                ],
            }

    ledger = module.RolloutLedger(Runtime(), tmp_path, list(range(15)), list(range(100, 150)))
    selection = module.run_search(ledger, "pick")

    assert selection["search_rollouts"] == 60
    assert selection["selected_schedule"] == [3.5, 3.0, 3.5, 3.0]
    assert selection["binary_candidate_replaced_uniform"]
    assert selection["binary_parameterization"]["fast_phases"] == ["pre_grasp", "transport"]
    assert selection["binary_parameterization"]["slow_phases"] == ["grasp_lift", "interaction"]


def test_shared_records_verify_runtime_schedule_and_seed(tmp_path):
    schedule = [2.0] * 4
    seeds = list(range(50))
    runtime_identity = {
        "task": "pick_and_place",
        "task_label": "pick",
        "run_manifest_sha256": "manifest",
        "policy_artifacts": {"policy": "hash"},
        "detector": {"checkpoint": "hash"},
    }
    identity = {**runtime_identity, "final_seeds": seeds}
    (tmp_path / "IDENTITY.json").write_text(json.dumps(identity))
    controller = tmp_path / "final" / "controllers" / module.schedule_sha256(schedule)
    controller.mkdir(parents=True)
    (controller / "SCHEDULE.json").write_text(json.dumps({"schedule": schedule}))
    states = controller / "states"
    states.mkdir()
    for seed in seeds:
        (states / f"{seed}.json").write_text(json.dumps({"seed": seed, "schedule": schedule}))

    values = module.shared_records(tmp_path, schedule, seeds, runtime_identity)

    assert len(values) == 50
    assert values[-1]["seed"] == 49
