import json

from act_speed_benchmark import canonical_sha256
from scripts import run_act_sail_warmstart_v33 as runner
from scripts import train_act_sail_priors_v33 as module


def synthetic_record(seed: int):
    telemetry = []
    qpos = [0.0] * 14
    for phase_index, phase in enumerate(module.PHASES):
        for step in range(12):
            qpos = list(qpos)
            if phase in {"pre_grasp", "transport"}:
                qpos[0] += 0.002
            else:
                qpos[1] += 0.02 if step % 2 == 0 else -0.02
            if phase == "grasp_lift" and step == 5:
                qpos[6] += 0.03
                qpos[13] += 0.03
            telemetry.append(
                {
                    "physics_step": phase_index * 12 + step + 1,
                    "policy_time": float(phase_index * 12 + step + 1),
                    "observed_phase": phase,
                    "task_reward": float(phase_index),
                    "robot_qpos": qpos,
                    "object_positions": [[0.0, 0.5, 0.05]],
                }
            )
    return {
        "seed": seed,
        "schedule": module.NATIVE_SCHEDULE,
        "success": True,
        "physics_error": None,
        "safety_violation": None,
        "attribution_telemetry": telemetry,
    }


def test_new_prior_training_is_deterministic_and_contact_conservative():
    records = [synthetic_record(seed) for seed in range(20)]
    first = module.train_phase_precision_prior(records)
    second = module.train_phase_precision_prior(records)
    assert first == second
    assert first["training_rollouts"] == 20
    assert first["paper_faithful_sail"] is False
    assert min(first["schedule"]) >= 1.5
    grasp = module.PHASES.index("grasp_lift")
    transport = module.PHASES.index("transport")
    assert first["phase_importance"][grasp] > first["phase_importance"][transport]
    assert first["schedule"][grasp] <= first["schedule"][transport]


def test_prior_training_banks_are_exact_and_disjoint_from_repo_banks():
    path = module.REPO_ROOT / "experiments/act_sail_warmstart_v33/PRIOR_BANKS.json"
    task_seeds = module.validate_prior_banks(path)
    assert set(task_seeds) == set(module.TASKS)
    assert all(len(seeds) == 20 for seeds in task_seeds.values())
    assert len(set().union(*(set(seeds) for seeds in task_seeds.values()))) == 60


def test_runner_accepts_only_hash_valid_new_prior_bundle(tmp_path):
    phase_prior = {
        "schema": "act-new-sail-inspired-phase-precision-prior-v33",
        "prior_kind": "new_sail_inspired_native_precision_head",
        "paper_faithful_sail": False,
        "schedule": [1.5, 2.0, 2.5, 1.5],
        "phase_importance": [0.6, 0.4, 0.2, 0.7],
    }
    phase_prior["prior_payload_sha256"] = canonical_sha256(phase_prior)
    task_result = {
        "schema": "act-new-sail-prior-task-result-v33",
        "task_label": "pick",
        "phase_prior": phase_prior,
    }
    task_result["result_payload_sha256"] = canonical_sha256(task_result)
    bundle = {
        "schema": "act-new-sail-inspired-offline-priors-v33",
        "offline_training_rollouts": 60,
        "online_search_rollouts": 0,
        "final_bank_opened": False,
        "tasks": {"pick": task_result},
    }
    bundle["payload_sha256"] = canonical_sha256(bundle)
    path = tmp_path / "OFFLINE_PRIORS.json"
    path.write_text(json.dumps(bundle))
    loaded, prior = runner.load_trained_prior(path, "pick")
    assert loaded == bundle
    assert prior["schedule"] == [1.5, 2.0, 2.5, 1.5]
