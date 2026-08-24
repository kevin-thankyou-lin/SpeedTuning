import pytest

from scripts.vlm_targeted_frontier import (
    causal_backoff,
    discovery_decision,
    freeze_target_proposal,
)


PHASES = ("pre_grasp", "grasp_lift", "transport", "interaction")


def rollout(seed, schedule=(2.5, 1.5, 2.5, 2.5), *, success=True):
    steps = 120
    return {
        "seed": seed,
        "schedule": list(schedule),
        "success": success,
        "physics_steps": steps,
        "first_success_step": steps if success else None,
        "safety_violation": None,
        "physics_error": None,
        "phase_decisions": [
            {"phase": phase, "physics_step": index * 30, "speed": schedule[index]}
            for index, phase in enumerate(PHASES)
        ],
    }


def test_vlm_can_nominate_aggressive_full_target_without_grid_search():
    proposal = freeze_target_proposal(
        [2.5, 1.5, 2.5, 2.5],
        [3, 1.5, 4, 4],
        [rollout(seed) for seed in range(20)],
        vlm_model_identity="vlm-test",
        prompt_sha256="a" * 64,
        evidence_sha256="b" * 64,
        phase_safe_probabilities={
            "pre_grasp": 0.9,
            "transport": 0.8,
            "interaction": 0.85,
        },
        phase_evidence={
            "pre_grasp": "large free-space margin",
            "transport": "stable retained grasp",
            "interaction": "terminal motion remains inside envelope",
        },
    )

    assert proposal["target_schedule"] == [3.0, 1.5, 4.0, 4.0]
    assert proposal["changed_phases"] == ["pre_grasp", "transport", "interaction"]
    assert proposal["predicted_steps_saved"] > 0
    assert proposal["qualification"] == "untested_acquisition_prior_only"
    assert len(proposal["proposal_sha256"]) == 64


def test_three_pose_discovery_is_not_a_reliability_claim():
    schedule = [3, 1.5, 4, 4]
    result = discovery_decision(
        schedule, [rollout(seed, schedule) for seed in (10, 11, 12)]
    )

    assert result["decision"] == "promote_to_registered_5_to_10_to_20_gate"
    assert result["successes"] == 3
    assert result["reliability_claim"] is None


def test_discovery_failure_repairs_only_earliest_divergent_phase():
    schedule = [3, 1.5, 4, 4]
    result = discovery_decision(
        schedule,
        [rollout(10, schedule), rollout(11, schedule, success=False), rollout(12, schedule)],
    )

    assert result["decision"] == "require_one_phase_causal_backoff"
    assert causal_backoff(schedule, "interaction") == [3.0, 1.5, 4.0, 3.5]


def test_discovery_requires_exactly_three_distinct_poses():
    schedule = [3, 1.5, 4, 4]
    with pytest.raises(ValueError, match="three distinct poses"):
        discovery_decision(schedule, [rollout(10, schedule)] * 3)


def test_runtime_incident_halts_instead_of_becoming_policy_failure():
    schedule = [3, 1.5, 4, 4]
    unsafe = rollout(11, schedule)
    unsafe["safety_violation"] = "workspace"
    result = discovery_decision(
        schedule, [rollout(10, schedule), unsafe, rollout(12, schedule)]
    )

    assert result["decision"] == "halt_runtime_incident"
