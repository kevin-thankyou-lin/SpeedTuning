"""Sample-efficient VLM target proposals for phase-speed frontiers.

The VLM may nominate a complete schedule from existing matched telemetry.  A
nomination is an acquisition prior, not reliability evidence: it first sees the
same three discovery poses, then a surviving schedule enters the unchanged
5->10->20 gate on fresh seeds.  A failed discovery screen permits only one
causally attributed, adjacent-rung backoff.
"""

from __future__ import annotations

import hashlib
import json
import statistics

from one_reset_phase_schedule import (
    ALLOWED_SPEEDS,
    PHASES,
    estimate_phase_workload,
    validate_schedule,
)
from scripts.staged_vlm_frontier import successful


DISCOVERY_POSES = 3


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _mean_phase_workloads(rollouts: list[dict]) -> dict[str, float]:
    successes = [item for item in rollouts if successful(item)]
    if not successes:
        raise ValueError("target proposal requires successful incumbent telemetry")
    return {
        phase: statistics.fmean(
            estimate_phase_workload(item)[phase] for item in successes
        )
        for phase in PHASES
    }


def freeze_target_proposal(
    incumbent_schedule,
    target_schedule,
    incumbent_rollouts: list[dict],
    *,
    vlm_model_identity: str,
    prompt_sha256: str,
    evidence_sha256: str,
    phase_safe_probabilities: dict[str, float],
    phase_evidence: dict[str, str],
) -> dict:
    """Freeze one full VLM target before any target rollout is opened.

    Full-vector nomination is allowed to save search rollouts.  It does not
    establish that simultaneous phase changes are causal or reliable.
    """

    incumbent = validate_schedule(incumbent_schedule)
    target = validate_schedule(target_schedule)
    if incumbent == target:
        raise ValueError("VLM target must differ from the incumbent")
    if not str(vlm_model_identity).strip():
        raise ValueError("VLM model identity is required")
    for label, value in (
        ("prompt_sha256", prompt_sha256),
        ("evidence_sha256", evidence_sha256),
    ):
        if len(str(value)) != 64:
            raise ValueError(f"{label} must be a SHA-256 hex digest")

    changed = [
        phase
        for phase, old, new in zip(PHASES, incumbent, target)
        if old != new
    ]
    probabilities = {}
    evidence = {}
    for phase in changed:
        if phase not in phase_safe_probabilities:
            raise ValueError(f"missing VLM safe probability for {phase}")
        probability = float(phase_safe_probabilities[phase])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid VLM safe probability for {phase}")
        if not str(phase_evidence.get(phase, "")).strip():
            raise ValueError(f"missing causal VLM evidence for {phase}")
        probabilities[phase] = probability
        evidence[phase] = str(phase_evidence[phase])

    workloads = _mean_phase_workloads(incumbent_rollouts)
    incumbent_steps = sum(
        workloads[phase] / speed
        for phase, speed in zip(PHASES, incumbent)
    )
    target_steps = sum(
        workloads[phase] / speed
        for phase, speed in zip(PHASES, target)
    )
    saved = incumbent_steps - target_steps
    conservative_joint_prior = min(probabilities.values())
    body = {
        "schema": "vlm-targeted-frontier-proposal-v1",
        "incumbent_schedule": list(incumbent),
        "target_schedule": list(target),
        "changed_phases": changed,
        "vlm_model_identity": str(vlm_model_identity),
        "prompt_sha256": str(prompt_sha256),
        "evidence_sha256": str(evidence_sha256),
        "phase_safe_probabilities": probabilities,
        "phase_evidence": evidence,
        "successful_incumbent_rollouts": sum(
            successful(item) for item in incumbent_rollouts
        ),
        "mean_native_equivalent_phase_workload": workloads,
        "predicted_incumbent_steps": incumbent_steps,
        "predicted_target_steps": target_steps,
        "predicted_steps_saved": saved,
        "predicted_relative_speedup": (
            None if target_steps <= 0 else incumbent_steps / target_steps
        ),
        "conservative_joint_safe_prior": conservative_joint_prior,
        "expected_steps_saved": conservative_joint_prior * saved,
        "discovery_pose_count": DISCOVERY_POSES,
        "next_gate_if_discovery_perfect": "registered_fresh_seed_5_to_10_to_20",
        "qualification": "untested_acquisition_prior_only",
    }
    return {**body, "proposal_sha256": _digest(body)}


def discovery_decision(target_schedule, discovery_rollouts: list[dict]) -> dict:
    """Decide a frozen target after exactly three distinct discovery poses."""

    target = validate_schedule(target_schedule)
    if len(discovery_rollouts) != DISCOVERY_POSES:
        raise ValueError("target discovery requires exactly three rollouts")
    seeds = [int(item["seed"]) for item in discovery_rollouts]
    if len(set(seeds)) != DISCOVERY_POSES:
        raise ValueError("target discovery requires three distinct poses")
    if any(validate_schedule(item["schedule"]) != target for item in discovery_rollouts):
        raise ValueError("discovery rollout schedule mismatch")
    if any(
        item.get("safety_violation") is not None
        or item.get("physics_error") is not None
        for item in discovery_rollouts
    ):
        return {
            "decision": "halt_runtime_incident",
            "successes": sum(successful(item) for item in discovery_rollouts),
            "discovery_only": True,
        }
    successes = sum(successful(item) for item in discovery_rollouts)
    return {
        "decision": (
            "promote_to_registered_5_to_10_to_20_gate"
            if successes == DISCOVERY_POSES
            else "require_one_phase_causal_backoff"
        ),
        "successes": successes,
        "discovery_only": True,
        "reliability_claim": None,
    }


def causal_backoff(target_schedule, earliest_divergence_phase: str) -> list[float]:
    """Lower only the attributed phase by one registered speed rung."""

    target = list(validate_schedule(target_schedule))
    if earliest_divergence_phase not in PHASES:
        raise ValueError("earliest divergence phase is invalid")
    index = PHASES.index(earliest_divergence_phase)
    rung = ALLOWED_SPEEDS.index(target[index])
    if rung == 0:
        raise ValueError("attributed phase is already at native speed")
    target[index] = ALLOWED_SPEEDS[rung - 1]
    return target
