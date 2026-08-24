#!/usr/bin/env python3
"""Run the bounded qualitative-VLM Pick candidate screen and selected gate."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import sha256
from one_reset_phase_schedule import rollout_metric_steps, validate_schedule
from scripts.act_vlm_frontier_server import (
    ACTFrontierRuntime,
    ACTThreeSceneServer,
    git_head,
)
from scripts.staged_vlm_frontier import StagedFrontier, successful, write_json
from scripts.three_scene_server import comma_ints, schedule_hash


QUALITATIVE_SPEEDS = {
    "protected": 1.5,
    "moderate": 2.5,
    "fast": 3.0,
    "ceiling": 4.0,
}

CANDIDATES = (
    {
        "id": "uniform_2x",
        "origin": "standard_uniform_comparator",
        "schedule": [2.0, 2.0, 2.0, 2.0],
    },
    {
        "id": "uniform_2p5x",
        "origin": "standard_uniform_comparator",
        "schedule": [2.5, 2.5, 2.5, 2.5],
    },
    {
        "id": "prior_incumbent",
        "origin": "known_posthoc_search_incumbent",
        "schedule": [2.5, 1.5, 2.5, 2.5],
    },
    {
        "id": "coarse_fast_protected_fast_fast",
        "origin": "coarse_label_ablation_not_independent_vlm_output",
        "labels": ["fast", "protected", "fast", "fast"],
        "schedule": [3.0, 1.5, 3.0, 3.0],
    },
    {
        "id": "user_hypothesis_3_1p5_4_4",
        "origin": "user_disclosed_hypothesis_not_independent_vlm_output",
        "labels": ["fast", "protected", "ceiling", "ceiling"],
        "schedule": [3.0, 1.5, 4.0, 4.0],
    },
)


class DirectTargetGate(StagedFrontier):
    """Reuse the receipt-bearing staged gate for one frozen discovery winner."""

    def _expected_kind(self, schedule, kind: str, phase: str | None) -> None:
        if not self.state["candidate_order"] and kind == "targeted_discovery_winner":
            validate_schedule(schedule)
            if phase is not None:
                raise ValueError("direct target gate does not accept a phase mutation")
            return
        super()._expected_kind(schedule, kind, phase)


def select_discovery_candidate(state: dict, candidates=CANDIDATES) -> dict | None:
    """Select the fastest clean 3/3 candidate under a frozen tie-break."""

    eligible = []
    for order, definition in enumerate(candidates):
        identifier = schedule_hash(definition["schedule"])
        item = state["candidates"].get(identifier)
        if item is None or len(item["discovery"]) != 3:
            raise RuntimeError(f"missing discovery result for {definition['id']}")
        rollouts = item["discovery"]
        if not all(successful(result) for result in rollouts):
            continue
        if any(result.get("physics_error") is not None for result in rollouts):
            continue
        mean_steps = statistics.fmean(rollout_metric_steps(result) for result in rollouts)
        eligible.append((mean_steps, len(set(definition["schedule"])), order, definition))
    if not eligible:
        return None
    _, _, _, selected = min(eligible, key=lambda item: item[:3])
    return dict(selected)


def discovery_summary(state: dict, selected: dict | None) -> dict:
    reports = []
    for definition in CANDIDATES:
        identifier = schedule_hash(definition["schedule"])
        rollouts = state["candidates"][identifier]["discovery"]
        successes = [result for result in rollouts if successful(result)]
        reports.append({
            **definition,
            "schedule_hash": identifier,
            "completed": len(rollouts),
            "successes": len(successes),
            "safety_violations": sum(
                result.get("safety_violation") is not None for result in rollouts
            ),
            "physics_errors": sum(
                result.get("physics_error") is not None for result in rollouts
            ),
            "successful_mean_first_success_steps": (
                None
                if not successes
                else statistics.fmean(rollout_metric_steps(result) for result in successes)
            ),
        })
    return {
        "schema": "act-vlm-targeted-pick-discovery-v1",
        "qualification": "three_pose_discovery_only_no_reliability_claim",
        "candidates": reports,
        "selected_for_fresh_gate": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--discovery-seeds", type=comma_ints, required=True)
    parser.add_argument("--gate-seeds", type=comma_ints, required=True)
    parser.add_argument("--reserved-final-seeds", type=comma_ints, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if git_head() != args.source_commit:
        raise RuntimeError("checked-out source does not match requested commit")
    if len(args.discovery_seeds) != 3 or len(set(args.discovery_seeds)) != 3:
        raise ValueError("exactly three unique discovery seeds are required")
    if len(args.gate_seeds) != 20 or len(set(args.gate_seeds)) != 20:
        raise ValueError("exactly twenty unique gate seeds are required")
    if len(args.reserved_final_seeds) != 100:
        raise ValueError("exactly one hundred final seeds must remain reserved")
    banks = [set(args.discovery_seeds), set(args.gate_seeds), set(args.reserved_final_seeds)]
    if any(banks[i] & banks[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("discovery, gate, and final seed banks must be disjoint")
    for definition in CANDIDATES:
        validate_schedule(definition["schedule"])
        if "labels" in definition:
            mapped = [QUALITATIVE_SPEEDS[label] for label in definition["labels"]]
            if mapped != definition["schedule"]:
                raise ValueError(f"qualitative mapping mismatch for {definition['id']}")

    runtime = ACTFrontierRuntime(
        source_commit=args.source_commit,
        run_manifest=args.run_manifest,
        task_label="pick",
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
    )
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    contract = Path("experiments/act_vlm_targeted_pick_screen_v1/CONTRACT.md")
    identity = {
        **runtime.identity(),
        "schema": "act-vlm-targeted-pick-screen-identity-v1",
        "method": "qualitative_target_plus_standard_comparators",
        "fairness": (
            "user schedule is disclosed hypothesis; no independent VLM discovery claim"
        ),
        "qualitative_speed_mapping": QUALITATIVE_SPEEDS,
        "ordered_candidates": [dict(item) for item in CANDIDATES],
        "discovery_seeds": args.discovery_seeds,
        "gate_seeds": args.gate_seeds,
        "reserved_final_seeds": args.reserved_final_seeds,
        "maximum_new_rollouts": 58,
        "contract_sha256": sha256(contract),
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("targeted Pick output identity mismatch")
    write_json(identity_path, identity)

    discovery_root = root / "discovery"
    discovery = ACTThreeSceneServer(
        discovery_root,
        runtime.task,
        args.discovery_seeds,
        list(range(140220200, 140220210)),
        58,
        runtime=runtime,
    )
    for definition in CANDIDATES:
        result = discovery.probe(definition["schedule"])
        raw = discovery.state["candidates"][result["schedule_hash"]]["discovery"]
        if any(item.get("physics_error") is not None for item in raw):
            raise RuntimeError(f"physics error in discovery candidate {definition['id']}")

    selected = select_discovery_candidate(discovery.state)
    discovery_result = discovery_summary(discovery.state, selected)
    write_json(root / "DISCOVERY.json", discovery_result)

    gate_result = None
    if selected is not None:
        gate = DirectTargetGate(root / "gate", runtime.task, args.gate_seeds, 60, runtime.rollout)
        gate_result = gate.gate(
            selected["schedule"], kind="targeted_discovery_winner"
        )
    result = {
        "schema": "act-vlm-targeted-pick-screen-result-v1",
        "fairness": identity["fairness"],
        "discovery": discovery_result,
        "gate": gate_result,
        "accelerated_search_qualified": (
            gate_result is not None and gate_result["verdict"] == "qualified"
        ),
        "deployment_claim": False,
        "final_bank_opened": False,
        "new_rollouts": (
            discovery.state["episodes_used"]
            + (0 if selected is None else gate.state["episodes_used"])
        ),
    }
    write_json(root / "RESULT.json", result)
    write_json(root / "COMPLETE.json", {
        "schema": "act-vlm-targeted-pick-screen-completion-v1",
        "identity_sha256": sha256(identity_path),
        "discovery_sha256": sha256(root / "DISCOVERY.json"),
        "result_sha256": sha256(root / "RESULT.json"),
        "new_rollouts": result["new_rollouts"],
        "final_bank_opened": False,
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
