#!/usr/bin/env python3
"""Run the phase-risk-biased 5->10->15 search on frozen ACT Pick."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import sha256
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head
from scripts.phase_adaptive15 import Adaptive15Search, generate_candidates, write_json
from scripts.three_scene_server import comma_ints


PHASE_RISK_LABELS = {
    "pre_grasp": "cautious",
    "grasp_lift": "protected",
    "transport": "open",
    "interaction": "open",
}
PROPOSAL_RECEIPT = {
    "schema": "qualitative-phase-risk-proposal-v1",
    "source": "user_guided_vlm_compatible_prior_from_existing_pick_evidence",
    "independent_schedule_discovery_claim": False,
    "labels_only": True,
    "rationale": {
        "pre_grasp": "approach remains spatially sensitive",
        "grasp_lift": "contact and grasp transition require protection",
        "transport": "stable retained grasp permits frontier expansion",
        "interaction": "post-transport motion permits frontier expansion",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--search-seeds", type=comma_ints, required=True)
    parser.add_argument("--reserved-final-seeds", type=comma_ints, required=True)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if git_head() != args.source_commit:
        raise RuntimeError("checked-out source does not match requested commit")
    if len(args.search_seeds) != 15 or len(set(args.search_seeds)) != 15:
        raise ValueError("exactly fifteen unique search seeds are required")
    if len(args.reserved_final_seeds) != 100:
        raise ValueError("exactly one hundred final seeds must remain reserved")
    if set(args.search_seeds) & set(args.reserved_final_seeds):
        raise ValueError("search and final seed banks overlap")

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
    identity = {
        **runtime.identity(),
        "schema": "act-phase-adaptive15-pick-identity-v1",
        "method": "qualitative_phase_risk_ladder_with_5_10_15_gate",
        "phase_risk_labels": PHASE_RISK_LABELS,
        "proposal_receipt": PROPOSAL_RECEIPT,
        "candidate_definitions": generate_candidates(PHASE_RISK_LABELS),
        "search_seeds": args.search_seeds,
        "reserved_final_seeds": args.reserved_final_seeds,
        "maximum_new_rollouts": args.budget,
        "contract_sha256": sha256(
            Path("experiments/act_phase_adaptive15_pick_v1/CONTRACT.md")
        ),
        "source_head": git_head(),
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("adaptive15 output identity mismatch")
    write_json(identity_path, identity)

    search = Adaptive15Search(
        root,
        runtime.task,
        args.search_seeds,
        args.budget,
        runtime.rollout,
        phase_risk_labels=PHASE_RISK_LABELS,
        proposal_receipt=PROPOSAL_RECEIPT,
    )
    result = search.run()
    write_json(root / "COMPLETE.json", {
        "schema": "act-phase-adaptive15-pick-completion-v1",
        "identity_sha256": sha256(identity_path),
        "state_sha256": sha256(root / "private" / "state.json"),
        "result_sha256": sha256(root / "public" / "RESULT.json"),
        "episodes_used": result["episodes_used"],
        "final_bank_opened": False,
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
