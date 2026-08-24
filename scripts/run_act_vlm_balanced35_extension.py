#!/usr/bin/env python3
"""Run the Pick [2.5,1.5,3.5,3.5] same-pose staged extension."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import sha256
from scripts.act_vlm_frontier_server import ACTFrontierRuntime
from scripts.staged_vlm_extension import SingleCandidateExtension
from scripts.staged_vlm_frontier import write_json


PARENT_SOURCE = "cda7f3e0c85a7efc420cbc34e565a368561ae607"
CANDIDATE = [2.5, 1.5, 3.5, 3.5]
PARENT_TARGET = [3.0, 1.5, 4.0, 4.0]
GATE_SEEDS = list(range(140220100, 140220120))


def validate_parent(parent: Path) -> tuple[dict, dict]:
    paths = {
        "IDENTITY.json": parent / "IDENTITY.json",
        "DISCOVERY.json": parent / "DISCOVERY.json",
        "RESULT.json": parent / "RESULT.json",
        "COMPLETE.json": parent / "COMPLETE.json",
        "gate/private/state.json": parent / "gate/private/state.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"parent artifact is absent: {path}")
    identity = json.loads(paths["IDENTITY.json"].read_text())
    result = json.loads(paths["RESULT.json"].read_text())
    complete = json.loads(paths["COMPLETE.json"].read_text())
    gate = json.loads(paths["gate/private/state.json"].read_text())
    if identity.get("source_commit") != PARENT_SOURCE:
        raise RuntimeError("parent source identity mismatch")
    if identity.get("gate_seeds") != GATE_SEEDS:
        raise RuntimeError("parent gate seed bank mismatch")
    if identity.get("reserved_final_seeds") != list(range(140210000, 140210100)):
        raise RuntimeError("parent final seed reservation mismatch")
    if complete.get("new_rollouts") != 58 or complete.get("final_bank_opened"):
        raise RuntimeError("parent completion contract mismatch")
    parent_gate = result.get("gate") or {}
    if (
        parent_gate.get("schedule") != PARENT_TARGET
        or parent_gate.get("completed") != 20
        or parent_gate.get("successes") != 17
        or parent_gate.get("verdict") != "rejected_at_20"
    ):
        raise RuntimeError("parent target-gate result mismatch")
    if gate.get("seeds") != GATE_SEEDS or len(gate.get("native", [])) != 20:
        raise RuntimeError("parent matched native gate bank mismatch")
    hashes = {name: sha256(path) for name, path in paths.items()}
    return gate, hashes


def extension_result(raw: dict, parent_hashes: dict) -> dict:
    return {
        "schema": "act-vlm-balanced35-extension-result-v1",
        "candidate": CANDIDATE,
        "completed": raw["completed"],
        "successes": raw["successes"],
        "verdict": raw["verdict"],
        "matched_native_speedup": raw["matched_native_speedup"],
        "safety_violations": raw["safety_violations"],
        "physics_errors": raw["physics_errors"],
        "qualified_for_continued_search": raw["verdict"] == "qualified",
        "parent_target": {
            "schedule": PARENT_TARGET,
            "successes": 17,
            "completed": 20,
            "verdict": "rejected_at_20",
        },
        "parent_hashes": parent_hashes,
        "new_candidate_rollouts": raw["new_candidate_rollouts"],
        "new_native_rollouts": 0,
        "final_bank_opened": False,
        "deployment_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    parent = args.parent_root.resolve()
    parent_state, parent_hashes = validate_parent(parent)
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
        "schema": "act-vlm-balanced35-extension-identity-v1",
        "candidate": CANDIDATE,
        "origin": "user_disclosed_posthoc_hypothesis",
        "parent_root": str(parent),
        "parent_source": PARENT_SOURCE,
        "parent_hashes": parent_hashes,
        "gate_seeds": GATE_SEEDS,
        "new_rollout_budget": 20,
        "native_rollouts_reexecuted": 0,
        "reserved_final_seeds_opened": False,
        "contract_sha256": sha256(
            Path("experiments/act_vlm_balanced35_extension_pick_v1/CONTRACT.md")
        ),
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("balanced-3.5 extension identity mismatch")
    write_json(identity_path, identity)

    extension = SingleCandidateExtension(
        root,
        parent_state,
        CANDIDATE,
        runtime.rollout,
        anchor_successes=18,
        anchor_speedup=0.0,
    )
    raw = extension.run()
    result = extension_result(raw, parent_hashes)
    write_json(root / "RESULT.json", result)
    write_json(root / "COMPLETE.json", {
        "schema": "act-vlm-balanced35-extension-completion-v1",
        "identity_sha256": sha256(identity_path),
        "state_sha256": sha256(root / "private/state.json"),
        "result_sha256": sha256(root / "RESULT.json"),
        "new_candidate_rollouts": result["new_candidate_rollouts"],
        "new_native_rollouts": 0,
        "final_bank_opened": False,
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

