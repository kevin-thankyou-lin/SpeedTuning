#!/usr/bin/env python3
"""Run same-pose uniform gates and compare with the Pick adaptive schedules."""

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


TARGETED_SOURCE = "cda7f3e0c85a7efc420cbc34e565a368561ae607"
BALANCED_SOURCE = "0c76a17d8903269ec3be3c1f49c46b33564f2e56"
GATE_SEEDS = list(range(140220100, 140220120))
UNIFORMS = ((2.0, 2.0, 2.0, 2.0), (2.5, 2.5, 2.5, 2.5))
BALANCED = (2.5, 1.5, 3.5, 3.5)
AGGRESSIVE = (3.0, 1.5, 4.0, 4.0)


def _load(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required artifact is absent: {path}")
    return json.loads(path.read_text())


def validate_parents(targeted: Path, balanced: Path) -> tuple[dict, dict]:
    targeted_paths = {
        "targeted/IDENTITY.json": targeted / "IDENTITY.json",
        "targeted/RESULT.json": targeted / "RESULT.json",
        "targeted/COMPLETE.json": targeted / "COMPLETE.json",
        "targeted/gate/private/state.json": targeted / "gate/private/state.json",
    }
    balanced_paths = {
        "balanced/IDENTITY.json": balanced / "IDENTITY.json",
        "balanced/RESULT.json": balanced / "RESULT.json",
        "balanced/COMPLETE.json": balanced / "COMPLETE.json",
        "balanced/private/state.json": balanced / "private/state.json",
    }
    targeted_identity = _load(targeted_paths["targeted/IDENTITY.json"])
    targeted_result = _load(targeted_paths["targeted/RESULT.json"])
    targeted_complete = _load(targeted_paths["targeted/COMPLETE.json"])
    gate_state = _load(targeted_paths["targeted/gate/private/state.json"])
    balanced_identity = _load(balanced_paths["balanced/IDENTITY.json"])
    balanced_result = _load(balanced_paths["balanced/RESULT.json"])
    balanced_complete = _load(balanced_paths["balanced/COMPLETE.json"])

    if targeted_identity.get("source_commit") != TARGETED_SOURCE:
        raise RuntimeError("targeted parent source mismatch")
    if targeted_identity.get("gate_seeds") != GATE_SEEDS:
        raise RuntimeError("targeted parent seed mismatch")
    if targeted_complete.get("new_rollouts") != 58 or targeted_complete.get("final_bank_opened"):
        raise RuntimeError("targeted parent completion mismatch")
    aggressive = targeted_result.get("gate") or {}
    if (
        aggressive.get("schedule") != list(AGGRESSIVE)
        or aggressive.get("successes") != 17
        or aggressive.get("completed") != 20
        or aggressive.get("verdict") != "rejected_at_20"
    ):
        raise RuntimeError("targeted aggressive result mismatch")
    if gate_state.get("seeds") != GATE_SEEDS or len(gate_state.get("native", [])) != 20:
        raise RuntimeError("targeted matched native bank mismatch")

    if balanced_identity.get("source_commit") != BALANCED_SOURCE:
        raise RuntimeError("balanced parent source mismatch")
    if balanced_identity.get("candidate") != list(BALANCED):
        raise RuntimeError("balanced parent candidate mismatch")
    if balanced_complete.get("new_native_rollouts") != 0 or balanced_complete.get("final_bank_opened"):
        raise RuntimeError("balanced parent completion mismatch")
    if (
        balanced_result.get("candidate") != list(BALANCED)
        or balanced_result.get("successes") != 18
        or balanced_result.get("completed") != 20
        or balanced_result.get("verdict") != "qualified"
    ):
        raise RuntimeError("balanced parent result mismatch")

    hashes = {
        name: sha256(path)
        for name, path in {**targeted_paths, **balanced_paths}.items()
    }
    return gate_state, {
        "hashes": hashes,
        "aggressive": aggressive,
        "balanced": balanced_result,
    }


def uniform_summary(raw: dict) -> dict:
    return {
        "schedule": raw["candidate"],
        "completed": raw["completed"],
        "successes": raw["successes"],
        "verdict": raw["verdict"],
        "matched_native_speedup": raw["matched_native_speedup"],
        "safety_violations": raw["safety_violations"],
        "physics_errors": raw["physics_errors"],
        "source": "new_same_pose_uniform_gate",
    }


def clean_qualified(item: dict) -> bool:
    return (
        item["completed"] == 20
        and item["successes"] >= 18
        and item["verdict"] == "qualified"
        and item["safety_violations"] == 0
        and item["physics_errors"] == 0
    )


def ranking_key(item: dict):
    return (
        item["successes"],
        item["matched_native_speedup"],
        -len(set(item["schedule"])),
    )


def pareto_frontier(items: list[dict]) -> list[dict]:
    eligible = [item for item in items if clean_qualified(item)]
    return [
        item
        for item in eligible
        if not any(
            other is not item
            and other["successes"] >= item["successes"]
            and other["matched_native_speedup"] >= item["matched_native_speedup"]
            and (
                other["successes"] > item["successes"]
                or other["matched_native_speedup"] > item["matched_native_speedup"]
            )
            for other in eligible
        )
    ]


def compare(items: list[dict]) -> dict:
    eligible = [item for item in items if clean_qualified(item)]
    selected = None if not eligible else max(eligible, key=ranking_key)
    uniforms = [item for item in eligible if len(set(item["schedule"])) == 1]
    best_uniform = None if not uniforms else max(uniforms, key=ranking_key)
    balanced = next(item for item in items if item["schedule"] == list(BALANCED))
    adaptive_strictly_beats_uniform = False
    if best_uniform is not None and clean_qualified(balanced):
        adaptive_strictly_beats_uniform = (
            balanced["successes"] >= best_uniform["successes"]
            and balanced["matched_native_speedup"] >= best_uniform["matched_native_speedup"]
            and (
                balanced["successes"] > best_uniform["successes"]
                or balanced["matched_native_speedup"] > best_uniform["matched_native_speedup"]
            )
        )
    return {
        "selection_rule": "clean qualified; exact successes, speedup, simplicity",
        "selected": selected,
        "best_uniform": best_uniform,
        "balanced_adaptive": balanced,
        "balanced_strictly_beats_best_uniform": adaptive_strictly_beats_uniform,
        "pareto_frontier": pareto_frontier(items),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--targeted-root", type=Path, required=True)
    parser.add_argument("--balanced-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    targeted = args.targeted_root.resolve()
    balanced = args.balanced_root.resolve()
    parent_state, parents = validate_parents(targeted, balanced)
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
        "schema": "act-vlm-paired-uniform-pick-identity-v1",
        "targeted_root": str(targeted),
        "balanced_root": str(balanced),
        "parent_hashes": parents["hashes"],
        "gate_seeds": GATE_SEEDS,
        "ordered_new_schedules": [list(schedule) for schedule in UNIFORMS],
        "maximum_new_rollouts": 40,
        "native_rollouts_reexecuted": 0,
        "reserved_final_seeds_opened": False,
        "contract_sha256": sha256(
            Path("experiments/act_vlm_paired_uniform_pick_v1/CONTRACT.md")
        ),
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("paired uniform identity mismatch")
    write_json(identity_path, identity)

    uniform_results = []
    for name, schedule in zip(("uniform2", "uniform2p5"), UNIFORMS):
        extension = SingleCandidateExtension(
            root / name,
            parent_state,
            schedule,
            runtime.rollout,
            anchor_successes=18,
            anchor_speedup=0.0,
        )
        uniform_results.append(uniform_summary(extension.run()))

    balanced_result = {
        key: parents["balanced"][key]
        for key in (
            "candidate", "completed", "successes", "verdict",
            "matched_native_speedup", "safety_violations", "physics_errors",
        )
    }
    balanced_result["schedule"] = balanced_result.pop("candidate")
    balanced_result["source"] = "reused_hash_pinned_balanced_extension"
    aggressive_result = {
        key: parents["aggressive"][key]
        for key in (
            "schedule", "completed", "successes", "verdict",
            "matched_native_speedup", "safety_violations", "physics_errors",
        )
    }
    aggressive_result["source"] = "reused_hash_pinned_targeted_screen"
    items = [*uniform_results, balanced_result, aggressive_result]
    result = {
        "schema": "act-vlm-paired-uniform-pick-result-v1",
        "candidates": items,
        "comparison": compare(items),
        "new_candidate_rollouts": sum(item["completed"] for item in uniform_results),
        "new_native_rollouts": 0,
        "final_bank_opened": False,
        "deployment_claim": False,
    }
    write_json(root / "RESULT.json", result)
    write_json(root / "COMPLETE.json", {
        "schema": "act-vlm-paired-uniform-pick-completion-v1",
        "identity_sha256": sha256(identity_path),
        "result_sha256": sha256(root / "RESULT.json"),
        "new_candidate_rollouts": result["new_candidate_rollouts"],
        "new_native_rollouts": 0,
        "final_bank_opened": False,
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

