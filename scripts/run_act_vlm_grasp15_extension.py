#!/usr/bin/env python3
"""Run the Pick 1.5x grasp-lift post-hoc extension with frozen ACT."""

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


def checked(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"parent artifact hash mismatch: {path}: {actual} != {expected}")
    return actual


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
    hashes = {
        "COMPLETE.json": "1d694a6b2ca32be65b706260fdd061be0c81bd70fc9a474304318ade7eee7ba1",
        "private/state.json": "ace434d57f46b53edd1243c783541694e724e6046a8d3f63882df8d01d8745c2",
        "public/SELECTION.json": "4f7fe5eaf8f0804fb8c12c59e5a4a3e7a48824d884211f2e5da525f5df273d8c",
    }
    for name, expected in hashes.items():
        checked(parent / name, expected)
    parent_complete = json.loads((parent / "COMPLETE.json").read_text())
    if parent_complete["episodes_used"] != 80 or parent_complete["final_bank_opened"]:
        raise RuntimeError("parent completion contract mismatch")
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
        "schema": "act-vlm-grasp15-extension-identity-v1",
        "candidate": [2.5, 1.5, 2.5, 2.5],
        "parent_root": str(parent),
        "parent_hashes": hashes,
        "contract_sha256": sha256(Path("experiments/act_vlm_grasp15_extension_pick_v1/CONTRACT.md")),
        "new_rollout_budget": 20,
        "native_rollouts_reexecuted": 0,
        "reserved_final_seeds_opened": False,
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("extension identity mismatch")
    write_json(identity_path, identity)
    extension = SingleCandidateExtension(
        root,
        json.loads((parent / "private/state.json").read_text()),
        [2.5, 1.5, 2.5, 2.5],
        runtime.rollout,
        anchor_successes=19,
        anchor_speedup=1.8725346968590213,
    )
    result = extension.run()
    write_json(root / "COMPLETE.json", {
        "schema": "act-vlm-grasp15-extension-completion-v1",
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
