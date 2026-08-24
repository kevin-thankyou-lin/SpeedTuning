#!/usr/bin/env python3
"""Serve the blinded staged VLM frontier over the frozen ACT policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import sha256
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head
from scripts.staged_vlm_frontier import StagedFrontier, write_json
from scripts.three_scene_server import comma_ints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--search-seeds", type=comma_ints, required=True)
    parser.add_argument("--reserved-final-seeds", type=comma_ints, required=True)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--autorun-anchor", action="store_true")
    parser.add_argument("--exit-after-selection", action="store_true")
    args = parser.parse_args()
    if len(args.reserved_final_seeds) != 100:
        raise ValueError("exactly 100 untouched final seeds must be reserved")
    if set(args.search_seeds) & set(args.reserved_final_seeds):
        raise ValueError("search and final seed banks overlap")

    runtime = ACTFrontierRuntime(
        source_commit=args.source_commit,
        run_manifest=args.run_manifest,
        task_label=args.task_label,
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
    )
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": "act-staged-vlm-frontier-identity-v1",
        "method": "blinded_oracle_phase_anchor_repair_promote",
        "search_seeds": args.search_seeds,
        "reserved_final_seeds": args.reserved_final_seeds,
        "budget": args.budget,
        "contract_sha256": sha256(Path("experiments/act_vlm_staged_frontier_pick_v1/CONTRACT.md")),
        "source_head": git_head(),
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("output root identity mismatch")
    write_json(identity_path, identity)

    server = StagedFrontier(root, runtime.task, args.search_seeds, args.budget, runtime.rollout)
    api = root / "api"
    (api / "requests").mkdir(parents=True, exist_ok=True)
    (api / "responses").mkdir(parents=True, exist_ok=True)
    write_json(api / "READY.json", {"ready": True, "identity_sha256": sha256(identity_path)})
    if args.autorun_anchor and not server.state["candidate_order"]:
        server.run_native()
        server.gate([2, 2, 2, 2], kind="anchor")
        write_json(root / "public" / "AUTORUN.json", server.info())

    while True:
        for request_path in sorted((api / "requests").glob("*.json")):
            response_path = api / "responses" / request_path.name
            if response_path.exists():
                continue
            try:
                request = json.loads(request_path.read_text())
                command = request.get("command")
                if command == "info":
                    result = server.info()
                elif command == "native":
                    result = server.run_native()
                elif command == "gate":
                    result = server.gate(
                        request.get("schedule"), kind=str(request.get("kind")),
                        phase=request.get("phase"), evidence=request.get("evidence"),
                    )
                elif command == "finalize":
                    result = server.finalize()
                else:
                    raise ValueError(f"unknown command: {command}")
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            write_json(response_path, response)
        selection = root / "public" / "SELECTION.json"
        if args.exit_after_selection and selection.exists():
            write_json(root / "COMPLETE.json", {
                "schema": "act-staged-vlm-frontier-completion-v1",
                "identity_sha256": sha256(identity_path),
                "state_sha256": sha256(root / "private" / "state.json"),
                "selection_sha256": sha256(selection),
                "episodes_used": server.state["episodes_used"],
                "final_bank_opened": False,
            })
            return 0
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
