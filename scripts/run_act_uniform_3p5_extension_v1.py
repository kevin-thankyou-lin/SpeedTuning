#!/usr/bin/env python3
"""Evaluate the missing uniform 3.5x point on a frozen STRIDER v4 final bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import run_act_strider_frontier_v2 as base


SCHEDULE = [3.5] * 4


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def expected_final_seeds(banks: dict, task_label: str) -> list[int]:
    spec = banks["tasks"][task_label]["final"]
    start = int(spec["start"])
    return list(range(start, start + int(spec["count"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument(
        "--task-label", choices=("pick", "tea", "insertion"), required=True
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--final-seeds", type=comma_ints, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    banks = json.loads(args.banks.read_text())
    expected = expected_final_seeds(banks, args.task_label)
    if args.final_seeds != expected or len(set(args.final_seeds)) != 50:
        raise ValueError("runtime final seeds do not match the frozen STRIDER v4 bank")

    from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head

    if git_head() != args.source_commit:
        raise RuntimeError("checked-out source does not match requested commit")
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
        "schema": "act-uniform-3p5-extension-identity-v1",
        "method": "uniform_3p5x",
        "schedule": SCHEDULE,
        "schedule_sha256": base.schedule_sha256(SCHEDULE),
        "contract_sha256": base.file_sha256(args.contract),
        "banks_sha256": base.file_sha256(args.banks),
        "final_seeds": args.final_seeds,
        "native_controls_rerun": False,
        "paired_native_source": "STRIDER v4 native_1x on the identical final seeds",
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("uniform 3.5x extension identity mismatch")
    base.write_json(identity_path, identity)

    ledger = base.RolloutLedger(runtime, root, [], args.final_seeds)
    controller, _ = ledger.evaluate_final(SCHEDULE)
    result = {
        "schema": "act-uniform-3p5-extension-result-v1",
        "task_label": args.task_label,
        "identity_sha256": base.file_sha256(identity_path),
        "controller": controller,
        "accounting": {
            "new_uniform_3p5_rollouts": 50,
            "native_controls_rerun": 0,
            "shared_final_seed_bank": True,
        },
    }
    result_path = root / "RESULT.json"
    base.write_json(result_path, result)
    base.write_json(
        root / "COMPLETE.json",
        {
            "schema": "act-uniform-3p5-extension-completion-v1",
            "identity_sha256": base.file_sha256(identity_path),
            "result_sha256": base.file_sha256(result_path),
            **result["accounting"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
