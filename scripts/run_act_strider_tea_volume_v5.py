#!/usr/bin/env python3
"""Run Tea STRIDER with the corrected cup-volume success criterion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import run_act_strider_frontier_v4 as v4


file_sha256 = v4.file_sha256
write_json = v4.write_json
comma_ints = v4.comma_ints
METRIC_REGRESSION_SEEDS = (160500100, 160500109)


def checked_success_criterion(path: Path) -> dict:
    criterion = json.loads(path.read_text())
    if criterion.get("schema") != "tea-cup-volume-success-v1":
        raise ValueError("unexpected Tea success criterion schema")
    repo_root = Path(__file__).resolve().parents[1]
    for relative_path, receipt in criterion["files"].items():
        actual = file_sha256(repo_root / relative_path)
        if actual != receipt["sha256"]:
            raise RuntimeError(
                f"Tea success criterion hash mismatch: {relative_path}: "
                f"{actual} != {receipt['sha256']}"
            )
    return criterion


def run_metric_regression(runtime, root: Path) -> dict:
    """Prove both previously inspected in-cup trajectories now count."""

    regression_root = root / "metric_regression"
    records = []
    new_rollouts = 0
    for seed in METRIC_REGRESSION_SEEDS:
        record_path = regression_root / "states" / f"{seed}.json"
        video_path = regression_root / f"tea_uniform2x_seed{seed}.mp4"
        if record_path.exists():
            record = json.loads(record_path.read_text())
        else:
            record = runtime.rollout(
                [2.0] * 4,
                seed,
                video_path=video_path,
                record_attribution_telemetry=True,
            )
            write_json(record_path, record)
            new_rollouts += 1
        if int(record.get("seed", -1)) != seed:
            raise RuntimeError("metric-regression seed identity mismatch")
        if list(map(float, record.get("schedule", ()))) != [2.0] * 4:
            raise RuntimeError("metric-regression schedule identity mismatch")
        if not record.get("success"):
            raise RuntimeError(
                f"Tea cup-volume metric did not accept diagnostic seed {seed}"
            )
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError(f"missing metric-regression video: {video_path}")
        records.append(record)
    report = {
        "schema": "tea-cup-volume-metric-regression-v1",
        "schedule": [2.0] * 4,
        "seeds": list(METRIC_REGRESSION_SEEDS),
        "successes": sum(bool(record["success"]) for record in records),
        "episodes": len(records),
        "new_rollouts_this_invocation": new_rollouts,
        "excluded_from_search_and_final": True,
        "records": records,
    }
    write_json(regression_root / "RECEIPT.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--success-criterion", type=Path, required=True)
    parser.add_argument("--search-seeds", type=comma_ints, required=True)
    parser.add_argument("--final-seeds", type=comma_ints, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if len(args.search_seeds) != 20 or len(set(args.search_seeds)) != 20:
        raise ValueError("STRIDER Tea v5 requires twenty unique search seeds")
    if len(args.final_seeds) != 50 or len(set(args.final_seeds)) != 50:
        raise ValueError("STRIDER Tea v5 requires fifty unique final seeds")
    if set(args.search_seeds) & set(args.final_seeds):
        raise ValueError("search and final banks must be disjoint")

    criterion = checked_success_criterion(args.success_criterion)
    banks = json.loads(args.banks.read_text())
    task_banks = banks["tasks"]["tea"]
    expected_search = list(
        range(task_banks["search"]["start"], task_banks["search"]["start"] + 20)
    )
    expected_final = list(
        range(task_banks["final"]["start"], task_banks["final"]["start"] + 50)
    )
    if args.search_seeds != expected_search or args.final_seeds != expected_final:
        raise ValueError("runtime seeds do not match frozen STRIDER Tea v5 banks")

    from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head

    if git_head() != args.source_commit:
        raise RuntimeError("checked-out source does not match requested commit")
    runtime = ACTFrontierRuntime(
        source_commit=args.source_commit,
        run_manifest=args.run_manifest,
        task_label="tea",
        detector_checkpoint=args.detector_checkpoint,
        detector_source=args.detector_source,
        device=args.device,
        critical_source_overrides={
            "sim_tasks.py": criterion["files"]["sim_tasks.py"]["sha256"]
        },
    )

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": "act-strider-tea-volume-identity-v5",
        "method": "strider_conservative_uniform_lower_bound",
        "contract_sha256": file_sha256(args.contract),
        "banks_sha256": file_sha256(args.banks),
        "success_criterion_sha256": file_sha256(args.success_criterion),
        "success_criterion": criterion,
        "search_seeds": args.search_seeds,
        "final_seeds": args.final_seeds,
        "search_budget": v4.SEARCH_BUDGET,
        "stages": [
            {"episodes": count, "minimum_successes": minimum}
            for count, minimum in v4.STAGES
        ],
        "prior_metric_results_reused": False,
        "metric_regression_seeds": list(METRIC_REGRESSION_SEEDS),
        "metric_regression_visible_to_selector": False,
        "selection_frozen_before_final": True,
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("STRIDER Tea v5 root identity mismatch")
    write_json(identity_path, identity)

    metric_regression = run_metric_regression(runtime, root)

    ledger = v4.RolloutLedger(
        runtime,
        root,
        args.search_seeds,
        args.final_seeds,
        record_search_telemetry=True,
    )
    selection = v4.run_search(ledger, "tea")
    selection["schema"] = "act-strider-tea-volume-selection-v5"
    selection["success_criterion_sha256"] = identity["success_criterion_sha256"]
    selection_path = root / "SELECTION.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise RuntimeError("sealed STRIDER Tea v5 selection changed during resume")
    write_json(selection_path, selection)
    selection_hash = file_sha256(selection_path)

    final = v4.run_final(ledger, selection)
    result = {
        "schema": "act-strider-tea-volume-result-v5",
        "identity_sha256": file_sha256(identity_path),
        "selection_sha256_before_final": selection_hash,
        "selection": selection,
        "final": final,
        "metric_regression": metric_regression,
        "accounting": {
            "metric_regression_rollouts": metric_regression["episodes"],
            "search_rollouts": ledger.search_rollouts_used(),
            "search_budget": v4.SEARCH_BUDGET,
            "new_final_rollouts": final["new_final_rollouts"],
            "total_new_rollouts": (
                metric_regression["episodes"]
                + ledger.search_rollouts_used()
                + final["new_final_rollouts"]
            ),
            "prior_metric_rollouts_reused": 0,
            "final_bank_opened_only_after_selection": True,
        },
    }
    result_path = root / "RESULT.json"
    write_json(result_path, result)
    write_json(
        root / "COMPLETE.json",
        {
            "schema": "act-strider-tea-volume-completion-v5",
            "identity_sha256": file_sha256(identity_path),
            "selection_sha256": selection_hash,
            "result_sha256": file_sha256(result_path),
            **result["accounting"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
