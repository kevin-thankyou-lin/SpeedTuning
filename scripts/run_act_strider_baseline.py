#!/usr/bin/env python3
"""Run one receipt-bearing STRIDER search and final ACT benchmark lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_speed_benchmark import sha256
from one_reset_phase_schedule import PHASES, rollout_metric_steps, validate_schedule
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, ACTThreeSceneServer, git_head
from scripts.three_scene_server import comma_ints, successful, write_json


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def episode_metric_steps(record: dict) -> int:
    if record.get("success") and record.get("first_success_step") is not None:
        return int(record["first_success_step"])
    return int(record["physics_steps"])


def summary(records: list[dict]) -> dict:
    successes = [record for record in records if successful(record)]
    return {
        "episodes": len(records),
        "successes": len(successes),
        "success_rate": len(successes) / len(records),
        "successful_mean_first_success_steps": (
            None
            if not successes
            else statistics.fmean(rollout_metric_steps(record) for record in successes)
        ),
        "total_episode_metric_steps": sum(episode_metric_steps(record) for record in records),
        "achieved_throughput_per_step": (
            len(successes) / sum(episode_metric_steps(record) for record in records)
        ),
        "safety_violations": sum(record.get("safety_violation") is not None for record in records),
        "physics_errors": sum(record.get("physics_error") is not None for record in records),
    }


def earliest_failed_phase(candidate: dict) -> str | None:
    """Return the earliest phase that did not advance in a failed rollout."""

    if successful(candidate):
        return None
    reached = [str(item["phase"]) for item in candidate.get("phase_decisions", ())]
    reached_indices = [PHASES.index(phase) for phase in reached if phase in PHASES]
    if not reached_indices:
        return PHASES[0]
    furthest = max(reached_indices)
    return PHASES[min(furthest, len(PHASES) - 1)]


def attribute_backoff_phase(base: dict, protected_phase: str) -> tuple[str, str]:
    failures = [item for item in base["discovery"] if not successful(item)]
    attributed = [earliest_failed_phase(item) for item in failures]
    attributed = [phase for phase in attributed if phase is not None]
    if attributed:
        phase = min(attributed, key=PHASES.index)
        return phase, (
            f"earliest failed phase across {len(failures)} accelerated discovery "
            f"counterexample(s) was {phase}"
        )
    return protected_phase, (
        f"all discovery poses succeeded; registered semantic-risk backoff protects {protected_phase}"
    )


def load_native_records(root: Path, seeds: list[int]) -> list[dict]:
    identity = json.loads((root / "identity.json").read_text())
    expected_identity = identity["identity_sha256"]
    records = []
    for seed in seeds:
        path = root / "states" / f"{seed}.json"
        record = json.loads(path.read_text())
        if record.get("seed") != seed or record.get("identity_sha256") != expected_identity:
            raise RuntimeError(f"native receipt identity mismatch: {path}")
        records.append(record)
    marker = json.loads((root / "COMPLETE.json").read_text())
    if marker.get("episodes") != len(seeds):
        raise RuntimeError("native completion count mismatch")
    return records


def run_search(runtime, root: Path, proposals: dict, discovery: list[int], ranking: list[int]):
    server = ACTThreeSceneServer(
        root,
        runtime.task,
        discovery,
        ranking,
        50,
        runtime=runtime,
    )
    candidates = proposals["candidate_schedules"]
    if len(candidates) != 5 or candidates[0] != [2.0, 2.0, 2.0, 2.0]:
        raise RuntimeError("proposal receipt must freeze five candidates with uniform 2x first")
    for schedule in candidates:
        server.probe(schedule)
    info = server.info()
    base_hash = info["preferred_backoff_base_hash"]
    if base_hash is None:
        raise RuntimeError("search produced no completed accelerated base")
    base = info["candidates"][base_hash]
    phase, evidence = attribute_backoff_phase(base, proposals["protected_phase"])
    server.backoff(base_hash, phase, evidence)
    finalists = server._required_ranking_hashes()
    ranking_result = server.rank(finalists)
    selection = json.loads((root / "public" / "SELECTION.json").read_text())
    return selection, {
        "proposal": proposals,
        "attributed_phase": phase,
        "attribution_evidence": evidence,
        "ranking": ranking_result,
        "episodes_used": server.state["episodes_used"],
        "state_sha256": sha256(server.state_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--proposal-receipt", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--discovery-seeds", type=comma_ints, required=True)
    parser.add_argument("--ranking-seeds", type=comma_ints, required=True)
    parser.add_argument("--final-seeds", type=comma_ints, required=True)
    parser.add_argument("--native-final-root", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if git_head() != args.source_commit:
        raise RuntimeError("checked-out source does not match requested commit")
    if len(args.discovery_seeds) != 3 or len(args.ranking_seeds) != 10:
        raise ValueError("STRIDER requires exactly three discovery and ten ranking seeds")
    if len(args.final_seeds) != 50:
        raise ValueError("STRIDER final evaluation requires exactly fifty seeds")
    banks = [set(args.discovery_seeds), set(args.ranking_seeds), set(args.final_seeds)]
    if any(banks[i] & banks[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("discovery, ranking, and final banks must be disjoint")

    proposal_receipt = json.loads(args.proposal_receipt.read_text())
    proposals = proposal_receipt["tasks"][args.task_label]
    for schedule in proposals["candidate_schedules"]:
        validate_schedule(schedule)
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
        "schema": "act-strider-baseline-identity-v1",
        "method": "strider",
        "contract_sha256": sha256(args.contract),
        "proposal_receipt_sha256": sha256(args.proposal_receipt),
        "task_proposal_sha256": canonical_sha256(proposals),
        "discovery_seeds": args.discovery_seeds,
        "ranking_seeds": args.ranking_seeds,
        "final_seeds": args.final_seeds,
        "native_final_root": str(args.native_final_root.resolve()),
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("STRIDER root identity mismatch")
    write_json(identity_path, identity)

    selection, search = run_search(
        runtime,
        root / "search",
        proposals,
        args.discovery_seeds,
        args.ranking_seeds,
    )
    selected_schedule = selection["benchmark_schedule"]
    final_root = root / "final" / "states"
    final_records = []
    for seed in args.final_seeds:
        path = final_root / f"{seed}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("seed") != seed or record.get("schedule") != selected_schedule:
                raise RuntimeError(f"STRIDER cached final identity mismatch: {path}")
        else:
            record = runtime.rollout(selected_schedule, seed)
            write_json(path, record)
        final_records.append(record)

    native_records = load_native_records(args.native_final_root, args.final_seeds)
    candidate_summary = summary(final_records)
    native_summary = summary(native_records)
    candidate_mean = candidate_summary["successful_mean_first_success_steps"]
    native_mean = native_summary["successful_mean_first_success_steps"]
    candidate_summary["successful_rollout_speedup"] = (
        None if candidate_mean is None or native_mean is None else native_mean / candidate_mean
    )
    candidate_summary["throughput_delta_percent_vs_native"] = 100.0 * (
        candidate_summary["achieved_throughput_per_step"]
        / native_summary["achieved_throughput_per_step"]
        - 1.0
    )
    result = {
        "schema": "act-strider-baseline-result-v1",
        "task_label": args.task_label,
        "identity_sha256": sha256(identity_path),
        "selected_schedule": selected_schedule,
        "accelerated_qualified_in_search": selection["accelerated_qualified"],
        "search": search,
        "final": candidate_summary,
        "native_reused": native_summary,
        "native_rollouts_reexecuted": 0,
        "final_rollouts": 50,
        "development_disclosure": proposal_receipt["disclosure"],
    }
    result_path = root / "RESULT.json"
    write_json(result_path, result)
    completion = {
        "schema": "act-strider-baseline-completion-v1",
        "identity_sha256": sha256(identity_path),
        "search_state_sha256": search["state_sha256"],
        "selection_sha256": sha256(root / "search" / "public" / "SELECTION.json"),
        "result_sha256": sha256(result_path),
        "search_rollouts": search["episodes_used"],
        "final_rollouts": 50,
        "native_rollouts_reexecuted": 0,
    }
    write_json(root / "COMPLETE.json", completion)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

