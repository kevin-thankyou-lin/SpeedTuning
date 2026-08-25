#!/usr/bin/env python3
"""Audit sealed ACT baselines plus STRIDER into a paper-facing table."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


TASKS = ("pick", "tea", "insertion")
METHODS = (
    ("uniform_sweep", "Uniform sweep", "base"),
    ("learned_phase_subtask", "Learned subtask", "repair"),
    ("learned_phase_tabular_rl", "Tabular RL", "base"),
    ("learned_phase_rainbow_rl", "Rainbow RL", "base"),
    ("awe_offline_proxy", "AWE offline proxy", "base"),
    ("sail_inspired_adaptive", "SAIL-inspired", "base"),
)


def load(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing required evidence: {path}")
    return json.loads(path.read_text())


def episode_steps(record: dict) -> int:
    if record.get("success") and record.get("first_success_step") is not None:
        return int(record["first_success_step"])
    return int(record["physics_steps"])


def summarize(records: list[dict], native: list[dict]) -> dict:
    successes = [record for record in records if record.get("success")]
    native_successes = [record for record in native if record.get("success")]
    throughput = len(successes) / sum(episode_steps(record) for record in records)
    native_throughput = len(native_successes) / sum(episode_steps(record) for record in native)
    mean = None if not successes else statistics.fmean(episode_steps(record) for record in successes)
    native_mean = statistics.fmean(episode_steps(record) for record in native_successes)
    return {
        "episodes": len(records),
        "successes": len(successes),
        "success_rate": len(successes) / len(records),
        "successful_mean_first_success_steps": mean,
        "successful_rollout_speedup": None if mean is None else native_mean / mean,
        "achieved_throughput_per_step": throughput,
        "throughput_delta_percent_vs_native": 100.0 * (throughput / native_throughput - 1.0),
        "safety_violations": sum(record.get("safety_violation") is not None for record in records),
        "physics_errors": sum(record.get("physics_error") is not None for record in records),
    }


def records(root: Path, seeds: list[int]) -> list[dict]:
    values = [load(root / "states" / f"{seed}.json") for seed in seeds]
    if [value.get("seed") for value in values] != seeds:
        raise RuntimeError(f"seed order mismatch: {root}")
    return values


def markdown(report: dict) -> str:
    lines = [
        "# Preliminary frozen-ACT speed results with STRIDER",
        "",
        "All rows use the same 50 final seeds within each task. Throughput charges successful episodes through first success and failures through their terminal horizon.",
        "",
        "| Task | Method | Success | SR | Successful-rollout speedup | Throughput delta vs 1x | Safety | Physics |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for method in report["tasks"][task]["methods"]:
            speedup = method["successful_rollout_speedup"]
            lines.append(
                f"| {task.title()} | {method['display_name']} | {method['successes']}/50 | "
                f"{method['success_rate']:.2f} | "
                f"{'--' if speedup is None else f'{speedup:.3f}x'} | "
                f"{method['throughput_delta_percent_vs_native']:+.1f}% | "
                f"{method['safety_violations']} | {method['physics_errors']} |"
            )
    lines.extend([
        "",
        "`AWE offline proxy` and `SAIL-inspired` are the benchmark's preregistered internal proxies; they are not claimed as paper-faithful AWE or SAIL implementations.",
        "The Pick STRIDER proposal is a disclosed development-task result. Treat this table as preliminary until a newly preregistered paper benchmark is run.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--base-source", required=True)
    parser.add_argument("--repair-source", required=True)
    parser.add_argument("--strider-root", type=Path, required=True)
    parser.add_argument("--strider-pick-source", required=True)
    parser.add_argument("--strider-tea-source", required=True)
    parser.add_argument("--strider-insertion-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_manifest = load(args.benchmark_root / "attempts" / args.base_source / "run_manifest.json")
    repair_manifest = load(args.benchmark_root / "attempts" / args.repair_source / "run_manifest.json")
    report = {
        "schema": "act-strider-preliminary-table-v1",
        "tasks": {},
        "baseline_sources": {"base": args.base_source, "repair": args.repair_source},
        "strider_sources": {
            "pick": args.strider_pick_source,
            "tea": args.strider_tea_source,
            "insertion": args.strider_insertion_source,
        },
    }
    for task in TASKS:
        seeds = base_manifest["tasks"][task]["final_bank"]["seeds"]
        if repair_manifest["tasks"][task]["final_bank"]["seeds"] != seeds or len(seeds) != 50:
            raise RuntimeError(f"manifest final-bank mismatch: {task}")
        native_root = args.benchmark_root / "runs" / args.repair_source / task / "native_1x" / "final"
        native = records(native_root, seeds)
        methods = []
        for method, display, source_kind in METHODS:
            source = args.base_source if source_kind == "base" else args.repair_source
            root = args.benchmark_root / "runs" / source / task / method / "final"
            value = {"method": method, "display_name": display, **summarize(records(root, seeds), native)}
            methods.append(value)
        strider_source = report["strider_sources"][task]
        strider_result = load(args.strider_root / "runs" / strider_source / task / "RESULT.json")
        if strider_result.get("final_rollouts") != 50 or strider_result.get("native_rollouts_reexecuted") != 0:
            raise RuntimeError(f"invalid STRIDER accounting: {task}")
        methods.append({
            "method": "strider",
            "display_name": "STRIDER",
            **strider_result["final"],
            "selected_schedule": strider_result["selected_schedule"],
            "search_rollouts": strider_result["search"]["episodes_used"],
        })
        report["tasks"][task] = {"final_seeds": seeds, "methods": methods}

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "RESULTS.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "RESULTS.md").write_text(markdown(report))
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
