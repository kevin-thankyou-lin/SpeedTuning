#!/usr/bin/env python3
"""Repair paired-uniform aggregate wording without executing rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


BALANCED = [2.5, 1.5, 3.5, 3.5]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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


def repaired_comparison(items: list[dict]) -> dict:
    eligible = [item for item in items if clean_qualified(item)]
    selected = None if not eligible else max(eligible, key=ranking_key)
    qualified_uniforms = [
        item for item in eligible if len(set(item["schedule"])) == 1
    ]
    best_uniform = (
        None
        if not qualified_uniforms
        else max(qualified_uniforms, key=ranking_key)
    )
    balanced = next(item for item in items if item["schedule"] == BALANCED)
    strict = None
    if best_uniform is not None and clean_qualified(balanced):
        strict = (
            balanced["successes"] >= best_uniform["successes"]
            and balanced["matched_native_speedup"] >= best_uniform["matched_native_speedup"]
            and (
                balanced["successes"] > best_uniform["successes"]
                or balanced["matched_native_speedup"] > best_uniform["matched_native_speedup"]
            )
        )
    wins_gate = clean_qualified(balanced) and not qualified_uniforms
    if best_uniform is not None:
        wins_gate = bool(strict)
    return {
        "selection_rule": "clean qualified; exact successes, speedup, simplicity",
        "selected": selected,
        "best_uniform": best_uniform,
        "balanced_adaptive": balanced,
        "balanced_strictly_beats_best_uniform": strict,
        "balanced_wins_registered_gate_over_all_uniforms": wins_gate,
        "pareto_frontier": pareto_frontier(items),
    }


def validate_source(source: dict) -> None:
    if source.get("schema") != "act-vlm-paired-uniform-pick-result-v1":
        raise RuntimeError("source result schema mismatch")
    expected = {
        (2.0, 2.0, 2.0, 2.0): (20, 17, "rejected_at_20"),
        (2.5, 2.5, 2.5, 2.5): (10, 7, "rejected_at_10"),
        (2.5, 1.5, 3.5, 3.5): (20, 18, "qualified"),
        (3.0, 1.5, 4.0, 4.0): (20, 17, "rejected_at_20"),
    }
    actual = {
        tuple(item["schedule"]): (
            item["completed"], item["successes"], item["verdict"]
        )
        for item in source.get("candidates", [])
    }
    if actual != expected:
        raise RuntimeError("source candidate table mismatch")
    if source.get("new_candidate_rollouts") != 30:
        raise RuntimeError("source rollout accounting mismatch")
    if source.get("new_native_rollouts") != 0 or source.get("final_bank_opened"):
        raise RuntimeError("source isolation contract mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    source_path = args.source_result.resolve()
    source = json.loads(source_path.read_text())
    validate_source(source)
    root = args.output_root.resolve()
    if root.exists():
        raise RuntimeError("report output root already exists")
    root.mkdir(parents=True)
    result = {
        **source,
        "schema": "act-vlm-paired-uniform-pick-result-v2",
        "comparison": repaired_comparison(source["candidates"]),
        "report_repair": {
            "source_result": str(source_path),
            "source_result_sha256": sha256(source_path),
            "source_commit": args.source_commit,
            "new_rollouts": 0,
        },
    }
    write_json(root / "RESULT.json", result)
    write_json(root / "COMPLETE.json", {
        "schema": "act-vlm-paired-uniform-report-completion-v2",
        "source_result_sha256": sha256(source_path),
        "result_sha256": sha256(root / "RESULT.json"),
        "new_rollouts": 0,
        "final_bank_opened": False,
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

