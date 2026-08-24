#!/usr/bin/env python3
"""Fail-closed audit and report generator for the frozen ACT speed benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


METHODS = (
    "uniform_sweep",
    "learned_phase_subtask",
    "learned_phase_tabular_rl",
    "learned_phase_rainbow_rl",
    "awe_offline_proxy",
    "sail_inspired_adaptive",
)
PHASE_METHODS = {
    "learned_phase_subtask",
    "learned_phase_tabular_rl",
    "learned_phase_rainbow_rl",
}
DETECTOR_HASHES = {
    "checkpoint": "c25c3f530da42eb7c60e5f70405b3a99c56ab72c1e53dfd27055dc3d99c3512d",
    "inference": "1398e1d1b5b4e682f009c6501598e651a516341f6d60822f40fc575a40061815",
    "model_source": "8a47f110f19f4e52a39b7e0e4f2273c2895690f6332ab17a4b71c8eb5ce4ae37",
}
EXPECTED_PARITY = {"pick": 49, "tea": 50, "insertion": 49}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing required evidence: {path}")
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_identity(path: Path, *, task: str, method: str, stage: str, seeds: list[int]) -> dict:
    value = load(path)
    recorded = value.get("identity_sha256")
    payload = dict(value)
    payload.pop("identity_sha256", None)
    require(recorded == canonical_sha256(payload), f"identity payload hash mismatch: {path}")
    require(value.get("task_label") == task, f"identity task mismatch: {path}")
    require(value.get("method") == method, f"identity method mismatch: {path}")
    require(value.get("stage") == stage, f"identity stage mismatch: {path}")
    bank = value.get("seed_bank", {})
    require(bank.get("seeds") == seeds, f"identity seed list mismatch: {path}")
    require(bank.get("sha256") == canonical_sha256(seeds), f"identity seed hash mismatch: {path}")
    expected_detector = DETECTOR_HASHES if method in PHASE_METHODS else None
    require(value.get("detector") == expected_detector, f"detector boundary mismatch: {path}")
    controller = value.get("controller", {})
    require(controller.get("cameras") == ["angle", "left_wrist", "right_wrist"], f"camera receipt mismatch: {path}")
    require(controller.get("progress_clock") == "nominal_policy_time", f"progress-clock receipt mismatch: {path}")
    require(controller.get("per_physics_step_inference") is True, f"inference cadence mismatch: {path}")
    require(math.isclose(controller.get("temporal_ensemble_m", -1), 0.01), f"temporal-ensemble receipt mismatch: {path}")
    require(controller.get("physics_error_policy") == "count_as_failure_and_continue", f"physics policy mismatch: {path}")
    return value


def verify_states(root: Path, seeds: list[int], identity: str) -> tuple[list[dict], list[str]]:
    states = root / "states"
    paths = sorted(states.glob("*.json"))
    expected_names = {f"{seed}.json" for seed in seeds}
    require({path.name for path in paths} == expected_names, f"state receipt set mismatch: {states}")
    records = []
    hashes = []
    for seed in seeds:
        path = states / f"{seed}.json"
        value = load(path)
        require(value.get("seed") == seed, f"state seed mismatch: {path}")
        require(value.get("identity_sha256") == identity, f"state identity mismatch: {path}")
        records.append(value)
        hashes.append(sha256(path))
    require(len(set(hashes)) == len(hashes), f"duplicate byte-identical state receipts: {states}")
    return records, hashes


def verify_summary(root: Path, records: list[dict], *, schema: str, identity: str) -> dict:
    result_path = root / "result.json"
    result = load(result_path)
    marker = load(root / "COMPLETE.json")
    successes = sum(bool(record.get("success")) for record in records)
    safety = sum(record.get("safety_violation") is not None for record in records)
    physics = sum("physics_error" in record for record in records)
    successful_steps = [record["first_success_step"] for record in records if record.get("success")]
    successful_mean = sum(successful_steps) / len(successful_steps) if successful_steps else None
    require(result.get("schema") == schema, f"result schema mismatch: {result_path}")
    require(result.get("identity_sha256") == identity, f"result identity mismatch: {result_path}")
    require(result.get("episodes") == 50 and result.get("exact_budget_complete") is True, f"non-exact result budget: {result_path}")
    require(result.get("successes") == successes, f"success count mismatch: {result_path}")
    require(math.isclose(result.get("success_rate", -1), successes / 50), f"success rate mismatch: {result_path}")
    require(result.get("safety_violations") == safety, f"safety count mismatch: {result_path}")
    require(result.get("physics_errors") == physics, f"physics count mismatch: {result_path}")
    if successful_mean is None:
        require(result.get("successful_mean_first_success_steps") is None, f"successful mean mismatch: {result_path}")
    else:
        require(math.isclose(result.get("successful_mean_first_success_steps", -1), successful_mean), f"successful mean mismatch: {result_path}")
    require(marker.get("identity_sha256") == identity and marker.get("episodes") == 50, f"completion marker mismatch: {root}")
    require(marker.get("result_sha256") == sha256(result_path), f"completion result hash mismatch: {root}")
    return result


def verify_search(root: Path, *, task: str, method: str, seeds: list[int]) -> tuple[dict, dict, list[str]]:
    identity = verify_identity(root / "identity.json", task=task, method=method, stage="search", seeds=seeds)
    require((root / "preregistration.json").is_file(), f"missing preregistration: {root}")
    records, hashes = verify_states(root, seeds, identity["identity_sha256"])
    result = verify_summary(root, records, schema="act-speed-search-result-v1", identity=identity["identity_sha256"])
    selected = root / "selected.json"
    require(result.get("selected_path") == str(selected), f"selected path mismatch: {root}")
    require(result.get("selected_sha256") == sha256(selected), f"selected hash mismatch: {root}")
    marker = load(root / "COMPLETE.json")
    require(marker.get("selected_sha256") == sha256(selected), f"completion selected hash mismatch: {root}")
    return identity, result, hashes


def verify_final(root: Path, search_root: Path, *, task: str, method: str, seeds: list[int]) -> tuple[dict, dict, list[str]]:
    identity = verify_identity(root / "identity.json", task=task, method=method, stage="final", seeds=seeds)
    records, hashes = verify_states(root, seeds, identity["identity_sha256"])
    result = verify_summary(root, records, schema="act-speed-final-result-v1", identity=identity["identity_sha256"])
    selected = search_root / "selected.json"
    require(result.get("method_artifact_path") == str(selected), f"final artifact path mismatch: {root}")
    require(result.get("method_artifact_sha256") == sha256(selected), f"final artifact hash mismatch: {root}")
    return identity, result, hashes


def verify_manifest(path: Path, expected_source: str, expected_sha256: str, contract: dict) -> dict:
    require(sha256(path) == expected_sha256, f"run manifest file hash mismatch: {path}")
    value = load(path)
    require(value.get("source", {}).get("commit") == expected_source, f"run manifest source mismatch: {path}")
    require(value.get("contract", {}).get("payload") == contract, f"run manifest contract mismatch: {path}")
    gate = value.get("parity_gate", {})
    require(gate.get("passed") is True, f"parity gate not passed: {path}")
    for task, expected in EXPECTED_PARITY.items():
        receipt = gate.get("tasks", {}).get(task, {})
        require(receipt.get("successes") == expected and receipt.get("episodes") == 50, f"exact parity mismatch for {task}: {path}")
        for key in ("identity_path", "result_path", "completion_path"):
            require(Path(receipt[key]).is_file(), f"missing linked parity evidence: {receipt[key]}")
    return value


def verify_import(root: Path, origin: Path, seeds: list[int]) -> dict:
    marker = load(root / "IMPORT.json")
    require(marker.get("count") == 20 and marker.get("rollouts_reexecuted") == 0, f"invalid receipt import marker: {root}")
    require(Path(marker.get("origin_root", "")) == origin, f"receipt import origin mismatch: {root}")
    origin_identity = load(origin / "identity.json")["identity_sha256"]
    origin_paths = sorted((origin / "states").glob("*.json"))
    require({path.name for path in origin_paths} == {f"{seed}.json" for seed in seeds[:20]}, f"origin must contain exactly the imported 20 receipts: {origin}")
    for seed in seeds[:20]:
        source = origin / "states" / f"{seed}.json"
        destination = root / "states" / f"{seed}.json"
        source_value = load(source)
        destination_value = load(destination)
        imported = destination_value.get("imported_rollout_receipt", {})
        require(source_value.get("identity_sha256") == origin_identity, f"origin identity mismatch: {source}")
        require(imported.get("origin_path") == str(source), f"imported origin path mismatch: {destination}")
        require(imported.get("origin_sha256") == sha256(source), f"imported origin hash mismatch: {destination}")
        require(imported.get("rollout_reexecuted") is False, f"imported receipt marked reexecuted: {destination}")
    return marker


def render_markdown(report: dict) -> str:
    lines = [
        "# Frozen multiview ACT speed benchmark results",
        "",
        "The table reports the untouched 50-state final banks. `Native mean` and method `FSS mean` are success-only first-success physics steps; speedup is `native mean / method mean`. Each task uses one shared matched native reference.",
        "",
        "| Task | Method | Successes/50 | SR | FSS mean | Native mean | Speedup | Safety | Physics | Manifest | Method artifact | Per-state evidence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in report["results"]:
        lines.append(
            "| {task} | `{method}` | {successes}/50 | {success_rate:.2f} | {method_mean:.2f} | {native_mean:.2f} | {speedup:.3f}x | {safety_violations} | {physics_errors} | `{manifest_path}` | `{method_artifact_path}` | `{states_path}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Audit receipt",
            "",
            f"- Audit JSON: `{report['audit_path']}`",
            f"- Base source manifest SHA-256: `{report['manifests']['base']['sha256']}`",
            f"- Repair source manifest SHA-256: `{report['manifests']['repair']['sha256']}`",
            f"- Registered search rollouts: {report['totals']['search_rollouts']}",
            f"- Registered method final rollouts: {report['totals']['method_final_rollouts']}",
            f"- Shared native final rollouts: {report['totals']['native_rollouts']}",
            f"- Safety violations: {report['totals']['safety_violations']}",
            f"- Physics errors: {report['totals']['physics_errors']}",
            "",
            "`awe_offline_proxy` is the preregistered offline proxy, not full SAIL. `sail_inspired_adaptive` has its own executable preregistration and provenance and is not claimed as paper-faithful SAIL.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def audit(args) -> dict:
    contract = load(args.contract)
    require(tuple(contract.get("methods", ())) == METHODS, "contract method registry mismatch")
    base_manifest_path = args.benchmark_root / "attempts" / args.base_source / "run_manifest.json"
    repair_manifest_path = args.benchmark_root / "attempts" / args.repair_source / "run_manifest.json"
    base_manifest = verify_manifest(base_manifest_path, args.base_source, args.base_manifest_sha256, contract)
    repair_manifest = verify_manifest(repair_manifest_path, args.repair_source, args.repair_manifest_sha256, contract)
    results = []
    evidence = {}
    all_receipt_hashes = []
    native_results = {}
    totals = {"search_rollouts": 0, "method_final_rollouts": 0, "native_rollouts": 0, "safety_violations": 0, "physics_errors": 0}

    for task in contract["tasks"]:
        task_manifest = repair_manifest["tasks"][task]
        search_seeds = task_manifest["search_bank"]["seeds"]
        final_seeds = task_manifest["final_bank"]["seeds"]
        require(set(search_seeds).isdisjoint(final_seeds), f"search/final seed overlap for {task}")
        require(base_manifest["tasks"][task]["search_bank"] == task_manifest["search_bank"], f"source manifests disagree on search bank for {task}")
        require(base_manifest["tasks"][task]["final_bank"] == task_manifest["final_bank"], f"source manifests disagree on final bank for {task}")
        native_root = args.benchmark_root / "runs" / args.repair_source / task / "native_1x" / "final"
        native_identity = verify_identity(native_root / "identity.json", task=task, method="native_1x", stage="native", seeds=final_seeds)
        native_records, native_hashes = verify_states(native_root, final_seeds, native_identity["identity_sha256"])
        native = verify_summary(native_root, native_records, schema="act-speed-native-result-v1", identity=native_identity["identity_sha256"])
        native_results[task] = native
        totals["native_rollouts"] += 50
        totals["safety_violations"] += native["safety_violations"]
        totals["physics_errors"] += native["physics_errors"]
        all_receipt_hashes.extend(native_hashes)

        for method in METHODS:
            source = args.repair_source if method == "learned_phase_subtask" else args.base_source
            cell = args.benchmark_root / "runs" / source / task / method
            search_root = cell / "search"
            final_root = cell / "final"
            search_identity, search, search_hashes = verify_search(search_root, task=task, method=method, seeds=search_seeds)
            final_identity, final, final_hashes = verify_final(final_root, search_root, task=task, method=method, seeds=final_seeds)
            require(search["policy_checkpoint_sha256"] == final["policy_checkpoint_sha256"], f"policy checkpoint changed between search/final: {task}/{method}")
            require(set(search_hashes).isdisjoint(final_hashes), f"duplicate receipt bytes across search/final: {task}/{method}")
            if method == "learned_phase_subtask":
                origin = args.benchmark_root / "runs" / args.base_source / task / method / "search"
                verify_import(search_root, origin, search_seeds)
            method_mean = final["successful_mean_first_success_steps"]
            native_mean = native["successful_mean_first_success_steps"]
            require(method_mean is not None and native_mean is not None, f"undefined success-only speedup: {task}/{method}")
            row = {
                "task": task,
                "method": method,
                "successes": final["successes"],
                "success_rate": final["success_rate"],
                "method_mean": method_mean,
                "native_mean": native_mean,
                "speedup": native_mean / method_mean,
                "safety_violations": final["safety_violations"],
                "physics_errors": final["physics_errors"],
                "manifest_path": final["manifest_path"],
                "method_artifact_path": final["method_artifact_path"],
                "method_artifact_sha256": final["method_artifact_sha256"],
                "states_path": final["states_path"],
                "search_states_path": search["states_path"],
                "policy_checkpoint_sha256": final["policy_checkpoint_sha256"],
                "search_identity_sha256": search_identity["identity_sha256"],
                "final_identity_sha256": final_identity["identity_sha256"],
            }
            results.append(row)
            evidence[f"{task}/{method}"] = {"search_root": str(search_root), "final_root": str(final_root)}
            totals["search_rollouts"] += 50
            totals["method_final_rollouts"] += 50
            totals["safety_violations"] += final["safety_violations"]
            totals["physics_errors"] += final["physics_errors"]
            all_receipt_hashes.extend(search_hashes + final_hashes)

    require(len(all_receipt_hashes) == 1950, "unexpected total receipt count")
    require(len(set(all_receipt_hashes)) == len(all_receipt_hashes), "byte-identical duplicate receipts across registered benchmark")
    report = {
        "schema": "act-speed-benchmark-audit-v1",
        "passed": True,
        "contract_sha256": sha256(args.contract),
        "base_source": args.base_source,
        "repair_source": args.repair_source,
        "manifests": {
            "base": {"path": str(base_manifest_path), "sha256": args.base_manifest_sha256},
            "repair": {"path": str(repair_manifest_path), "sha256": args.repair_manifest_sha256},
        },
        "totals": totals,
        "results": results,
        "native_results": native_results,
        "evidence": evidence,
        "receipt_set_sha256": canonical_sha256(all_receipt_hashes),
        "audit_path": str(args.output / "audit.json"),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--base-source", required=True)
    parser.add_argument("--repair-source", required=True)
    parser.add_argument("--base-manifest-sha256", required=True)
    parser.add_argument("--repair-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args)
    atomic_write(args.output / "audit.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_write(args.output / "RESULTS.md", render_markdown(report))
    print(json.dumps({"passed": True, "audit_path": report["audit_path"], "totals": report["totals"], "results": report["results"]}, sort_keys=True), flush=True)
    print("ACT_SPEED_BENCHMARK_AUDIT_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
