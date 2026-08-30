#!/usr/bin/env python3
"""Train fresh SAIL-inspired phase priors from charged native ACT rollouts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SPEEDTUNING_SPEED_VALUES", "1,1.5,2,2.5,3")

from act_speed_benchmark import canonical_sha256, sha256  # noqa: E402
from scripts.act_vlm_frontier_server import ACTFrontierRuntime, git_head  # noqa: E402
from scripts.run_act_speed_benchmark_cell import atomic_json, immutable_json  # noqa: E402
from scripts.run_act_sail_warmstart_v33 import PHASES  # noqa: E402


TASKS = ("pick", "tea", "insertion")
NATIVE_SCHEDULE = [1.0, 1.0, 1.0, 1.0]
PRIOR_ROLLOUTS_PER_TASK = 20
CURVATURE_QUANTILE = 0.75
GRIPPER_DELTA_THRESHOLD = 0.01
SPEED_THRESHOLDS = (
    (0.50, 1.5),
    (0.30, 2.0),
    (0.15, 2.5),
    (0.00, 3.0),
)


def checked_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing required JSON: {path}")
    return json.loads(path.read_text())


def bank_seeds(value) -> set[int]:
    """Expand seed-bank JSON while ignoring scalar metadata integers."""

    if isinstance(value, dict):
        if set(("start", "count")) <= set(value):
            start = int(value["start"])
            count = int(value["count"])
            return set(range(start, start + count))
        result: set[int] = set()
        for child in value.values():
            result |= bank_seeds(child)
        return result
    if isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return set(map(int, value))
    if isinstance(value, list):
        result: set[int] = set()
        for child in value:
            result |= bank_seeds(child)
        return result
    return set()


def validate_prior_banks(path: Path) -> dict[str, list[int]]:
    spec = checked_json(path)
    if spec.get("rollouts_per_task") != PRIOR_ROLLOUTS_PER_TASK:
        raise RuntimeError("v33 prior bank must allocate exactly 20 rollouts/task")
    task_seeds = {}
    all_current: set[int] = set()
    for task in TASKS:
        entry = spec["tasks"][task]
        seeds = list(range(int(entry["start"]), int(entry["start"]) + int(entry["count"])))
        if len(seeds) != PRIOR_ROLLOUTS_PER_TASK or len(set(seeds)) != len(seeds):
            raise RuntimeError(f"invalid v33 prior seeds for {task}")
        if all_current & set(seeds):
            raise RuntimeError("v33 prior task banks overlap")
        task_seeds[task] = seeds
        all_current |= set(seeds)

    historical: set[int] = set()
    for other in sorted((REPO_ROOT / "experiments").glob("**/BANKS.json")):
        if other.resolve() == path.resolve():
            continue
        historical |= bank_seeds(checked_json(other))
    overlap = all_current & historical
    if overlap:
        raise RuntimeError(f"v33 prior bank overlaps historical seeds: {sorted(overlap)[:5]}")
    return task_seeds


def load_contiguous_states(directory: Path, seeds: list[int], identity: str) -> list[dict]:
    records = []
    missing = False
    for seed in seeds:
        path = directory / f"{seed}.json"
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError("v33 prior states contain a non-contiguous suffix")
        value = checked_json(path)
        if value.get("seed") != seed or value.get("identity_sha256") != identity:
            raise RuntimeError(f"v33 prior state identity mismatch: {path}")
        records.append(value)
    return records


def _schedule_from_importance(value: float) -> float:
    for threshold, speed in SPEED_THRESHOLDS:
        if float(value) >= threshold:
            return float(speed)
    raise AssertionError("unreachable speed threshold")


def train_phase_precision_prior(records: list[dict]) -> dict:
    """Fit a phase-only Bernoulli precision head from causal robot motion."""

    if len(records) != PRIOR_ROLLOUTS_PER_TASK:
        raise RuntimeError("phase prior requires exactly 20 completed native rollouts")
    if any(not item.get("success") for item in records):
        raise RuntimeError("native 1x prior bank is unreliable; refusing to train")
    if any(item.get("physics_error") is not None for item in records):
        raise RuntimeError("physics error in native prior bank")
    if any(item.get("safety_violation") is not None for item in records):
        raise RuntimeError("safety violation in native prior bank")

    trajectories = []
    all_qpos = []
    for record in records:
        success_step = record.get("first_success_step")
        telemetry = [
            item
            for item in record.get("attribution_telemetry", [])
            if success_step is None or int(item["physics_step"]) <= int(success_step)
        ]
        if not telemetry:
            raise RuntimeError("native prior rollout lacks attribution telemetry")
        qpos = np.asarray([item["robot_qpos"] for item in telemetry], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != 14:
            raise RuntimeError("native prior qpos telemetry must have shape (T, 14)")
        phases = [str(item["observed_phase"]) for item in telemetry]
        if any(phase not in PHASES for phase in phases):
            raise RuntimeError("native prior telemetry contains an unknown phase")
        trajectories.append((qpos, phases))
        all_qpos.append(qpos)

    stacked = np.concatenate(all_qpos, axis=0)
    qpos_scale = np.maximum(np.std(stacked, axis=0), 1e-6)
    prepared = []
    all_curvature = []
    for qpos, phases in trajectories:
        normalized = qpos / qpos_scale
        velocity = np.diff(normalized, axis=0, prepend=normalized[:1])
        curvature = np.linalg.norm(
            np.diff(velocity, axis=0, prepend=velocity[:1]), axis=1
        )
        gripper_delta = np.max(
            np.abs(np.diff(qpos[:, (6, 13)], axis=0, prepend=qpos[:1, (6, 13)])),
            axis=1,
        )
        prepared.append((phases, curvature, gripper_delta))
        all_curvature.extend(float(value) for value in curvature[2:])
    if not all_curvature:
        raise RuntimeError("insufficient native trajectory length for prior training")
    curvature_threshold = float(np.quantile(all_curvature, CURVATURE_QUANTILE))

    samples = np.zeros(len(PHASES), dtype=np.int64)
    positives = np.zeros(len(PHASES), dtype=np.int64)
    for phases, curvature, gripper_delta in prepared:
        labels = (curvature >= curvature_threshold) | (
            gripper_delta >= GRIPPER_DELTA_THRESHOLD
        )
        for phase, label in zip(phases, labels, strict=True):
            index = PHASES.index(phase)
            samples[index] += 1
            positives[index] += int(label)
    if np.any(samples == 0):
        raise RuntimeError("native prior bank did not observe every phase")

    # Beta(1, 1) posterior means are the deterministic fitted phase head.
    importance = (positives.astype(np.float64) + 1.0) / (
        samples.astype(np.float64) + 2.0
    )
    logits = np.log(importance / (1.0 - importance))
    labels_total = int(np.sum(positives))
    samples_total = int(np.sum(samples))
    bce = -float(
        np.sum(
            positives * np.log(importance)
            + (samples - positives) * np.log(1.0 - importance)
        )
        / samples_total
    )
    schedule = [_schedule_from_importance(value) for value in importance]
    result = {
        "schema": "act-new-sail-inspired-phase-precision-prior-v33",
        "prior_kind": "new_sail_inspired_native_precision_head",
        "paper_faithful_sail": False,
        "runtime_privileged_signals": False,
        "training_source": "fresh_disjoint_native_1x_simulation_rollouts",
        "phase_order": list(PHASES),
        "training_rollouts": len(records),
        "precision_target": {
            "definition": (
                "pre_success_top_quartile_normalized_qpos_curvature_"
                "or_gripper_transition"
            ),
            "curvature_quantile": CURVATURE_QUANTILE,
            "curvature_threshold": curvature_threshold,
            "gripper_delta_threshold": GRIPPER_DELTA_THRESHOLD,
        },
        "model": {
            "family": "beta_smoothed_phase_bernoulli_precision_head",
            "alpha": 1.0,
            "beta": 1.0,
            "phase_samples": samples.tolist(),
            "phase_positive_labels": positives.tolist(),
            "phase_logits": [float(value) for value in logits],
            "training_binary_cross_entropy": bce,
            "positive_labels": labels_total,
            "samples": samples_total,
            "qpos_scale": [float(value) for value in qpos_scale],
        },
        "phase_importance": [float(value) for value in importance],
        "speed_mapping": [
            {"minimum_importance": float(threshold), "speed": float(speed)}
            for threshold, speed in SPEED_THRESHOLDS
        ],
        "schedule": schedule,
    }
    result["prior_payload_sha256"] = canonical_sha256(result)
    return result


def train_task(
    runtime: ACTFrontierRuntime,
    root: Path,
    task: str,
    seeds: list[int],
    prior_banks: Path,
) -> dict:
    output = root / task
    identity = {
        **runtime.identity(),
        "schema": "act-new-sail-prior-training-identity-v33",
        "task_label": task,
        "schedule": NATIVE_SCHEDULE,
        "seed_bank": {"seeds": seeds, "sha256": canonical_sha256(seeds)},
        "prior_banks_sha256": sha256(prior_banks),
        "offline_training_rollouts": PRIOR_ROLLOUTS_PER_TASK,
        "online_search_or_final_bank_opened": False,
        "historical_rollouts_reexecuted": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    identity_path = output / "IDENTITY.json"
    if identity_path.exists():
        if checked_json(identity_path) != identity:
            raise RuntimeError(f"existing v33 prior identity differs: {identity_path}")
    else:
        immutable_json(identity_path, identity)

    records = load_contiguous_states(output / "states", seeds, identity["identity_sha256"])
    for seed in seeds[len(records) :]:
        record = runtime.rollout(
            NATIVE_SCHEDULE,
            seed,
            record_attribution_telemetry=True,
        )
        if record.get("seed") != seed or list(record.get("schedule", ())) != NATIVE_SCHEDULE:
            raise RuntimeError("v33 prior runtime returned a different rollout identity")
        record["identity_sha256"] = identity["identity_sha256"]
        immutable_json(output / "states" / f"{seed}.json", record)
        records.append(record)
        progress = {
            "task": task,
            "completed": len(records),
            "successes": sum(bool(item.get("success")) for item in records),
            "physics_errors": sum(item.get("physics_error") is not None for item in records),
            "safety_violations": sum(item.get("safety_violation") is not None for item in records),
        }
        atomic_json(output / "progress.json", progress)
        print(json.dumps({"stage": "prior_training", **progress}), flush=True)
        if record.get("physics_error") is not None:
            raise RuntimeError("physics error in v33 prior training; receipt preserved")

    prior = train_phase_precision_prior(records)
    state_receipts = [
        {"seed": seed, "sha256": sha256(output / "states" / f"{seed}.json")}
        for seed in seeds
    ]
    result = {
        "schema": "act-new-sail-prior-task-result-v33",
        "task_label": task,
        "training_identity_sha256": sha256(identity_path),
        "state_receipts_sha256": canonical_sha256(state_receipts),
        "native_successes": sum(bool(item.get("success")) for item in records),
        "native_rollouts": len(records),
        "physics_errors": sum(item.get("physics_error") is not None for item in records),
        "safety_violations": sum(item.get("safety_violation") is not None for item in records),
        "phase_prior": prior,
    }
    result["result_payload_sha256"] = canonical_sha256(result)
    result_path = output / "RESULT.json"
    if result_path.exists():
        if checked_json(result_path) != result:
            raise RuntimeError(f"existing v33 prior result differs: {result_path}")
    else:
        immutable_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--prior-banks", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if git_head() != args.source_commit:
        raise RuntimeError("v33 prior checked-out source differs from requested commit")
    task_seeds = validate_prior_banks(args.prior_banks)
    root = args.root.resolve()
    tasks = {}
    for task in TASKS:
        runtime = ACTFrontierRuntime(
            source_commit=args.source_commit,
            run_manifest=args.run_manifest,
            task_label=task,
            detector_checkpoint=args.detector_checkpoint,
            detector_source=args.detector_source,
            device=args.device,
        )
        tasks[task] = train_task(runtime, root, task, task_seeds[task], args.prior_banks)

    bundle = {
        "schema": "act-new-sail-inspired-offline-priors-v33",
        "paper_faithful_sail": False,
        "label": "newly trained SAIL-inspired native precision prior",
        "offline_training_rollouts": PRIOR_ROLLOUTS_PER_TASK * len(TASKS),
        "offline_training_rollouts_per_task": PRIOR_ROLLOUTS_PER_TASK,
        "online_search_rollouts": 0,
        "final_bank_opened": False,
        "historical_rollouts_reexecuted": 0,
        "prior_banks_sha256": sha256(args.prior_banks),
        "tasks": tasks,
    }
    bundle["payload_sha256"] = canonical_sha256(bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if checked_json(args.output) != bundle:
            raise RuntimeError(f"existing v33 prior bundle differs: {args.output}")
    else:
        immutable_json(args.output, bundle)
    complete = {
        "schema": "act-new-sail-prior-training-completion-v33",
        "offline_training_rollouts": 60,
        "online_search_rollouts": 0,
        "final_bank_opened": False,
        "historical_rollouts_reexecuted": 0,
        "physics_errors": sum(item["physics_errors"] for item in tasks.values()),
        "safety_violations": sum(item["safety_violations"] for item in tasks.values()),
        "offline_priors_sha256": sha256(args.output),
        "offline_priors_payload_sha256": bundle["payload_sha256"],
    }
    complete_path = root / "COMPLETE.json"
    if complete_path.exists():
        if checked_json(complete_path) != complete:
            raise RuntimeError(f"existing v33 prior completion differs: {complete_path}")
    else:
        immutable_json(complete_path, complete)
    print("V33_NEW_SAIL_PRIORS=" + json.dumps(bundle, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
