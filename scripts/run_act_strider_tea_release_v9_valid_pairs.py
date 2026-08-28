#!/usr/bin/env python3
"""Evaluate uniform and delayed Tea release on 50 simulator-valid pairs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from scripts import run_act_strider_frontier_v4 as v4
from scripts import run_act_strider_tea_release_v9 as v9
from scripts import run_act_strider_tea_volume_v5 as tea


def checked_record(path: Path, schedule: list[float], seed: int) -> dict:
    record = json.loads(path.read_text())
    if int(record.get("seed", -1)) != seed:
        raise RuntimeError(f"seed identity mismatch: {path}")
    if list(map(float, record.get("schedule", ()))) != schedule:
        raise RuntimeError(f"schedule identity mismatch: {path}")
    return record


def copy_video(source: Path, destination: Path) -> tuple[str, int]:
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError(f"missing source video: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if v4.file_sha256(destination) != v4.file_sha256(source):
        raise RuntimeError(f"copied video hash mismatch: {destination}")
    return v4.file_sha256(destination), destination.stat().st_size


def method_paths(root: Path, name: str, seed: int) -> tuple[Path, Path]:
    method_root = root / "attempts" / name
    return method_root / "states" / f"{seed}.json", method_root / "videos" / f"{seed}.mp4"


def load_or_run(
    *,
    runtime,
    root: Path,
    method: str,
    schedule: list[float],
    seed: int,
    source_record: Path | None = None,
    source_video: Path | None = None,
) -> tuple[dict, bool]:
    record_path, video_path = method_paths(root, method, seed)
    if record_path.exists():
        record = checked_record(record_path, schedule, seed)
        if "physics_error" not in record:
            if not video_path.is_file() or video_path.stat().st_size <= 0:
                raise RuntimeError(f"valid receipt lacks video: {record_path}")
            if record.get("video_sha256") != v4.file_sha256(video_path):
                raise RuntimeError(f"video identity mismatch: {video_path}")
        return record, False

    if source_record is not None:
        if source_video is None:
            raise RuntimeError("source record requires a source video")
        record = checked_record(source_record, schedule, seed)
        if "physics_error" in record:
            raise RuntimeError("physics-error cache imports are not valid pairs")
        video_hash, video_bytes = copy_video(source_video, video_path)
        record = {**record, "video_path": str(video_path), "video_sha256": video_hash, "video_bytes": video_bytes, "cache_source_record": str(source_record)}
        v4.write_json(record_path, record)
        return record, False

    if video_path.exists() and not record_path.exists():
        raise RuntimeError(f"unreceipted video requires audit: {video_path}")
    record = runtime.rollout(schedule, seed, video_path=video_path)
    if list(map(float, record.get("schedule", ()))) != schedule:
        raise RuntimeError("runtime returned a different schedule")
    if "physics_error" in record:
        record = {**record, "simulator_invalid": True, "video_missing_allowed_for_physics_error": not video_path.is_file()}
    else:
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError(f"valid rollout lacks video: {video_path}")
        record = {**record, "video_sha256": v4.file_sha256(video_path), "video_bytes": video_path.stat().st_size}
    v4.write_json(record_path, record)
    return record, True


def parent_method_root(parent_root: Path, schedule: list[float]) -> Path:
    return parent_root / "final" / "controllers" / v4.schedule_sha256(schedule)


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--aborted-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--replacement-contract", type=Path, required=True)
    parser.add_argument("--success-criterion", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    contract = json.loads(args.replacement_contract.read_text())
    if contract.get("schema") != "act-strider-tea-physics-error-replacement-v1":
        raise ValueError("unexpected replacement contract")
    if contract.get("replacement_outcomes_visible_when_rule_frozen") is not False:
        raise ValueError("replacement rule must be frozen before reserve outcomes")
    if v4.file_sha256(args.parent_root / "RESULT.json") != contract["parent_result_sha256"]:
        raise RuntimeError("parent result hash mismatch")
    if v4.file_sha256(args.aborted_root / "IDENTITY.json") != contract["aborted_identity_sha256"]:
        raise RuntimeError("aborted identity hash mismatch")
    if v4.file_sha256(args.aborted_root / "run.log") != contract["aborted_log_sha256"]:
        raise RuntimeError("aborted incident log hash mismatch")

    uniform = list(v4.validate_schedule(contract["uniform_schedule"]))
    delayed = list(v4.validate_schedule(contract["delayed_release_schedule"]))
    if uniform != v9.UNIFORM or delayed != v9.DELAYED_RELEASE:
        raise RuntimeError("replacement schedules changed")
    primary = list(range(contract["primary_seed_start"], contract["primary_seed_start"] + contract["primary_seed_count"]))
    reserve = list(range(contract["reserve_seed_start"], contract["reserve_seed_start"] + contract["reserve_seed_count"]))
    excluded_registered = set(map(int, contract["excluded_primary_seeds"]))
    if excluded_registered != {int(contract["aborted_physics_error_seed"])}:
        raise RuntimeError("registered simulator-invalid seed mismatch")

    tea.SUCCESS_CRITERION_SCHEMA = "tea-cup-center-success-v1"
    criterion = tea.checked_success_criterion(args.success_criterion)
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
        critical_source_overrides={"sim_tasks.py": criterion["files"]["sim_tasks.py"]["sha256"]},
    )
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        **runtime.identity(),
        "schema": "act-strider-tea-valid-pairs-identity-v1",
        "method": "matched_valid_pairs_with_simulator_error_replacement",
        "replacement_contract_sha256": v4.file_sha256(args.replacement_contract),
        "target_valid_pairs": int(contract["target_valid_pairs"]),
        "primary_seeds": primary,
        "reserve_seeds": reserve,
        "registered_excluded_seeds": sorted(excluded_registered),
        "uniform_schedule": uniform,
        "delayed_release_schedule": delayed,
    }
    identity_path = root / "IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("valid-pairs root identity mismatch")
    v4.write_json(identity_path, identity)

    parent_uniform_root = parent_method_root(args.parent_root, uniform)
    aborted_delayed_root = args.aborted_root / "final" / "controllers" / v4.schedule_sha256(delayed)
    cached_delayed = set(map(int, contract["prior_valid_delayed_cache_seeds"]))
    valid_uniform: list[dict] = []
    valid_delayed: list[dict] = []
    valid_seeds: list[int] = []
    invalid_pairs = [{
        "seed": int(contract["aborted_physics_error_seed"]),
        "reason": "physics_error",
        "source": "aborted_incident_log",
        "log_sha256": contract["aborted_log_sha256"],
        "counted_in_valid_denominator": False,
    }]
    cache_hits = 0
    new_rollouts = 0
    candidates = [seed for seed in primary if seed not in excluded_registered] + reserve
    for seed in candidates:
        if len(valid_seeds) >= int(contract["target_valid_pairs"]):
            break
        uniform_source_record = uniform_source_video = None
        if seed in primary:
            uniform_source_record = parent_uniform_root / "states" / f"{seed}.json"
            uniform_source_video = parent_uniform_root / "videos" / f"{seed}.mp4"
        uniform_record, ran = load_or_run(
            runtime=runtime, root=root, method="uniform_1p5x", schedule=uniform,
            seed=seed, source_record=uniform_source_record, source_video=uniform_source_video,
        )
        new_rollouts += int(ran)
        cache_hits += int(not ran)

        delayed_source_record = delayed_source_video = None
        if seed in cached_delayed:
            delayed_source_record = aborted_delayed_root / "states" / f"{seed}.json"
            delayed_source_video = aborted_delayed_root / "videos" / f"{seed}.mp4"
        delayed_record, ran = load_or_run(
            runtime=runtime, root=root, method="delayed_release", schedule=delayed,
            seed=seed, source_record=delayed_source_record, source_video=delayed_source_video,
        )
        new_rollouts += int(ran)
        cache_hits += int(not ran)

        physics = []
        for name, record in (("uniform_1p5x", uniform_record), ("delayed_release", delayed_record)):
            if "physics_error" in record:
                physics.append({"method": name, "physics_error": record["physics_error"]})
        if physics:
            invalid_pairs.append({"seed": seed, "reason": "physics_error", "details": physics, "counted_in_valid_denominator": False})
            continue
        valid_seeds.append(seed)
        valid_uniform.append(uniform_record)
        valid_delayed.append(delayed_record)

    if len(valid_seeds) != int(contract["target_valid_pairs"]):
        raise RuntimeError("reserve bank exhausted before 50 simulator-valid pairs")
    uniform_summary = v4.summarize(valid_uniform)
    delayed_summary = v4.summarize(valid_delayed)
    parent = json.loads((args.parent_root / "RESULT.json").read_text())
    native = parent["final"]["methods"]["native_1x"]["summary"]
    for summary in (uniform_summary, delayed_summary):
        summary["successful_rollout_speedup"] = native["successful_mean_first_success_steps"] / summary["successful_mean_first_success_steps"]
        summary["throughput_delta_percent_vs_native"] = 100.0 * (summary["achieved_throughput_per_step"] / native["achieved_throughput_per_step"] - 1.0)
    result = {
        "schema": "act-strider-tea-valid-pairs-result-v1",
        "identity_sha256": v4.file_sha256(identity_path),
        "valid_pair_seeds": valid_seeds,
        "invalid_pairs": invalid_pairs,
        "uniform_1p5x": uniform_summary,
        "delayed_release": delayed_summary,
        "delayed_success_delta": delayed_summary["successes"] - uniform_summary["successes"],
        "delayed_throughput_delta_percent_points_vs_uniform": delayed_summary["throughput_delta_percent_vs_native"] - uniform_summary["throughput_delta_percent_vs_native"],
        "accounting": {
            "valid_pairs": len(valid_seeds),
            "simulator_invalid_pairs": len(invalid_pairs),
            "new_rollouts": new_rollouts,
            "cache_hits": cache_hits,
            "prior_unreceipted_physics_error_attempts": 1,
            "all_attempts_included_in_physical_cost": True,
        },
    }
    result_path = root / "RESULT.json"
    v4.write_json(result_path, result)
    v4.write_json(root / "COMPLETE.json", {"schema": "act-strider-tea-valid-pairs-completion-v1", "identity_sha256": v4.file_sha256(identity_path), "result_sha256": v4.file_sha256(result_path), **result["accounting"]})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
