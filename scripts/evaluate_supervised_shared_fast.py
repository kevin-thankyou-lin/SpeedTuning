#!/usr/bin/env python3
"""Evaluate one shared fast speed with a learned binary protection veto."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy_speed_env import create_speed_env  # noqa: E402
from scripts.capture_phase_dataset import OfflinePhaseOracle, snapshot  # noqa: E402
from scripts.evaluate_reference_aligned_schedule import EncoderClient  # noqa: E402
from scripts.extract_scripted_action_chunks import chunk_feature  # noqa: E402
from supervised_phase_controller import (  # noqa: E402
    CausalTemporalFeatureBuffer,
    ConservativeBinaryDecoder,
    PortableStandardizedLogisticRegression,
    compose_online_features,
    mapped_protected_speed,
    shared_fast_speed,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_env(task: str, seed: int, *, render_images: bool):
    environment_task = "tea_bag" if task == "tea_bag_randomized" else task
    return create_speed_env(
        task_name=environment_task,
        seed=seed,
        render_images=render_images,
        randomize_object_pose=task == "tea_bag_randomized",
        speed_values=(1.0,),
        terminate_on_success=False,
    )


def run_native(task: str, seed: int) -> dict:
    env = make_env(task, seed, render_images=False)
    try:
        env.reset()
        done = False
        info = {"success": False, "task_reward": 0.0}
        while not done:
            _, _, done, info = env.step(1.0, quantized=False)
        return {
            "arm": "native_1x",
            "seed": seed,
            "success": bool(info["success"]),
            "physics_steps": int(env.physics_steps),
            "policy_time": float(env.policy_time),
            "task_reward": float(info["task_reward"]),
            "mean_speed": float(np.mean(env.speed_list)),
        }
    finally:
        env.close()


def run_candidate(
    task: str,
    seed: int,
    *,
    model,
    method: str,
    decoder_config: dict,
    encoder,
    offsets: tuple[int, ...],
    fast_speed: float,
    protected_speed: float,
    protected_speed_map: dict[str, float],
    cadence: int,
    oracle_config: dict,
) -> dict:
    uses_visual = method in ("visual", "fused")
    env = make_env(task, seed, render_images=uses_visual)
    try:
        env.reset()
        policy = env.action_source.policy
        policy.generate_trajectory(env.cur_ts)
        decoder = ConservativeBinaryDecoder(
            model.classes_,
            risk_threshold=float(decoder_config["risk_threshold"]),
            exit_threshold=float(decoder_config["exit_threshold"]),
            exit_stability=int(decoder_config["exit_stability"]),
        )
        visual_buffer = CausalTemporalFeatureBuffer()
        action_buffer = CausalTemporalFeatureBuffer()
        oracle = OfflinePhaseOracle(oracle_config)
        task_reward = float(env.cur_ts.reward or 0.0)
        oracle_label = oracle.label(snapshot(env, task_reward))
        done = False
        info = {"success": False, "task_reward": task_reward}
        trace = []
        while not done:
            visual = (
                encoder.encode(env.cur_ts.observation["images"]["angle"])
                if uses_visual
                else np.empty(0, dtype=np.float32)
            )
            action = chunk_feature(policy, float(env.policy_time), offsets)
            features = compose_online_features(
                method,
                visual,
                action,
                visual_buffer,
                action_buffer,
            )
            probabilities = model.predict_proba(features.reshape(1, -1))[0]
            label = decoder.update(probabilities)
            speed = mapped_protected_speed(
                label,
                fast_speed=fast_speed,
                default_protected_speed=protected_speed,
                protected_speed_map=protected_speed_map,
            )
            decision = {
                "physics_steps": int(env.physics_steps),
                "policy_time": float(env.policy_time),
                "task_reward": task_reward,
                "prediction": label,
                "oracle_label": oracle_label,
                "speed": speed,
                "probabilities": {
                    str(name): float(value)
                    for name, value in zip(model.classes_, probabilities)
                },
                "decoder": decoder.state(),
            }
            executed = 0
            for _ in range(cadence):
                _, _, done, info = env.step(speed, quantized=False)
                executed += 1
                task_reward = float(info["task_reward"])
                oracle_label = oracle.label(snapshot(env, task_reward))
                if done:
                    break
            decision["executed_physics_steps"] = executed
            trace.append(decision)

        true_risk = np.asarray([item["oracle_label"] != "fast" for item in trace])
        pred_risk = np.asarray([item["prediction"] != "fast" for item in trace])
        protected_recall = (
            None if not np.any(true_risk) else float(np.mean(pred_risk[true_risk]))
        )
        false_fast_rate = (
            None if not np.any(true_risk) else float(np.mean(~pred_risk[true_risk]))
        )
        false_slow_rate = (
            None if np.all(true_risk) else float(np.mean(pred_risk[~true_risk]))
        )
        return {
            "arm": "learned_shared_fast",
            "seed": seed,
            "success": bool(info["success"]),
            "physics_steps": int(env.physics_steps),
            "policy_time": float(env.policy_time),
            "task_reward": task_reward,
            "mean_speed": float(np.mean(env.speed_list)),
            "decision_frames": len(trace),
            "oracle_metrics_without_preentry_margin": {
                "protected_frames": int(np.sum(true_risk)),
                "protected_recall": protected_recall,
                "false_fast_rate": false_fast_rate,
                "false_slow_rate": false_slow_rate,
            },
            "trace": trace,
        }
    finally:
        env.close()


def summarize(native: list[dict], candidate: list[dict]) -> dict:
    native_success_steps = [item["physics_steps"] for item in native if item["success"]]
    native_mean = (
        None if not native_success_steps else float(np.mean(native_success_steps))
    )
    candidate_mean = float(np.mean([item["physics_steps"] for item in candidate]))
    return {
        "native_1x_success_rate": float(np.mean([item["success"] for item in native])),
        "candidate_success_rate": float(
            np.mean([item["success"] for item in candidate])
        ),
        "native_successful_mean_physics_steps": native_mean,
        "candidate_all_mean_physics_steps": candidate_mean,
        "duration_normalized_speedup": (
            None if native_mean is None else native_mean / candidate_mean
        ),
        "candidate_mean_executed_speed": float(
            np.mean([item["mean_speed"] for item in candidate])
        ),
        "new_rollouts": len(native) + len(candidate),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--portable-model", type=Path, required=True)
    parser.add_argument("--portable-receipt", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--action-receipt", type=Path, required=True)
    parser.add_argument("--method", choices=("visual", "action", "fused"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--fast-speed", type=float, default=2.0)
    parser.add_argument("--protected-speed", type=float, default=1.0)
    parser.add_argument(
        "--protected-speed-override",
        action="append",
        default=[],
        metavar="LABEL=SPEED",
    )
    parser.add_argument("--cadence", type=int, default=5)
    parser.add_argument("--encoder-socket", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.cadence <= 0:
        parser.error("cadence must be positive")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite a prior attempt")
    args.protected_speed_map = {}
    for item in args.protected_speed_override:
        label, separator, value = item.partition("=")
        if not separator or label in args.protected_speed_map:
            parser.error("protected speed overrides must be unique LABEL=SPEED values")
        try:
            speed = float(value)
        except ValueError:
            parser.error(f"invalid protected speed override: {item}")
        if not np.isfinite(speed) or speed <= 0:
            parser.error("protected speed overrides must be finite and positive")
        args.protected_speed_map[label] = speed
    return args


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    model_dir = args.model_dir.resolve()
    training_result_path = model_dir / "results.json"
    training_result = json.loads(training_result_path.read_text())
    manifest_path = args.dataset_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    receipt_path = args.action_receipt.resolve()
    action_receipt = json.loads(receipt_path.read_text())
    if manifest["task"] != args.task or training_result["task"] != args.task:
        raise ValueError("task does not match model and dataset")
    if sha256(manifest_path) != training_result["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash does not match training result")
    if action_receipt["output_sha256"] != training_result["action_features_sha256"]:
        raise ValueError("action feature receipt does not match training result")

    method_result = training_result["results"][args.method]
    model_path = model_dir / method_result["model"]
    if sha256(model_path) != method_result["model_sha256"]:
        raise ValueError("model hash does not match training result")
    portable_path = args.portable_model.resolve()
    portable_receipt_path = args.portable_receipt.resolve()
    portable_receipt = json.loads(portable_receipt_path.read_text())
    if portable_receipt["source_model_sha256"] != sha256(model_path):
        raise ValueError("portable model source does not match training model")
    if portable_receipt["output_sha256"] != sha256(portable_path):
        raise ValueError("portable model hash does not match receipt")
    model = PortableStandardizedLogisticRegression.load(portable_path)
    unknown_overrides = set(args.protected_speed_map) - (
        set(model.classes_) - {"fast"}
    )
    if unknown_overrides:
        raise ValueError(f"unknown protected labels in speed map: {sorted(unknown_overrides)}")

    oracle_path = Path(manifest["controller"])
    if sha256(oracle_path) != manifest["controller_sha256"]:
        raise ValueError("oracle controller hash does not match dataset manifest")
    oracle_config = json.loads(oracle_path.read_text())
    offsets = tuple(int(value) for value in action_receipt["offsets_policy_steps"])
    uses_visual = args.method in ("visual", "fused")
    if uses_visual and args.encoder_socket is None:
        raise ValueError("visual and fused methods require --encoder-socket")
    encoder = EncoderClient(args.encoder_socket) if uses_visual else None

    native = []
    candidate = []
    for seed in args.seeds:
        native.append(run_native(args.task, seed))
        candidate.append(
            run_candidate(
                args.task,
                seed,
                model=model,
                method=args.method,
                decoder_config=method_result["validation_decoder"],
                encoder=encoder,
                offsets=offsets,
                fast_speed=args.fast_speed,
                protected_speed=args.protected_speed,
                protected_speed_map=args.protected_speed_map,
                cadence=args.cadence,
                oracle_config=oracle_config,
            )
        )
        print(
            json.dumps(
                {
                    "task": args.task,
                    "seed": seed,
                    "native": native[-1],
                    "candidate": {
                        key: value
                        for key, value in candidate[-1].items()
                        if key != "trace"
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result = {
        "schema": "speedtuning-supervised-shared-fast-v1",
        "task": args.task,
        "method": args.method,
        "seeds": args.seeds,
        "controller": {
            "fast_speed": args.fast_speed,
            "protected_speed": args.protected_speed,
            "protected_speed_map": {
                str(label): args.protected_speed_map.get(
                    str(label), args.protected_speed
                )
                for label in model.classes_
                if label != "fast"
            },
            "protected_labels": [
                str(value) for value in model.classes_ if value != "fast"
            ],
            "all_protected_labels_share_speed": len(
                {
                    args.protected_speed_map.get(str(label), args.protected_speed)
                    for label in model.classes_
                    if label != "fast"
                }
            )
            == 1,
            "decision_cadence_physics_steps": args.cadence,
            "decoder": method_result["validation_decoder"],
            "runtime_speed_inputs": (
                ["scripted_policy_action_chunk"]
                if args.method == "action"
                else (
                    ["angle_rgb"]
                    if args.method == "visual"
                    else ["angle_rgb", "scripted_policy_action_chunk"]
                )
            ),
            "runtime_privileged_speed_inputs": False,
        },
        "native_1x": native,
        "candidate": candidate,
        "summary": summarize(native, candidate),
        "provenance": {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "training_result": str(training_result_path),
            "training_result_sha256": sha256(training_result_path),
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": sha256(manifest_path),
            "action_receipt": str(receipt_path),
            "action_receipt_sha256": sha256(receipt_path),
            "oracle_controller_for_metrics_only": str(oracle_path),
            "oracle_controller_sha256": sha256(oracle_path),
            "portable_model": str(portable_path),
            "portable_model_sha256": sha256(portable_path),
            "portable_receipt": str(portable_receipt_path),
            "portable_receipt_sha256": sha256(portable_receipt_path),
        },
        "oracle_metric_note": (
            "The privileged oracle is evaluated for auditing only and never selects speed. "
            "Runtime oracle metrics omit the retrospective one-frame preentry margin "
            "used in training."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True)
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "COMPLETE").write_text(f"{sha256(result_path)}  results.json\n")
    print(json.dumps({"summary": result["summary"]}, sort_keys=True), flush=True)
    if encoder is not None:
        encoder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
