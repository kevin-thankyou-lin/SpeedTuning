#!/usr/bin/env python3
"""Serve a fresh three-scene VLM frontier search over the frozen ACT policy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_integration import build_original_act_speed_adapter  # noqa: E402
from act_speed_benchmark import sha256  # noqa: E402
from learned_phase_observation import LearnedPhaseEncoder  # noqa: E402
from one_reset_phase_schedule import run_phase_schedule  # noqa: E402
from original_act import set_seed  # noqa: E402
from scripts.run_act_speed_benchmark_cell import DETECTOR_HASHES  # noqa: E402
from scripts.three_scene_server import (  # noqa: E402
    ThreeSceneServer,
    comma_ints,
    write_json,
)


CRITICAL_SOURCES = (
    "act_integration.py",
    "detr/models/backbone.py",
    "detr/models/detr_vae.py",
    "detr/models/transformer.py",
    "detr/util/misc.py",
    "original_act.py",
    "policy.py",
    "policy_speed_env.py",
    "sim_env.py",
    "sim_tasks.py",
    "speed_policy.py",
)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def checked_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"artifact hash mismatch: {path}: {actual} != {expected}")
    return actual


class ACTFrontierRuntime:
    """Hash-checked ACT and learned-phase runtime shared by every rollout."""

    def __init__(
        self,
        *,
        source_commit: str,
        run_manifest: Path,
        task_label: str,
        detector_checkpoint: Path,
        detector_source: Path,
        device: str,
        checkout_commit: str | None = None,
        critical_source_overrides: dict[str, str] | None = None,
    ):
        expected_checkout = checkout_commit or source_commit
        if git_head() != expected_checkout:
            raise RuntimeError("checked-out source does not match requested commit")
        self.source_commit = source_commit
        self.run_manifest = run_manifest.resolve()
        self.manifest = json.loads(self.run_manifest.read_text())
        if not self.manifest.get("parity_gate", {}).get("passed"):
            raise RuntimeError("ACT source manifest lacks the passed uniform-1x gate")
        tracked = self.manifest["source"]["tracked_file_sha256"]
        overrides = dict(critical_source_overrides or {})
        unknown_overrides = set(overrides) - set(CRITICAL_SOURCES)
        if unknown_overrides:
            raise ValueError(
                f"unknown critical source overrides: {sorted(unknown_overrides)}"
            )
        self.critical_source_hashes = {}
        for name in CRITICAL_SOURCES:
            expected = overrides.get(name, tracked[name])
            self.critical_source_hashes[name] = checked_hash(
                REPO_ROOT / name, expected
            )

        if task_label not in self.manifest["tasks"]:
            raise ValueError(f"unknown task label: {task_label}")
        self.task_label = task_label
        self.task_manifest = self.manifest["tasks"][task_label]
        self.task = self.task_manifest["task"]
        policy_root = Path(self.task_manifest["policy_root"])
        artifacts = self.task_manifest["artifacts"]
        checkpoint = policy_root / "checkpoints/policy_best.ckpt"
        stats = policy_root / "checkpoints/dataset_stats.pkl"
        config = policy_root / "checkpoints/policy_config.json"
        for name, path in (
            ("policy_best.ckpt", checkpoint),
            ("dataset_stats.pkl", stats),
            ("policy_config.json", config),
        ):
            checked_hash(path, artifacts[name])

        detector_checkpoint = detector_checkpoint.resolve()
        detector_source = detector_source.resolve()
        checked_hash(detector_checkpoint, DETECTOR_HASHES["checkpoint"])
        checked_hash(
            detector_source / "phase_detector/rgb_inference.py",
            DETECTOR_HASHES["inference"],
        )
        checked_hash(
            detector_source / "phase_detector/rgb_proprio.py",
            DETECTOR_HASHES["model_source"],
        )
        self.detector = {
            "checkpoint_path": str(detector_checkpoint),
            "source_root": str(detector_source),
            "checkpoint_sha256": DETECTOR_HASHES["checkpoint"],
            "inference_sha256": DETECTOR_HASHES["inference"],
            "model_source_sha256": DETECTOR_HASHES["model_source"],
            "device": device,
            "history_stride": 5,
            "cpu_threads_per_worker": 2,
            "render_camera_names": ["angle"],
        }
        set_seed(1000)
        self.adapter = build_original_act_speed_adapter(
            task_name=self.task,
            checkpoint=checkpoint,
            stats_path=stats,
            policy_config_path=config,
            temporal_ensemble_m=0.01,
            device=device,
        )

    def encoder(self):
        return LearnedPhaseEncoder(**self.detector)

    def rollout(
        self,
        schedule,
        seed: int,
        *,
        object_pose=None,
        video_path=None,
        record_attribution_telemetry=False,
    ):
        return run_phase_schedule(
            self.task,
            schedule,
            seed,
            object_pose=object_pose,
            video_path=video_path,
            observation_encoder=self.encoder(),
            chunk_predictor=self.adapter,
            terminate_on_success=False,
            record_attribution_telemetry=record_attribution_telemetry,
        )

    def identity(self) -> dict:
        return {
            "schema": "act-vlm-frontier-runtime-v1",
            "source_commit": self.source_commit,
            "run_manifest": str(self.run_manifest),
            "run_manifest_sha256": sha256(self.run_manifest),
            "task_label": self.task_label,
            "task": self.task,
            "policy_artifacts": self.task_manifest["artifacts"],
            "critical_source_hashes": self.critical_source_hashes,
            "detector": DETECTOR_HASHES,
            "controller": {
                "base_policy": "frozen_multiview_act",
                "cameras": ["angle", "left_wrist", "right_wrist"],
                "progress_clock": "nominal_policy_time",
                "per_physics_step_inference": True,
                "temporal_ensemble_m": 0.01,
                "phase_effector_source": "joint_fk_body_xpos",
            },
        }


class ACTThreeSceneServer(ThreeSceneServer):
    def __init__(self, *args, runtime: ACTFrontierRuntime, **kwargs):
        self.runtime = runtime
        super().__init__(*args, detector=runtime.detector, **kwargs)

    def _phase_encoder(self):
        return self.runtime.encoder()

    def _run(self, schedule, seed: int, pose, media_name: str | None = None) -> dict:
        return self.runtime.rollout(
            schedule,
            seed,
            object_pose=pose,
            video_path=(
                None if media_name is None else self.public_media / media_name
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--task-label", choices=("pick", "tea", "insertion"), required=True)
    parser.add_argument("--discovery-seeds", type=comma_ints, required=True)
    parser.add_argument("--ranking-seeds", type=comma_ints, required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exit-after-selection", action="store_true")
    args = parser.parse_args()

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
    identity_path = root / "IDENTITY.json"
    identity = {
        **runtime.identity(),
        "discovery_seeds": args.discovery_seeds,
        "ranking_seeds": args.ranking_seeds,
        "budget": args.budget,
        "method": "vlm_derivative_frontier_expansion",
        "status": "exploratory_extension_not_original_benchmark",
    }
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("frontier output root identity mismatch")
    else:
        write_json(identity_path, identity)

    api = root / "api"
    (api / "requests").mkdir(parents=True, exist_ok=True)
    (api / "responses").mkdir(parents=True, exist_ok=True)
    server = ACTThreeSceneServer(
        root,
        runtime.task,
        args.discovery_seeds,
        args.ranking_seeds,
        args.budget,
        runtime=runtime,
    )
    write_json(
        api / "READY.json",
        {"ready": True, "task": runtime.task, "identity_sha256": sha256(identity_path)},
    )
    while True:
        for request_path in sorted((api / "requests").glob("*.json")):
            response_path = api / "responses" / request_path.name
            if response_path.exists():
                continue
            try:
                result = server.handle(json.loads(request_path.read_text()))
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            write_json(response_path, response)
        if args.exit_after_selection and (root / "public/SELECTION.json").exists():
            write_json(
                root / "COMPLETE.json",
                {
                    "schema": "act-vlm-frontier-ranking-completion-v1",
                    "identity_sha256": sha256(identity_path),
                    "selection_sha256": sha256(root / "public/SELECTION.json"),
                },
            )
            return 0
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
