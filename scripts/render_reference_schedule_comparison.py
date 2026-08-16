#!/usr/bin/env python3
"""Render a labeled, media-only three-arm reference-schedule comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference_alignment import OnlineReferenceAligner  # noqa: E402
from reference_schedule import (  # noqa: E402
    CausalTemporalPool,
    EventController,
    expand_protected_speed_map,
    select_aligned_speed,
)
from scripts.evaluate_reference_aligned_schedule import (  # noqa: E402
    EncoderClient,
    make_env,
    snapshot,
)


TITLES = {
    "privileged": "1. Privileged state boundaries",
    "aligned_exact": "2. RN18 exact reference boundaries",
    "aligned_margin": "3. RN18 p90 expansion + confidence fallback",
}


def annotate(frame: np.ndarray, lines: list[str], color: tuple[int, int, int]) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=16)
    line_height = 22
    box_height = 12 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0, 185))
    for index, line in enumerate(lines):
        draw.text((10, 6 + index * line_height), line, fill=(*color, 255), font=font)
    return np.asarray(image)


def capture_arm(
    *,
    task: str,
    seed: int,
    arm: str,
    controller_config: dict,
    encoder: EncoderClient,
    reference_descriptors: np.ndarray,
    speed_map: np.ndarray,
    frame_stride: int,
    pool_frames: int,
    confidence_threshold: float,
    output: Path,
) -> dict:
    visual = arm.startswith("aligned_")
    ceiling = float(controller_config["ceiling"])
    speeds = tuple(sorted({1.0, ceiling, *map(float, speed_map)}))
    env = make_env(task, seed, speeds, True)
    privileged = EventController(controller_config)
    pool = CausalTemporalPool(pool_frames)
    aligner = (
        OnlineReferenceAligner(
            reference_descriptors,
            max_advance=5,
            max_backtrack=1,
            emission_temperature=0.07,
            expected_advance=1.0,
            transition_scale=1.5,
            backward_penalty=2.0,
            initialization_fraction=0.12,
            updates_per_second=10.0,
        )
        if visual
        else None
    )
    chosen_speed = 1.0
    confidence = None
    predicted = None
    fallback = False
    last_reward = 0.0
    last_success = False
    writer = imageio.get_writer(output, fps=25, codec="libx264", quality=7)
    frames = 0
    first_success = None
    try:
        env.reset()
        done = False
        while not done:
            pre = snapshot(env, last_reward, last_success)
            oracle_speed, oracle_reason, _ = privileged.select(pre)
            if arm == "privileged":
                chosen_speed = oracle_speed
                reason = oracle_reason
                confidence = None
                predicted = None
                fallback = False
            elif env.physics_steps % frame_stride == 0:
                assert aligner is not None
                frame = env.cur_ts.observation["images"]["angle"]
                result = aligner.update_embedding(pool.update(encoder.encode(frame)))
                threshold = confidence_threshold if arm == "aligned_margin" else None
                chosen_speed, fallback = select_aligned_speed(
                    speed_map,
                    result.reference_index,
                    result.confidence,
                    confidence_threshold=threshold,
                    fallback_speed=1.0,
                )
                confidence = result.confidence
                predicted = result.reference_position
                reason = "FALLBACK 1x" if fallback else "RN18 lookup"
            if env.physics_steps % 2 == 0:
                true_position = float(np.clip(env.policy_time / env.episode_len, 0.0, 1.0))
                lines = [
                    TITLES[arm],
                    f"speed {chosen_speed:.1f}x | reward {last_reward:.0f}/{env.env.task.max_reward} | step {env.physics_steps}",
                ]
                if visual:
                    lines.append(
                        f"ref predicted {predicted:.3f} | true {true_position:.3f} | confidence {confidence:.3f}"
                    )
                    lines.append(reason)
                else:
                    lines.append(f"exact observable gate | {reason}")
                color = (255, 190, 70) if fallback else (120, 255, 150)
                writer.append_data(annotate(env.cur_ts.observation["images"]["angle"], lines, color))
                frames += 1
            _, _, done, info = env._step_physics(chosen_speed)
            last_reward = float(info["task_reward"])
            last_success = bool(info["success"])
            if last_reward >= env.env.task.max_reward:
                first_success = int(env.physics_steps)
                done = True
        status = "SUCCESS" if first_success is not None else "FAILURE"
        final_frame = annotate(
            env.cur_ts.observation["images"]["angle"],
            [TITLES[arm], f"{status} | final reward {last_reward:.0f}/{env.env.task.max_reward}", f"physics steps {env.physics_steps}"],
            (100, 255, 130) if first_success is not None else (255, 100, 100),
        )
        for _ in range(35):
            writer.append_data(final_frame)
            frames += 1
        return {
            "arm": arm,
            "success": first_success is not None,
            "first_success_steps": first_success,
            "attempted_steps": int(env.physics_steps),
            "video": str(output),
            "frames": frames,
        }
    finally:
        writer.close()
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--encoder-socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p90-margin", type=float, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    args = parser.parse_args()

    sys.path.insert(0, str(args.runtime_root.resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    controller = json.loads(args.controller.read_text())
    reference = np.load(args.reference)
    descriptors = reference["descriptors"]
    exact_map = reference["speeds"]
    margin_indices = int(np.ceil(args.p90_margin * max(1, len(exact_map) - 1)))
    expanded_map = expand_protected_speed_map(
        exact_map,
        ceiling=float(controller["ceiling"]),
        margin_indices=margin_indices,
    )
    encoder = EncoderClient(args.encoder_socket)
    results = []
    try:
        for arm in ("privileged", "aligned_exact", "aligned_margin"):
            speed_map = expanded_map if arm == "aligned_margin" else exact_map
            results.append(
                capture_arm(
                    task=args.task,
                    seed=args.seed,
                    arm=arm,
                    controller_config=controller,
                    encoder=encoder,
                    reference_descriptors=descriptors,
                    speed_map=speed_map,
                    frame_stride=5,
                    pool_frames=10,
                    confidence_threshold=args.confidence_threshold,
                    output=args.output / f"{arm}.mp4",
                )
            )
    finally:
        encoder.close()

    max_duration = max(item["frames"] for item in results) / 25.0
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(args.output / "privileged.mp4"),
            "-i", str(args.output / "aligned_exact.mp4"),
            "-i", str(args.output / "aligned_margin.mp4"),
            "-filter_complex",
            "[0:v]tpad=stop_mode=clone:stop_duration=60[p];"
            "[1:v]tpad=stop_mode=clone:stop_duration=60[e];"
            "[2:v]tpad=stop_mode=clone:stop_duration=60[m];"
            "[p][e][m]hstack=inputs=3[v]",
            "-map", "[v]", "-t", f"{max_duration:.3f}",
            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            str(args.output / "comparison.mp4"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "r3-reference-schedule-media-replay-v1",
                "task": args.task,
                "seed": args.seed,
                "scientific_reset": False,
                "media_only": True,
                "p90_margin": args.p90_margin,
                "confidence_threshold": args.confidence_threshold,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
