#!/usr/bin/env python3
"""Bridge blinded robot-failure evidence to a remote Codex coordinator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.qwen_vlm_failure_attribution import (
    PHASES,
    build_prompt,
    canonical_sha256,
    phase_timeline,
    sha256_file,
    validate_attribution,
)


SCHEMA = "codex-agent-attribution-exchange-v1"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _phase_at_step(record: dict, step: int) -> str:
    phase = PHASES[0]
    for item in phase_timeline(record):
        if int(item["physics_step"]) > step:
            break
        phase = item["phase"]
    return phase


def _sample_indices(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("video contains no frames")
    if frame_count <= count:
        return list(range(frame_count))
    return sorted(
        {
            round(index * (frame_count - 1) / (count - 1))
            for index in range(count)
        }
    )


def make_contact_sheet(
    video_path: Path,
    record: dict,
    output_path: Path,
    *,
    label: str,
    sample_count: int = 12,
    cell_width: int = 360,
) -> dict:
    """Render a chronological labeled contact sheet without hidden state."""

    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        frames = [frame.to_image().convert("RGB") for frame in container.decode(stream)]
        rate = float(stream.average_rate) if stream.average_rate else 0.0
    selected = _sample_indices(len(frames), sample_count)
    rows = math.ceil(len(selected) / 4)
    label_height = 38
    first = frames[selected[0]]
    cell_height = round(first.height * cell_width / first.width)
    sheet = Image.new("RGB", (4 * cell_width, rows * (cell_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    samples = []
    for slot, frame_index in enumerate(selected):
        frame = frames[frame_index]
        frame.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        x = (slot % 4) * cell_width
        y = (slot // 4) * (cell_height + label_height)
        sheet.paste(frame, (x, y))
        phase = _phase_at_step(record, frame_index)
        seconds = frame_index / rate if rate > 0 else 0.0
        caption = f"{label} | step {frame_index} | {seconds:.2f}s | {phase}"
        draw.rectangle((x, y + cell_height, x + cell_width, y + cell_height + label_height), fill="white")
        draw.text((x + 4, y + cell_height + 8), caption, fill="black")
        samples.append(
            {
                "frame_index": frame_index,
                "seconds": seconds,
                "learned_phase": phase,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=92, optimize=True)
    return {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "source_video_sha256": sha256_file(video_path),
        "source_frame_count": len(frames),
        "source_fps": rate,
        "samples": samples,
    }


def sanitized_record(record: dict) -> dict:
    """Expose policy-visible phase evidence, never simulator object state."""

    return {
        "seed": int(record["seed"]),
        "schedule": list(map(float, record["schedule"])),
        "success": bool(record["success"]),
        "physics_steps": int(record["physics_steps"]),
        "first_success_step": record.get("first_success_step"),
        "phase_timeline": phase_timeline(record),
    }


class CodexExchangeAttributor:
    """Write blinded evidence bundles and await signed Codex responses."""

    def __init__(
        self,
        exchange_root: Path,
        *,
        model: str,
        poll_seconds: float = 2.0,
        timeout_seconds: float = 3600.0,
    ):
        self.exchange_root = exchange_root.resolve()
        self.model = model
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def identity(self) -> dict:
        return {
            "backend": "codex_agent_external_coordinator",
            "model": self.model,
            "input": "paired_same_seed_contact_sheets_plus_sanitized_learned_phase_timelines",
            "simulator_hidden_state_exposed": False,
            "output_schema": SCHEMA,
            "causal_phase_constraint": "same_as_or_earlier_than_observed_failure_phase",
        }

    def diagnose(
        self,
        *,
        task_label: str,
        reference_record: dict,
        candidate_record: dict,
        reference_video: Path,
        candidate_video: Path,
    ) -> dict:
        seed = int(candidate_record["seed"])
        request_id = canonical_sha256(
            {
                "task": task_label,
                "seed": seed,
                "reference_video_sha256": sha256_file(reference_video),
                "candidate_video_sha256": sha256_file(candidate_video),
                "model": self.model,
            }
        )
        root = self.exchange_root / request_id
        request_path = root / "REQUEST.json"
        response_path = root / "RESPONSE.json"
        if not request_path.exists():
            reference_sheet = make_contact_sheet(
                reference_video, reference_record, root / "reference_contact_sheet.jpg", label="REFERENCE SUCCESS"
            )
            failure_sheet = make_contact_sheet(
                candidate_video, candidate_record, root / "failure_contact_sheet.jpg", label="ACCELERATED FAILURE"
            )
            prompt = build_prompt(task_label, reference_record, candidate_record)
            request = {
                "schema": SCHEMA,
                "request_id": request_id,
                "model": self.model,
                "prompt": prompt,
                "task_label": task_label,
                "seed": seed,
                "reference": sanitized_record(reference_record),
                "candidate": sanitized_record(candidate_record),
                "images": [reference_sheet, failure_sheet],
            }
            request["payload_sha256"] = canonical_sha256(request)
            _write_json(request_path, request)
            _write_json(root / "READY.json", {"request_id": request_id, "payload_sha256": request["payload_sha256"]})
        request = json.loads(request_path.read_text())
        deadline = time.monotonic() + self.timeout_seconds
        while not response_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Codex attribution timed out: {request_id}")
            time.sleep(self.poll_seconds)
        response = json.loads(response_path.read_text())
        if response.get("request_id") != request_id:
            raise RuntimeError("Codex response request ID mismatch")
        if response.get("request_payload_sha256") != request["payload_sha256"]:
            raise RuntimeError("Codex response request hash mismatch")
        attribution = validate_attribution(response["attribution"])
        return {
            "schema": "codex-agent-failure-attribution-v1",
            "seed": seed,
            "request_id": request_id,
            "request_payload_sha256": request["payload_sha256"],
            "response_sha256": sha256_file(response_path),
            "coordinator": response["coordinator"],
            "attribution": attribution,
        }

    def close(self) -> None:
        return None
