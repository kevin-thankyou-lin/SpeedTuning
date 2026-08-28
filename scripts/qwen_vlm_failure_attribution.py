#!/usr/bin/env python3
"""Blinded video attribution for acceleration-induced robot failures.

The VLM receives a same-seed successful reference and accelerated failure.  It
reports both the phase where the failure is visible and the earliest phase
whose acceleration could have caused it.  The causal phase is constrained to
the observed phase or an earlier phase; a later phase cannot explain an
already-visible divergence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from collections import Counter
from pathlib import Path


PHASES = ("pre_grasp", "grasp_lift", "transport", "interaction")

TASK_GOALS = {
    "pick": "pick up the red cube and place it inside the green target box",
    "tea": "place the tea-bag center inside the oriented volume of the mug",
    "insertion": "insert the red peg into the blue socket and complete insertion",
}

PHASE_DESCRIPTIONS = {
    "pre_grasp": "approach and align before grasp closure",
    "grasp_lift": "close the gripper, establish the grasp, and lift clear",
    "transport": "carry the grasped object toward the goal",
    "interaction": "perform the terminal placement, release, or insertion",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def phase_timeline(record: dict) -> list[dict]:
    """Return the learned detector's causal phase-change timeline."""

    result = []
    for item in record.get("phase_decisions", ()):
        phase = str(item.get("phase"))
        if phase in PHASES:
            result.append(
                {
                    "phase": phase,
                    "physics_step": int(item.get("physics_step", 0)),
                    "speed": float(item.get("speed", 1.0)),
                }
            )
    return result


def build_prompt(
    task_label: str,
    reference_record: dict,
    candidate_record: dict,
) -> str:
    if task_label not in TASK_GOALS:
        raise ValueError(f"unknown task: {task_label}")
    if int(reference_record.get("seed", -1)) != int(candidate_record.get("seed", -2)):
        raise ValueError("VLM attribution requires the same initial-state seed")
    phase_lines = "\n".join(
        f"- {phase}: {PHASE_DESCRIPTIONS[phase]}" for phase in PHASES
    )
    return f"""You diagnose execution-speed failures in a robot manipulation task.

Task goal: {TASK_GOALS[task_label]}.

The learned online phase detector uses these phases in causal order:
{phase_lines}

Video A is a successful reference rollout. Video B uses the same initial-state
seed but a more aggressive speed schedule and failed the task checker. The
videos are evidence; the phase timelines below come from the learned detector,
not simulator-oracle phase labels.

Reference timeline: {json.dumps(phase_timeline(reference_record), separators=(',', ':'))}
Failed timeline: {json.dumps(phase_timeline(candidate_record), separators=(',', ':'))}

Identify:
1. observed_failure_phase: the first phase in which the failure is visibly
   observable;
2. causal_phase: the earliest phase whose excessive speed plausibly caused the
   observed failure.

The causal phase may equal the observed phase or be earlier. The causal phase
must never be later. For example, an object visibly missing the receptacle during interaction
may have been caused by a weak grasp during grasp_lift or unstable transport.
Do not infer success from proximity alone; apply the task goal above.

Return one JSON object only with exactly these keys:
observed_failure_phase, causal_phase, confidence, evidence.
confidence must be a number in [0,1]. evidence must cite visible differences
between Videos A and B without mentioning hidden simulator state."""


def extract_json(text: str) -> dict:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S)
    if fenced:
        stripped = fenced.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("VLM output did not contain a JSON object")
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def validate_attribution(value: dict) -> dict:
    expected = {
        "observed_failure_phase",
        "causal_phase",
        "confidence",
        "evidence",
    }
    if set(value) != expected:
        raise ValueError(f"VLM attribution keys must be exactly {sorted(expected)}")
    observed = str(value["observed_failure_phase"])
    causal = str(value["causal_phase"])
    if observed not in PHASES or causal not in PHASES:
        raise ValueError("VLM attribution returned an unknown phase")
    if PHASES.index(causal) > PHASES.index(observed):
        raise ValueError("causal phase cannot occur after observed failure phase")
    confidence = float(value["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("VLM confidence must be in [0,1]")
    evidence = str(value["evidence"]).strip()
    if not evidence:
        raise ValueError("VLM evidence cannot be empty")
    return {
        "observed_failure_phase": observed,
        "causal_phase": causal,
        "confidence": confidence,
        "evidence": evidence,
    }


def aggregate_attributions(items: list[dict], fallback_phase: str) -> dict:
    """Select a repair phase by majority causal attribution, ties earlier."""

    if fallback_phase not in PHASES:
        raise ValueError("invalid fallback phase")
    checked = [validate_attribution(item) for item in items]
    if not checked:
        return {
            "method": "vlm_no_matched_pair_semantic_fallback",
            "selected_phase": fallback_phase,
            "counts": {},
            "attributions": [],
        }
    counts = Counter(item["causal_phase"] for item in checked)
    selected = max(PHASES, key=lambda phase: (counts[phase], -PHASES.index(phase)))
    return {
        "method": "blinded_vlm_majority_causal_phase_tie_earlier",
        "selected_phase": selected,
        "counts": {phase: counts[phase] for phase in PHASES if counts[phase]},
        "attributions": checked,
    }


class QwenVideoAttributor:
    """Lazy local Qwen2.5-VL inference with immutable receipt metadata."""

    def __init__(self, model_path: Path, *, device: str = "cuda"):
        self.model_path = model_path.resolve()
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.model_path),
            dtype=torch.bfloat16,
            device_map=self.device,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        self._processor = AutoProcessor.from_pretrained(
            str(self.model_path), local_files_only=True, use_fast=False
        )

    def identity(self) -> dict:
        files = {}
        for name in (
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
        ):
            path = self.model_path / name
            if path.is_file():
                files[name] = sha256_file(path)
        shards = sorted(self.model_path.glob("model-*.safetensors"))
        files.update({path.name: sha256_file(path) for path in shards})
        return {
            "model_family": "Qwen2.5-VL-3B-Instruct",
            "model_path": str(self.model_path),
            "device": self.device,
            "local_files_only": True,
            "greedy_decoding": True,
            "sampled_video_fps": 1.0,
            "max_video_pixels": 320 * 480,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("transformers", "qwen-vl-utils", "torch", "av")
            },
            "files_sha256": files,
        }

    def close(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._processor = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def diagnose(
        self,
        *,
        task_label: str,
        reference_record: dict,
        candidate_record: dict,
        reference_video: Path,
        candidate_video: Path,
    ) -> dict:
        if not reference_video.is_file() or not candidate_video.is_file():
            raise FileNotFoundError("VLM attribution requires both matched videos")
        prompt = build_prompt(task_label, reference_record, candidate_record)
        self._load()
        import torch
        from qwen_vl_utils import process_vision_info

        content = [
            {"type": "text", "text": prompt + "\nVideo A: successful reference."},
            {
                "type": "video",
                "video": str(reference_video.resolve()),
                "fps": 1.0,
                "max_pixels": 320 * 480,
            },
            {"type": "text", "text": "Video B: failed accelerated rollout."},
            {
                "type": "video",
                "video": str(candidate_video.resolve()),
                "fps": 1.0,
                "max_pixels": 320 * 480,
            },
        ]
        messages = [{"role": "user", "content": content}]
        rendered = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = self._processor(
            text=[rendered],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=192, do_sample=False
            )
        generated = output[:, inputs.input_ids.shape[1] :]
        raw = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        parsed = validate_attribution(extract_json(raw))
        body = {
            "schema": "strider-qwen-failure-attribution-v1",
            "seed": int(candidate_record["seed"]),
            "task_label": task_label,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "reference_video": str(reference_video.resolve()),
            "reference_video_sha256": sha256_file(reference_video),
            "candidate_video": str(candidate_video.resolve()),
            "candidate_video_sha256": sha256_file(candidate_video),
            "reference_schedule": list(map(float, reference_record["schedule"])),
            "candidate_schedule": list(map(float, candidate_record["schedule"])),
            "attribution": parsed,
            "raw_response": raw,
        }
        return {**body, "receipt_sha256": canonical_sha256(body)}
