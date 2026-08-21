#!/usr/bin/env python3
"""Trusted file-queue runner for one-reset phase-conditioned VLM search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from one_reset_phase_schedule import (
    ALLOWED_SPEEDS,
    PHASES,
    run_phase_schedule,
    sample_object_pose,
    validate_schedule,
)
from learned_phase_observation import LearnedPhaseEncoder


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def schedule_hash(schedule) -> str:
    return hashlib.sha256(canonical(list(validate_schedule(schedule)))).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class OneResetServer:
    def __init__(self, root: Path, task: str, training_seed: int, budget: int, detector=None):
        self.root = root
        self.task = task
        self.training_seed = int(training_seed)
        self.budget = int(budget)
        self.detector = detector
        self.state_path = root / "private" / "state.json"
        self.public_media = root / "public" / "media"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = self._initialize()

    def _phase_encoder(self):
        return None if self.detector is None else LearnedPhaseEncoder(**self.detector)

    def _initialize(self) -> dict:
        object_pose = sample_object_pose(self.task, self.training_seed)
        native = run_phase_schedule(
            self.task,
            (1.0, 1.0, 1.0, 1.0),
            self.training_seed,
            object_pose=object_pose,
            video_path=self.public_media / "native.mp4",
            observation_encoder=self._phase_encoder(),
        )
        native_hash = schedule_hash(native["schedule"])
        state = {
            "schema": "one-reset-phase-vlm-v1",
            "task": self.task,
            "training_seed": self.training_seed,
            "object_pose": list(object_pose),
            "budget": self.budget,
            "results": {native_hash: native},
            "selected_schedule_hash": None,
            "phase_observation": (
                {"type": "oracle_phase_one_hot"}
                if self.detector is None
                else self._phase_encoder().spec()
            ),
        }
        write_json(self.state_path, state)
        return state

    def _public_result(self, schedule_id: str, result: dict) -> dict:
        video_path = result["video_path"]
        return {
            "schedule_hash": schedule_id,
            "schedule": result["schedule"],
            "success": result["success"],
            "raw_task_success": result["raw_task_success"],
            "physics_steps": result["physics_steps"],
            "success_only_acceleration": result["success_only_acceleration"],
            "safety_violation": result["safety_violation"],
            "phase_decisions": result["phase_decisions"],
            "video_path": (
                None if video_path is None else f"media/{Path(video_path).name}"
            ),
        }

    def info(self) -> dict:
        results = {
            key: self._public_result(key, value)
            for key, value in self.state["results"].items()
        }
        return {
            "task": self.task,
            "phases": list(PHASES),
            "allowed_speeds": list(ALLOWED_SPEEDS),
            "learning_condition": "one frozen reset reused for every schedule",
            "budget": self.budget,
            "budget_used": len(results),
            "budget_remaining": self.budget - len(results),
            "results": results,
            "selected_schedule_hash": self.state["selected_schedule_hash"],
            "phase_observation": self.state["phase_observation"],
        }

    def test(self, schedule) -> dict:
        schedule = validate_schedule(schedule)
        identifier = schedule_hash(schedule)
        if identifier in self.state["results"]:
            return {"cache_hit": True, **self._public_result(
                identifier, self.state["results"][identifier]
            )}
        if len(self.state["results"]) >= self.budget:
            raise ValueError("one-reset learning budget exhausted")
        result = run_phase_schedule(
            self.task,
            schedule,
            self.training_seed,
            object_pose=self.state["object_pose"],
            video_path=self.public_media / f"{identifier[:12]}.mp4",
            observation_encoder=self._phase_encoder(),
        )
        self.state["results"][identifier] = result
        write_json(self.state_path, self.state)
        return {"cache_hit": False, **self._public_result(identifier, result)}

    def select(self, schedule_id: str) -> dict:
        result = self.state["results"].get(schedule_id)
        if result is None:
            raise ValueError("selected schedule has not been tested")
        if not result["success"] or result["safety_violation"] is not None:
            raise ValueError("selected schedule must succeed safely on the learning reset")
        self.state["selected_schedule_hash"] = schedule_id
        write_json(self.state_path, self.state)
        public = self._public_result(schedule_id, result)
        write_json(self.root / "public" / "SELECTION.json", public)
        return public

    def handle(self, request: dict) -> dict:
        command = request.get("command")
        if command == "info":
            return self.info()
        if command == "test":
            return self.test(request.get("schedule"))
        if command == "select":
            return self.select(str(request.get("schedule_hash")))
        raise ValueError(f"unknown command: {command}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=("pick_and_place", "tea_bag", "insertion"), required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--detector-json", type=Path)
    args = parser.parse_args()
    api = args.root / "api"
    requests = api / "requests"
    responses = api / "responses"
    requests.mkdir(parents=True, exist_ok=True)
    responses.mkdir(parents=True, exist_ok=True)
    detector = None if args.detector_json is None else json.loads(args.detector_json.read_text())
    server = OneResetServer(args.root, args.task, args.training_seed, args.budget, detector)
    write_json(api / "READY.json", {"ready": True, "task": args.task})
    while True:
        for request_path in sorted(requests.glob("*.json")):
            response_path = responses / request_path.name
            if response_path.exists():
                continue
            try:
                result = server.handle(json.loads(request_path.read_text()))
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            write_json(response_path, response)
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
