#!/usr/bin/env python3
"""Trusted staged runner for three-scene VLM schedule search."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learned_phase_observation import LearnedPhaseEncoder
from one_reset_phase_schedule import (
    ALLOWED_SPEEDS,
    PHASES,
    estimate_phase_workload,
    run_phase_schedule,
    sample_object_pose,
    validate_schedule,
)

DISCOVERY_LIMIT = 30
PROBE_LIMIT = 9
SCREEN_LIMIT = 5
REFINEMENT_EPISODE_LIMIT = 8
RANKING_STATES = 10


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def schedule_hash(schedule) -> str:
    return hashlib.sha256(canonical(list(validate_schedule(schedule)))).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def successful(result: dict) -> bool:
    return bool(result["success"]) and result["safety_violation"] is None


class ThreeSceneServer:
    def __init__(
        self,
        root: Path,
        task: str,
        discovery_seeds: list[int],
        ranking_seeds: list[int],
        budget: int,
        detector: dict | None,
    ):
        if len(discovery_seeds) != 3:
            raise ValueError("exactly three discovery seeds are required")
        if len(ranking_seeds) != RANKING_STATES:
            raise ValueError(f"exactly {RANKING_STATES} ranking seeds are required")
        if set(discovery_seeds) & set(ranking_seeds):
            raise ValueError("discovery and ranking seeds must be disjoint")
        self.root = root
        self.task = task
        self.discovery_seeds = [int(value) for value in discovery_seeds]
        self.ranking_seeds = [int(value) for value in ranking_seeds]
        self.budget = int(budget)
        self.detector = detector
        self.state_path = root / "private" / "state.json"
        self.public_media = root / "public" / "media"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            self._validate_identity()
            if self._merge_native_candidate():
                self._persist()
        else:
            self.state = self._initialize()

    def _validate_identity(self) -> None:
        expected = {
            "task": self.task,
            "discovery_seeds": self.discovery_seeds,
            "ranking_seeds": self.ranking_seeds,
            "budget": self.budget,
        }
        actual = {key: self.state[key] for key in expected}
        if actual != expected:
            raise ValueError("runner resume identity mismatch")

    def _phase_encoder(self):
        return None if self.detector is None else LearnedPhaseEncoder(**self.detector)

    def _run(self, schedule, seed: int, pose, media_name: str | None = None) -> dict:
        return run_phase_schedule(
            self.task,
            schedule,
            seed,
            object_pose=pose,
            video_path=(None if media_name is None else self.public_media / media_name),
            observation_encoder=self._phase_encoder(),
        )

    def _initialize(self) -> dict:
        poses = [sample_object_pose(self.task, seed) for seed in self.discovery_seeds]
        native = [
            self._run((1, 1, 1, 1), seed, pose, f"native-{index}.mp4")
            for index, (seed, pose) in enumerate(zip(self.discovery_seeds, poses))
        ]
        state = {
            "schema": "three-scene-vlm-search-v1",
            "task": self.task,
            "discovery_seeds": self.discovery_seeds,
            "ranking_seeds": self.ranking_seeds,
            "discovery_poses": [list(value) for value in poses],
            "budget": self.budget,
            "episodes_used": 3,
            "native": native,
            "candidates": {
                schedule_hash((1, 1, 1, 1)): {
                    "schedule": [1.0, 1.0, 1.0, 1.0],
                    "discovery": native,
                }
            },
            "probe_hashes": [],
            "screen_hashes": [],
            "refinement_episodes": 0,
            "ranking": None,
            "selected_schedule_hash": None,
            "phase_observation": (
                {"type": "oracle_phase_one_hot"}
                if self.detector is None
                else self._phase_encoder().spec()
            ),
        }
        write_json(self.state_path, state)
        return state

    def _merge_native_candidate(self) -> bool:
        """Expose the already-paid native 3/3 baseline as a lawful finalist."""

        identifier = schedule_hash((1, 1, 1, 1))
        candidate = self.state["candidates"].get(identifier)
        if candidate is not None and all(candidate["discovery"]):
            return False
        self.state["candidates"][identifier] = {
            "schedule": [1.0, 1.0, 1.0, 1.0],
            "discovery": self.state["native"],
        }
        return True

    def _public_rollout(self, result: dict) -> dict:
        value = {
            "schedule": result["schedule"],
            "success": result["success"],
            "raw_task_success": result["raw_task_success"],
            "physics_steps": result["physics_steps"],
            "safety_violation": result["safety_violation"],
            "phase_decisions": result["phase_decisions"],
            "phase_workload_steps": estimate_phase_workload(result),
        }
        if result.get("video_path") is not None:
            value["video_path"] = f"media/{Path(result['video_path']).name}"
        return value

    def _public_candidate(self, identifier: str) -> dict:
        candidate = self.state["candidates"][identifier]
        discovery = [
            self._public_rollout(value) if value is not None else None
            for value in candidate["discovery"]
        ]
        return {
            "schedule_hash": identifier,
            "schedule": candidate["schedule"],
            "discovery": discovery,
            "discovery_successes": sum(
                value is not None and successful(value) for value in candidate["discovery"]
            ),
            "discovery_completed": sum(value is not None for value in candidate["discovery"]),
        }

    def _persist(self) -> None:
        write_json(self.state_path, self.state)

    def _ensure_discovery_budget(self, cost: int) -> None:
        if self.state["ranking"] is not None:
            raise ValueError("ranking is already frozen")
        if self.state["episodes_used"] + cost > DISCOVERY_LIMIT:
            raise ValueError("discovery allocation exhausted; rank two eligible finalists")

    def _candidate(self, schedule) -> tuple[str, dict]:
        schedule = list(validate_schedule(schedule))
        identifier = schedule_hash(schedule)
        candidate = self.state["candidates"].setdefault(
            identifier,
            {"schedule": schedule, "discovery": [None, None, None]},
        )
        return identifier, candidate

    def info(self) -> dict:
        return {
            "task": self.task,
            "phases": list(PHASES),
            "allowed_speeds": list(ALLOWED_SPEEDS),
            "learning_condition": "three frozen discovery scenes plus a fresh ten-pose finalist bank",
            "budget": self.budget,
            "budget_used": self.state["episodes_used"],
            "budget_remaining": self.budget - self.state["episodes_used"],
            "stage_limits": {
                "native_previews": 3,
                "probed_schedules": PROBE_LIMIT,
                "screened_schedules": SCREEN_LIMIT,
                "refinement_episodes": REFINEMENT_EPISODE_LIMIT,
                "discovery_episode_ceiling": DISCOVERY_LIMIT,
                "ranking_states_per_finalist": RANKING_STATES,
            },
            "native": [self._public_rollout(value) for value in self.state["native"]],
            "candidates": {
                key: self._public_candidate(key) for key in self.state["candidates"]
            },
            "ranking": self.state["ranking"],
            "selected_schedule_hash": self.state["selected_schedule_hash"],
            "phase_observation": self.state["phase_observation"],
        }

    def probe(self, schedule) -> dict:
        identifier, candidate = self._candidate(schedule)
        if candidate["discovery"][0] is not None:
            return {"cache_hit": True, **self._public_candidate(identifier)}
        if len(self.state["probe_hashes"]) >= PROBE_LIMIT:
            raise ValueError("nine-schedule scene-A probe limit reached")
        self._ensure_discovery_budget(1)
        candidate["discovery"][0] = self._run(
            candidate["schedule"],
            self.discovery_seeds[0],
            self.state["discovery_poses"][0],
            f"{identifier[:12]}-a.mp4",
        )
        self.state["probe_hashes"].append(identifier)
        self.state["episodes_used"] += 1
        self._persist()
        return {"cache_hit": False, **self._public_candidate(identifier)}

    def screen(self, identifier: str) -> dict:
        candidate = self.state["candidates"].get(identifier)
        if candidate is None or identifier not in self.state["probe_hashes"]:
            raise ValueError("screen requires a probed schedule hash")
        if not successful(candidate["discovery"][0]):
            raise ValueError("scene-A failure is not eligible for screening")
        if identifier not in self.state["screen_hashes"]:
            if len(self.state["screen_hashes"]) >= SCREEN_LIMIT:
                raise ValueError("five-schedule B/C screen limit reached")
            self.state["screen_hashes"].append(identifier)
        for index in (1, 2):
            if candidate["discovery"][index] is not None:
                if not successful(candidate["discovery"][index]):
                    break
                continue
            self._ensure_discovery_budget(1)
            candidate["discovery"][index] = self._run(
                candidate["schedule"],
                self.discovery_seeds[index],
                self.state["discovery_poses"][index],
                f"{identifier[:12]}-{chr(ord('a') + index)}.mp4",
            )
            self.state["episodes_used"] += 1
            self._persist()
            if not successful(candidate["discovery"][index]):
                break
        self._persist()
        return self._public_candidate(identifier)

    def refine(self, schedule) -> dict:
        identifier, candidate = self._candidate(schedule)
        if any(value is not None for value in candidate["discovery"]):
            return {"cache_hit": True, **self._public_candidate(identifier)}
        for index in range(3):
            if self.state["refinement_episodes"] >= REFINEMENT_EPISODE_LIMIT:
                break
            self._ensure_discovery_budget(1)
            candidate["discovery"][index] = self._run(
                candidate["schedule"],
                self.discovery_seeds[index],
                self.state["discovery_poses"][index],
                f"{identifier[:12]}-r{index}.mp4",
            )
            self.state["episodes_used"] += 1
            self.state["refinement_episodes"] += 1
            self._persist()
            if not successful(candidate["discovery"][index]):
                break
        return {"cache_hit": False, **self._public_candidate(identifier)}

    def _rank_summary(self, identifier: str, rollouts: list[dict]) -> dict:
        successful_rollouts = [value for value in rollouts if successful(value)]
        return {
            "schedule_hash": identifier,
            "schedule": self.state["candidates"][identifier]["schedule"],
            "successes": len(successful_rollouts),
            "trials": len(rollouts),
            "safety_violations": sum(value["safety_violation"] is not None for value in rollouts),
            "successful_mean_steps": (
                None
                if not successful_rollouts
                else statistics.fmean(value["physics_steps"] for value in successful_rollouts)
            ),
            "rollouts": [self._public_rollout(value) for value in rollouts],
        }

    @staticmethod
    def _rank_key(summary: dict):
        mean = summary["successful_mean_steps"]
        return (
            -summary["safety_violations"],
            summary["successes"],
            -math.inf if mean is None else -mean,
            -len(set(summary["schedule"])),
        )

    def rank(self, identifiers: list[str]) -> dict:
        if self.state["ranking"] is not None:
            return {**self.state["ranking"], "cache_hit": True}
        if len(identifiers) != 2 or len(set(identifiers)) != 2:
            raise ValueError("rank requires exactly two distinct schedule hashes")
        for identifier in identifiers:
            candidate = self.state["candidates"].get(identifier)
            if candidate is None or not all(
                value is not None and successful(value) for value in candidate["discovery"]
            ):
                raise ValueError("each finalist must have three safe discovery successes")
        if self.state["episodes_used"] > DISCOVERY_LIMIT:
            raise ValueError("ranking reserve was violated")
        if self.state["episodes_used"] + 2 * RANKING_STATES > self.budget:
            raise ValueError("insufficient reserved budget for finalist ranking")
        summaries = []
        for identifier in identifiers:
            schedule = self.state["candidates"][identifier]["schedule"]
            rollouts = [
                self._run(schedule, seed, None)
                for seed in self.ranking_seeds
            ]
            self.state["episodes_used"] += len(rollouts)
            summaries.append(self._rank_summary(identifier, rollouts))
            self._persist()
        selected = max(summaries, key=self._rank_key)
        ranking = {
            "cache_hit": False,
            "selection_rule": "fewest safety violations, then most successes, then lowest successful mean steps",
            "finalists": summaries,
            "selected_schedule_hash": selected["schedule_hash"],
            "selected_schedule": selected["schedule"],
            "budget_used": self.state["episodes_used"],
        }
        self.state["ranking"] = ranking
        self.state["selected_schedule_hash"] = selected["schedule_hash"]
        self._persist()
        write_json(
            self.root / "public" / "SELECTION.json",
            {
                "schedule_hash": selected["schedule_hash"],
                "schedule": selected["schedule"],
                "ranking": ranking,
            },
        )
        return ranking

    def handle(self, request: dict) -> dict:
        command = request.get("command")
        if command == "info":
            return self.info()
        if command == "probe":
            return self.probe(request.get("schedule"))
        if command == "screen":
            return self.screen(str(request.get("schedule_hash")))
        if command == "refine":
            return self.refine(request.get("schedule"))
        if command == "rank":
            return self.rank([str(value) for value in request.get("schedule_hashes", [])])
        raise ValueError(f"unknown command: {command}")


def comma_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=("pick_and_place", "tea_bag", "insertion"), required=True)
    parser.add_argument("--discovery-seeds", type=comma_ints, required=True)
    parser.add_argument("--ranking-seeds", type=comma_ints, required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--detector-json", type=Path)
    args = parser.parse_args()
    api = args.root / "api"
    requests = api / "requests"
    responses = api / "responses"
    requests.mkdir(parents=True, exist_ok=True)
    responses.mkdir(parents=True, exist_ok=True)
    detector = None if args.detector_json is None else json.loads(args.detector_json.read_text())
    server = ThreeSceneServer(
        args.root,
        args.task,
        args.discovery_seeds,
        args.ranking_seeds,
        args.budget,
        detector,
    )
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
