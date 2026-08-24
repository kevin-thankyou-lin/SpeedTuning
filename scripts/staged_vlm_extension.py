#!/usr/bin/env python3
"""Receipt-bearing single-candidate continuation of a sealed staged search."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

from one_reset_phase_schedule import rollout_metric_steps, validate_schedule
from scripts.staged_vlm_frontier import stage_verdict, successful, write_json


class SingleCandidateExtension:
    def __init__(
        self,
        root: Path,
        parent_state: dict,
        candidate,
        rollout,
        *,
        anchor_successes: int,
        anchor_speedup: float,
    ):
        self.root = root
        self.parent = parent_state
        self.candidate = validate_schedule(candidate)
        self.rollout = rollout
        self.anchor_successes = int(anchor_successes)
        self.anchor_speedup = float(anchor_speedup)
        if len(self.parent["seeds"]) != 20 or len(self.parent["native"]) != 20:
            raise ValueError("parent lacks the exact 20-seed matched native bank")
        if any(not successful(item) for item in self.parent["native"]):
            raise ValueError("parent matched native bank is not 20/20 clean")
        self.state_path = root / "private" / "state.json"
        self.receipt_root = root / "private" / "rollouts"
        self.media_root = root / "public" / "media"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state["candidate"] != list(self.candidate):
                raise RuntimeError("extension resume candidate mismatch")
            count = len(list(self.receipt_root.glob("*.json")))
            if count != self.state["episodes_used"]:
                self.state["episodes_used"] = count
                self._persist()
        else:
            self.state = {
                "schema": "act-vlm-single-candidate-extension-v1",
                "candidate": list(self.candidate),
                "episodes_used": 0,
                "rollouts": [],
                "verdict": None,
                "halt_reason": None,
            }
            self._persist()

    def _persist(self):
        write_json(self.state_path, self.state)

    def _run_one(self, index: int) -> dict:
        seed = int(self.parent["seeds"][index])
        receipt = self.receipt_root / f"{index:02d}-{seed}.json"
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if value["seed"] != seed or value["schedule"] != list(self.candidate):
                raise RuntimeError("extension cached receipt identity mismatch")
            return value
        if self.state["episodes_used"] >= 20:
            raise RuntimeError("extension rollout budget exhausted")
        value = self.rollout(
            self.candidate,
            seed,
            object_pose=self.parent["poses"][index],
            video_path=self.media_root / f"candidate-{index:02d}-{seed}.mp4",
        )
        if value["seed"] != seed or value["schedule"] != list(self.candidate):
            raise RuntimeError("extension rollout identity mismatch")
        write_json(receipt, value)
        self.state["episodes_used"] += 1
        if value.get("safety_violation") is not None or value.get("physics_error") is not None:
            self.state["halt_reason"] = {
                "type": "runtime_incident",
                "seed": seed,
                "safety_violation": value.get("safety_violation"),
                "physics_error": value.get("physics_error"),
            }
        self._persist()
        return value

    def run(self) -> dict:
        if self.state["halt_reason"]:
            raise RuntimeError(f"extension halted: {self.state['halt_reason']}")
        for target in (5, 10, 20):
            while len(self.state["rollouts"]) < target:
                self.state["rollouts"].append(self._run_one(len(self.state["rollouts"])))
                self._persist()
                if self.state["halt_reason"]:
                    raise RuntimeError("extension produced a runtime incident")
            successes = sum(successful(item) for item in self.state["rollouts"])
            self.state["verdict"] = stage_verdict(target, successes)
            self._persist()
            if self.state["verdict"] != "continue":
                break
        result = self.summary()
        write_json(self.root / "RESULT.json", result)
        return result

    def summary(self) -> dict:
        rollouts = self.state["rollouts"]
        successes = [item for item in rollouts if successful(item)]
        native = {item["seed"]: item for item in self.parent["native"]}
        matched_native = [native[item["seed"]] for item in successes]
        speedup = None
        if successes:
            speedup = (
                statistics.fmean(rollout_metric_steps(item) for item in matched_native)
                / statistics.fmean(rollout_metric_steps(item) for item in successes)
            )
        incidents = sum(
            item.get("safety_violation") is not None or item.get("physics_error") is not None
            for item in rollouts
        )
        replaces = (
            len(successes) >= self.anchor_successes
            and speedup is not None
            and speedup >= self.anchor_speedup
            and incidents == 0
            and len(rollouts) == 20
        )
        return {
            "schema": "act-vlm-grasp15-extension-result-v1",
            "candidate": list(self.candidate),
            "completed": len(rollouts),
            "successes": len(successes),
            "verdict": self.state["verdict"],
            "matched_native_speedup": speedup,
            "safety_violations": sum(item.get("safety_violation") is not None for item in rollouts),
            "physics_errors": sum(item.get("physics_error") is not None for item in rollouts),
            "anchor": {"schedule": [2.0] * 4, "successes": self.anchor_successes, "matched_native_speedup": self.anchor_speedup},
            "candidate_replaces_anchor": replaces,
            "selected_schedule": list(self.candidate) if replaces else [2.0] * 4,
            "new_candidate_rollouts": len(rollouts),
            "new_native_rollouts": 0,
            "final_bank_opened": False,
        }
