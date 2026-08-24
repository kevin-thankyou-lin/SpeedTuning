#!/usr/bin/env python3
"""Receipt-bearing 5->10->20 reliability gate for phase-speed frontiers."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path

from one_reset_phase_schedule import (
    ALLOWED_SPEEDS,
    PHASES,
    estimate_phase_workload,
    rollout_metric_steps,
    sample_object_pose,
    validate_schedule,
)

ANCHOR = (2.0, 2.0, 2.0, 2.0)
NATIVE = (1.0, 1.0, 1.0, 1.0)
STAGES = (5, 10, 20)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def successful(result: dict) -> bool:
    return (
        bool(result.get("success"))
        and result.get("safety_violation") is None
        and result.get("physics_error") is None
    )


def schedule_hash(schedule) -> str:
    return digest(list(validate_schedule(schedule)))


def stage_verdict(completed: int, successes: int) -> str:
    if completed == 5:
        return "continue" if successes >= 3 else "rejected_at_5"
    if completed == 10:
        return "continue" if successes >= 9 else "rejected_at_10"
    if completed == 20:
        return "qualified" if successes >= 18 else "rejected_at_20"
    raise ValueError(f"unsupported staged count: {completed}")


class StagedFrontier:
    def __init__(self, root: Path, task: str, seeds: list[int], budget: int, rollout):
        if len(seeds) != 20 or len(set(seeds)) != 20:
            raise ValueError("exactly 20 unique matched search seeds are required")
        if budget < 60:
            raise ValueError("budget must cover native controls, anchor, and one challenger")
        self.root = root
        self.task = task
        self.seeds = [int(seed) for seed in seeds]
        self.budget = int(budget)
        self.rollout = rollout
        self.state_path = root / "private" / "state.json"
        self.rollout_root = root / "private" / "rollouts"
        self.media_root = root / "public" / "media"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            expected = {"task": task, "seeds": self.seeds, "budget": self.budget}
            if any(self.state[key] != value for key, value in expected.items()):
                raise RuntimeError("staged frontier resume identity mismatch")
            receipt_count = len(list((root / "private" / "rollouts").glob("*/*.json")))
            if receipt_count != self.state["episodes_used"]:
                # A crash may land after the immutable per-rollout receipt but
                # before the aggregate state update. Receipts are authoritative:
                # charge them to the budget and reuse them; never rerun them.
                self.state["episodes_used"] = receipt_count
                self._persist()
        else:
            poses = [sample_object_pose(task, seed) for seed in self.seeds]
            self.state = {
                "schema": "act-staged-vlm-frontier-v1",
                "task": task,
                "seeds": self.seeds,
                "poses": [list(pose) for pose in poses],
                "budget": self.budget,
                "episodes_used": 0,
                "native": [],
                "candidates": {},
                "candidate_order": [],
                "incumbent_hash": None,
                "first_rejected_uniform_hash": None,
                "repaired_phases": [],
                "midpoint_used": False,
                "halt_reason": None,
            }
            self._persist()

    def _persist(self):
        write_json(self.state_path, self.state)

    def _public_rollout(self, result: dict) -> dict:
        return {
            key: result.get(key)
            for key in (
                "seed", "schedule", "success", "raw_task_success", "physics_steps",
                "first_success_step", "safety_violation", "physics_error",
                "phase_decisions", "video_path",
            )
            if key in result
        }

    def _run_one(self, schedule, index: int, label: str) -> dict:
        if self.state["episodes_used"] >= self.budget:
            raise RuntimeError("pre-final rollout budget exhausted")
        seed = self.seeds[index]
        identifier = schedule_hash(schedule)
        receipt = self.rollout_root / identifier / f"{index:02d}-{seed}.json"
        if receipt.exists():
            value = json.loads(receipt.read_text())
            if value["seed"] != seed or value["schedule"] != list(validate_schedule(schedule)):
                raise RuntimeError("cached rollout receipt identity mismatch")
            return value
        media = self.media_root / f"{label}-{index:02d}-{seed}.mp4"
        value = self.rollout(
            schedule,
            seed,
            object_pose=self.state["poses"][index],
            video_path=media,
        )
        if value["seed"] != seed or value["schedule"] != list(validate_schedule(schedule)):
            raise RuntimeError("rollout returned mismatched identity")
        write_json(receipt, value)
        self.state["episodes_used"] += 1
        if value.get("safety_violation") is not None or value.get("physics_error") is not None:
            self.state["halt_reason"] = {
                "type": "runtime_incident",
                "schedule_hash": identifier,
                "seed": seed,
                "safety_violation": value.get("safety_violation"),
                "physics_error": value.get("physics_error"),
            }
        self._persist()
        return value

    @staticmethod
    def _summary(schedule, rollouts: list[dict], verdict: str) -> dict:
        successes = [item for item in rollouts if successful(item)]
        return {
            "schedule_hash": schedule_hash(schedule),
            "schedule": list(validate_schedule(schedule)),
            "completed": len(rollouts),
            "successes": len(successes),
            "verdict": verdict,
            "successful_mean_first_success_steps": (
                None if not successes else statistics.fmean(rollout_metric_steps(x) for x in successes)
            ),
            "safety_violations": sum(x.get("safety_violation") is not None for x in rollouts),
            "physics_errors": sum(x.get("physics_error") is not None for x in rollouts),
        }

    def run_native(self) -> dict:
        while len(self.state["native"]) < 20:
            index = len(self.state["native"])
            self.state["native"].append(self._run_one(NATIVE, index, "native"))
            self._persist()
            if self.state["halt_reason"]:
                raise RuntimeError("native control produced a runtime incident")
        summary = self._summary(NATIVE, self.state["native"], "reference")
        if summary["successes"] < 18:
            self.state["halt_reason"] = {"type": "unreliable_native", "summary": summary}
            self._persist()
            raise RuntimeError("matched native control is below 18/20")
        return summary

    def _expected_kind(self, schedule, kind: str, phase: str | None) -> None:
        schedule = validate_schedule(schedule)
        order = self.state["candidate_order"]
        if not order:
            if kind != "anchor" or schedule != ANCHOR:
                raise ValueError("first candidate must be the blinded uniform 2x anchor")
            return
        incumbent = self.state["candidates"][self.state["incumbent_hash"]]
        base = tuple(incumbent["schedule"])
        if kind == "uniform":
            if self.state["first_rejected_uniform_hash"] is not None:
                raise ValueError("uniform ladder already rejected; causal repair is required")
            if len(set(base)) != 1:
                raise ValueError("uniform expansion requires a uniform incumbent")
            rung = ALLOWED_SPEEDS.index(base[0]) + 1
            expected = (ALLOWED_SPEEDS[rung],) * 4
            if schedule != expected:
                raise ValueError(f"next uniform schedule must be {expected}")
            return
        if kind == "repair":
            rejected_hash = self.state["first_rejected_uniform_hash"]
            if rejected_hash is None:
                raise ValueError("repair requires the first rejected uniform candidate")
            if phase not in PHASES:
                raise ValueError("repair phase is invalid")
            rejected = list(self.state["candidates"][rejected_hash]["schedule"])
            index = PHASES.index(phase)
            rung = ALLOWED_SPEEDS.index(rejected[index])
            if rung == 0:
                raise ValueError("attributed phase is already at native speed")
            rejected[index] = ALLOWED_SPEEDS[rung - 1]
            if schedule != tuple(rejected):
                raise ValueError(f"repair must lower only {phase} to {rejected[index]}")
            return
        if kind == "promote":
            ranking = self.promotion_scores()["ordered_phases"]
            if not ranking or phase != ranking[0]:
                raise ValueError(f"promotion phase must be {ranking[0] if ranking else None}")
            expected = list(base)
            index = PHASES.index(phase)
            expected[index] = ALLOWED_SPEEDS[ALLOWED_SPEEDS.index(expected[index]) + 1]
            if schedule != tuple(expected):
                raise ValueError(f"promotion schedule must be {tuple(expected)}")
            return
        raise ValueError(f"unsupported proposal kind: {kind}")

    def gate(self, schedule, *, kind: str, phase: str | None = None, evidence: str | None = None) -> dict:
        if self.state["halt_reason"]:
            raise RuntimeError(f"lane is halted: {self.state['halt_reason']}")
        self.run_native()
        schedule = validate_schedule(schedule)
        identifier = schedule_hash(schedule)
        existing = self.state["candidates"].get(identifier)
        if existing is None:
            self._expected_kind(schedule, kind, phase)
        elif existing["kind"] != kind or existing["phase"] != phase:
            raise ValueError("cached candidate proposal identity mismatch")
        if kind == "repair" and not str(evidence or "").strip():
            raise ValueError("causal repair requires VLM evidence")
        if identifier not in self.state["candidates"]:
            self.state["candidates"][identifier] = {
                "schedule": list(schedule), "kind": kind, "phase": phase,
                "evidence": evidence, "rollouts": [], "verdict": None,
            }
            self.state["candidate_order"].append(identifier)
            self._persist()
        candidate = self.state["candidates"][identifier]
        for target in STAGES:
            while len(candidate["rollouts"]) < target:
                index = len(candidate["rollouts"])
                candidate["rollouts"].append(self._run_one(schedule, index, identifier[:10]))
                self._persist()
                if self.state["halt_reason"]:
                    raise RuntimeError("candidate produced a runtime incident")
            successes = sum(successful(item) for item in candidate["rollouts"])
            verdict = stage_verdict(target, successes)
            candidate["verdict"] = verdict
            self._persist()
            if verdict != "continue":
                break
        if candidate["verdict"] == "qualified":
            self.state["incumbent_hash"] = identifier
            if kind == "repair" and phase not in self.state["repaired_phases"]:
                self.state["repaired_phases"].append(phase)
        elif kind == "uniform" and self.state["first_rejected_uniform_hash"] is None:
            self.state["first_rejected_uniform_hash"] = identifier
        self._persist()
        return self.candidate_summary(identifier)

    def candidate_summary(self, identifier: str) -> dict:
        item = self.state["candidates"][identifier]
        result = self._summary(item["schedule"], item["rollouts"], item["verdict"])
        result.update({"kind": item["kind"], "phase": item["phase"], "evidence": item["evidence"]})
        native_by_seed = {item["seed"]: item for item in self.state["native"]}
        pairs = [
            (native_by_seed[item["seed"]], item)
            for item in item["rollouts"]
            if successful(item) and successful(native_by_seed[item["seed"]])
        ]
        result["matched_success_pairs"] = len(pairs)
        result["matched_native_speedup"] = (
            None if not pairs else
            statistics.fmean(rollout_metric_steps(a) for a, _ in pairs)
            / statistics.fmean(rollout_metric_steps(b) for _, b in pairs)
        )
        return result

    def promotion_scores(self) -> dict:
        if self.state["incumbent_hash"] is None:
            return {"ordered_phases": [], "scores": {}}
        item = self.state["candidates"][self.state["incumbent_hash"]]
        successes = [x for x in item["rollouts"] if successful(x)]
        workloads = {phase: statistics.fmean(estimate_phase_workload(x)[phase] for x in successes) for phase in PHASES}
        schedule = item["schedule"]
        scores = {}
        for phase, old in zip(PHASES, schedule):
            if phase in self.state["repaired_phases"] or old == ALLOWED_SPEEDS[-1]:
                continue
            new = ALLOWED_SPEEDS[ALLOWED_SPEEDS.index(old) + 1]
            scores[phase] = workloads[phase] * (1.0 / old - 1.0 / new)
        ordered = sorted(scores, key=lambda phase: (-scores[phase], PHASES.index(phase)))
        return {"ordered_phases": ordered, "scores": scores, "aggregation": "mean successful current-run phase workload"}

    def info(self) -> dict:
        return {
            "schema": self.state["schema"],
            "episodes_used": self.state["episodes_used"],
            "budget": self.budget,
            "native": None if len(self.state["native"]) < 20 else self._summary(NATIVE, self.state["native"], "reference"),
            "candidates": [self.candidate_summary(x) for x in self.state["candidate_order"]],
            "incumbent_hash": self.state["incumbent_hash"],
            "first_rejected_uniform_hash": self.state["first_rejected_uniform_hash"],
            "repaired_phases": self.state["repaired_phases"],
            "promotion_scores": self.promotion_scores(),
            "halt_reason": self.state["halt_reason"],
        }

    def finalize(self) -> dict:
        qualified = [self.candidate_summary(x) for x in self.state["candidate_order"] if self.state["candidates"][x]["verdict"] == "qualified"]
        if not qualified:
            selection = {"selected": None, "fallback": list(NATIVE), "reason": "no qualified accelerated schedule"}
        else:
            anchor = next((x for x in qualified if x["schedule"] == list(ANCHOR)), None)
            eligible = [x for x in qualified if anchor is None or (x["successes"] >= anchor["successes"] and x["matched_native_speedup"] >= anchor["matched_native_speedup"])]
            selected = max(eligible, key=lambda x: (x["successes"], x["matched_native_speedup"], -len(set(x["schedule"]))))
            selection = {"selected": selected, "uniform_anchor": anchor, "fallback": list(ANCHOR) if anchor else list(NATIVE), "final_bank_opened": False, "qualification": "continued_search_only_not_certified"}
        write_json(self.root / "public" / "SELECTION.json", selection)
        return selection
