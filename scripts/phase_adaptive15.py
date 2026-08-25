#!/usr/bin/env python3
"""Phase-risk-biased speed search with a strict 5->10->15 gate."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path

from one_reset_phase_schedule import (
    PHASES,
    rollout_metric_steps,
    sample_object_pose,
    validate_schedule,
)


STAGES = (5, 10, 15)
MIN_SUCCESSES = {5: 4, 10: 9, 15: 14}
AGGRESSION_LEVELS = (2.0, 2.5, 3.0, 3.5, 4.0)
RISK_CAPS = {"protected": 1.5, "cautious": 2.5, "open": 4.0}
NATIVE = (1.0, 1.0, 1.0, 1.0)


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


def recoverable_workspace_exit(result: dict) -> bool:
    violation = result.get("safety_violation")
    return (
        isinstance(violation, str)
        and violation.startswith("object_")
        and violation.endswith("_outside_preregistered_workspace")
    )


def runtime_incident(result: dict) -> bool:
    return (
        result.get("physics_error") is not None
        or (
            result.get("safety_violation") is not None
            and not recoverable_workspace_exit(result)
        )
    )


def schedule_hash(schedule) -> str:
    return digest(list(validate_schedule(schedule)))


def stage_verdict(completed: int, successes: int) -> str:
    if completed not in MIN_SUCCESSES:
        raise ValueError(f"unsupported staged count: {completed}")
    if successes < MIN_SUCCESSES[completed]:
        return f"rejected_at_{completed}"
    return "qualified" if completed == STAGES[-1] else "continue"


def candidate_verdict(rollouts: list[dict], completed: int) -> str:
    if any(recoverable_workspace_exit(item) for item in rollouts):
        return f"rejected_at_{completed}_workspace_exit"
    return stage_verdict(completed, sum(successful(item) for item in rollouts))


def generate_candidates(phase_risk_labels: dict[str, str]) -> list[dict]:
    """Generate numeric schedules from qualitative phase risk and a speed ladder."""

    if set(phase_risk_labels) != set(PHASES):
        raise ValueError("phase risk labels must cover the exact phase vocabulary")
    caps = []
    for phase in PHASES:
        label = str(phase_risk_labels[phase])
        if label not in RISK_CAPS:
            raise ValueError(f"unsupported risk label for {phase}: {label}")
        caps.append(RISK_CAPS[label])

    definitions = [
        {
            "id": "uniform_2x",
            "family": "uniform_comparator",
            "schedule": [2.0] * len(PHASES),
        },
        {
            "id": "uniform_2p5x",
            "family": "uniform_comparator",
            "schedule": [2.5] * len(PHASES),
        },
    ]
    seen = {tuple(item["schedule"]) for item in definitions}
    for level in AGGRESSION_LEVELS:
        schedule = tuple(min(level, cap) for cap in caps)
        if schedule in seen:
            continue
        seen.add(schedule)
        definitions.append({
            "id": f"phase_risk_level_{str(level).replace('.', 'p')}x",
            "family": "phase_risk_ladder",
            "aggression_level": level,
            "schedule": list(validate_schedule(schedule)),
        })
    return definitions


class Adaptive15Search:
    def __init__(
        self,
        root: Path,
        task: str,
        seeds: list[int],
        budget: int,
        rollout,
        *,
        phase_risk_labels: dict[str, str],
        proposal_receipt: dict,
    ):
        if len(seeds) != 15 or len(set(seeds)) != 15:
            raise ValueError("exactly 15 unique matched search seeds are required")
        self.definitions = generate_candidates(phase_risk_labels)
        maximum = 15 * (1 + len(self.definitions))
        if budget < maximum:
            raise ValueError(f"budget must cover the fail-open maximum of {maximum}")
        self.root = root
        self.task = task
        self.seeds = [int(seed) for seed in seeds]
        self.budget = int(budget)
        self.rollout = rollout
        self.state_path = root / "private" / "state.json"
        self.rollout_root = root / "private" / "rollouts"
        self.media_root = root / "public" / "media"
        identity = {
            "task": task,
            "seeds": self.seeds,
            "budget": self.budget,
            "phase_risk_labels": phase_risk_labels,
            "proposal_receipt": proposal_receipt,
            "candidate_definitions": self.definitions,
            "stages": list(STAGES),
            "minimum_successes": {str(k): v for k, v in MIN_SUCCESSES.items()},
        }
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if any(self.state[key] != value for key, value in identity.items()):
                raise RuntimeError("adaptive15 resume identity mismatch")
            receipts = len(list(self.rollout_root.glob("*/*.json")))
            if receipts != self.state["episodes_used"]:
                self.state["episodes_used"] = receipts
                self._persist()
        else:
            self.state = {
                "schema": "act-phase-adaptive15-search-state-v1",
                **identity,
                "poses": [list(sample_object_pose(task, seed)) for seed in self.seeds],
                "episodes_used": 0,
                "native": [],
                "candidates": {
                    item["id"]: {**item, "rollouts": [], "verdict": None}
                    for item in self.definitions
                },
                "halt_reason": None,
            }
            self._persist()

    def _persist(self) -> None:
        write_json(self.state_path, self.state)

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
        value = self.rollout(
            schedule,
            seed,
            object_pose=self.state["poses"][index],
            video_path=self.media_root / f"{label}-{index:02d}-{seed}.mp4",
        )
        if value["seed"] != seed or value["schedule"] != list(validate_schedule(schedule)):
            raise RuntimeError("rollout returned mismatched identity")
        write_json(receipt, value)
        self.state["episodes_used"] += 1
        if runtime_incident(value):
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
                None
                if not successes
                else statistics.fmean(rollout_metric_steps(item) for item in successes)
            ),
            "safety_violations": sum(
                item.get("safety_violation") is not None for item in rollouts
            ),
            "physics_errors": sum(
                item.get("physics_error") is not None for item in rollouts
            ),
        }

    def run_native(self) -> dict:
        while len(self.state["native"]) < 15:
            index = len(self.state["native"])
            self.state["native"].append(self._run_one(NATIVE, index, "native"))
            self._persist()
            if self.state["halt_reason"]:
                raise RuntimeError("native control produced a runtime incident")
        result = self._summary(NATIVE, self.state["native"], "reference")
        if result["successes"] < 14:
            self.state["halt_reason"] = {"type": "unreliable_native", "summary": result}
            self._persist()
            raise RuntimeError("matched native control is below 14/15")
        return result

    def candidate_summary(self, candidate_id: str) -> dict:
        item = self.state["candidates"][candidate_id]
        result = self._summary(item["schedule"], item["rollouts"], item["verdict"])
        result.update({
            "id": candidate_id,
            "family": item["family"],
            "aggression_level": item.get("aggression_level"),
        })
        native_by_seed = {item["seed"]: item for item in self.state["native"]}
        pairs = [
            (native_by_seed[item["seed"]], item)
            for item in item["rollouts"]
            if successful(item) and successful(native_by_seed[item["seed"]])
        ]
        result["matched_success_pairs"] = len(pairs)
        result["matched_native_speedup"] = (
            None
            if not pairs
            else statistics.fmean(rollout_metric_steps(a) for a, _ in pairs)
            / statistics.fmean(rollout_metric_steps(b) for _, b in pairs)
        )
        return result

    def run(self) -> dict:
        if self.state["halt_reason"]:
            raise RuntimeError(f"lane is halted: {self.state['halt_reason']}")
        native = self.run_native()
        active = [item["id"] for item in self.definitions]
        for target in STAGES:
            next_active = []
            for candidate_id in active:
                item = self.state["candidates"][candidate_id]
                while len(item["rollouts"]) < target:
                    index = len(item["rollouts"])
                    item["rollouts"].append(
                        self._run_one(item["schedule"], index, candidate_id)
                    )
                    self._persist()
                    if self.state["halt_reason"]:
                        raise RuntimeError("candidate produced a runtime incident")
                item["verdict"] = candidate_verdict(item["rollouts"], target)
                self._persist()
                if item["verdict"] == "continue":
                    next_active.append(candidate_id)
            active = next_active

        candidates = [
            self.candidate_summary(item["id"]) for item in self.definitions
        ]
        qualified = [item for item in candidates if item["verdict"] == "qualified"]
        uniforms = [item for item in qualified if item["family"] == "uniform_comparator"]
        best_uniform = (
            None
            if not uniforms
            else max(
                uniforms,
                key=lambda item: (
                    item["successes"], item["matched_native_speedup"],
                    -len(set(item["schedule"])),
                ),
            )
        )
        eligible = qualified
        if best_uniform is not None:
            eligible = [
                item
                for item in qualified
                if item["successes"] >= best_uniform["successes"]
                and item["matched_native_speedup"] >= best_uniform["matched_native_speedup"]
            ]
        selected = (
            None
            if not eligible
            else max(
                eligible,
                key=lambda item: (
                    item["successes"], item["matched_native_speedup"],
                    -len(set(item["schedule"])),
                ),
            )
        )
        result = {
            "schema": "act-phase-adaptive15-search-result-v1",
            "native": native,
            "candidates": candidates,
            "best_qualified_uniform": best_uniform,
            "selected_search_incumbent": selected,
            "fallback": list(NATIVE) if selected is None else selected["schedule"],
            "episodes_used": self.state["episodes_used"],
            "maximum_rollout_budget": self.budget,
            "gate": {
                "stages": list(STAGES),
                "minimum_successes": {str(k): v for k, v in MIN_SUCCESSES.items()},
                "qualification": "search_only_not_certification",
            },
            "final_bank_opened": False,
            "deployment_claim": False,
        }
        write_json(self.root / "public" / "RESULT.json", result)
        return result
