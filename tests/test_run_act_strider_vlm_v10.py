import json
from pathlib import Path

from scripts import run_act_strider_frontier_v4 as v4
from scripts.codex_agent_failure_attribution import sanitized_record
from scripts.run_act_strider_vlm_v10 import ValidVideoLedger, run_search


class FakeRuntime:
    def __init__(self, *, physics_error_seed=None):
        self.physics_error_seed = physics_error_seed

    def rollout(
        self,
        schedule,
        seed,
        *,
        video_path,
        record_attribution_telemetry=False,
    ):
        schedule = list(map(float, schedule))
        video_path = Path(video_path)
        if seed == self.physics_error_seed and schedule == [2.0] * 4:
            return {
                "seed": seed,
                "schedule": schedule,
                "success": False,
                "physics_steps": 1,
                "first_success_step": None,
                "safety_violation": None,
                "physics_error": "QACC",
                "phase_decisions": [],
                "video_path": str(video_path),
            }
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake-mp4")
        index = seed % 100
        if schedule == [2.5] * 4:
            success = index < 17
        elif schedule == [2.5, 2.5, 2.5, 2.0]:
            success = index < 18
        else:
            success = True
        steps = round(400 / (sum(schedule) / 4))
        record = {
            "seed": seed,
            "schedule": schedule,
            "success": success,
            "physics_steps": steps,
            "first_success_step": steps if success else None,
            "safety_violation": None,
            "phase_decisions": [
                {"phase": phase, "physics_step": i * 25, "speed": schedule[i]}
                for i, phase in enumerate(v4.PHASES)
            ],
            "video_path": str(video_path),
        }
        if record_attribution_telemetry:
            record["attribution_telemetry"] = [
                {
                    "physics_step": i * 25 + 1,
                    "policy_time": float(i),
                    "observed_phase": phase,
                    "task_reward": 0.0,
                    "object_positions": [[0.0, 0.0, 0.0]],
                }
                for i, phase in enumerate(v4.PHASES)
            ]
        return record


class FakeAttributor:
    def __init__(self):
        self.calls = 0

    def diagnose(self, **kwargs):
        self.calls += 1
        candidate = kwargs["candidate_record"]
        return {
            "schema": "fake",
            "seed": candidate["seed"],
            "attribution": {
                "observed_failure_phase": "interaction",
                "causal_phase": "grasp_lift",
                "confidence": 0.9,
                "evidence": "grasp becomes loose before terminal interaction",
            },
        }

    def close(self):
        pass


def test_search_compares_vlm_cause_to_telemetry_and_selects_better_repair(tmp_path):
    seeds = list(range(100, 140))
    ledger = ValidVideoLedger(FakeRuntime(), tmp_path, seeds, list(range(1000, 1070)))
    result = run_search(ledger, "tea", FakeAttributor())

    assert result["rejected_uniform"]["schedule"] == [2.5] * 4
    assert result["vlm_attribution"]["selected_phase"] == "grasp_lift"
    assert result["telemetry_attribution"]["selected_phase"] == "interaction"
    assert result["attribution_comparison"]["vlm_and_telemetry_agree"] is False
    assert result["attribution_comparison"]["distinct_repairs_tested"] == 2
    assert result["selected_role"] == "vlm_causal_repair"
    assert result["selected_schedule"] == [2.5, 2.0, 2.5, 2.5]
    assert result["search_valid_rollouts"] == 80


def test_final_excludes_qacc_pair_and_uses_registered_reserve(tmp_path):
    primary = list(range(1000, 1050))
    reserve = list(range(1050, 1070))
    ledger = ValidVideoLedger(
        FakeRuntime(physics_error_seed=1003),
        tmp_path,
        list(range(100, 140)),
        primary + reserve,
    )
    final = ledger.evaluate_final_paired(
        {"native_1x": [1.0] * 4, "uniform_incumbent": [2.0] * 4}
    )

    assert len(final["valid_pair_seeds"]) == 50
    assert 1003 not in final["valid_pair_seeds"]
    assert 1050 in final["valid_pair_seeds"]
    assert final["simulator_invalid_pairs"] == [
        {
            "seed": 1003,
            "reason": "physics_error",
            "details": [
                {
                    "schedule_sha256": v4.schedule_sha256([2.0] * 4),
                    "physics_error": "QACC",
                }
            ],
            "counted_in_scientific_denominator": False,
        }
    ]
    assert final["scientific_rollouts"] == 100
    assert json.loads((tmp_path / "final" / "controllers" / v4.schedule_sha256([2.0] * 4) / "states" / "1003.json").read_text())["simulator_invalid"]


def test_codex_record_is_sanitized_of_simulator_telemetry(tmp_path):
    record = FakeRuntime().rollout(
        [2.0] * 4,
        100,
        video_path=tmp_path / "unused.mp4",
        record_attribution_telemetry=True,
    )
    sanitized = sanitized_record(record)

    assert set(sanitized) == {
        "seed",
        "schedule",
        "success",
        "physics_steps",
        "first_success_step",
        "phase_timeline",
    }
    assert "attribution_telemetry" not in sanitized
    assert "object_positions" not in json.dumps(sanitized)


def test_vlm_attribution_is_capped_at_three_matched_failures(tmp_path):
    class FiveFailureRuntime(FakeRuntime):
        def rollout(self, schedule, seed, **kwargs):
            record = super().rollout(schedule, seed, **kwargs)
            if list(map(float, schedule)) == [2.5] * 4:
                record["success"] = seed % 100 < 15
                record["first_success_step"] = (
                    record["physics_steps"] if record["success"] else None
                )
            return record

    attributor = FakeAttributor()
    ledger = ValidVideoLedger(
        FiveFailureRuntime(), tmp_path, list(range(100, 140)), list(range(1000, 1070))
    )
    result = run_search(ledger, "tea", attributor)

    assert result["rejected_uniform"]["summary"]["successes"] == 15
    assert attributor.calls == 3
    assert len(result["vlm_pair_receipts"]) == 3
