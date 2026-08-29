import json
from pathlib import Path

from scripts import run_act_strider_representative_confirmation_v17 as v17


def summary(successes, throughput=1.0, safety=0, physics=0):
    return {
        "successes": successes,
        "achieved_throughput_per_step": throughput,
        "safety_violations": safety,
        "physics_errors": physics,
    }


def test_four_pose_gate_continues_close_results_and_rejects_futility():
    assert v17.gate_decision(
        4, {"uniform": summary(4), "adaptive": summary(3, 2.0)}
    ) == "continue"
    assert v17.gate_decision(
        4, {"uniform": summary(4), "adaptive": summary(2, 2.0)}
    ) == "reject_adaptive_futility"


def test_eight_pose_gate_is_reliability_first():
    assert v17.gate_decision(
        8, {"uniform": summary(8), "adaptive": summary(7, 2.0)}
    ) == "reject_adaptive_paired_reliability_regression"
    assert v17.gate_decision(
        8, {"uniform": summary(7), "adaptive": summary(7, 1.03)}
    ) == "select_adaptive"
    assert v17.gate_decision(
        8, {"uniform": summary(6), "adaptive": summary(7, 1.0)}
    ) == "select_adaptive_uniform_ineligible"


def test_incident_rejects_adaptive_at_every_stage():
    for stage in (4, 8):
        assert v17.gate_decision(
            stage, {"uniform": summary(stage), "adaptive": summary(stage, 2.0, safety=1)}
        ) == "reject_adaptive_incident"


def test_early_rejection_keeps_qualified_uniform_or_native_fallback():
    summaries = {"uniform": summary(4), "adaptive": summary(2)}
    assert v17.selected_name(
        "reject_adaptive_futility",
        summaries,
        stage=4,
        native_deployment_fallback=False,
    ) == "uniform"
    assert v17.selected_name(
        "reject_adaptive_futility",
        summaries,
        stage=4,
        native_deployment_fallback=True,
    ) == "native"


class FakeRuntime:
    def __init__(self):
        self.calls = 0

    def rollout(self, schedule, seed, *, object_pose, video_path, record_attribution_telemetry):
        self.calls += 1
        video_path.write_bytes(b"video")
        return {
            "seed": seed,
            "schedule": schedule,
            "success": True,
            "physics_steps": 10,
            "first_success_step": 10,
            "safety_violation": None,
            "video_path": str(video_path),
        }


def test_representative_pose_cache_does_not_duplicate_rollout(tmp_path: Path):
    runtime = FakeRuntime()
    panel = {"panel_ids": [11], "object_pose_vectors": [[0.1] * 7]}
    ledger = v17.RepresentativePoseLedger(runtime, tmp_path, panel)
    schedule = [1.0] * 4
    first, first_ran = ledger.run_or_load("uniform", schedule, 11)
    second, second_ran = ledger.run_or_load("uniform", schedule, 11)
    assert first == second
    assert first_ran is True
    assert second_ran is False
    assert runtime.calls == 1
    state = json.loads(next(tmp_path.glob("controllers/*/states/11.json")).read_text())
    assert state["representative_pose"] == [0.1] * 7
