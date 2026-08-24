import json

import pytest

from scripts import staged_vlm_frontier as module


def fake_rollout(schedule, seed, *, object_pose=None, video_path=None):
    del object_pose
    speed = float(schedule[0])
    success = not (speed >= 2.5 and seed % 20 in (1, 6, 7))
    steps = int(300 / max(speed, 1))
    return {
        "task": "pick_and_place", "seed": seed, "schedule": list(schedule),
        "success": success, "raw_task_success": success,
        "physics_steps": steps, "first_success_step": steps if success else None,
        "safety_violation": None,
        "phase_decisions": [
            {"phase": phase, "physics_step": i * steps // 4, "speed": schedule[i]}
            for i, phase in enumerate(module.PHASES)
        ],
        "video_path": None if video_path is None else str(video_path),
    }


@pytest.fixture
def frontier(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "sample_object_pose", lambda task, seed: (float(seed),) * 7)
    return module.StagedFrontier(tmp_path, "pick_and_place", list(range(100, 120)), 80, fake_rollout)


def test_stage_thresholds():
    assert module.stage_verdict(5, 2) == "rejected_at_5"
    assert module.stage_verdict(5, 3) == "continue"
    assert module.stage_verdict(10, 8) == "rejected_at_10"
    assert module.stage_verdict(10, 9) == "continue"
    assert module.stage_verdict(20, 17) == "rejected_at_20"
    assert module.stage_verdict(20, 18) == "qualified"


def test_anchor_then_uniform_reject_then_single_phase_repair(frontier):
    anchor = frontier.gate([2, 2, 2, 2], kind="anchor")
    assert anchor["verdict"] == "qualified"
    assert frontier.state["episodes_used"] == 40
    rejected = frontier.gate([2.5, 2.5, 2.5, 2.5], kind="uniform")
    assert rejected["verdict"] == "rejected_at_10"
    assert frontier.state["episodes_used"] == 50
    repaired = frontier.gate(
        [2, 2.5, 2.5, 2.5], kind="repair", phase="pre_grasp",
        evidence="earliest divergence is approach overshoot",
    )
    assert repaired["verdict"] == "qualified"
    assert frontier.state["episodes_used"] == 70
    assert frontier.state["repaired_phases"] == ["pre_grasp"]


def test_receipts_prevent_reruns(frontier):
    first = frontier.gate([2, 2, 2, 2], kind="anchor")
    used = frontier.state["episodes_used"]
    second = frontier.gate([2, 2, 2, 2], kind="anchor")
    assert first == second
    assert frontier.state["episodes_used"] == used
    assert len(list((frontier.root / "private/rollouts").glob("*/*.json"))) == 40


def test_resume_charges_receipt_written_before_aggregate_state(frontier):
    frontier.gate([2, 2, 2, 2], kind="anchor")
    state = json.loads(frontier.state_path.read_text())
    state["episodes_used"] -= 1
    module.write_json(frontier.state_path, state)

    resumed = module.StagedFrontier(
        frontier.root, "pick_and_place", list(range(100, 120)), 80, fake_rollout
    )

    assert resumed.state["episodes_used"] == 40


def test_selection_never_downgrades_uniform(frontier):
    frontier.gate([2, 2, 2, 2], kind="anchor")
    frontier.gate([2.5, 2.5, 2.5, 2.5], kind="uniform")
    frontier.gate([2, 2.5, 2.5, 2.5], kind="repair", phase="pre_grasp", evidence="overshoot")
    result = frontier.finalize()
    assert result["selected"]["successes"] >= result["uniform_anchor"]["successes"]
    assert result["selected"]["matched_native_speedup"] >= result["uniform_anchor"]["matched_native_speedup"]
    assert json.loads((frontier.root / "public/SELECTION.json").read_text()) == result


def test_runtime_incident_halts(frontier):
    original = frontier.rollout
    def unsafe(*args, **kwargs):
        result = original(*args, **kwargs)
        result["safety_violation"] = "workspace"
        return result
    frontier.rollout = unsafe
    with pytest.raises(RuntimeError, match="runtime incident"):
        frontier.run_native()
    assert frontier.state["halt_reason"]["type"] == "runtime_incident"
