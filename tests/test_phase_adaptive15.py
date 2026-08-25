import json

from scripts import phase_adaptive15 as module


LABELS = {
    "pre_grasp": "cautious",
    "grasp_lift": "protected",
    "transport": "open",
    "interaction": "open",
}


def fake_rollout(schedule, seed, *, object_pose=None, video_path=None):
    del object_pose
    schedule = list(schedule)
    failure = schedule == [2.5, 2.5, 2.5, 2.5] and seed % 15 in (0, 1)
    steps = int(320 / (sum(schedule) / len(schedule)))
    return {
        "seed": seed,
        "schedule": schedule,
        "success": not failure,
        "physics_steps": steps,
        "first_success_step": None if failure else steps,
        "safety_violation": None,
        "physics_error": None,
        "video_path": None if video_path is None else str(video_path),
    }


def test_strict_5_to_10_to_15_thresholds():
    assert module.stage_verdict(5, 3) == "rejected_at_5"
    assert module.stage_verdict(5, 4) == "continue"
    assert module.stage_verdict(10, 8) == "rejected_at_10"
    assert module.stage_verdict(10, 9) == "continue"
    assert module.stage_verdict(15, 13) == "rejected_at_15"
    assert module.stage_verdict(15, 14) == "qualified"


def test_workspace_exit_is_failed_evidence_not_lane_halting_incident():
    result = fake_rollout([3, 1.5, 4, 4], 3)
    result["success"] = False
    result["safety_violation"] = "object_0_outside_preregistered_workspace"
    assert not module.successful(result)
    assert module.recoverable_workspace_exit(result)
    assert not module.runtime_incident(result)
    assert module.candidate_verdict([result], 5) == "rejected_at_5_workspace_exit"

    result["safety_violation"] = "unclassified_controller_fault"
    assert module.runtime_incident(result)


def test_phase_labels_generate_ladder_instead_of_one_fixed_schedule():
    candidates = module.generate_candidates(LABELS)
    assert candidates[:2] == [
        {"id": "uniform_2x", "family": "uniform_comparator", "schedule": [2.0] * 4},
        {"id": "uniform_2p5x", "family": "uniform_comparator", "schedule": [2.5] * 4},
    ]
    assert [2.5, 1.5, 3.5, 3.5] in [item["schedule"] for item in candidates]
    assert [2.5, 1.5, 4.0, 4.0] in [item["schedule"] for item in candidates]


def test_search_prunes_bad_uniform_after_five_and_preserves_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module, "sample_object_pose", lambda task, seed: (float(seed),) * 7
    )
    search = module.Adaptive15Search(
        tmp_path,
        "pick_and_place",
        list(range(15)),
        120,
        fake_rollout,
        phase_risk_labels=LABELS,
        proposal_receipt={"schema": "test"},
    )
    result = search.run()
    uniform = next(item for item in result["candidates"] if item["id"] == "uniform_2p5x")
    assert uniform["completed"] == 5
    assert uniform["verdict"] == "rejected_at_5"
    assert result["selected_search_incumbent"]["family"] == "phase_risk_ladder"
    assert not result["final_bank_opened"]
    assert len(list((tmp_path / "private/rollouts").glob("*/*.json"))) == result["episodes_used"]

    resumed = module.Adaptive15Search(
        tmp_path,
        "pick_and_place",
        list(range(15)),
        120,
        fake_rollout,
        phase_risk_labels=LABELS,
        proposal_receipt={"schema": "test"},
    )
    assert resumed.run() == result
    assert json.loads((tmp_path / "public/RESULT.json").read_text()) == result
