from scripts.run_act_strider_codex_v14_six_schedule import promotion_candidates


def test_promotions_freeze_repaired_phase_and_rank_by_saved_steps():
    selected = [2.0, 1.5, 2.0, 2.0]
    workload = {
        "pre_grasp": 112.0,
        "grasp_lift": 80.0,
        "transport": 37.0,
        "interaction": 57.0,
    }
    proposed, full = promotion_candidates(
        selected,
        workload,
        {"grasp_lift"},
        {tuple(selected)},
        2,
    )

    assert [item["phase"] for item in proposed] == ["pre_grasp", "interaction"]
    assert all(item["phase"] != "grasp_lift" for item in full)
    assert proposed[0]["schedule"] == [2.5, 1.5, 2.0, 2.0]


def test_existing_schedule_is_not_proposed_twice():
    selected = [2.0, 2.0, 2.0, 1.5]
    existing = {
        tuple(selected),
        (2.5, 2.0, 2.0, 1.5),
    }
    proposed, _ = promotion_candidates(
        selected,
        {phase: 1.0 for phase in ("pre_grasp", "grasp_lift", "transport", "interaction")},
        {"interaction"},
        existing,
        2,
    )

    assert all(item["schedule"] != [2.5, 2.0, 2.0, 1.5] for item in proposed)
