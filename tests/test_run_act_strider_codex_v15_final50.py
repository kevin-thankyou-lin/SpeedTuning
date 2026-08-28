import json
from pathlib import Path

from scripts import run_act_strider_codex_v15_final50 as v15
from scripts import run_act_strider_frontier_v4 as v4
from scripts import run_act_strider_vlm_v10 as base


def selection(selected, incumbent):
    return {
        "selected_schedule": selected,
        "uniform_incumbent": None if incumbent is None else {"schedule": incumbent},
    }


def test_pick_registry_has_four_unique_controllers():
    schedules = v15.named_schedules(
        selection([2.0, 1.5, 2.0, 2.0], [1.5] * 4),
        {"selected_schedule": [2.5, 1.5, 2.0, 2.0]},
    )
    assert len({v4.schedule_sha256(value) for value in schedules.values()}) == 4


def test_tea_registry_deduplicates_uniform_and_v13():
    schedules = v15.named_schedules(
        selection([2.0] * 4, [2.0] * 4),
        {"selected_schedule": [2.0, 2.5, 2.0, 2.0]},
    )
    assert schedules["uniform_incumbent"] == schedules["strider_v13"]
    assert len({v4.schedule_sha256(value) for value in schedules.values()}) == 3


def test_insertion_registry_deduplicates_v13_and_v14():
    schedules = v15.named_schedules(
        selection([2.0, 2.0, 2.0, 1.5], [1.5] * 4),
        {"selected_schedule": [2.0, 2.0, 2.0, 1.5]},
    )
    assert schedules["strider_v13"] == schedules["strider_v14"]
    assert len({v4.schedule_sha256(value) for value in schedules.values()}) == 3


def test_v15_banks_have_fifty_primary_and_disjoint_reserves():
    path = Path("experiments/act_strider_codex_v15_final50/BANKS.json")
    banks = json.loads(path.read_text())
    seen = set()
    for task in ("pick", "tea", "insertion"):
        task_banks = banks["tasks"][task]
        primary = base._range(task_banks["final_primary"])
        reserve = base._range(task_banks["final_reserve"])
        assert len(primary) == 50
        assert len(reserve) == 20
        assert not seen.intersection(primary + reserve)
        seen.update(primary + reserve)
