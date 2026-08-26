import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "act_strider_tea_volume_v6"


def _seeds(bank):
    return set(range(bank["start"], bank["start"] + bank["count"]))


def test_v6_overlap_criterion_receipt_remains_historical():
    criterion = json.loads((EXPERIMENT_ROOT / "SUCCESS_CRITERION.json").read_text())
    assert criterion["center_inside_required"] is False
    assert criterion["tea_bag_half_extents_m"] == [0.02, 0.02, 0.02]
    assert criterion["files"]["sim_tasks.py"]["sha256"] == (
        "cdc2cf03ae560903503905496b2ce4087269c0963be3c5f2b73a8398f9a538ab"
    )


def test_v6_banks_are_fresh_and_disjoint_from_v4_and_v5():
    v6 = json.loads((EXPERIMENT_ROOT / "BANKS.json").read_text())["tasks"]["tea"]
    prior = []
    for name in ("act_strider_frontier_v4", "act_strider_tea_volume_v5"):
        prior.append(
            json.loads(
                (REPO_ROOT / "experiments" / name / "BANKS.json").read_text()
            )["tasks"]["tea"]
        )
    v6_search, v6_final = _seeds(v6["search"]), _seeds(v6["final"])
    assert not v6_search & v6_final
    for bank in prior:
        assert not v6_search & _seeds(bank["search"])
        assert not v6_search & _seeds(bank["final"])
        assert not v6_final & _seeds(bank["search"])
        assert not v6_final & _seeds(bank["final"])
