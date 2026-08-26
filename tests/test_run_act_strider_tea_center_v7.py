import json
from pathlib import Path

from scripts import run_act_strider_tea_center_v7 as module
from scripts import run_act_strider_tea_volume_v5 as implementation


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "act_strider_tea_center_v7"


def _seeds(bank):
    return set(range(bank["start"], bank["start"] + bank["count"]))


def test_v7_center_criterion_hashes_are_frozen_and_current():
    old_schema = implementation.SUCCESS_CRITERION_SCHEMA
    try:
        implementation.SUCCESS_CRITERION_SCHEMA = "tea-cup-center-success-v1"
        criterion = implementation.checked_success_criterion(
            EXPERIMENT_ROOT / "SUCCESS_CRITERION.json"
        )
    finally:
        implementation.SUCCESS_CRITERION_SCHEMA = old_schema
    assert criterion["center_inside_required"] is True
    assert criterion["overlap_only_is_success"] is False


def test_v7_banks_are_fresh_and_disjoint_from_v4_v5_v6():
    v7 = json.loads((EXPERIMENT_ROOT / "BANKS.json").read_text())["tasks"]["tea"]
    prior = []
    for name in (
        "act_strider_frontier_v4",
        "act_strider_tea_volume_v5",
        "act_strider_tea_volume_v6",
    ):
        prior.append(
            json.loads((REPO_ROOT / "experiments" / name / "BANKS.json").read_text())[
                "tasks"
            ]["tea"]
        )
    v7_search, v7_final = _seeds(v7["search"]), _seeds(v7["final"])
    assert not v7_search & v7_final
    for bank in prior:
        assert not v7_search & _seeds(bank["search"])
        assert not v7_search & _seeds(bank["final"])
        assert not v7_final & _seeds(bank["search"])
        assert not v7_final & _seeds(bank["final"])


def test_v7_semantic_regression_rejects_overlap_only(tmp_path):
    old_schema = implementation.METRIC_REGRESSION_SCHEMA
    try:
        implementation.METRIC_REGRESSION_SCHEMA = (
            "tea-cup-center-semantic-regression-v1"
        )
        report = module.run_center_semantic_regression(None, tmp_path)
    finally:
        implementation.METRIC_REGRESSION_SCHEMA = old_schema
    cases = {case["name"]: case for case in report["cases"]}
    assert cases["center_inside"]["center_inside"] is True
    assert cases["rim_overlap_only"]["oriented_boxes_overlap"] is True
    assert cases["rim_overlap_only"]["center_inside"] is False
    assert cases["side_overlap_only"]["oriented_boxes_overlap"] is True
    assert cases["side_overlap_only"]["center_inside"] is False
    assert report["episodes"] == 0
