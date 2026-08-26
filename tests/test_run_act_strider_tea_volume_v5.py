import json
from pathlib import Path

from scripts import run_act_strider_tea_volume_v5 as module


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "act_strider_tea_volume_v5"


def _seeds(bank):
    return set(range(bank["start"], bank["start"] + bank["count"]))


def test_success_criterion_hashes_are_frozen_and_current():
    criterion = module.checked_success_criterion(
        EXPERIMENT_ROOT / "SUCCESS_CRITERION.json"
    )
    assert criterion["legacy_bottom_contact_required"] is False
    assert criterion["cup_local_bounds_m"]["z"] == [0.005, 0.08]


def test_v5_banks_are_fresh_and_disjoint_from_v4():
    v5 = json.loads((EXPERIMENT_ROOT / "BANKS.json").read_text())["tasks"]["tea"]
    v4 = json.loads(
        (REPO_ROOT / "experiments" / "act_strider_frontier_v4" / "BANKS.json").read_text()
    )["tasks"]["tea"]

    v5_search, v5_final = _seeds(v5["search"]), _seeds(v5["final"])
    assert not v5_search & v5_final
    assert not v5_search & _seeds(v4["search"])
    assert not v5_search & _seeds(v4["final"])
    assert not v5_final & _seeds(v4["search"])
    assert not v5_final & _seeds(v4["final"])


def test_metric_regression_requires_both_prior_in_cup_trajectories(tmp_path):
    class Runtime:
        def rollout(
            self,
            schedule,
            seed,
            *,
            video_path,
            record_attribution_telemetry,
        ):
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"mp4")
            return {
                "seed": seed,
                "schedule": list(schedule),
                "success": True,
                "attribution_telemetry": [],
            }

    report = module.run_metric_regression(Runtime(), tmp_path)

    assert report["seeds"] == [160500100, 160500109]
    assert report["successes"] == 2
    assert report["episodes"] == 2
    assert report["excluded_from_search_and_final"] is True
    assert (tmp_path / "metric_regression" / "RECEIPT.json").is_file()
