import json
from pathlib import Path

from scripts import run_act_strider_tea_release_v9 as module
from scripts import run_act_strider_tea_volume_v5 as implementation


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "act_strider_tea_release_v9"


def _seeds(bank):
    return set(range(bank["start"], bank["start"] + bank["count"]))


def _report(schedule, successes, throughput, qualified=True):
    return {
        "schedule": list(schedule),
        "schedule_sha256": module.v4.schedule_sha256(schedule),
        "qualified": qualified,
        "summary": {
            "successes": successes,
            "success_rate": successes / 20,
            "achieved_throughput_per_step": throughput,
            "safety_violations": 0,
            "physics_errors": 0,
        },
    }


def test_v9_center_criterion_hashes_are_frozen_and_current():
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


def test_v9_banks_are_fresh_and_disjoint_from_prior_strider_banks():
    current = json.loads((EXPERIMENT_ROOT / "BANKS.json").read_text())["tasks"][
        "tea"
    ]
    current_search = _seeds(current["search"])
    current_final = _seeds(current["final"])
    assert not current_search & current_final
    for path in sorted((REPO_ROOT / "experiments").glob("act_strider_*")):
        if path == EXPERIMENT_ROOT or not (path / "BANKS.json").exists():
            continue
        tasks = json.loads((path / "BANKS.json").read_text()).get("tasks", {})
        if "tea" not in tasks:
            continue
        for split in ("search", "final"):
            if split not in tasks["tea"]:
                continue
            prior = _seeds(tasks["tea"][split])
            assert not current_search & prior, (path.name, split)
            assert not current_final & prior, (path.name, split)


def test_delayed_release_requires_reliability_lift_over_qualified_uniform():
    native = _report(module.NATIVE, 20, 0.002)
    uniform = _report(module.UNIFORM, 19, 0.003)
    same_reliability = _report(module.DELAYED_RELEASE, 19, 0.004)
    lifted_but_slow = _report(module.DELAYED_RELEASE, 20, 0.00305)
    lifted = _report(module.DELAYED_RELEASE, 20, 0.0032)
    assert not module.repair_replaces_uniform(same_reliability, uniform, native)
    assert not module.repair_replaces_uniform(lifted_but_slow, uniform, native)
    assert module.repair_replaces_uniform(lifted, uniform, native)


def test_delayed_release_can_replace_unqualified_uniform_when_it_beats_native():
    native = _report(module.NATIVE, 20, 0.002)
    uniform = _report(module.UNIFORM, 18, 0.003, qualified=False)
    repair = _report(module.DELAYED_RELEASE, 19, 0.0025)
    assert module.repair_replaces_uniform(repair, uniform, native)


def test_video_final_ledger_records_actual_episode_media(tmp_path):
    class Runtime:
        def rollout(self, schedule, seed, *, video_path):
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"actual-final-video")
            return {
                "seed": seed,
                "schedule": list(schedule),
                "success": True,
                "first_success_step": 100,
                "physics_steps": 120,
                "safety_violation": None,
                "phase_decisions": [],
                "video_path": str(video_path),
            }

    ledger = module.VideoFinalLedger(Runtime(), tmp_path, [], [101, 102])
    result, records = ledger.evaluate_final(module.DELAYED_RELEASE)
    assert result["actual_episode_videos"] == 2
    assert len(records) == 2
    assert all(record["video_sha256"] for record in records)
    assert len(list((tmp_path / "final").glob("controllers/*/videos/*.mp4"))) == 2


def test_full_final_includes_3p5_and_deduplicates_selected_uniform():
    class Ledger:
        final_seeds = list(range(50))

        def __init__(self):
            self.calls = []

        def evaluate_final(self, schedule):
            self.calls.append(list(schedule))
            speed = sum(schedule) / 4
            return {
                "schedule": list(schedule),
                "schedule_sha256": module.v4.schedule_sha256(schedule),
                "summary": {
                    "episodes": 50,
                    "successes": 50,
                    "success_rate": 1.0,
                    "successful_mean_first_success_steps": 400 / speed,
                    "total_episode_metric_steps": 50 * 400 / speed,
                    "achieved_throughput_per_step": speed / 400,
                    "safety_violations": 0,
                    "physics_errors": 0,
                },
                "actual_episode_videos": 50,
            }, []

    ledger = Ledger()
    result = module.run_video_final(
        ledger, {"selected_schedule": module.UNIFORM}
    )
    assert [3.5] * 4 in ledger.calls
    assert result["unique_controllers_evaluated"] == 6
    assert result["new_final_rollouts"] == 300
    assert result["actual_episode_videos"] == 300
    assert result["same_gpu_controller_concurrency"] is False
    assert result["methods"]["strider_selected"]["alias_of"] == "uniform_1p5x"
