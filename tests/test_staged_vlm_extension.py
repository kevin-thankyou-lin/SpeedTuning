import json

from scripts import staged_vlm_extension as module


def parent():
    seeds = list(range(20))
    native = [{"seed": seed, "success": True, "physics_steps": 300, "first_success_step": 270, "safety_violation": None} for seed in seeds]
    return {"seeds": seeds, "poses": [[float(seed)] * 7 for seed in seeds], "native": native}


def rollout(schedule, seed, *, object_pose=None, video_path=None):
    del object_pose
    success = seed != 19
    return {"seed": seed, "schedule": list(schedule), "success": success, "physics_steps": 140, "first_success_step": 130 if success else None, "safety_violation": None, "video_path": str(video_path)}


def test_candidate_replaces_anchor_only_on_no_regression(tmp_path):
    extension = module.SingleCandidateExtension(tmp_path, parent(), [2.5, 1.5, 2.5, 2.5], rollout, anchor_successes=19, anchor_speedup=1.87)
    result = extension.run()
    assert result["successes"] == 19
    assert result["candidate_replaces_anchor"]
    assert result["selected_schedule"] == [2.5, 1.5, 2.5, 2.5]
    assert result["new_native_rollouts"] == 0


def test_candidate_with_lower_reliability_keeps_anchor(tmp_path):
    def less_reliable(*args, **kwargs):
        result = rollout(*args, **kwargs)
        if result["seed"] == 18:
            result["success"] = False
            result["first_success_step"] = None
        return result
    extension = module.SingleCandidateExtension(tmp_path, parent(), [2.5, 1.5, 2.5, 2.5], less_reliable, anchor_successes=19, anchor_speedup=1.87)
    result = extension.run()
    assert result["successes"] == 18
    assert not result["candidate_replaces_anchor"]
    assert result["selected_schedule"] == [2.0] * 4


def test_resume_uses_receipts_without_rerun(tmp_path):
    calls = []
    def counted(*args, **kwargs):
        calls.append(args[1])
        return rollout(*args, **kwargs)
    first = module.SingleCandidateExtension(tmp_path, parent(), [2.5, 1.5, 2.5, 2.5], counted, anchor_successes=19, anchor_speedup=1.87)
    result = first.run()
    assert len(calls) == 20
    resumed = module.SingleCandidateExtension(tmp_path, parent(), [2.5, 1.5, 2.5, 2.5], counted, anchor_successes=19, anchor_speedup=1.87)
    assert resumed.run() == result
    assert len(calls) == 20
    assert len(list((tmp_path / "private/rollouts").glob("*.json"))) == 20
