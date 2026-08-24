import pytest

from scripts import act_vlm_frontier_server as module


def test_checked_hash_rejects_artifact_mismatch(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")

    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        module.checked_hash(artifact, "0" * 64)


def test_act_server_delegates_rollout_to_frozen_runtime(tmp_path, monkeypatch):
    class Runtime:
        detector = {"checkpoint_path": "sealed"}
        task = "pick_and_place"

        def encoder(self):
            class Encoder:
                @staticmethod
                def spec():
                    return {"type": "sealed_rgb_proprio_phase_one_hot"}

            return Encoder()

        def rollout(self, schedule, seed, *, object_pose=None, video_path=None):
            return {
                "task": self.task,
                "seed": seed,
                "schedule": list(schedule),
                "success": True,
                "raw_task_success": True,
                "physics_steps": 100,
                "first_success_step": 75,
                "success_only_acceleration": 4.0,
                "safety_violation": None,
                "phase_decisions": [
                    {"phase": "pre_grasp", "physics_step": 0, "speed": schedule[0]},
                    {"phase": "grasp_lift", "physics_step": 25, "speed": schedule[1]},
                    {"phase": "transport", "physics_step": 50, "speed": schedule[2]},
                    {"phase": "interaction", "physics_step": 75, "speed": schedule[3]},
                ],
                "video_path": None if video_path is None else str(video_path),
            }

    monkeypatch.setattr(
        "scripts.three_scene_server.sample_object_pose",
        lambda task, seed: (float(seed),) * 7,
    )
    runtime = Runtime()
    server = module.ACTThreeSceneServer(
        tmp_path,
        runtime.task,
        [101, 102, 103],
        list(range(200, 210)),
        50,
        runtime=runtime,
    )

    result = server.probe([2, 2, 2, 2])

    assert result["discovery_successes"] == 3
    assert result["schedule"] == [2.0, 2.0, 2.0, 2.0]
    assert result["discovery"][0]["physics_steps"] == 100
    assert result["discovery"][0]["first_success_step"] == 75
    score = server.score(
        result["schedule_hash"], [3, 2, 2, 2], safe_success_probability=0.75
    )
    assert score["mean_expected_absolute_steps_saved"] > 0
