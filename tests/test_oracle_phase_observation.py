import numpy as np

from oracle_phase_observation import OraclePhaseEncoder, OraclePhaseStateEncoder


class FakeLearnedPredictor:
    phases = ("pre_grasp", "grasp_lift", "transport", "interaction")
    proprio_feature_names = (
        "effector.left.delta", "effector.left.x", "effector.left.y", "effector.left.z",
        "effector.right.delta", "effector.right.x", "effector.right.y", "effector.right.z",
        "gripper.left", "gripper.right",
    )
    history = 8

    @staticmethod
    def causal_history_windows(values, history, stride):
        del stride
        return np.repeat(values[:1], history, axis=0)[None]

    @staticmethod
    def predict(image, history):
        assert image.dtype == np.uint8
        assert history.shape == (8, 10)
        return {"phase_index": 2}


def observation(objects, left=(0, 0, 0), right=(0, 0, 0)):
    env_state = np.concatenate(
        [np.asarray([*position, 1, 0, 0, 0], dtype=float) for position in objects]
    )
    return {
        "env_state": env_state,
        "mocap_pose_left": np.asarray([*left, 1, 0, 0, 0], dtype=float),
        "mocap_pose_right": np.asarray([*right, 1, 0, 0, 0], dtype=float),
    }


def test_pick_phase_encoder_latches_ordered_boundaries():
    encoder = OraclePhaseEncoder("pick_and_place")
    assert encoder(observation([(0, 0, 0)], right=(1, 0, 0))).tolist() == [1, 0, 0, 0]
    assert encoder(observation([(0, 0, 0)], right=(0.01, 0, 0))).tolist() == [0, 1, 0, 0]
    assert encoder(observation([(0, 0, 0.04)])).tolist() == [0, 0, 1, 0]
    assert encoder(observation([(0, 0, 0.04)], left=(0.01, 0, 0.04))).tolist() == [0, 0, 0, 1]


def test_hybrid_encoder_appends_phase_to_full_state():
    encoder = OraclePhaseStateEncoder("pick_and_place")
    value = observation([(0, 0, 0)], right=(1, 0, 0))
    value["qpos"] = np.zeros(14)
    value["qvel"] = np.zeros(14)
    encoded = encoder(value)
    assert encoded.shape == (39,)
    assert encoded[-4:].tolist() == [1, 0, 0, 0]
    assert encoder.decision_token() == 0


def test_learned_phase_encoder_exposes_raw_detector_argmax():
    from learned_phase_observation import LearnedPhaseEncoder

    encoder = LearnedPhaseEncoder(
        checkpoint_path=".", source_root=".",
        checkpoint_sha256="0" * 64, inference_sha256="1" * 64,
        model_source_sha256="2" * 64, predictor=FakeLearnedPredictor(),
    )
    value = observation([(0, 0, 0)], left=(0.1, 0.2, 0.3), right=(0.4, 0.5, 0.6))
    value["qpos"] = np.zeros(14)
    value["images"] = {"angle": np.zeros((84, 84, 3), dtype=np.uint8)}

    assert encoder(value).tolist() == [0, 0, 1, 0]
    assert encoder.output_dim(123) == 4
    assert encoder.decision_token() == 2
    assert encoder.spec()["temporal_postprocessing"] == "none_raw_argmax"
    assert encoder.spec()["render_camera_names"] == ["angle"]
    assert encoder.spec()["cpu_threads_per_worker"] == 2
    assert encoder.spec()["effector_position_source"] == "joint_fk_body_xpos_or_legacy_mocap_pose"


def test_learned_phase_encoder_accepts_joint_fk_effector_positions():
    from learned_phase_observation import LearnedPhaseEncoder

    encoder = LearnedPhaseEncoder(
        checkpoint_path=".", source_root=".",
        checkpoint_sha256="0" * 64, inference_sha256="1" * 64,
        model_source_sha256="2" * 64, predictor=FakeLearnedPredictor(),
    )
    value = observation([(0, 0, 0)])
    value.pop("mocap_pose_left")
    value.pop("mocap_pose_right")
    value["effector_position_left"] = np.asarray([0.1, 0.2, 0.3])
    value["effector_position_right"] = np.asarray([0.4, 0.5, 0.6])
    value["qpos"] = np.zeros(14)
    value["images"] = {"angle": np.zeros((84, 84, 3), dtype=np.uint8)}

    assert encoder(value).tolist() == [0, 0, 1, 0]
