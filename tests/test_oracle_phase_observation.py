import numpy as np

from oracle_phase_observation import OraclePhaseEncoder


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
