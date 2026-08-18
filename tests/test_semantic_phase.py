import numpy as np
import pytest

from chunked_policy import SemanticPhaseChunkRunner
from semantic_phase import (
    FuturePhasePrediction,
    FuturePhaseSequencePredictor,
    SemanticPhaseStrideController,
    future_phase_targets,
    parse_future_offsets,
)


class ConstantHead:
    def __init__(self, label):
        self.label = label

    def predict(self, features):
        return np.repeat(self.label, len(features))


def test_future_phase_targets_follow_nominal_policy_time_and_pad_terminal_phase():
    records = [{"policy_time": value} for value in (0, 2, 4, 6)]
    labels = ["reach", "reach", "align", "insert"]
    targets = future_phase_targets(records, labels, (0, 1, 2, 4))
    np.testing.assert_array_equal(targets[0], ["reach", "reach", "reach", "align"])
    np.testing.assert_array_equal(targets[2], ["align", "align", "insert", "insert"])
    np.testing.assert_array_equal(targets[-1], ["insert"] * 4)


def test_future_phase_sequence_predictor_returns_one_id_per_offset():
    predictor = FuturePhaseSequencePredictor(
        offsets=(0, 1, 2),
        models=(ConstantHead("transport"), ConstantHead("align"), ConstantHead("insert")),
    )
    prediction = predictor.predict_one(np.ones(8))
    assert prediction == FuturePhasePrediction(
        offsets=(0, 1, 2),
        phase_ids=("transport", "align", "insert"),
    )


def test_stride_slows_before_crossing_into_a_precise_phase():
    controller = SemanticPhaseStrideController(
        {"transport": 3.0, "align": 1.0, "insert": 1.0}
    )
    actions = np.zeros((8, 14))
    prediction = FuturePhasePrediction(
        (0, 1, 2, 3),
        ("transport", "transport", "align", "align"),
    )
    assert controller.choose_stride(actions, prediction) == 1.0


def test_stride_uses_fast_phase_when_dense_lookahead_is_safe():
    controller = SemanticPhaseStrideController({"transport": 3.0, "align": 1.0})
    prediction = FuturePhasePrediction(
        (0, 1, 2, 3),
        ("transport", "transport", "transport", "transport"),
    )
    assert controller.choose_stride(np.zeros((8, 14)), prediction) == 3.0


def test_stride_fails_closed_when_lookahead_is_not_dense():
    controller = SemanticPhaseStrideController({"transport": 3.0})
    prediction = FuturePhasePrediction((0, 2, 3), ("transport",) * 3)
    assert controller.choose_stride(np.zeros((8, 14)), prediction) == 1.0


def test_stride_preserves_gripper_transition_at_native_speed():
    controller = SemanticPhaseStrideController({"transport": 3.0})
    actions = np.zeros((8, 14))
    actions[2:, 6] = 1.0
    prediction = FuturePhasePrediction((0, 1, 2, 3), ("transport",) * 4)
    assert controller.choose_stride(actions, prediction) == 1.0


def test_semantic_runner_uses_phase_stride_and_replans_after_chunk():
    actions = np.repeat(np.arange(5, dtype=float)[:, None], 14, axis=1)
    actions[:, (6, 13)] = 0.0
    prediction = FuturePhasePrediction((0, 1, 2, 3), ("transport",) * 4)
    runner = SemanticPhaseChunkRunner(
        lambda observation: actions,
        lambda observation, chunk: prediction,
        SemanticPhaseStrideController({"transport": 3.0}),
    )
    assert runner.action({})[0] == 0.0
    assert runner.action({})[0] == 3.0
    assert runner.decision_speed == 3.0
    assert runner.phase_prediction == prediction


@pytest.mark.parametrize("value", ("", "1,2", "0,2,1", "0,1,1", "0,-1"))
def test_future_offsets_reject_invalid_contracts(value):
    with pytest.raises(ValueError):
        parse_future_offsets(value)
