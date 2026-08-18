import numpy as np

from scripts.train_supervised_phase_intent import (
    binary_metrics,
    compose_features,
    conservative_decode,
    labels_for_records,
    monotonic_phase_decode,
    ordered_phase_metrics,
    temporal_features,
    warp_sequence,
)


def test_warp_sequence_keeps_causal_labels():
    values = np.arange(12).reshape(6, 2)
    labels = np.array(["fast", "fast", "segment_0", "segment_0", "fast", "fast"])
    warped, warped_labels = warp_sequence(values, labels, 2)
    np.testing.assert_array_equal(warped, values[[0, 2, 4]])
    np.testing.assert_array_equal(warped_labels, labels[[0, 2, 4]])


def test_temporal_features_use_only_current_and_past():
    values = np.arange(20, dtype=np.float32).reshape(5, 4)
    first = temporal_features(values[:4])
    changed_future = values.copy()
    changed_future[4] = 999
    np.testing.assert_array_equal(first, temporal_features(changed_future)[:4])


def test_action_visual_fusion_preserves_both_features():
    visual = np.arange(20, dtype=np.float32).reshape(5, 4)
    action = np.arange(15, dtype=np.float32).reshape(5, 3)
    fused = compose_features("fused", visual, action)
    assert fused.shape == (5, 21)
    np.testing.assert_array_equal(fused[:, :12], temporal_features(visual))


def test_conservative_decoder_enters_early_and_exits_stably():
    classes = np.array(["fast", "segment_0"])
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.85, 0.15],
        ]
    )
    prediction = conservative_decode(
        probabilities,
        classes,
        risk_threshold=0.25,
        exit_threshold=0.75,
        exit_stability=2,
    )
    np.testing.assert_array_equal(
        prediction,
        ["fast", "segment_0", "segment_0", "segment_0", "fast"],
    )


def test_binary_metrics_penalize_false_fast():
    result = binary_metrics(
        np.array(["fast", "segment_0", "segment_0"]),
        np.array(["fast", "fast", "segment_0"]),
    )
    assert result["protected_recall"] == 0.5
    assert result["false_fast_rate"] == 0.5


def test_reward_phase4_labels_clip_terminal_rewards():
    records = [{"task_reward": value} for value in (0.0, 1.0, 2.0, 3.0, 4.0)]
    np.testing.assert_array_equal(
        labels_for_records(records, "reward-phase4"),
        np.asarray(["phase_1", "phase_2", "phase_3", "phase_4", "phase_4"]),
    )


def test_semantic_phase_labels_use_stable_named_ids():
    records = [
        {"semantic_phase_id": "approach_object"},
        {"semantic_phase_id": "pregrasp_align"},
        {"semantic_phase_id": "grasp_confirm"},
    ]
    np.testing.assert_array_equal(
        labels_for_records(records, "semantic-phase"),
        np.asarray(["approach_object", "pregrasp_align", "grasp_confirm"]),
    )


def test_monotonic_phase_decoder_advances_one_phase_at_a_time():
    classes = np.asarray(["phase_1", "phase_2", "phase_3", "phase_4"])
    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0],
            [0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.1, 0.9],
        ]
    )
    prediction = monotonic_phase_decode(
        probabilities,
        classes,
        advance_threshold=0.5,
        advance_stability=1,
    )
    np.testing.assert_array_equal(prediction, classes)
    assert np.all(np.diff([int(value[-1]) for value in prediction]) >= 0)


def test_ordered_phase_metrics_report_early_advance_and_no_backward_jump():
    truth = [np.asarray(["phase_1", "phase_2", "phase_3", "phase_4"])]
    prediction = [np.asarray(["phase_1", "phase_3", "phase_3", "phase_4"])]
    score = ordered_phase_metrics(truth, prediction)
    assert score["false_advance_rate"] == 0.25
    assert score["mean_absolute_phase_error"] == 0.25
    assert score["backward_jumps"] == 0
