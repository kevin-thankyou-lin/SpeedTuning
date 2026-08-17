import numpy as np

from scripts.summarize_supervised_shared_fast import audit_result

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
from supervised_phase_controller import (
    CausalTemporalFeatureBuffer,
    ConservativeBinaryDecoder,
    mapped_protected_speed,
    PortableStandardizedLogisticRegression,
    resolve_conservative_decoder_config,
    shared_fast_speed,
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


def test_online_temporal_buffer_matches_batch_features():
    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    expected = temporal_features(values)
    buffer = CausalTemporalFeatureBuffer()
    actual = np.stack([buffer.update(value) for value in values])
    np.testing.assert_array_equal(actual, expected)


def test_stateful_online_decoder_matches_batch_decoder():
    classes = np.asarray(["fast", "segment_0", "segment_1"])
    probabilities = np.asarray(
        [
            [0.9, 0.05, 0.05],
            [0.4, 0.5, 0.1],
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.85, 0.1, 0.05],
        ]
    )
    expected = conservative_decode(
        probabilities,
        classes,
        risk_threshold=0.45,
        exit_threshold=0.75,
        exit_stability=2,
    )
    decoder = ConservativeBinaryDecoder(
        classes,
        risk_threshold=0.45,
        exit_threshold=0.75,
        exit_stability=2,
    )
    actual = np.asarray([decoder.update(row) for row in probabilities])
    np.testing.assert_array_equal(actual, expected)


def test_decoder_margin_overrides_are_explicit_and_leave_source_unchanged():
    base = {
        "risk_threshold": 0.45,
        "exit_threshold": 0.6,
        "exit_stability": 1,
        "protected_recall": 1.0,
    }
    resolved, overrides = resolve_conservative_decoder_config(
        base,
        risk_threshold=0.35,
        exit_stability=2,
    )
    assert base["risk_threshold"] == 0.45
    assert base["exit_stability"] == 1
    assert resolved["risk_threshold"] == 0.35
    assert resolved["exit_threshold"] == 0.6
    assert resolved["exit_stability"] == 2
    assert overrides == {"risk_threshold": 0.35, "exit_stability": 2}


def test_shared_fast_audit_distinguishes_false_fast_and_safe_segment_swap():
    result = {
        "controller": {
            "fast_speed": 3.0,
            "protected_labels": ["segment_0", "segment_1"],
            "protected_speed_map": {"segment_0": 1.5, "segment_1": 1.0},
        },
        "summary": {"new_rollouts": 1},
        "candidate": [
            {
                "seed": 7,
                "success": True,
                "trace": [
                    {"oracle_label": "fast", "prediction": "fast", "speed": 3.0},
                    {
                        "oracle_label": "segment_0",
                        "prediction": "segment_1",
                        "speed": 1.0,
                    },
                    {
                        "oracle_label": "segment_1",
                        "prediction": "fast",
                        "speed": 3.0,
                    },
                ],
            }
        ],
    }
    audit = audit_result(result)
    assert audit["false_fast_rate"] == 0.5
    assert audit["protected_recall"] == 0.5
    assert audit["protected_segment_exact_accuracy"] == 0.0
    assert audit["speed_choice_accuracy"] == 1 / 3


def test_shared_fast_mapping_collapses_protected_segment_speeds():
    assert shared_fast_speed("fast", fast_speed=2.0, protected_speed=1.0) == 2.0
    assert shared_fast_speed("segment_0", fast_speed=2.0, protected_speed=1.0) == 1.0
    assert shared_fast_speed("segment_1", fast_speed=2.0, protected_speed=1.0) == 1.0


def test_mapped_protected_speeds_keep_one_fast_ceiling():
    kwargs = {
        "fast_speed": 3.0,
        "default_protected_speed": 1.0,
        "protected_speed_map": {"segment_0": 1.5},
    }
    assert mapped_protected_speed("fast", **kwargs) == 3.0
    assert mapped_protected_speed("segment_0", **kwargs) == 1.5
    assert mapped_protected_speed("segment_1", **kwargs) == 1.0


def test_portable_multiclass_logistic_regression_probabilities():
    model = PortableStandardizedLogisticRegression(
        classes=np.asarray(["fast", "segment_0", "segment_1"]),
        mean=np.asarray([1.0, 2.0]),
        scale=np.asarray([2.0, 4.0]),
        coef=np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
        intercept=np.asarray([0.1, 0.2, 0.3]),
    )
    probabilities = model.predict_proba(np.asarray([[3.0, 6.0]]))
    logits = np.asarray([[1.1, 1.2, -1.7]])
    expected = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probabilities, expected)
    np.testing.assert_array_equal(model.classes_, model.classes)
