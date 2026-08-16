import numpy as np

from reference_alignment import OnlineReferenceAligner, ReferencePositionSpeedPolicy


def smooth_reference(length=60, dimensions=8):
    positions = np.linspace(0.0, 1.0, length)
    columns = [
        np.sin((index + 1) * np.pi * positions)
        if index % 2 == 0
        else np.cos((index + 1) * np.pi * positions)
        for index in range(dimensions)
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def test_causal_alignment_tracks_time_warp_without_large_backward_jumps():
    reference = smooth_reference()
    query_indices = np.asarray(
        [0, 1, 2, 3, 4, 5, 5, 6, 7, 9, 11, 13, 16, 19, 22, 25, 29, 33, 38, 43, 49, 55, 59]
    )
    aligner = OnlineReferenceAligner(
        reference,
        max_advance=6,
        max_backtrack=1,
        emission_temperature=0.04,
    )
    results = [aligner.update_embedding(reference[index]) for index in query_indices]
    predicted = np.asarray([item.reference_position for item in results])
    truth = query_indices / (len(reference) - 1)

    assert np.mean(np.abs(predicted - truth)) < 0.04
    assert np.min(np.diff(predicted)) > -0.04
    assert all(0.0 <= item.confidence <= 1.0 for item in results)
    assert results[-1].reference_index >= 55


def test_frame_api_uses_only_trailing_clip():
    encoded_clips = []

    def encoder(frames):
        encoded_clips.append(tuple(float(frame[0, 0, 0]) for frame in frames))
        value = float(frames[-1][0, 0, 0])
        return np.asarray([value, 1.0 - value], dtype=np.float32)

    reference_frames = np.asarray(
        [np.full((2, 2, 3), value, dtype=np.float32) for value in np.linspace(0.0, 1.0, 8)]
    )
    aligner = OnlineReferenceAligner(
        reference_frames,
        clip_encoder=encoder,
        clip_frames=3,
        max_advance=3,
    )
    reference_calls = list(encoded_clips)
    encoded_clips.clear()
    aligner.update(reference_frames[:4])

    assert reference_calls[0] == (0.0,)
    assert all(len(item) <= 3 for item in reference_calls)
    assert encoded_clips[0] == (0.0,)
    assert encoded_clips[-1] == tuple(float(value) for value in reference_frames[1:4, 0, 0, 0])


def test_segment_lookup_is_external_and_editable_without_retraining():
    policy = ReferencePositionSpeedPolicy(
        [(0.20, 0.35, 1.0), (0.70, 0.80, 1.5)],
        default=4.0,
    )
    assert policy(0.1) == 4.0
    assert policy(0.25) == 1.0
    assert policy(0.75) == 1.5
