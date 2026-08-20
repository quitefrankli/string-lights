import cv2
import numpy as np
import pytest

import string_lights.planar as planar

from string_lights.planar import (
    PlanarTrackerConfig,
    TrackState,
    create_planar_profile,
    track_planar_frames,
)


_DIRECT_MATCHING = PlanarTrackerConfig(relocalize_interval=1)


def _appearance_reference(
    seed: int, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    frame = np.full((180, 280, 3), 224, dtype=np.uint8)
    frame[25:155, 30:250] = rng.integers(15, 240, size=(130, 220, 3), dtype=np.uint8)
    cv2.putText(frame, label, (53, 104), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[25:155, 30:250] = 255
    strings = np.array(
        [[[48.0, y], [232.0, y - 9.0]] for y in np.linspace(65.0, 120.0, 6)],
        dtype=np.float64,
    )
    return frame, mask, strings


def _warp(frame: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(frame, homography, (frame.shape[1], frame.shape[0]))


def _ambiguous_profiles():
    rng = np.random.default_rng(504)
    reference = np.full((220, 300, 3), 225, dtype=np.uint8)
    reference[20:110, 25:275] = rng.integers(
        15, 240, size=(90, 250, 3), dtype=np.uint8
    )
    reference[130:195, 25:275] = rng.integers(
        15, 240, size=(65, 250, 3), dtype=np.uint8
    )
    mask_a = np.zeros(reference.shape[:2], dtype=np.uint8)
    mask_a[20:110, 25:275] = 255
    mask_b = np.zeros(reference.shape[:2], dtype=np.uint8)
    mask_b[130:195, 25:275] = 255
    strings_a = np.array(
        [[[45.0, y], [255.0, y]] for y in np.linspace(50.0, 90.0, 6)],
        dtype=np.float64,
    )
    strings_b = strings_a + np.array([0.0, 95.0])
    only_b_visible = reference.copy()
    only_b_visible[20:110, 25:275] = 225
    return (
        reference,
        only_b_visible,
        create_planar_profile(reference, mask_a, strings_a),
        create_planar_profile(reference, mask_b, strings_b),
        strings_b,
    )


def test_reference_bank_covers_appearance_intervals_that_each_profile_misses() -> None:
    reference_a, mask_a, strings = _appearance_reference(101, "A")
    reference_b, mask_b, _ = _appearance_reference(202, "B")
    profile_a = create_planar_profile(reference_a, mask_a, strings)
    profile_b = create_planar_profile(reference_b, mask_b, strings)
    transforms = [
        np.eye(3, dtype=np.float64),
        np.array([[1.0, 0.018, 5.0], [-0.012, 1.0, 3.0], [0.00005, -0.00004, 1.0]]),
        np.eye(3, dtype=np.float64),
        np.array([[0.99, -0.022, 8.0], [0.015, 1.01, -2.0], [-0.00004, 0.00006, 1.0]]),
    ]
    frames = [
        _warp(reference_a, transforms[0]),
        _warp(reference_a, transforms[1]),
        _warp(reference_b, transforms[2]),
        _warp(reference_b, transforms[3]),
    ]

    only_a = track_planar_frames(frames, profile_a, _DIRECT_MATCHING)
    only_b = track_planar_frames(frames, profile_b, _DIRECT_MATCHING)
    combined = track_planar_frames(frames, (profile_a, profile_b), _DIRECT_MATCHING)

    assert [frame.state for frame in only_a.frames] == [
        TrackState.OBSERVED,
        TrackState.OBSERVED,
        TrackState.UNRESOLVED,
        TrackState.UNRESOLVED,
    ]
    assert [frame.state for frame in only_b.frames] == [
        TrackState.UNRESOLVED,
        TrackState.UNRESOLVED,
        TrackState.OBSERVED,
        TrackState.OBSERVED,
    ]
    assert all(frame.state == TrackState.OBSERVED for frame in combined.frames)
    assert [frame.reference_index for frame in combined.frames] == [0, 0, 1, 1]
    for actual, homography in zip(combined.projected_lines, transforms):
        expected = cv2.perspectiveTransform(
            strings.reshape(-1, 1, 2), homography
        ).reshape(6, 2, 2)
        assert np.max(np.linalg.norm(actual - expected, axis=-1)) <= 1.0


def test_reference_bank_fails_closed_when_no_profile_resolves_the_frame() -> None:
    reference_a, mask_a, strings = _appearance_reference(303, "A")
    reference_b, mask_b, _ = _appearance_reference(404, "B")
    profiles = (
        create_planar_profile(reference_a, mask_a, strings),
        create_planar_profile(reference_b, mask_b, strings),
    )
    blank = np.full_like(reference_a, 127)

    track = track_planar_frames([blank, blank], profiles, _DIRECT_MATCHING)

    assert all(frame.state == TrackState.UNRESOLVED for frame in track.frames)
    assert all(frame.homography is None for frame in track.frames)
    assert all(frame.reference_index is None for frame in track.frames)
    assert np.isnan(track.projected_lines).all()


def test_reference_bank_keeps_a_valid_reference_across_ambiguous_frames() -> None:
    ambiguous, only_b_visible, profile_a, profile_b, strings_b = (
        _ambiguous_profiles()
    )
    assert (
        track_planar_frames([ambiguous], profile_a, _DIRECT_MATCHING).frames[0].state
        == TrackState.OBSERVED
    )
    assert (
        track_planar_frames([ambiguous], profile_b, _DIRECT_MATCHING).frames[0].state
        == TrackState.OBSERVED
    )

    track = track_planar_frames(
        [only_b_visible, ambiguous, ambiguous, ambiguous],
        (profile_a, profile_b),
        _DIRECT_MATCHING,
    )

    assert all(frame.state == TrackState.OBSERVED for frame in track.frames)
    assert [frame.reference_index for frame in track.frames] == [1, 1, 1, 1]
    expected = np.repeat(strings_b[None, ...], len(track.frames), axis=0)
    np.testing.assert_allclose(track.projected_lines, expected, atol=1.0)


def test_reference_bank_switches_only_after_three_better_frames() -> None:
    ambiguous, only_b_visible, profile_a, profile_b, _ = _ambiguous_profiles()
    config = PlanarTrackerConfig(
        relocalize_interval=1,
        reference_switch_margin=0.0,
        reference_switch_confirm_frames=3,
    )

    track = track_planar_frames(
        [only_b_visible, ambiguous, ambiguous, ambiguous],
        (profile_a, profile_b),
        config,
    )

    assert all(frame.state == TrackState.OBSERVED for frame in track.frames)
    assert [frame.reference_index for frame in track.frames] == [1, 1, 1, 0]


def test_reference_bank_retains_active_reference_through_a_brief_dropout() -> None:
    ambiguous, only_b_visible, profile_a, profile_b, strings_b = (
        _ambiguous_profiles()
    )
    blank = np.full_like(ambiguous, 127)

    track = track_planar_frames(
        [only_b_visible, blank, ambiguous],
        (profile_a, profile_b),
        _DIRECT_MATCHING,
    )

    assert [frame.state for frame in track.frames] == [
        TrackState.OBSERVED,
        TrackState.UNRESOLVED,
        TrackState.OBSERVED,
    ]
    assert [frame.reference_index for frame in track.frames] == [1, None, 1]
    np.testing.assert_allclose(track.frames[0].projected_lines, strings_b, atol=1.0)
    assert np.isnan(track.frames[1].projected_lines).all()
    np.testing.assert_allclose(track.frames[2].projected_lines, strings_b, atol=1.0)


def test_absent_reference_relocalization_is_rate_limited(monkeypatch) -> None:
    reference_a, mask_a, strings = _appearance_reference(808, "A")
    reference_b, mask_b, _ = _appearance_reference(909, "B")
    profiles = (
        create_planar_profile(reference_a, mask_a, strings),
        create_planar_profile(reference_b, mask_b, strings),
    )
    frames = [reference_a.copy() for _ in range(8)]
    calls_by_descriptors: dict[int, int] = {}
    original_match_reference = planar._match_reference

    def counted_match_reference(*args, **kwargs):
        descriptor_id = id(args[4])
        calls_by_descriptors[descriptor_id] = (
            calls_by_descriptors.get(descriptor_id, 0) + 1
        )
        return original_match_reference(*args, **kwargs)

    monkeypatch.setattr(planar, "_match_reference", counted_match_reference)

    track = track_planar_frames(frames, profiles)

    assert all(frame.state == TrackState.OBSERVED for frame in track.frames)
    assert [frame.reference_index for frame in track.frames] == [0] * len(frames)
    assert len(calls_by_descriptors) == 2
    active_calls, absent_calls = calls_by_descriptors.values()
    assert active_calls == 1
    assert absent_calls <= 2


def test_reference_bank_rejects_an_empty_profile_sequence() -> None:
    with pytest.raises(ValueError, match="at least one planar profile"):
        track_planar_frames([], (), _DIRECT_MATCHING)


def test_reference_bank_rejects_mismatched_dimensions_before_reading_frames() -> None:
    reference_a, mask_a, strings_a = _appearance_reference(606, "A")
    reference_b, mask_b, strings_b = _appearance_reference(707, "B")
    width_scale = 300.0 / reference_b.shape[1]
    height_scale = 200.0 / reference_b.shape[0]
    reference_b = cv2.resize(reference_b, (300, 200))
    mask_b = cv2.resize(mask_b, (300, 200), interpolation=cv2.INTER_NEAREST)
    strings_b = strings_b * np.array([width_scale, height_scale])
    profiles = (
        create_planar_profile(reference_a, mask_a, strings_a),
        create_planar_profile(reference_b, mask_b, strings_b),
    )
    consumed = False

    def frames():
        nonlocal consumed
        consumed = True
        yield reference_a

    with pytest.raises(ValueError, match="reference dimensions must match"):
        track_planar_frames(frames(), profiles, _DIRECT_MATCHING)

    assert not consumed
