import cv2
import numpy as np
import pytest

from string_lights.planar import (
    TrackState,
    create_planar_profile,
    load_planar_profile,
    load_planar_track,
    save_planar_profile,
    save_planar_track,
    track_planar_frames,
    validate_projection,
)


def _textured_plane() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    frame = np.full((240, 320, 3), 225, dtype=np.uint8)
    texture = rng.integers(20, 235, size=(170, 240, 3), dtype=np.uint8)
    frame[35:205, 40:280] = texture
    cv2.putText(frame, "G7", (66, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.circle(frame, (236, 158), 24, (5, 240, 70), 4)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[35:205, 40:280] = 255
    ys = np.linspace(78.0, 158.0, 6)
    strings = np.array([[[65.0, y], [255.0, y - 13.0]] for y in ys], dtype=np.float64)
    return frame, mask, strings


def test_planar_profile_projects_strings_through_known_video_warps() -> None:
    reference, mask, strings = _textured_plane()
    transforms = [
        np.eye(3, dtype=np.float64),
        np.array([[1.01, 0.025, 5.0], [-0.015, 1.02, 4.0], [0.00008, -0.00012, 1.0]]),
        np.array([[0.98, -0.035, 12.0], [0.02, 0.99, -3.0], [-0.00010, 0.00015, 1.0]]),
    ]
    frames = [
        cv2.warpPerspective(reference, H, (reference.shape[1], reference.shape[0]))
        for H in transforms
    ]

    profile = create_planar_profile(reference, mask, strings)
    track = track_planar_frames(frames, profile)

    assert len(track.frames) == len(frames)
    assert all(frame.state == TrackState.OBSERVED for frame in track.frames)
    for actual, H in zip(track.projected_lines, transforms):
        expected = cv2.perspectiveTransform(strings.reshape(-1, 1, 2), H).reshape(6, 2, 2)
        assert np.max(np.linalg.norm(actual - expected, axis=-1)) <= 1.0


def test_tracking_localizes_the_first_video_frame_against_the_profile() -> None:
    reference, mask, strings = _textured_plane()
    H = np.array(
        [[0.97, 0.04, 14.0], [-0.02, 1.01, 6.0], [0.00012, -0.00008, 1.0]],
        dtype=np.float64,
    )
    first_video_frame = cv2.warpPerspective(reference, H, (reference.shape[1], reference.shape[0]))

    profile = create_planar_profile(reference, mask, strings)
    track = track_planar_frames([first_video_frame], profile)

    expected = cv2.perspectiveTransform(strings.reshape(-1, 1, 2), H).reshape(6, 2, 2)
    assert track.frames[0].state == TrackState.OBSERVED
    assert np.max(np.linalg.norm(track.projected_lines[0] - expected, axis=-1)) <= 1.0


def test_blank_frame_is_unresolved_and_following_textured_frame_reacquires() -> None:
    reference, mask, strings = _textured_plane()
    H = np.array(
        [[0.98, 0.035, 11.0], [-0.025, 1.01, 7.0], [0.00010, -0.00008, 1.0]],
        dtype=np.float64,
    )
    blank = np.full_like(reference, 127)
    transformed = cv2.warpPerspective(reference, H, (reference.shape[1], reference.shape[0]))

    profile = create_planar_profile(reference, mask, strings)
    track = track_planar_frames([reference, blank, transformed], profile)

    assert [frame.state for frame in track.frames] == [
        TrackState.OBSERVED,
        TrackState.UNRESOLVED,
        TrackState.OBSERVED,
    ]
    assert track.frames[1].homography is None
    assert np.isnan(track.frames[1].projected_lines).all()
    expected = cv2.perspectiveTransform(strings.reshape(-1, 1, 2), H).reshape(6, 2, 2)
    assert np.max(np.linalg.norm(track.frames[2].projected_lines - expected, axis=-1)) <= 1.0


def test_stationary_features_outside_profile_mask_do_not_anchor_tracking() -> None:
    rng = np.random.default_rng(91)
    reference = np.full((240, 420, 3), 225, dtype=np.uint8)
    reference[35:205, 25:175] = rng.integers(
        10, 245, size=(170, 150, 3), dtype=np.uint8
    )
    reference[15:225, 215:405] = rng.integers(
        0, 255, size=(210, 190, 3), dtype=np.uint8
    )
    cv2.putText(reference, "MASK", (35, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(reference, "STATIONARY", (220, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    mask = np.zeros(reference.shape[:2], dtype=np.uint8)
    mask[35:205, 25:175] = 255
    strings = np.array(
        [[[45.0, y], [155.0, y - 8.0]] for y in np.linspace(70.0, 170.0, 6)],
        dtype=np.float64,
    )
    H = np.array([[1.0, 0.0, 18.0], [0.0, 1.0, -9.0], [0.0, 0.0, 1.0]])

    frame = np.full_like(reference, 225)
    frame[15:225, 215:405] = reference[15:225, 215:405]
    moving_plane = np.full_like(reference, 225)
    moving_plane[35:205, 25:175] = reference[35:205, 25:175]
    moving_mask = np.zeros(reference.shape[:2], dtype=np.uint8)
    moving_mask[35:205, 25:175] = 255
    warped_plane = cv2.warpPerspective(moving_plane, H, (reference.shape[1], reference.shape[0]))
    warped_mask = cv2.warpPerspective(
        moving_mask, H, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_NEAREST
    )
    frame[warped_mask.astype(bool)] = warped_plane[warped_mask.astype(bool)]

    profile = create_planar_profile(reference, mask, strings)
    track = track_planar_frames([frame], profile)

    expected = cv2.perspectiveTransform(strings.reshape(-1, 1, 2), H).reshape(6, 2, 2)
    assert track.frames[0].state == TrackState.OBSERVED
    assert np.max(np.linalg.norm(track.frames[0].projected_lines - expected, axis=-1)) <= 1.0


def test_profile_creation_rejects_a_featureless_tracking_region() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
    strings = np.array([[[20.0, y], [140.0, y]] for y in np.linspace(45, 75, 6)])

    with pytest.raises(ValueError, match="too few trackable features"):
        create_planar_profile(frame, mask, strings)


def test_projection_validation_reports_pixel_error_and_coverage() -> None:
    reference, mask, strings = _textured_plane()
    H = np.array(
        [[1.0, 0.02, 8.0], [-0.01, 1.0, 3.0], [0.00005, -0.00007, 1.0]],
        dtype=np.float64,
    )
    frame = cv2.warpPerspective(reference, H, (reference.shape[1], reference.shape[0]))
    expected = cv2.perspectiveTransform(strings.reshape(-1, 1, 2), H).reshape(1, 6, 2, 2)

    track = track_planar_frames([frame], create_planar_profile(reference, mask, strings))
    report = validate_projection(track, expected)

    assert report.evaluated_frames == 1
    assert report.coverage == 1.0
    assert report.median_perpendicular_error_px <= 1.0
    assert report.p95_endpoint_error_px <= 1.0
    assert report.p95_bridge_endpoint_error_px <= 1.0
    assert report.p95_nut_endpoint_error_px <= 1.0


def test_planar_profile_round_trips_as_a_versioned_artifact(tmp_path) -> None:
    reference, mask, strings = _textured_plane()
    profile = create_planar_profile(reference, mask, strings)
    path = tmp_path / "guitar-profile.npz"

    save_planar_profile(path, profile)
    restored = load_planar_profile(path)

    np.testing.assert_array_equal(restored.reference_gray, profile.reference_gray)
    np.testing.assert_array_equal(restored.tracking_mask, profile.tracking_mask)
    np.testing.assert_allclose(restored.string_lines, profile.string_lines)


def test_tracking_rejects_matches_clustered_at_one_end_of_the_strings() -> None:
    reference, mask, strings = _textured_plane()
    partial = np.full_like(reference, 225)
    partial[:, :125] = reference[:, :125]

    track = track_planar_frames([partial], create_planar_profile(reference, mask, strings))

    assert track.frames[0].state == TrackState.UNRESOLVED
    assert np.isnan(track.projected_lines[0]).all()


def test_tracking_rejects_reused_target_feature_that_collapses_plane() -> None:
    reference = np.full((280, 480, 3), 220, dtype=np.uint8)
    repeated_feature = np.full((31, 31, 3), 220, dtype=np.uint8)
    cv2.line(repeated_feature, (3, 15), (27, 15), (20, 20, 20), 3)
    cv2.line(repeated_feature, (15, 15), (15, 21), (20, 20, 20), 3)
    for y in range(25, 239, 45):
        for x in range(25, 439, 45):
            reference[y : y + 31, x : x + 31] = repeated_feature

    frame = np.full_like(reference, 220)
    frame[120:151, 210:241] = repeated_feature
    cv2.putText(frame, "Z", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 5)
    mask = np.full(reference.shape[:2], 255, dtype=np.uint8)
    strings = np.array(
        [[[50.0, y], [430.0, y]] for y in np.linspace(110.0, 170.0, 6)],
        dtype=np.float64,
    )

    track = track_planar_frames([frame], create_planar_profile(reference, mask, strings))

    assert track.frames[0].state == TrackState.UNRESOLVED
    assert track.frames[0].homography is None
    assert np.isnan(track.frames[0].projected_lines).all()


def test_planar_track_round_trips_with_video_metadata(tmp_path) -> None:
    reference, mask, strings = _textured_plane()
    track = track_planar_frames([reference], create_planar_profile(reference, mask, strings))
    path = tmp_path / "projection-track.npz"

    save_planar_track(path, track, {"w": 320, "h": 240, "fps": 30.0, "total": 1})
    restored, metadata = load_planar_track(path)

    assert restored.frames[0].state == TrackState.OBSERVED
    assert restored.frames[0].reference_index == 0
    np.testing.assert_allclose(restored.projected_lines, track.projected_lines)
    assert metadata == {"w": 320, "h": 240, "fps": 30.0, "total": 1}


def test_planar_track_loader_maps_legacy_observed_frames_to_first_reference(
    tmp_path,
) -> None:
    reference, mask, strings = _textured_plane()
    track = track_planar_frames(
        [reference], create_planar_profile(reference, mask, strings)
    )
    path = tmp_path / "legacy-projection-track.npz"
    save_planar_track(path, track, {"w": 320, "h": 240, "fps": 30.0, "total": 1})
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files if key != "reference_indices"}
    payload["format_version"] = np.int64(1)
    np.savez_compressed(path, **payload)

    restored, _ = load_planar_track(path)

    assert restored.frames[0].reference_index == 0
