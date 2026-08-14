from pathlib import Path

import cv2
import numpy as np
import pytest

from string_lights.planar import track_planar_frames, validate_projection
from string_lights.planar_io import (
    create_planar_profile_from_annotation,
    projected_strings_from_pose_cache,
)


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data/input/gunjo_10s.mp4"
POSES = ROOT / "data/components/poses/gunjo_10s.npz"
ANNOTATION = ROOT / "tests/fixtures/gunjo_planar.json"


def test_gunjo_validation_fixtures_are_frame_aligned() -> None:
    if not VIDEO.exists() or not POSES.exists():
        pytest.skip("local gunjo validation fixtures are unavailable")
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1920
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 1080
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 300
    finally:
        cap.release()
    with np.load(POSES) as poses:
        assert poses["rvecs"].shape == (300, 3)
        assert poses["tvecs"].shape == (300, 3)
        assert int(poses["total"]) == 300
        assert np.isfinite(poses["rvecs"]).all()
        assert np.isfinite(poses["tvecs"]).all()


@pytest.mark.slow
def test_gunjo_planar_tracking_is_markerless_and_matches_pose_oracle() -> None:
    if not VIDEO.exists() or not POSES.exists():
        pytest.skip("local gunjo validation fixtures are unavailable")
    profile = create_planar_profile_from_annotation(VIDEO, ANNOTATION)

    cap = cv2.VideoCapture(str(VIDEO))

    def board_suppressed_frames():
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            frame[530:890, 40:360] = (160, 190, 220)
            yield frame

    try:
        track = track_planar_frames(board_suppressed_frames(), profile)
    finally:
        cap.release()

    expected, _ = projected_strings_from_pose_cache(POSES)
    report = validate_projection(track, expected)

    assert report.coverage >= 0.95
    assert report.median_perpendicular_error_px <= 3.0
    assert report.p95_perpendicular_error_px <= 6.5
    assert report.p95_bridge_endpoint_error_px <= 4.0
    # The board occupies a small region near the bridge; its pose extrapolates
    # to the nut with a measured systematic disagreement from natural features.
    assert report.p95_nut_endpoint_error_px <= 10.5
