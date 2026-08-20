import numpy as np
import pytest

from string_lights.planar import (
    PlanarFrame,
    PlanarTrack,
    TrackDiagnostics,
    TrackState,
    load_planar_track,
    save_planar_track,
)


_FINGERPRINTS = ("a" * 64, "b" * 64)
_METADATA = {"w": 320, "h": 240, "fps": 30.0, "total": 3}


def _diagnostics() -> TrackDiagnostics:
    return TrackDiagnostics(30, 25, 0.8, 0.5, 0.75)


def _track() -> PlanarTrack:
    lines = np.arange(24, dtype=np.float64).reshape(6, 2, 2)
    frames = (
        PlanarFrame(np.eye(3), lines, TrackState.OBSERVED, _diagnostics(), 0),
        PlanarFrame(np.eye(3), lines + 1, TrackState.OBSERVED, _diagnostics(), 1),
        PlanarFrame(
            None,
            np.full((6, 2, 2), np.nan),
            TrackState.UNRESOLVED,
            _diagnostics(),
            None,
        ),
    )
    return PlanarTrack(frames, reference_fingerprints=_FINGERPRINTS)


def _artifact_payload(version: int = 2) -> dict[str, np.ndarray]:
    lines = np.arange(72, dtype=np.float64).reshape(3, 6, 2, 2)
    lines[2] = np.nan
    homographies = np.repeat(np.eye(3)[None, ...], 3, axis=0)
    homographies[2] = np.nan
    payload = {
        "format_version": np.array(version, dtype=np.int64),
        "homographies": homographies,
        "lines": lines,
        "states": np.array(["observed", "observed", "unresolved"]),
        "correspondences": np.array([30, 30, 0], dtype=np.int32),
        "inliers": np.array([25, 25, 0], dtype=np.int32),
        "inlier_ratios": np.array([0.8, 0.8, 0.0]),
        "median_reprojection_errors_px": np.array([0.5, 0.5, np.inf]),
        "string_axis_spans": np.array([0.75, 0.75, 0.0]),
        "w": np.array(320, dtype=np.int64),
        "h": np.array(240, dtype=np.int64),
        "fps": np.array(30.0),
        "total": np.array(3, dtype=np.int64),
    }
    if version == 2:
        payload.update(
            reference_indices=np.array([0, 1, -1], dtype=np.int16),
            reference_count=np.array(2, dtype=np.int64),
            reference_fingerprints=np.array(_FINGERPRINTS),
        )
    return payload


def test_planar_track_round_trip_preserves_reference_provenance(tmp_path) -> None:
    path = tmp_path / "projection-track.npz"

    save_planar_track(path, _track(), _METADATA)

    with np.load(path, allow_pickle=False) as artifact:
        assert int(artifact["format_version"]) == 2
        assert int(artifact["reference_count"]) == 2
        assert artifact["reference_fingerprints"].astype(str).tolist() == list(
            _FINGERPRINTS
        )
        assert artifact["reference_indices"].tolist() == [0, 1, -1]

    restored, metadata = load_planar_track(path)
    assert restored.reference_fingerprints == _FINGERPRINTS
    assert [frame.reference_index for frame in restored.frames] == [0, 1, None]
    assert metadata == _METADATA


def test_v1_planar_track_uses_legacy_reference_provenance(tmp_path) -> None:
    path = tmp_path / "legacy-track.npz"
    np.savez_compressed(path, **_artifact_payload(version=1))

    restored, _ = load_planar_track(path)

    assert restored.reference_fingerprints == ("legacy-v1:unknown",)
    assert [frame.reference_index for frame in restored.frames] == [0, 0, None]


@pytest.mark.parametrize(
    "overrides",
    [
        {"reference_indices": np.array([0, 1], dtype=np.int16)},
        {"reference_indices": np.array([0, -1, -1], dtype=np.int16)},
        {"reference_indices": np.array([0, 1, 0], dtype=np.int16)},
        {"reference_indices": np.array([0, 2, -1], dtype=np.int16)},
        {"reference_count": np.array(3, dtype=np.int64)},
    ],
    ids=[
        "wrong-index-count",
        "observed-without-reference",
        "unresolved-with-reference",
        "reference-out-of-range",
        "fingerprint-count-mismatch",
    ],
)
def test_v2_planar_track_rejects_malformed_reference_provenance(
    tmp_path, overrides
) -> None:
    path = tmp_path / "malformed-track.npz"
    payload = _artifact_payload()
    payload.update(overrides)
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError):
        load_planar_track(path)
