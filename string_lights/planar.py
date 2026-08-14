from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROFILE_FORMAT_VERSION = 1
TRACK_FORMAT_VERSION = 1


class TrackState(StrEnum):
    OBSERVED = "observed"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class PlanarTrackerConfig:
    min_inliers: int = 20
    min_inlier_ratio: float = 0.30
    max_median_reprojection_error_px: float = 1.25
    min_string_axis_span: float = 0.55
    relocalize_interval: int = 15
    lowe_ratio: float = 0.74
    ransac_threshold_px: float = 1.5
    max_flow_round_trip_error_px: float = 0.75


@dataclass(frozen=True, slots=True)
class PlanarProfile:
    reference_gray: np.ndarray
    tracking_mask: np.ndarray
    string_lines: np.ndarray


@dataclass(frozen=True, slots=True)
class TrackDiagnostics:
    correspondences: int
    inliers: int
    inlier_ratio: float
    median_reprojection_error_px: float
    string_axis_span: float


@dataclass(frozen=True, slots=True)
class PlanarFrame:
    homography: np.ndarray | None
    projected_lines: np.ndarray
    state: TrackState
    diagnostics: TrackDiagnostics


@dataclass(frozen=True, slots=True)
class PlanarTrack:
    frames: tuple[PlanarFrame, ...]

    @property
    def projected_lines(self) -> np.ndarray:
        return np.stack([frame.projected_lines for frame in self.frames])


@dataclass(frozen=True, slots=True)
class ValidationReport:
    total_frames: int
    evaluated_frames: int
    coverage: float
    median_perpendicular_error_px: float
    p95_perpendicular_error_px: float
    max_perpendicular_error_px: float
    median_endpoint_error_px: float
    p95_endpoint_error_px: float
    max_endpoint_error_px: float
    p95_bridge_endpoint_error_px: float
    p95_nut_endpoint_error_px: float


@dataclass(frozen=True, slots=True)
class _FitResult:
    frame: PlanarFrame
    reference_points: np.ndarray
    frame_points: np.ndarray


def create_planar_profile(
    reference_frame: np.ndarray,
    tracking_mask: np.ndarray,
    string_lines: np.ndarray,
) -> PlanarProfile:
    gray = _gray(reference_frame)
    mask = np.asarray(tracking_mask, dtype=np.uint8)
    lines = np.asarray(string_lines, dtype=np.float64)
    if mask.shape != gray.shape:
        raise ValueError("tracking_mask must match the reference frame dimensions")
    if lines.shape != (6, 2, 2) or not np.isfinite(lines).all():
        raise ValueError("string_lines must contain six finite endpoint pairs")
    mask = (mask > 0).astype(np.uint8) * 255
    keypoints = cv2.SIFT_create(nfeatures=5_000, contrastThreshold=0.02).detect(gray, mask)
    if len(keypoints) < 8:
        raise ValueError("the planar profile contains too few trackable features")
    return PlanarProfile(gray.copy(), mask, lines.copy())


def save_planar_profile(path: str | Path, profile: PlanarProfile) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        format_version=np.int64(PROFILE_FORMAT_VERSION),
        reference_gray=profile.reference_gray,
        tracking_mask=profile.tracking_mask,
        string_lines=profile.string_lines,
    )
    return target


def load_planar_profile(path: str | Path) -> PlanarProfile:
    with np.load(Path(path), allow_pickle=False) as data:
        version = int(data["format_version"])
        if version != PROFILE_FORMAT_VERSION:
            raise ValueError(f"unsupported planar profile format version: {version}")
        reference_gray = data["reference_gray"]
        tracking_mask = data["tracking_mask"]
        string_lines = data["string_lines"]
    return create_planar_profile(reference_gray, tracking_mask, string_lines)


def save_planar_track(
    path: str | Path,
    track: PlanarTrack,
    metadata: dict[str, int | float],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame_count = len(track.frames)
    if int(metadata["total"]) != frame_count:
        raise ValueError("track length must match metadata total")
    homographies = np.full((frame_count, 3, 3), np.nan, dtype=np.float64)
    for index, frame in enumerate(track.frames):
        if frame.homography is not None:
            homographies[index] = frame.homography
    np.savez_compressed(
        target,
        format_version=np.int64(TRACK_FORMAT_VERSION),
        homographies=homographies,
        lines=track.projected_lines,
        states=np.array([frame.state.value for frame in track.frames]),
        correspondences=np.array(
            [frame.diagnostics.correspondences for frame in track.frames], dtype=np.int32
        ),
        inliers=np.array([frame.diagnostics.inliers for frame in track.frames], dtype=np.int32),
        inlier_ratios=np.array(
            [frame.diagnostics.inlier_ratio for frame in track.frames], dtype=np.float64
        ),
        median_reprojection_errors_px=np.array(
            [frame.diagnostics.median_reprojection_error_px for frame in track.frames],
            dtype=np.float64,
        ),
        string_axis_spans=np.array(
            [frame.diagnostics.string_axis_span for frame in track.frames], dtype=np.float64
        ),
        w=np.int64(metadata["w"]),
        h=np.int64(metadata["h"]),
        fps=np.float64(metadata["fps"]),
        total=np.int64(metadata["total"]),
    )
    return target


def load_planar_track(path: str | Path) -> tuple[PlanarTrack, dict[str, int | float]]:
    with np.load(Path(path), allow_pickle=False) as data:
        version = int(data["format_version"])
        if version != TRACK_FORMAT_VERSION:
            raise ValueError(f"unsupported planar track format version: {version}")
        homographies = data["homographies"]
        lines = data["lines"]
        states = data["states"]
        correspondences = data["correspondences"]
        inliers = data["inliers"]
        ratios = data["inlier_ratios"]
        errors = data["median_reprojection_errors_px"]
        spans = data["string_axis_spans"]
        metadata: dict[str, int | float] = {
            "w": int(data["w"]),
            "h": int(data["h"]),
            "fps": float(data["fps"]),
            "total": int(data["total"]),
        }
    if len(lines) != metadata["total"]:
        raise ValueError("planar track arrays do not match metadata total")
    frames = []
    for index in range(len(lines)):
        state = TrackState(str(states[index]))
        H = None if state == TrackState.UNRESOLVED else homographies[index].copy()
        frames.append(
            PlanarFrame(
                homography=H,
                projected_lines=lines[index].copy(),
                state=state,
                diagnostics=TrackDiagnostics(
                    int(correspondences[index]),
                    int(inliers[index]),
                    float(ratios[index]),
                    float(errors[index]),
                    float(spans[index]),
                ),
            )
        )
    return PlanarTrack(tuple(frames)), metadata


def track_planar_frames(
    frames: Iterable[np.ndarray],
    profile: PlanarProfile,
    config: PlanarTrackerConfig = PlanarTrackerConfig(),
) -> PlanarTrack:
    sift = cv2.SIFT_create(nfeatures=5_000, contrastThreshold=0.02)
    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        profile.reference_gray, profile.tracking_mask
    )
    if reference_descriptors is None or len(reference_keypoints) < 8:
        raise ValueError("the planar profile contains too few trackable features")

    output: list[PlanarFrame] = []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    previous_gray: np.ndarray | None = None
    tracked_reference = np.empty((0, 2), dtype=np.float32)
    tracked_frame = np.empty((0, 2), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        gray = _gray(frame)
        if gray.shape != profile.reference_gray.shape:
            raise ValueError("video frames must match the planar profile dimensions")

        should_relocalize = (
            previous_gray is None
            or frame_index % max(1, config.relocalize_interval) == 0
            or len(tracked_frame) < 8
        )
        direct_points: tuple[np.ndarray, np.ndarray] | None = None
        flow_points: tuple[np.ndarray, np.ndarray] | None = None
        if should_relocalize:
            direct_points = _match_reference(
                gray,
                sift,
                matcher,
                reference_keypoints,
                reference_descriptors,
                config,
            )
        elif previous_gray is not None:
            flow_points = _track_points(
                previous_gray,
                gray,
                tracked_reference,
                tracked_frame,
                config,
            )

        points = direct_points if direct_points is not None else flow_points
        if points is None:
            points = _empty_points()
        result = _fit_correspondences(*points, profile, config)
        if result.frame.state == TrackState.UNRESOLVED:
            if direct_points is None:
                direct_points = _match_reference(
                    gray,
                    sift,
                    matcher,
                    reference_keypoints,
                    reference_descriptors,
                    config,
                )
                result = _fit_correspondences(*direct_points, profile, config)
            elif previous_gray is not None and len(tracked_frame) >= 8:
                flow_points = _track_points(
                    previous_gray,
                    gray,
                    tracked_reference,
                    tracked_frame,
                    config,
                )
                flow_result = _fit_correspondences(*flow_points, profile, config)
                if flow_result.frame.state == TrackState.OBSERVED:
                    result = flow_result

        output.append(result.frame)
        if result.frame.state == TrackState.OBSERVED:
            tracked_reference = result.reference_points
            tracked_frame = result.frame_points
        else:
            tracked_reference, tracked_frame = _empty_points()
        previous_gray = gray
    return PlanarTrack(tuple(output))


def _empty_points() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
    )


def _match_reference(
    gray: np.ndarray,
    sift: cv2.SIFT,
    matcher: cv2.BFMatcher,
    reference_keypoints: tuple[cv2.KeyPoint, ...],
    reference_descriptors: np.ndarray,
    config: PlanarTrackerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    frame_keypoints, frame_descriptors = sift.detectAndCompute(gray, None)
    if frame_descriptors is None or len(frame_keypoints) < 2:
        return _empty_points()

    matches = []
    for pair in matcher.knnMatch(reference_descriptors, frame_descriptors, k=2):
        if len(pair) == 2 and pair[0].distance < config.lowe_ratio * pair[1].distance:
            matches.append(pair[0])
    if not matches:
        return _empty_points()
    reference_points = np.array(
        [reference_keypoints[match.queryIdx].pt for match in matches], dtype=np.float32
    )
    frame_points = np.array(
        [frame_keypoints[match.trainIdx].pt for match in matches], dtype=np.float32
    )
    return reference_points, frame_points


def _track_points(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    reference_points: np.ndarray,
    previous_points: np.ndarray,
    config: PlanarTrackerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if len(previous_points) < 8 or len(reference_points) != len(previous_points):
        return _empty_points()

    previous = np.asarray(previous_points, dtype=np.float32).reshape(-1, 1, 2)
    lk_options = {
        "winSize": (31, 31),
        "maxLevel": 4,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    current, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, gray, previous, None, **lk_options
    )
    if current is None or forward_status is None:
        return _empty_points()
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        gray, previous_gray, current, None, **lk_options
    )
    if backward is None or backward_status is None:
        return _empty_points()

    current_flat = current.reshape(-1, 2)
    backward_flat = backward.reshape(-1, 2)
    round_trip_error = np.linalg.norm(backward_flat - previous_points, axis=1)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(current_flat).all(axis=1)
        & np.isfinite(backward_flat).all(axis=1)
        & (round_trip_error <= config.max_flow_round_trip_error_px)
    )
    return reference_points[valid].copy(), current_flat[valid].copy()


def _fit_correspondences(
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    profile: PlanarProfile,
    config: PlanarTrackerConfig,
) -> _FitResult:
    source = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(frame_points, dtype=np.float32).reshape(-1, 2)
    if len(source) != len(target):
        raise ValueError("reference and frame correspondences must have equal length")
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    correspondences = len(source)
    if correspondences < 8:
        return _FitResult(_unresolved_frame(correspondences), *_empty_points())

    homography, inlier_mask = cv2.findHomography(
        source,
        target,
        cv2.USAC_MAGSAC,
        config.ransac_threshold_px,
        maxIters=10_000,
        confidence=0.999,
    )
    if homography is None or inlier_mask is None or not np.isfinite(homography).all():
        return _FitResult(_unresolved_frame(correspondences), *_empty_points())
    if abs(float(homography[2, 2])) < 1e-12:
        return _FitResult(_unresolved_frame(correspondences), *_empty_points())
    homography = homography / homography[2, 2]

    inlier = inlier_mask.reshape(-1).astype(bool)
    inliers = int(inlier.sum())
    ratio = inliers / correspondences
    if inliers:
        predicted = cv2.perspectiveTransform(source.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(predicted - target, axis=1)
        median_error = float(np.median(errors[inlier]))
        axis_span = _string_axis_span(source[inlier], profile.string_lines)
    else:
        median_error = float("nan")
        axis_span = 0.0

    accepted = (
        inliers >= config.min_inliers
        and ratio >= config.min_inlier_ratio
        and median_error <= config.max_median_reprojection_error_px
        and axis_span >= config.min_string_axis_span
    )
    if not accepted:
        return _FitResult(
            _unresolved_frame(
                correspondences, inliers, ratio, median_error, axis_span
            ),
            *_empty_points(),
        )

    frame = _observed_frame(
        homography,
        profile.string_lines,
        correspondences,
        inliers,
        median_error,
        axis_span,
    )
    if not np.isfinite(frame.projected_lines).all():
        return _FitResult(
            _unresolved_frame(
                correspondences, inliers, ratio, median_error, axis_span
            ),
            *_empty_points(),
        )
    return _FitResult(frame, source[inlier].copy(), target[inlier].copy())


def validate_projection(
    track: PlanarTrack,
    expected_lines: np.ndarray,
    *,
    samples_per_string: int = 25,
) -> ValidationReport:
    expected = np.asarray(expected_lines, dtype=np.float64)
    if expected.shape != (len(track.frames), 6, 2, 2):
        raise ValueError("expected_lines must have shape (frames, 6, 2, 2)")

    perpendicular_errors: list[np.ndarray] = []
    endpoint_errors: list[np.ndarray] = []
    bridge_errors: list[np.ndarray] = []
    nut_errors: list[np.ndarray] = []
    evaluable = np.isfinite(expected).all(axis=(1, 2, 3))
    evaluated = 0
    alpha = np.linspace(0.0, 1.0, samples_per_string)[None, :, None]
    for frame, truth, can_evaluate in zip(track.frames, expected, evaluable):
        if not can_evaluate or frame.state != TrackState.OBSERVED:
            continue
        actual = frame.projected_lines
        if not np.isfinite(actual).all():
            continue
        evaluated += 1
        frame_endpoint_errors = np.linalg.norm(actual - truth, axis=-1)
        endpoint_errors.append(frame_endpoint_errors.ravel())
        bridge_errors.append(frame_endpoint_errors[:, 0])
        nut_errors.append(frame_endpoint_errors[:, 1])

        actual_samples = (
            actual[:, None, 0] * (1.0 - alpha) + actual[:, None, 1] * alpha
        )
        direction = truth[:, 1] - truth[:, 0]
        length = np.linalg.norm(direction, axis=1)
        valid_lines = length > 1e-9
        offset = actual_samples - truth[:, None, 0]
        cross = np.abs(
            direction[:, None, 0] * offset[:, :, 1]
            - direction[:, None, 1] * offset[:, :, 0]
        )
        perpendicular_errors.append((cross[valid_lines] / length[valid_lines, None]).ravel())

    perpendicular = np.concatenate(perpendicular_errors) if perpendicular_errors else np.array([np.nan])
    endpoints = np.concatenate(endpoint_errors) if endpoint_errors else np.array([np.nan])
    bridges = np.concatenate(bridge_errors) if bridge_errors else np.array([np.nan])
    nuts = np.concatenate(nut_errors) if nut_errors else np.array([np.nan])
    total_evaluable = int(evaluable.sum())
    coverage = evaluated / total_evaluable if total_evaluable else 0.0
    return ValidationReport(
        total_frames=len(track.frames),
        evaluated_frames=evaluated,
        coverage=coverage,
        median_perpendicular_error_px=float(np.nanmedian(perpendicular)),
        p95_perpendicular_error_px=float(np.nanpercentile(perpendicular, 95)),
        max_perpendicular_error_px=float(np.nanmax(perpendicular)),
        median_endpoint_error_px=float(np.nanmedian(endpoints)),
        p95_endpoint_error_px=float(np.nanpercentile(endpoints, 95)),
        max_endpoint_error_px=float(np.nanmax(endpoints)),
        p95_bridge_endpoint_error_px=float(np.nanpercentile(bridges, 95)),
        p95_nut_endpoint_error_px=float(np.nanpercentile(nuts, 95)),
    )


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.asarray(frame, dtype=np.uint8)
    return cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_BGR2GRAY)


def _observed_frame(
    homography: np.ndarray,
    string_lines: np.ndarray,
    correspondences: int,
    inliers: int,
    median_error: float,
    axis_span: float,
) -> PlanarFrame:
    lines = cv2.perspectiveTransform(
        string_lines.reshape(-1, 1, 2).astype(np.float64), homography
    ).reshape(6, 2, 2)
    ratio = inliers / correspondences if correspondences else 1.0
    return PlanarFrame(
        homography=homography,
        projected_lines=lines,
        state=TrackState.OBSERVED,
        diagnostics=TrackDiagnostics(correspondences, inliers, ratio, median_error, axis_span),
    )


def _unresolved_frame(
    correspondences: int = 0,
    inliers: int = 0,
    inlier_ratio: float = 0.0,
    median_error: float = float("nan"),
    axis_span: float = 0.0,
) -> PlanarFrame:
    return PlanarFrame(
        homography=None,
        projected_lines=np.full((6, 2, 2), np.nan, dtype=np.float64),
        state=TrackState.UNRESOLVED,
        diagnostics=TrackDiagnostics(
            correspondences, inliers, inlier_ratio, median_error, axis_span
        ),
    )


def _string_axis_span(points: np.ndarray, string_lines: np.ndarray) -> float:
    bridge = string_lines[:, 0].mean(axis=0)
    nut = string_lines[:, 1].mean(axis=0)
    axis = nut - bridge
    length = float(np.linalg.norm(axis))
    if length < 1e-9 or len(points) == 0:
        return 0.0
    positions = (points - bridge) @ (axis / length)
    return float(np.ptp(positions) / length)
