from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np


PROFILE_FORMAT_VERSION = 1
TRACK_FORMAT_VERSION = 2


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
    unresolved_relocalize_interval: int = 5
    reference_switch_margin: float = 0.15
    reference_switch_confirm_frames: int = 3
    reference_memory_frames: int = 5


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
    reference_index: int | None = None


@dataclass(frozen=True, slots=True)
class PlanarTrack:
    frames: tuple[PlanarFrame, ...]
    reference_fingerprints: tuple[str, ...] = ()

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


@dataclass(slots=True)
class _ReferenceState:
    profile: PlanarProfile
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray
    tracked_reference: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    tracked_frame: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    last_direct_frame: int = -1_000_000


@dataclass(slots=True)
class _ReferenceSelector:
    active_index: int | None = None
    pending_index: int | None = None
    pending_frames: int = 0
    missing_frames: int = 0


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
    fingerprints = track.reference_fingerprints
    if frame_count and not fingerprints:
        raise ValueError("planar track reference provenance is missing")
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("planar track reference fingerprints must be unique")
    reference_indices = np.empty(frame_count, dtype=np.int16)
    homographies = np.full((frame_count, 3, 3), np.nan, dtype=np.float64)
    for index, frame in enumerate(track.frames):
        if frame.state == TrackState.OBSERVED:
            if frame.reference_index is None or not (
                0 <= frame.reference_index < len(fingerprints)
            ):
                raise ValueError("observed planar frame has an invalid reference index")
            if frame.homography is None:
                raise ValueError("observed planar frame must contain a homography")
            reference_indices[index] = frame.reference_index
            homographies[index] = frame.homography
        else:
            if frame.reference_index is not None:
                raise ValueError("unresolved planar frame cannot name a reference")
            reference_indices[index] = -1
    np.savez_compressed(
        target,
        format_version=np.int64(TRACK_FORMAT_VERSION),
        reference_count=np.int64(len(fingerprints)),
        reference_fingerprints=np.asarray(fingerprints, dtype=np.str_),
        homographies=homographies,
        lines=track.projected_lines,
        states=np.array([frame.state.value for frame in track.frames]),
        reference_indices=reference_indices,
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
        if version not in (1, TRACK_FORMAT_VERSION):
            raise ValueError(f"unsupported planar track format version: {version}")
        homographies = data["homographies"]
        lines = data["lines"]
        states = data["states"]
        if version >= 2:
            if "reference_count" not in data or "reference_fingerprints" not in data:
                raise ValueError("planar track reference provenance is missing")
            reference_count = int(data["reference_count"])
            reference_fingerprints = tuple(
                str(value) for value in data["reference_fingerprints"]
            )
            reference_indices = data["reference_indices"]
        else:
            reference_count = 1
            reference_fingerprints = ("legacy-v1:unknown",)
            reference_indices = np.where(
                states.astype(str) == TrackState.OBSERVED.value, 0, -1
            )
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
    arrays = (
        homographies,
        lines,
        states,
        reference_indices,
        correspondences,
        inliers,
        ratios,
        errors,
        spans,
    )
    if any(len(values) != metadata["total"] for values in arrays):
        raise ValueError("planar track arrays do not match metadata total")
    if reference_count < 1 or reference_count != len(reference_fingerprints):
        raise ValueError("planar track reference provenance is inconsistent")
    if len(set(reference_fingerprints)) != len(reference_fingerprints):
        raise ValueError("planar track reference fingerprints must be unique")
    frames = []
    for index in range(len(lines)):
        state = TrackState(str(states[index]))
        reference_index = int(reference_indices[index])
        if state == TrackState.OBSERVED and not (
            0 <= reference_index < reference_count
        ):
            raise ValueError("observed planar frame has an invalid reference index")
        if state == TrackState.UNRESOLVED and reference_index != -1:
            raise ValueError("unresolved planar frame cannot name a reference")
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
                reference_index=None if reference_index < 0 else reference_index,
            )
        )
    return PlanarTrack(tuple(frames), reference_fingerprints), metadata


def track_planar_frames(
    frames: Iterable[np.ndarray],
    profile: PlanarProfile | Sequence[PlanarProfile],
    config: PlanarTrackerConfig = PlanarTrackerConfig(),
) -> PlanarTrack:
    profiles = _normalize_profiles(profile)
    fingerprints = tuple(_profile_fingerprint(reference) for reference in profiles)
    sift = cv2.SIFT_create(nfeatures=5_000, contrastThreshold=0.02)
    references = []
    for reference in profiles:
        keypoints, descriptors = sift.detectAndCompute(
            reference.reference_gray, reference.tracking_mask
        )
        if descriptors is None or len(keypoints) < 8:
            raise ValueError("the planar profile contains too few trackable features")
        references.append(
            _ReferenceState(reference, tuple(keypoints), descriptors)
        )

    output: list[PlanarFrame] = []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    previous_gray: np.ndarray | None = None
    selector = _ReferenceSelector()
    for frame_index, frame in enumerate(frames):
        gray = _gray(frame)
        if gray.shape != profiles[0].reference_gray.shape:
            raise ValueError("video frames must match the planar profile dimensions")

        features_computed = False
        frame_keypoints: tuple[cv2.KeyPoint, ...] = ()
        frame_descriptors: np.ndarray | None = None

        def direct_match(
            reference: _ReferenceState,
        ) -> tuple[np.ndarray, np.ndarray]:
            nonlocal features_computed, frame_keypoints, frame_descriptors
            reference.last_direct_frame = frame_index
            if not features_computed:
                keypoints, frame_descriptors = sift.detectAndCompute(gray, None)
                frame_keypoints = tuple(keypoints)
                features_computed = True
            return _match_reference(
                frame_keypoints,
                frame_descriptors,
                matcher,
                reference.keypoints,
                reference.descriptors,
                config,
            )

        candidates = []
        direct_attempted = []
        for reference in references:
            has_tracked_points = len(reference.tracked_frame) >= 8
            retry_interval = (
                1
                if len(references) == 1
                else max(1, config.unresolved_relocalize_interval)
            )
            retry_due = (
                frame_index - reference.last_direct_frame
                >= retry_interval
            )
            should_relocalize = (
                previous_gray is None
                or frame_index % max(1, config.relocalize_interval) == 0
                or (not has_tracked_points and retry_due)
            )
            direct_points: tuple[np.ndarray, np.ndarray] | None = None
            flow_points: tuple[np.ndarray, np.ndarray] | None = None
            attempted_direct = False
            attempted_flow = False
            if should_relocalize:
                direct_points = direct_match(reference)
                attempted_direct = True
            elif previous_gray is not None and has_tracked_points:
                flow_points = _track_points(
                    previous_gray,
                    gray,
                    reference.tracked_reference,
                    reference.tracked_frame,
                    config,
                )
                attempted_flow = True

            points = direct_points if direct_points is not None else flow_points
            if points is None:
                points = _empty_points()
            result = _fit_correspondences(*points, reference.profile, config)
            if result.frame.state == TrackState.UNRESOLVED:
                if attempted_flow and not attempted_direct:
                    direct_points = direct_match(reference)
                    attempted_direct = True
                    result = _fit_correspondences(
                        *direct_points, reference.profile, config
                    )
                elif attempted_direct and previous_gray is not None and has_tracked_points:
                    flow_points = _track_points(
                        previous_gray,
                        gray,
                        reference.tracked_reference,
                        reference.tracked_frame,
                        config,
                    )
                    flow_result = _fit_correspondences(
                        *flow_points, reference.profile, config
                    )
                    if flow_result.frame.state == TrackState.OBSERVED:
                        result = flow_result

            candidates.append(result)
            direct_attempted.append(attempted_direct)
            if result.frame.state == TrackState.OBSERVED:
                reference.tracked_reference = result.reference_points
                reference.tracked_frame = result.frame_points
            else:
                reference.tracked_reference, reference.tracked_frame = _empty_points()

        active = selector.active_index
        if (
            active is not None
            and candidates[active].frame.state == TrackState.UNRESOLVED
            and not any(
                candidate.frame.state == TrackState.OBSERVED
                for candidate in candidates
            )
        ):
            for index, reference in enumerate(references):
                if direct_attempted[index]:
                    continue
                points = direct_match(reference)
                result = _fit_correspondences(*points, reference.profile, config)
                candidates[index] = result
                if result.frame.state == TrackState.OBSERVED:
                    reference.tracked_reference = result.reference_points
                    reference.tracked_frame = result.frame_points
                else:
                    reference.tracked_reference, reference.tracked_frame = _empty_points()

        output.append(_select_reference_frame(candidates, selector, config))
        previous_gray = gray
    return PlanarTrack(tuple(output), fingerprints)


def _normalize_profiles(
    profile: PlanarProfile | Sequence[PlanarProfile],
) -> tuple[PlanarProfile, ...]:
    profiles = (profile,) if isinstance(profile, PlanarProfile) else tuple(profile)
    if not profiles:
        raise ValueError("at least one planar profile is required")
    if any(not isinstance(reference, PlanarProfile) for reference in profiles):
        raise TypeError("all planar profiles must be PlanarProfile instances")
    shape = profiles[0].reference_gray.shape
    if any(reference.reference_gray.shape != shape for reference in profiles[1:]):
        raise ValueError("all planar profile reference dimensions must match")
    fingerprints = [_profile_fingerprint(reference) for reference in profiles]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("planar profiles must be unique")
    return profiles


def _profile_fingerprint(profile: PlanarProfile) -> str:
    digest = hashlib.sha256()
    for values in (
        profile.reference_gray,
        profile.tracking_mask,
        profile.string_lines,
    ):
        array = np.ascontiguousarray(values)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.view(np.uint8))
    return f"sha256:{digest.hexdigest()}"


def _select_reference_frame(
    candidates: list[_FitResult],
    selector: _ReferenceSelector,
    config: PlanarTrackerConfig,
) -> PlanarFrame:
    observed = [
        index
        for index, result in enumerate(candidates)
        if result.frame.state == TrackState.OBSERVED
    ]
    if not observed:
        selector.pending_index = None
        selector.pending_frames = 0
        selector.missing_frames += 1
        if selector.missing_frames > max(0, config.reference_memory_frames):
            selector.active_index = None
        best = max(
            range(len(candidates)),
            key=lambda index: _reference_selection_key(
                candidates[index].frame, index, config
            ),
        )
        return replace(candidates[best].frame, reference_index=None)

    selector.missing_frames = 0
    best = max(
        observed,
        key=lambda index: _reference_selection_key(
            candidates[index].frame, index, config
        ),
    )
    active = selector.active_index
    if active not in observed:
        selector.active_index = best
        selector.pending_index = None
        selector.pending_frames = 0
    elif best == active:
        selector.pending_index = None
        selector.pending_frames = 0
    else:
        active_score = _reference_robustness(candidates[active].frame, config)
        best_score = _reference_robustness(candidates[best].frame, config)
        margin = max(0.0, config.reference_switch_margin)
        if best_score < active_score * (1.0 + margin):
            selector.pending_index = None
            selector.pending_frames = 0
        else:
            if selector.pending_index == best:
                selector.pending_frames += 1
            else:
                selector.pending_index = best
                selector.pending_frames = 1
            if selector.pending_frames >= max(
                1, config.reference_switch_confirm_frames
            ):
                selector.active_index = best
                selector.pending_index = None
                selector.pending_frames = 0

    chosen = selector.active_index
    if chosen is None:
        chosen = best
        selector.active_index = best
    return replace(candidates[chosen].frame, reference_index=chosen)


def _reference_selection_key(
    frame: PlanarFrame,
    index: int,
    config: PlanarTrackerConfig,
) -> tuple[float, float, float, int, float, int]:
    error = frame.diagnostics.median_reprojection_error_px
    finite_error = error if np.isfinite(error) else float("inf")
    return (
        _reference_robustness(frame, config),
        -finite_error,
        frame.diagnostics.string_axis_span,
        frame.diagnostics.inliers,
        frame.diagnostics.inlier_ratio,
        -index,
    )


def _reference_robustness(
    frame: PlanarFrame,
    config: PlanarTrackerConfig,
) -> float:
    diagnostics = frame.diagnostics
    error = diagnostics.median_reprojection_error_px
    if not np.isfinite(error):
        return 0.0
    return min(
        diagnostics.inliers / max(1, config.min_inliers),
        diagnostics.inlier_ratio / max(config.min_inlier_ratio, 1e-12),
        diagnostics.string_axis_span / max(config.min_string_axis_span, 1e-12),
        config.max_median_reprojection_error_px / max(error, 1e-6),
    )


def _empty_points() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
    )


def _match_reference(
    frame_keypoints: tuple[cv2.KeyPoint, ...],
    frame_descriptors: np.ndarray | None,
    matcher: cv2.BFMatcher,
    reference_keypoints: tuple[cv2.KeyPoint, ...],
    reference_descriptors: np.ndarray,
    config: PlanarTrackerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if frame_descriptors is None or len(frame_keypoints) < 2:
        return _empty_points()

    matches_by_target: dict[int, cv2.DMatch] = {}
    for pair in matcher.knnMatch(reference_descriptors, frame_descriptors, k=2):
        if len(pair) == 2 and pair[0].distance < config.lowe_ratio * pair[1].distance:
            match = pair[0]
            previous = matches_by_target.get(match.trainIdx)
            if previous is None or match.distance < previous.distance:
                matches_by_target[match.trainIdx] = match
    matches = sorted(matches_by_target.values(), key=lambda match: match.queryIdx)
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
