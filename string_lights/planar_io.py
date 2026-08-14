from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .board import camera_matrix
from .config import (
    NUM_STRINGS,
    STRING_BRIDGE_BOT,
    STRING_BRIDGE_TOP,
    STRING_NUT_BOT,
    STRING_NUT_TOP,
)
from .planar import PlanarProfile, PlanarTrack, create_planar_profile, track_planar_frames


PROFILES_DIR = Path("data/components/profiles")
PROJECTIONS_DIR = Path("data/components/projections")


def profile_path(name: str) -> Path:
    return PROFILES_DIR / f"{name}.npz"


def projection_path(stem: str) -> Path:
    return PROJECTIONS_DIR / f"{stem}.npz"


def create_planar_profile_from_annotation(
    video_path: str | Path,
    annotation_path: str | Path,
) -> PlanarProfile:
    with Path(annotation_path).open(encoding="utf-8") as source:
        annotation = json.load(source)
    frame_index = int(annotation.get("frame_index", 0))
    frame = _read_video_frame(video_path, frame_index)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    tracking_polygons = _polygons(annotation.get("tracking_polygons"), "tracking_polygons")
    cv2.fillPoly(mask, tracking_polygons, 255)
    exclusions = annotation.get("exclusion_polygons", [])
    if exclusions:
        cv2.fillPoly(mask, _polygons(exclusions, "exclusion_polygons"), 0)
    lines = np.asarray(annotation.get("string_lines"), dtype=np.float64)
    return create_planar_profile(frame, mask, lines)


def track_planar_video(
    video_path: str | Path,
    profile: PlanarProfile,
) -> tuple[PlanarTrack, dict[str, int | float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    metadata: dict[str, int | float] = {
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "total": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    def frames():
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame

    try:
        track = track_planar_frames(frames(), profile)
    finally:
        cap.release()
    metadata["total"] = len(track.frames)
    return track, metadata


def projected_strings_from_pose_cache(
    pose_path: str | Path,
) -> tuple[np.ndarray, dict[str, int | float]]:
    with np.load(Path(pose_path), allow_pickle=False) as poses:
        rvecs = poses["rvecs"]
        tvecs = poses["tvecs"]
        metadata: dict[str, int | float] = {
            "w": int(poses["w"]),
            "h": int(poses["h"]),
            "fps": float(poses["fps"]),
            "total": int(poses["total"]),
        }
    if len(rvecs) != metadata["total"] or len(tvecs) != metadata["total"]:
        raise ValueError("pose cache arrays do not match metadata total")

    bridge_top = np.asarray(STRING_BRIDGE_TOP, dtype=np.float64)
    bridge_bottom = np.asarray(STRING_BRIDGE_BOT, dtype=np.float64)
    nut_top = np.asarray(STRING_NUT_TOP, dtype=np.float64)
    nut_bottom = np.asarray(STRING_NUT_BOT, dtype=np.float64)
    object_lines = np.empty((NUM_STRINGS, 2, 3), dtype=np.float64)
    for index in range(NUM_STRINGS):
        alpha = index / (NUM_STRINGS - 1) if NUM_STRINGS > 1 else 0.0
        object_lines[index, 0] = bridge_top + alpha * (bridge_bottom - bridge_top)
        object_lines[index, 1] = nut_top + alpha * (nut_bottom - nut_top)

    K = camera_matrix(metadata["w"], metadata["h"])
    projected = np.full((metadata["total"], NUM_STRINGS, 2, 2), np.nan, dtype=np.float64)
    distortion = np.zeros(5, dtype=np.float64)
    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            continue
        points, _ = cv2.projectPoints(object_lines.reshape(-1, 3), rvec, tvec, K, distortion)
        projected[index] = points.reshape(NUM_STRINGS, 2, 2)
    return projected, metadata


def _read_video_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"could not read frame {frame_index} from {video_path}")
    return frame


def _polygons(value, field: str) -> list[np.ndarray]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must contain at least one polygon")
    polygons = []
    for polygon in value:
        points = np.asarray(polygon, dtype=np.int32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError(f"each {field} polygon must contain at least three points")
        polygons.append(points)
    return polygons
