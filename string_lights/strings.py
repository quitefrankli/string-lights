import cv2
import numpy as np

from .config import *

NUM_STRINGS = 6
STRING_SPACING = SQUARE_SIZE * 0.83
STRING0_OFFSET = np.array([CHARUCO_BOARD_WIDTH * 3, SQUARE_SIZE * 0.04, 0], dtype=np.float64)
STRING_LENGTH = CHARUCO_BOARD_WIDTH * 6.8
STRING_COLOR = (200, 220, 255)  # warm white (BGR)
STRING_CORE_COLOR = (255, 255, 255)
STRING_ALPHA = 0.8
FADE_DURATION = 0.5  # seconds


def draw_strings(frames: list[np.ndarray],
                 poses: list[tuple[np.ndarray, np.ndarray]], strings: list[list[int]],
                 K: np.ndarray,
                 fps: float = 30.0,
                 fade: str = "exponential") -> None:
    dist = np.zeros(5, dtype=np.float64)
    fade_frames = int(fps * FADE_DURATION)
    last_active: dict[int, int] = {}

    def get_convergence_factor(idx: int) -> float:
        return [
            1.0, 0.8, 0.6, 0.6, 0.8, 1.0
        ][idx] * 1.0

    def project_string(i: int, rvec, tvec):
        y0 = -STRING0_OFFSET[1] - i * STRING_SPACING * 0.86
        y1 = -STRING0_OFFSET[1] - i * STRING_SPACING
        p0 = np.array([-STRING0_OFFSET[0], y0, STRING0_OFFSET[2]], dtype=np.float64)
        p1 = np.array([STRING_LENGTH - STRING0_OFFSET[0], y1, STRING0_OFFSET[2]], dtype=np.float64)
        pts_3d = np.array([p0, p1], dtype=np.float64).reshape(-1, 3)
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
        return tuple(pts_2d[0].ravel().astype(int)), tuple(pts_2d[1].ravel().astype(int))

    for frame_idx, (frame, (rvec, tvec), active) in enumerate(zip(frames, poses, strings)):
        if rvec is None:
            for i in active:
                last_active[i] = frame_idx
            continue

        for i in active:
            last_active[i] = frame_idx

        to_draw: list[tuple[int, float]] = []
        for i in active:
            to_draw.append((i, STRING_ALPHA))
        for i, last_frame in last_active.items():
            if i in active:
                continue
            elapsed = frame_idx - last_frame
            if 0 < elapsed <= fade_frames:
                if fade == "exponential":
                    alpha = STRING_ALPHA * np.exp(-3.0 * elapsed / fade_frames)
                else:
                    alpha = STRING_ALPHA * (1.0 - elapsed / fade_frames)
                to_draw.append((i, alpha))

        # Project all strings and find bounding box for ROI blur
        h, w = frame.shape[:2]
        endpoints = []
        xs, ys = [], []
        for i, alpha in to_draw:
            a, b = project_string(i, rvec, tvec)
            endpoints.append((a, b, alpha))
            xs.extend((a[0], b[0]))
            ys.extend((a[1], b[1]))

        if not endpoints:
            continue

        # ROI with padding for blur kernel spread
        pad = 18 * 4  # ~4 sigma for outer bloom
        x0 = max(0, min(xs) - pad)
        y0 = max(0, min(ys) - pad)
        x1 = min(w, max(xs) + pad)
        y1 = min(h, max(ys) + pad)
        rh, rw = y1 - y0, x1 - x0

        outer_bloom = np.zeros((rh, rw, 3), dtype=np.uint8)
        inner_glow = np.zeros((rh, rw, 3), dtype=np.uint8)
        for a, b, alpha in endpoints:
            scaled_color = tuple(int(c * alpha) for c in STRING_COLOR)
            a_r = (a[0] - x0, a[1] - y0)
            b_r = (b[0] - x0, b[1] - y0)
            cv2.line(outer_bloom, a_r, b_r, scaled_color, 22, cv2.LINE_AA)
            cv2.line(inner_glow, a_r, b_r, scaled_color, 8, cv2.LINE_AA)

        outer_bloom = cv2.GaussianBlur(outer_bloom, (0, 0), sigmaX=18)
        inner_glow = cv2.GaussianBlur(inner_glow, (0, 0), sigmaX=5)

        # Saturating uint8 add — no int32 intermediates
        roi = frame[y0:y1, x0:x1]
        scaled_outer = cv2.multiply(outer_bloom, np.array([0.25, 0.25, 0.25, 0], dtype=np.float64))
        scaled_inner = cv2.multiply(inner_glow, np.array([0.7, 0.7, 0.7, 0], dtype=np.float64))
        cv2.add(roi, scaled_outer.astype(np.uint8), dst=roi)
        cv2.add(roi, scaled_inner.astype(np.uint8), dst=roi)

        # White core — group by alpha to minimise frame copies
        by_alpha: dict[float, list[tuple]] = {}
        for a, b, alpha in endpoints:
            by_alpha.setdefault(round(alpha, 4), []).append((a, b))
        for alpha, lines in by_alpha.items():
            core = frame.copy()
            for a, b in lines:
                cv2.line(core, a, b, STRING_CORE_COLOR, 2, cv2.LINE_AA)
            cv2.addWeighted(core, alpha, frame, 1 - alpha, 0, frame)
