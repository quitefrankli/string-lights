import cv2
import numpy as np

from .config import *

NUM_STRINGS = 6
STRING_SPACING = SQUARE_SIZE * 0.83
STRING0_OFFSET = np.array([CHARUCO_BOARD_WIDTH * 3, SQUARE_SIZE * 0.04, 0], dtype=np.float64)
STRING_LENGTH = CHARUCO_BOARD_WIDTH * 6.8
STRING_COLOR = (0, 255, 0)  # lime green
STRING_ALPHA = 0.8
FADE_DURATION = 2.0  # seconds


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

        for i, alpha in to_draw:
            overlay = frame.copy()
            a, b = project_string(i, rvec, tvec)
            cv2.line(overlay, a, b, STRING_COLOR, 4, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
