import cv2
import numpy as np

from .config import *

NUM_STRINGS = 6
STRING_SPACING = SQUARE_SIZE * 0.83
STRING_Y_OFFSET = - SQUARE_SIZE * 0.04
STRING_X_OFFSET = CHARUCO_BOARD_WIDTH * 3
STRING_LENGTH = CHARUCO_BOARD_WIDTH * 6.8
STRING_COLOR = (0, 255, 0)  # lime green
STRING_ALPHA = 0.4


def draw_strings(frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray) -> None:
    dist = np.zeros(5, dtype=np.float64)

    def get_convergence_factor(idx: int) -> float:
        # effect should should follow a linear ramp down then up, peaking at outer strings
        return [
            1.0, 0.8, 0.6, 0.6, 0.8, 1.0
        ][idx] * 1.0

    overlay = frame.copy()
    for i in range(NUM_STRINGS):
        y0 = STRING_Y_OFFSET - i * STRING_SPACING * 0.86
        y1 = STRING_Y_OFFSET - i * STRING_SPACING
        p0 = np.array([-STRING_X_OFFSET, y0, 0], dtype=np.float64)
        p1 = np.array([STRING_LENGTH-STRING_X_OFFSET, y1, 0], dtype=np.float64)
        pts_3d = np.array([p0, p1], dtype=np.float64).reshape(-1, 3)
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
        p1_2d = tuple(pts_2d[0].ravel().astype(int))
        p2_2d = tuple(pts_2d[1].ravel().astype(int))
        cv2.line(overlay, p1_2d, p2_2d, STRING_COLOR, 4, cv2.LINE_AA)
    cv2.addWeighted(overlay, STRING_ALPHA, frame, 1 - STRING_ALPHA, 0, frame)
