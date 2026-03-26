import cv2
import numpy as np

from .config import *

NUM_STRINGS = 6
STRING_SPACING = SQUARE_SIZE * 0.7
STRING_Y_OFFSET = - SQUARE_SIZE * 0.1
STRING_X_OFFSET = CHARUCO_BOARD_WIDTH * 3
STRING_LENGTH = CHARUCO_BOARD_WIDTH * 8
STRING_COLOR = (0, 255, 0)  # lime green
STRING_ALPHA = 0.4


def draw_strings(frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray) -> None:
    dist = np.zeros(5, dtype=np.float64)

    overlay = frame.copy()
    for i in range(NUM_STRINGS):
        y = STRING_Y_OFFSET - i * STRING_SPACING
        pts_3d = np.array([[-STRING_X_OFFSET, y, 0], [STRING_LENGTH-STRING_X_OFFSET, y, 0]], dtype=np.float64)
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
        p1 = tuple(pts_2d[0].ravel().astype(int))
        p2 = tuple(pts_2d[1].ravel().astype(int))
        cv2.line(overlay, p1, p2, STRING_COLOR, 2)
    cv2.addWeighted(overlay, STRING_ALPHA, frame, 1 - STRING_ALPHA, 0, frame)
