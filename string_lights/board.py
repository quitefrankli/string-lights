import cv2
import numpy as np

from .config import BOARD_COLS, BOARD_ROWS, SQUARE_SIZE, MARKER_SIZE, ARUCO_DICT, FOCAL_PX_AT_1920


def build_board() -> tuple[cv2.aruco.Dictionary, dict[int, np.ndarray]]:
    adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS), SQUARE_SIZE, MARKER_SIZE, adict
    )
    board.setLegacyPattern(True)
    obj_pts  = board.getObjPoints()
    ids_list = board.getIds().flatten()
    id_to_3d = {ids_list[i]: obj_pts[i].astype(np.float32) for i in range(len(ids_list))}
    return adict, id_to_3d


def make_detector(adict: cv2.aruco.Dictionary) -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin  = 3
    params.adaptiveThreshWinSizeMax  = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate    = 0.01
    params.errorCorrectionRate       = 1.0
    return cv2.aruco.ArucoDetector(adict, params)


def camera_matrix(w: int, h: int) -> np.ndarray:
    f = FOCAL_PX_AT_1920 * w / 1920.0
    return np.array([[f, 0, w / 2],
                     [0, f, h / 2],
                     [0, 0,     1]], dtype=np.float64)
