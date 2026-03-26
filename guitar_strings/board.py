import cv2
import numpy as np

BOARD_COLS  = 6
BOARD_ROWS  = 6
SQUARE_SIZE = 0.012   # metres
MARKER_SIZE = 0.009   # metres
ARUCO_DICT  = cv2.aruco.DICT_5X5_50

FOCAL_PX_AT_1920 = 1800.0


def build_board():
    adict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS), SQUARE_SIZE, MARKER_SIZE, adict
    )
    board.setLegacyPattern(True)
    obj_pts  = board.getObjPoints()
    ids_list = board.getIds().flatten()
    id_to_3d = {ids_list[i]: obj_pts[i].astype(np.float32) for i in range(len(ids_list))}
    return adict, id_to_3d


def make_detector(adict):
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin  = 3
    params.adaptiveThreshWinSizeMax  = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate    = 0.01
    params.errorCorrectionRate       = 1.0
    return cv2.aruco.ArucoDetector(adict, params)


def camera_matrix(w, h):
    f = FOCAL_PX_AT_1920 * w / 1920.0
    return np.array([[f, 0, w / 2],
                     [0, f, h / 2],
                     [0, 0,     1]], dtype=np.float64)
