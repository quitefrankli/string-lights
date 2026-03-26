import cv2
import numpy as np

from .config import MAX_ROTATION_JUMP, MAX_TRANSLATION_JUMP

Pose = tuple[np.ndarray | None, np.ndarray | None]


def estimate_pose(gray: np.ndarray, detector: cv2.aruco.ArucoDetector, id_to_3d: dict[int, np.ndarray], K: np.ndarray) -> Pose:
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None, None

    all_obj, all_img = [], []
    for i, mid in enumerate(ids.flatten()):
        if mid in id_to_3d:
            all_obj.append(id_to_3d[mid])
            all_img.append(corners[i][0])

    if len(all_obj) < 4:
        return None, None

    obj = np.array(all_obj, dtype=np.float32).reshape(-1, 3)
    img = np.array(all_img, dtype=np.float32).reshape(-1, 2)
    dist = np.zeros(5, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    return (rvec, tvec) if ok else (None, None)


def is_pose_valid(rvec: np.ndarray, tvec: np.ndarray, prev_rvec: np.ndarray | None, prev_tvec: np.ndarray | None) -> bool:
    # return True
    r = np.degrees(rvec.flatten())
    return 18 < r[0] < 22
    R, _ = cv2.Rodrigues(rvec)
    if (R @ np.array([0.0, 0.0, 1.0]))[2] > 0.0:
        return False

    if prev_rvec is not None:
        R_prev, _ = cv2.Rodrigues(prev_rvec)
        rvec_rel, _ = cv2.Rodrigues(R @ R_prev.T)
        if np.linalg.norm(rvec_rel) > MAX_ROTATION_JUMP:
            return False
        if np.linalg.norm(tvec - prev_tvec) > MAX_TRANSLATION_JUMP:
            return False

    return True
