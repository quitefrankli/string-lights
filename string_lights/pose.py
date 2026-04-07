import cv2
import numpy as np

from .config import MAX_TRANSLATION_JUMP

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


def compute_median_pose(raw_poses: list[Pose]) -> np.ndarray | None:
    origins = []
    for rvec, tvec in raw_poses:
        if rvec is not None:
            origins.append(tvec.flatten())
    if not origins:
        return None
    return np.median(origins, axis=0)


def is_pose_valid(tvec: np.ndarray, median_origin: np.ndarray) -> bool:
    return np.linalg.norm(tvec.flatten() - median_origin) < MAX_TRANSLATION_JUMP
