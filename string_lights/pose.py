import cv2
import numpy as np

Pose = tuple[np.ndarray | None, np.ndarray | None]


class OneEuroFilter:
    """Adaptive low-pass filter: cutoff rises with motion speed.

    See https://gery.casiez.net/1euro/ — `min_cutoff` sets the floor (more
    smoothing when still), `beta` controls how aggressively the cutoff opens up
    during fast motion (less lag).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def _rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
    r = rvec.flatten()
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    s = np.sin(angle * 0.5)
    return np.array([np.cos(angle * 0.5), *(r / angle * s)])


def _quat_to_rvec(q: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-12)
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, q[0])
    return q[1:] / s * angle


def smooth_poses(
    poses: list["Pose"],
    fps: float,
    t_min_cutoff: float,
    t_beta: float,
    r_min_cutoff: float,
    r_beta: float,
) -> list["Pose"]:
    """One-Euro smooth (rvec, tvec). Rotation goes through quaternions with
    antipodal alignment to avoid q/-q flips. Filters reset on invalid gaps."""
    dt = 1.0 / fps
    f_t = OneEuroFilter(min_cutoff=t_min_cutoff, beta=t_beta)
    f_q = OneEuroFilter(min_cutoff=r_min_cutoff, beta=r_beta)
    out: list[Pose] = []
    q_prev: np.ndarray | None = None
    for rvec, tvec in poses:
        if rvec is None:
            f_t.reset(); f_q.reset()
            q_prev = None
            out.append((None, None))
            continue
        q = _rvec_to_quat(rvec)
        if q_prev is not None and float(np.dot(q, q_prev)) < 0.0:
            q = -q
        q_s = f_q(q, dt)
        q_s = q_s / (np.linalg.norm(q_s) + 1e-12)
        q_prev = q_s
        rvec_s = _quat_to_rvec(q_s).reshape(3, 1)
        tvec_s = f_t(tvec.flatten(), dt).reshape(3, 1)
        out.append((rvec_s, tvec_s))
    return out


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


def pose_jump(prev: Pose, cur: Pose) -> tuple[float, float]:
    """Translation (metres) and rotation (radians) distance between two poses."""
    dt = float(np.linalg.norm(cur[1].flatten() - prev[1].flatten()))
    R_prev = cv2.Rodrigues(prev[0])[0]
    R_cur = cv2.Rodrigues(cur[0])[0]
    cos_theta = (np.trace(R_cur @ R_prev.T) - 1.0) * 0.5
    dr = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return dt, dr
