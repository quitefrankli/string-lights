import cv2
import numpy as np

from .config import (
    SQUARE_SIZE, NUM_STRINGS,
    STRING_BRIDGE_TOP, STRING_BRIDGE_BOT, STRING_NUT_TOP, STRING_NUT_BOT,
)

LINE_COLOR = (0, 255, 0)
LINE_WIDTH = 4
from .board import build_board, make_detector, camera_matrix
from .pose import estimate_pose

WIN = "String Tuner"

SAMPLE_COUNT = 20
HANDLE_RADIUS = 9
HIT_RADIUS = 18


def _find_posed_frame(path: str):
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = camera_matrix(w, h)
    adict, id_to_3d = build_board()
    detector = make_detector(adict)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rvec, tvec = estimate_pose(gray, detector, id_to_3d, K)
        if rvec is not None:
            samples.append((frame, rvec, tvec))
            if len(samples) >= SAMPLE_COUNT:
                break
    cap.release()

    if not samples:
        return None

    tvecs = np.array([t.flatten() for _, _, t in samples])
    median_tvec = np.median(tvecs, axis=0)
    best = min(range(len(samples)), key=lambda i: np.linalg.norm(tvecs[i] - median_tvec))
    frame, rvec, tvec = samples[best]
    return frame, rvec, tvec, K


def _project(p3d: np.ndarray, rvec, tvec, K) -> tuple[int, int]:
    pts, _ = cv2.projectPoints(p3d.reshape(1, 3), rvec, tvec, K, np.zeros(5))
    return tuple(pts[0].ravel().astype(int))


def _unproject_to_z0(u: int, v: int, rvec, tvec, K) -> np.ndarray | None:
    """Intersect the camera ray through pixel (u, v) with the board's z=0 plane.
    Returns a (3,) point in board coords, or None if the ray is parallel."""
    K_inv = np.linalg.inv(K)
    dir_cam = K_inv @ np.array([float(u), float(v), 1.0])
    R = cv2.Rodrigues(rvec)[0]
    t = tvec.flatten()
    origin_board = -R.T @ t          # camera centre in board frame
    dir_board = R.T @ dir_cam        # ray direction in board frame
    if abs(dir_board[2]) < 1e-9:
        return None
    s = -origin_board[2] / dir_board[2]
    return origin_board + s * dir_board


def _render(frame, rvec, tvec, K, corners: dict[str, np.ndarray]) -> np.ndarray:
    out = frame.copy()
    dist = np.zeros(5, dtype=np.float64)
    cv2.drawFrameAxes(out, K, dist, rvec, tvec, SQUARE_SIZE * 3)

    bt, bb = corners["bridge_top"], corners["bridge_bot"]
    nt, nb = corners["nut_top"],    corners["nut_bot"]

    for i in range(NUM_STRINGS):
        s = i / (NUM_STRINGS - 1) if NUM_STRINGS > 1 else 0.0
        p_bridge = bt + s * (bb - bt)
        p_nut    = nt + s * (nb - nt)
        pts_2d, _ = cv2.projectPoints(np.array([p_bridge, p_nut]), rvec, tvec, K, dist)
        a = tuple(pts_2d[0].ravel().astype(int))
        b = tuple(pts_2d[1].ravel().astype(int))
        cv2.line(out, a, b, LINE_COLOR, LINE_WIDTH, cv2.LINE_AA)
    return out


def run_tuner(input_path: str) -> None:
    result = _find_posed_frame(input_path)
    if result is None:
        print("No frame with a valid ArUco pose found.")
        return
    frame, rvec, tvec, K = result

    corners: dict[str, np.ndarray] = {
        "bridge_top": np.array(STRING_BRIDGE_TOP, dtype=np.float64),
        "bridge_bot": np.array(STRING_BRIDGE_BOT, dtype=np.float64),
        "nut_top":    np.array(STRING_NUT_TOP,    dtype=np.float64),
        "nut_bot":    np.array(STRING_NUT_BOT,    dtype=np.float64),
    }

    state = {"dragging": None, "hover": None}

    def _nearest(u: int, v: int) -> str | None:
        best, best_d = None, HIT_RADIUS
        for name, p3d in corners.items():
            cu, cv_ = _project(p3d, rvec, tvec, K)
            d = ((u - cu) ** 2 + (v - cv_) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best = name
        return best

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = _nearest(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if state["dragging"] is not None:
                p = _unproject_to_z0(x, y, rvec, tvec, K)
                if p is not None:
                    p[2] = 0.0   # lock to board's z=0 plane
                    corners[state["dragging"]] = p
            else:
                state["hover"] = _nearest(x, y)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        rendered = _render(frame, rvec, tvec, K, corners)
        for name, p3d in corners.items():
            cu, cv_ = _project(p3d, rvec, tvec, K)
            if name == state["dragging"]:
                color = (0, 255, 255)
            elif name == state["hover"]:
                color = (0, 220, 0)
            else:
                color = (200, 200, 200)
            cv2.circle(rendered, (cu, cv_), HANDLE_RADIUS, color, -1)
            cv2.circle(rendered, (cu, cv_), HANDLE_RADIUS + 1, (0, 0, 0), 1)
            cv2.putText(rendered, name, (cu + 12, cv_ - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.putText(rendered, "drag corner handles to align with strings | ESC to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(WIN, rendered)
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("\nFinal values for config.py:")
    for name in ("bridge_top", "bridge_bot", "nut_top", "nut_bot"):
        p = corners[name]
        print(f"STRING_{name.upper()} = ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})")
