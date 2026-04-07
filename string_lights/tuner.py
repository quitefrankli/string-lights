import cv2
import numpy as np

from .config import (
    SQUARE_SIZE, CHARUCO_BOARD_WIDTH, NUM_STRINGS,
    STRING_SPACING_FACTOR, STRING0_OFFSET_X_FACTOR, STRING0_OFFSET_Y_FACTOR,
    STRING_LENGTH_FACTOR, STRING_CONVERGENCE_FACTOR,
    STRING_COLOR, STRING_CORE_COLOR, STRING_ALPHA,
)
from .board import build_board, make_detector, camera_matrix
from .pose import estimate_pose

WIN = "String Tuner"


SAMPLE_COUNT = 20


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


def _render(frame, rvec, tvec, K, spacing_f, offset_x_f, offset_y_f, length_f, convergence_f):
    out = frame.copy()
    dist = np.zeros(5, dtype=np.float64)
    cv2.drawFrameAxes(out, K, dist, rvec, tvec, SQUARE_SIZE * 3)
    spacing = SQUARE_SIZE * spacing_f
    offset = np.array([CHARUCO_BOARD_WIDTH * offset_x_f, SQUARE_SIZE * offset_y_f, 0], dtype=np.float64)
    length = CHARUCO_BOARD_WIDTH * length_f
    h, w = out.shape[:2]

    endpoints = []
    xs, ys = [], []
    for i in range(NUM_STRINGS):
        y0 = -offset[1] - i * spacing * convergence_f
        y1 = -offset[1] - i * spacing
        p0 = np.array([-offset[0], y0, offset[2]], dtype=np.float64)
        p1 = np.array([length - offset[0], y1, offset[2]], dtype=np.float64)
        pts_2d, _ = cv2.projectPoints(np.array([p0, p1]), rvec, tvec, K, dist)
        a = tuple(pts_2d[0].ravel().astype(int))
        b = tuple(pts_2d[1].ravel().astype(int))
        endpoints.append((a, b))
        xs.extend((a[0], b[0]))
        ys.extend((a[1], b[1]))

    pad = 18 * 4
    x0 = max(0, min(xs) - pad)
    y0_ = max(0, min(ys) - pad)
    x1 = min(w, max(xs) + pad)
    y1_ = min(h, max(ys) + pad)
    rh, rw = y1_ - y0_, x1 - x0

    outer_bloom = np.zeros((rh, rw, 3), dtype=np.uint8)
    inner_glow = np.zeros((rh, rw, 3), dtype=np.uint8)
    for a, b in endpoints:
        color = tuple(int(c * STRING_ALPHA) for c in STRING_COLOR)
        a_r = (a[0] - x0, a[1] - y0_)
        b_r = (b[0] - x0, b[1] - y0_)
        cv2.line(outer_bloom, a_r, b_r, color, 22, cv2.LINE_AA)
        cv2.line(inner_glow, a_r, b_r, color, 8, cv2.LINE_AA)

    outer_bloom = cv2.GaussianBlur(outer_bloom, (0, 0), sigmaX=18)
    inner_glow = cv2.GaussianBlur(inner_glow, (0, 0), sigmaX=5)

    roi = out[y0_:y1_, x0:x1]
    cv2.add(roi, cv2.multiply(outer_bloom, np.array([0.25, 0.25, 0.25, 0], dtype=np.float64)).astype(np.uint8), dst=roi)
    cv2.add(roi, cv2.multiply(inner_glow, np.array([0.7, 0.7, 0.7, 0], dtype=np.float64)).astype(np.uint8), dst=roi)

    core = out.copy()
    for a, b in endpoints:
        cv2.line(core, a, b, STRING_CORE_COLOR, 2, cv2.LINE_AA)
    cv2.addWeighted(core, STRING_ALPHA, out, 1 - STRING_ALPHA, 0, out)
    return out


def run_tuner(input_path: str) -> None:
    result = _find_posed_frame(input_path)
    if result is None:
        print("No frame with a valid ArUco pose found.")
        return
    frame, rvec, tvec, K = result

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    # OpenCV trackbars are integer-only; scale ×100
    cv2.createTrackbar("spacing", WIN, int(STRING_SPACING_FACTOR * 100), 200, lambda _: None)
    cv2.createTrackbar("offset_x", WIN, int(STRING0_OFFSET_X_FACTOR * 100) + 500, 1500, lambda _: None)
    cv2.createTrackbar("offset_y", WIN, int(STRING0_OFFSET_Y_FACTOR * 100) + 200, 400, lambda _: None)
    cv2.createTrackbar("length", WIN, int(STRING_LENGTH_FACTOR * 100), 2000, lambda _: None)
    cv2.createTrackbar("convergence", WIN, int(STRING_CONVERGENCE_FACTOR * 100), 200, lambda _: None)

    while True:
        spacing_f = cv2.getTrackbarPos("spacing", WIN) / 100.0
        offset_x_f = (cv2.getTrackbarPos("offset_x", WIN) - 500) / 100.0
        offset_y_f = (cv2.getTrackbarPos("offset_y", WIN) - 200) / 100.0
        length_f = cv2.getTrackbarPos("length", WIN) / 100.0
        convergence_f = cv2.getTrackbarPos("convergence", WIN) / 100.0

        rendered = _render(frame, rvec, tvec, K, spacing_f, offset_x_f, offset_y_f, length_f, convergence_f)
        labels = [
            f"1 spacing:      {spacing_f:.2f}",
            f"2 offset_x:     {offset_x_f:.2f}",
            f"3 offset_y:     {offset_y_f:.2f}",
            f"4 length:       {length_f:.2f}",
            f"5 convergence:  {convergence_f:.2f}",
        ]
        for j, txt in enumerate(labels):
            cv2.putText(rendered, txt, (10, 30 + j * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(WIN, rendered)
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("\nFinal values for config.py:")
    print(f"STRING_SPACING_FACTOR = {spacing_f}")
    print(f"STRING0_OFFSET_X_FACTOR = {offset_x_f}")
    print(f"STRING0_OFFSET_Y_FACTOR = {offset_y_f}")
    print(f"STRING_LENGTH_FACTOR = {length_f}")
    print(f"STRING_CONVERGENCE_FACTOR = {convergence_f}")
