import cv2
import click
import numpy as np
from pathlib import Path

# ── Board config ──────────────────────────────────────────────────────────────
# 6×6 ChArUco board, 12 mm squares, 9 mm markers, DICT_5×5_50
# Board was generated with the legacy OpenCV pattern (pre-4.x convention).
BOARD_COLS  = 6
BOARD_ROWS  = 6
SQUARE_SIZE = 0.012   # metres
MARKER_SIZE = 0.009   # metres
ARUCO_DICT  = cv2.aruco.DICT_5X5_50

# ── Camera intrinsics ─────────────────────────────────────────────────────────
# iPhone 15 main camera: f=6.765 mm, sensor 7.21 mm wide → f_px ≈ 1800 at 1920 px
FOCAL_PX_AT_1920 = 1800.0

# ── String line endpoints in board-local coordinates (metres) ─────────────────
# Back-projected from frame 0:   +X toward sound hole,  +Y toward headstock.
# Extend slightly beyond the guitar landmarks for visual clarity.
LINE_START = np.array([[0.16,  -0.05, 0.0]], dtype=np.float64)  # near bridge / sound hole
LINE_END   = np.array([[-0.10,  0.10, 0.0]], dtype=np.float64)  # near headstock

# ── Visualisation ─────────────────────────────────────────────────────────────
LINE_COLOR     = (0, 255, 0)   # BGR green
LINE_THICKNESS = 4


def build_board():
    """Return (adict, id_to_3d) for the physical ChArUco board."""
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
    """ArUco detector tuned for robustness to mild motion blur."""
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


def estimate_pose(gray, detector, id_to_3d, K):
    """Detect markers and solve PnP. Returns (rvec, tvec) or (None, None)."""
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


def project_pt(pt_3d, rvec, tvec, K):
    dist = np.zeros(5, dtype=np.float64)
    p, _ = cv2.projectPoints(pt_3d, rvec, tvec, K, dist)
    return tuple(np.clip(p.reshape(2), -10000, 10000).astype(int).tolist())


def process_video(input_path: str, output_path: str):
    cap   = cv2.VideoCapture(input_path)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    K        = camera_matrix(w, h)
    adict, id_to_3d = build_board()
    detector = make_detector(adict)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Processing {total} frames  ({w}×{h} @ {fps:.0f} fps)  →  {output_path}")
    detected = 0

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rvec, tvec = estimate_pose(gray, detector, id_to_3d, K)

        if rvec is not None:
            detected += 1
            dist = np.zeros(5, dtype=np.float64)
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, SQUARE_SIZE * 6)

        out.write(frame)
        if (frame_idx + 1) % 60 == 0:
            print(f"  {frame_idx+1}/{total}  board found in {detected} frames so far")

    cap.release()
    out.release()
    print(f"Done.  Board pose found in {detected}/{total} frames ({100*detected//total}%).")


@click.command()
@click.argument("filename")
def main(filename: str):
    input_dir = Path("data/input")
    
    # Determine the actual input file path
    if Path(filename).suffix:
        # Filename has an extension, use it as-is
        input_path = input_dir / filename
    else:
        # No extension provided, try common video extensions
        common_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        input_path = None
        for ext in common_extensions:
            candidate = input_dir / (filename + ext)
            if candidate.exists():
                input_path = candidate
                break
        
        # If no file found with extensions, try the filename as-is
        if input_path is None:
            input_path = input_dir / filename
    
    # Check if input file exists
    if not input_path.exists():
        raise click.ClickException(f"Input file not found: {input_path}")
    
    # Extract stem from the actual resolved path for the output
    stem = input_path.stem
    output_path = Path("data/output") / f"{stem}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_video(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
