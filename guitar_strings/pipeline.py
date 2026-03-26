import cv2
import numpy as np

from .board import build_board, make_detector, camera_matrix, SQUARE_SIZE
from .pose import estimate_pose, is_pose_valid


def pass1_raw_poses(cap, total, detector, id_to_3d, K):
    """Read every frame and return raw PnP estimates; (None, None) when detection fails."""
    raw = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            raw.append((None, None))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw.append(estimate_pose(gray, detector, id_to_3d, K))
        if (i + 1) % 60 == 0:
            found = sum(1 for r, t in raw if r is not None)
            print(f"  pass1  {i+1}/{total}  raw detections: {found}")
    return raw


def pass2_resolve_poses(raw_poses):
    """Derive a stable pose for every frame from the raw estimates.

    Strategy: forward pass that accepts a new raw pose only when it passes
    the jump/orientation validity check, then holds the last accepted pose
    for frames where detection failed or the pose was rejected.
    """
    resolved = []
    last_rvec, last_tvec = None, None
    for rvec, tvec in raw_poses:
        if rvec is not None and is_pose_valid(rvec, tvec, last_rvec, last_tvec):
            last_rvec, last_tvec = rvec, tvec
        resolved.append((last_rvec, last_tvec))
    return resolved


def pass3_write_output(cap, resolved_poses, K, output_path, fps, w, h):
    """Seek back to the start and write annotated frames."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    dist   = np.zeros(5, dtype=np.float64)
    total  = len(resolved_poses)

    for frame_idx, (rvec, tvec) in enumerate(resolved_poses):
        ret, frame = cap.read()
        if not ret:
            break
        if rvec is not None:
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, SQUARE_SIZE * 6)
        out.write(frame)
        if (frame_idx + 1) % 60 == 0:
            print(f"  pass3  {frame_idx+1}/{total}  written")

    out.release()


def process_video(input_path: str, output_path: str):
    cap   = cv2.VideoCapture(input_path)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = 60  # dev limit

    K = camera_matrix(w, h)
    adict, id_to_3d = build_board()
    detector = make_detector(adict)

    print(f"Processing {total} frames  ({w}×{h} @ {fps:.0f} fps)  →  {output_path}")

    raw_poses      = pass1_raw_poses(cap, total, detector, id_to_3d, K)
    resolved_poses = pass2_resolve_poses(raw_poses)

    detected = sum(1 for r, _ in resolved_poses if r is not None)
    print(f"  pass2 complete: stable pose in {detected}/{total} frames")

    pass3_write_output(cap, resolved_poses, K, output_path, fps, w, h)
    cap.release()

    print(f"Done.  Board pose found in {detected}/{total} frames ({100*detected//total}%).")
