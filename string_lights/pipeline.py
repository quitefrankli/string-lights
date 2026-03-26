import cv2
import numpy as np
import subprocess
import tempfile

from .board import build_board, make_detector, camera_matrix, SQUARE_SIZE
from .config import POSE_RESOLUTION, PoseResolution, MASK_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, MASK_FRAME_SKIP
from .masking import resolve_device, load_models, get_mask
from .pose import estimate_pose, is_pose_valid, Pose
from .audio import get_strings_to_highlight
from .strings import draw_strings


def pass1_raw_poses(cap: cv2.VideoCapture, total: int, detector: cv2.aruco.ArucoDetector, id_to_3d: dict[int, np.ndarray], K: np.ndarray) -> list[Pose]:
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
            print(f"  pass1 {i+1}/{total} raw detections: {found}")
    return raw


def pass2_resolve_poses(raw_poses: list[Pose], mode: PoseResolution = POSE_RESOLUTION) -> list[Pose]:
    """Derive a stable pose for every frame from the raw estimates."""
    n = len(raw_poses)

    # Forward pass: accept poses that pass validity check
    accepted: list[Pose] = []
    last_rvec, last_tvec = None, None
    for rvec, tvec in raw_poses:
        if rvec is not None and is_pose_valid(rvec, tvec, last_rvec, last_tvec):
            last_rvec, last_tvec = rvec, tvec
            accepted.append((rvec, tvec))
        else:
            last_rvec, last_tvec = None, None
            accepted.append((None, None))

    if mode == PoseResolution.OMIT:
        return accepted

    if mode == PoseResolution.HOLD:
        resolved: list[Pose] = []
        last: Pose = (None, None)
        for pose in accepted:
            if pose[0] is not None:
                last = pose
            resolved.append(last)
        return resolved

    # INTERPOLATE: fill gaps between valid frames
    resolved = list(accepted)
    i = 0
    while i < n:
        if resolved[i][0] is not None:
            i += 1
            continue
        prev_idx = next((j for j in range(i - 1, -1, -1) if resolved[j][0] is not None), None)
        next_idx = next((j for j in range(i, n) if resolved[j][0] is not None), None)
        gap_end  = (next_idx - 1) if next_idx is not None else n - 1
        if prev_idx is not None and next_idx is not None:
            r0, t0 = resolved[prev_idx]
            r1, t1 = resolved[next_idx]
            span = next_idx - prev_idx
            for k in range(i, gap_end + 1):
                alpha = (k - prev_idx) / span
                resolved[k] = (r0 + alpha * (r1 - r0), t0 + alpha * (t1 - t0))
        i = gap_end + 1
    return resolved


def pass3_hand_masks(cap: cv2.VideoCapture, total: int, w: int, h: int) -> list[np.ndarray]:
    """Generate per-frame hand masks using GroundingDINO + SAM."""
    device = resolve_device()
    models = load_models(device)
    gd_processor, gd_model, sam_processor, sam_model = models

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    masks: list[np.ndarray] = []
    current_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            masks.append(current_mask)
            continue
        if i % MASK_FRAME_SKIP == 0:
            current_mask = get_mask(
                frame, MASK_PROMPT,
                gd_processor, gd_model, sam_processor, sam_model,
                device, BOX_THRESHOLD, TEXT_THRESHOLD,
            )
        masks.append(current_mask)
        if (i + 1) % 60 == 0:
            print(f"  pass3 {i+1}/{total}  hand masks")
    return masks


def pass4_write_output(cap: cv2.VideoCapture,
                       resolved_poses: list[Pose],
                       hand_masks: list[np.ndarray],
                       K: np.ndarray,
                       output_path: str,
                       fps: float,
                       w: int,
                       h: int) -> None:
    """Seek back to the start and write annotated frames."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total = len(resolved_poses)

    frames = []
    originals = []
    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        originals.append(frame.copy())
        frames.append(frame)

    strings = get_strings_to_highlight(len(frames), fps)
    draw_strings(frames, resolved_poses, strings, K, fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame_idx, frame in enumerate(frames):
        mask = hand_masks[frame_idx]
        if mask.any():
            mask_bool = mask.astype(bool)
            frame[mask_bool] = originals[frame_idx][mask_bool]
        out.write(frame)
    out.release()


def process_video(input_path: str, 
                  output_path: str, 
                  frames: int | None = None, 
                  disable_masking: bool = False) -> None:
    cap   = cv2.VideoCapture(input_path)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames is not None:
        total = min(total, frames)

    K = camera_matrix(w, h)
    adict, id_to_3d = build_board()
    detector = make_detector(adict)

    print(f"Processing {total} frames  ({w}×{h} @ {fps:.0f} fps)  →  {output_path}")

    raw_poses      = pass1_raw_poses(cap, total, detector, id_to_3d, K)
    resolved_poses = pass2_resolve_poses(raw_poses)

    detected = sum(1 for r, _ in resolved_poses if r is not None)
    print(f"  pass2 complete: stable pose in {detected}/{total} frames")

    if disable_masking:
        hand_masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(total)]
    else:
        hand_masks = pass3_hand_masks(cap, total, w, h)
    print(f"  pass3 complete: hand masks for {total} frames")

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp_path = tmp.name
        pass4_write_output(cap, resolved_poses, hand_masks, K, tmp_path, fps, w, h)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-i", input_path,
             "-map", "0:v:0", "-map", "1:a?",
             "-c:v", "copy", "-c:a", "copy", "-shortest", output_path],
            check=True, capture_output=True,
        )
    cap.release()

    print(f"Done.  Board pose found in {detected}/{total} frames ({100*detected//total}%).")
