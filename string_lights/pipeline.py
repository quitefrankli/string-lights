import cv2
import numpy as np
import subprocess
from pathlib import Path
from typing import Generator

from .board import build_board, make_detector, camera_matrix, SQUARE_SIZE
from .config import POSE_RESOLUTION, PoseResolution, MASK_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, MASK_FRAME_SKIP
from .masking import resolve_device, load_models, get_mask
from .pose import estimate_pose, is_pose_valid, compute_median_pose, Pose
from .audio import get_strings_to_highlight, get_random_strings
from .strings import draw_strings_frame


def pass1_raw_poses(cap: cv2.VideoCapture, total: int, detector: cv2.aruco.ArucoDetector, id_to_3d: dict[int, np.ndarray], K: np.ndarray) -> list[Pose]:
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
    n = len(raw_poses)

    median_origin = compute_median_pose(raw_poses)
    accepted: list[Pose] = []
    for rvec, tvec in raw_poses:
        if rvec is not None and median_origin is not None and is_pose_valid(tvec, median_origin):
            accepted.append((rvec, tvec))
        else:
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


def _stream_frames(cap: cv2.VideoCapture, total: int) -> Generator[np.ndarray, None, None]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            return
        yield frame


def _stream_with_masks(
    frames: Generator[np.ndarray, None, None],
    w: int, h: int,
    debug_writer: cv2.VideoWriter | None = None,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    device = resolve_device()
    gd_processor, gd_model, sam_processor, sam_model = load_models(device)
    current_mask = np.zeros((h, w), dtype=np.uint8)
    for i, frame in enumerate(frames):
        if i % MASK_FRAME_SKIP == 0:
            current_mask = get_mask(
                frame, MASK_PROMPT,
                gd_processor, gd_model, sam_processor, sam_model,
                device, BOX_THRESHOLD, TEXT_THRESHOLD,
                debug_writer=debug_writer,
            )
        elif debug_writer:
            vis = frame.copy()
            overlay = vis.copy()
            overlay[current_mask.astype(bool)] = (0, 0, 200)
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
            debug_writer.write(vis)
        if (i + 1) % 60 == 0:
            print(f"  pass3 {i+1}  hand masks")
        yield frame, current_mask


def process_video(input_path: str,
                  output_path: str,
                  frames: int | None = None,
                  disable_masking: bool = False,
                  random_strings: bool = False,
                  debug_masks: bool = False) -> None:
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

    npy_path = Path(input_path).with_suffix(".npy")
    if random_strings or not npy_path.exists():
        if not random_strings:
            print(f"  no tab data at {npy_path}, using random strings")
        strings = get_random_strings(total, fps)
    else:
        strings = get_strings_to_highlight(input_path, total, fps)

    debug_writer = None
    if debug_masks:
        debug_out = str(Path(output_path).with_suffix("").with_suffix("")) + ".debug.mp4"
        debug_writer = cv2.VideoWriter(debug_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    empty_mask = np.zeros((h, w), dtype=np.uint8)
    if disable_masking:
        frames_masks = ((frame, empty_mask) for frame in _stream_frames(cap, total))
    else:
        frames_masks = _stream_with_masks(_stream_frames(cap, total), w, h, debug_writer)

    dist     = np.zeros(5, dtype=np.float64)
    axis_len = SQUARE_SIZE * 3
    last_active: dict[int, int] = {}

    try:
        if not debug_masks:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
                "-i", "pipe:0",
                "-i", input_path,
                "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "copy", "-shortest",
                output_path,
            ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for frame_idx, (frame, mask) in enumerate(frames_masks):
                    original = frame.copy()
                    rvec, tvec = resolved_poses[frame_idx]
                    if rvec is not None:
                        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, axis_len)
                    draw_strings_frame(frame, frame_idx, rvec, tvec, strings[frame_idx], last_active, K, fps)
                    if mask.any():
                        frame[mask.astype(bool)] = original[mask.astype(bool)]
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                proc.stdin.close()
            except BrokenPipeError:
                pass
            stderr = proc.stderr.read().decode(errors="replace")
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed:\n{stderr}")
        else:
            for _ in frames_masks:
                pass
    finally:
        if debug_writer:
            debug_writer.release()

    cap.release()
    print(f"Done.  Board pose found in {detected}/{total} frames ({100*detected//total}%).")
