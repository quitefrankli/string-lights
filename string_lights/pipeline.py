import cv2
import numpy as np
import subprocess
from pathlib import Path
from typing import Generator, Iterator

from .board import build_board, make_detector, camera_matrix, SQUARE_SIZE
from .config import POSE_RESOLUTION, PoseResolution, MASK_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, MASK_FRAME_SKIP
from .masking import resolve_device, load_models, get_mask
from .pose import estimate_pose, is_pose_valid, compute_median_pose, Pose
from .audio import get_strings_to_highlight, get_random_strings
from .strings import draw_strings_frame


COMPONENTS_DIR = Path("data/components")


def poses_path(stem: str) -> Path:
    return COMPONENTS_DIR / "poses" / f"{stem}.npz"


def masks_path(stem: str) -> Path:
    return COMPONENTS_DIR / "masks" / f"{stem}.npy"


def masks_meta_path(stem: str) -> Path:
    return COMPONENTS_DIR / "masks" / f"{stem}.meta.npz"


def masks_cached(stem: str) -> bool:
    return masks_path(stem).exists() and masks_meta_path(stem).exists()


def _video_meta(cap: cv2.VideoCapture, frames: int | None = None) -> dict:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames is not None:
        total = min(total, frames)
    return {"w": w, "h": h, "fps": fps, "total": total}


def _stream_frames(cap: cv2.VideoCapture, total: int) -> Generator[np.ndarray, None, None]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            return
        yield frame


# ── Pose stage ─────────────────────────────────────────────────────────────

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


def compute_poses(input_path: str, frames: int | None = None) -> tuple[list[Pose], dict]:
    cap = cv2.VideoCapture(input_path)
    meta = _video_meta(cap, frames)
    K = camera_matrix(meta["w"], meta["h"])
    adict, id_to_3d = build_board()
    detector = make_detector(adict)
    raw = pass1_raw_poses(cap, meta["total"], detector, id_to_3d, K)
    resolved = pass2_resolve_poses(raw)
    cap.release()
    detected = sum(1 for r, _ in resolved if r is not None)
    print(f"  pose pass complete: stable in {detected}/{meta['total']} frames")
    return resolved, meta


def save_poses(stem: str, poses: list[Pose], meta: dict) -> Path:
    path = poses_path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(poses)
    rvecs = np.full((n, 3), np.nan, dtype=np.float64)
    tvecs = np.full((n, 3), np.nan, dtype=np.float64)
    for i, (r, t) in enumerate(poses):
        if r is not None:
            rvecs[i] = np.asarray(r).flatten()
            tvecs[i] = np.asarray(t).flatten()
    np.savez_compressed(
        path, rvecs=rvecs, tvecs=tvecs,
        w=meta["w"], h=meta["h"], fps=meta["fps"], total=meta["total"],
    )
    print(f"  poses cached → {path}")
    return path


def load_poses(stem: str) -> tuple[list[Pose], dict]:
    path = poses_path(stem)
    data = np.load(path)
    rvecs = data["rvecs"]
    tvecs = data["tvecs"]
    poses: list[Pose] = []
    for r, t in zip(rvecs, tvecs):
        if np.any(np.isnan(r)):
            poses.append((None, None))
        else:
            poses.append((r.reshape(3, 1), t.reshape(3, 1)))
    meta = {
        "w": int(data["w"]), "h": int(data["h"]),
        "fps": float(data["fps"]), "total": int(data["total"]),
    }
    return poses, meta


# ── Mask stage ─────────────────────────────────────────────────────────────

def compute_masks(
    input_path: str,
    stem: str | None = None,
    frames: int | None = None,
    debug_writer: cv2.VideoWriter | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute SAM2 masks at every MASK_FRAME_SKIP-th frame.

    Streams mask writes to a memmap file (if stem given) to avoid accumulating
    all frames in RAM. Returns (keyframes (K,h,w) uint8, indices (K,) int32, meta).
    """
    cap = cv2.VideoCapture(input_path)
    meta = _video_meta(cap, frames)
    total, h, w = meta["total"], meta["h"], meta["w"]

    device = resolve_device()
    gd_processor, gd_model, sam_processor, sam_model = load_models(device)

    # Pre-allocate max possible keyframes; actual count tracked by k.
    # If stem given, stream directly to disk via memmap to avoid RAM accumulation.
    n_max = max(1, -(-total // MASK_FRAME_SKIP))  # ceiling division
    if stem is not None:
        out_path = masks_path(stem)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        buf = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.uint8, shape=(n_max, h, w))
    else:
        buf = np.zeros((n_max, h, w), dtype=np.uint8)

    indices_list: list[int] = []
    k = 0
    current = np.zeros((h, w), dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i % MASK_FRAME_SKIP == 0:
            current = get_mask(
                frame, MASK_PROMPT,
                gd_processor, gd_model, sam_processor, sam_model,
                device, BOX_THRESHOLD, TEXT_THRESHOLD,
                debug_writer=debug_writer,
            )
            buf[k] = current
            indices_list.append(i)
            k += 1
            if k % 30 == 0 or i == total - 1:
                n_detected = sum(1 for j in range(k) if buf[j].any())
                print(f"  masks: {i+1}/{total} frames, {n_detected}/{k} had detections")
        elif debug_writer is not None:
            vis = frame.copy()
            overlay = vis.copy()
            overlay[current.astype(bool)] = (0, 0, 200)
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
            debug_writer.write(vis)
    cap.release()

    if k == 0:
        k = 1
        indices_list.append(0)

    if isinstance(buf, np.memmap):
        buf.flush()

    return buf[:k], np.array(indices_list, dtype=np.int32), meta


def save_masks(stem: str, indices: np.ndarray, meta: dict) -> Path:
    # masks .npy is already written to disk by compute_masks (via memmap flush);
    # only the small metadata file needs to be saved here.
    npy_path = masks_path(stem)
    meta_path = masks_meta_path(stem)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(meta_path), indices=indices,
        w=meta["w"], h=meta["h"], fps=meta["fps"], total=meta["total"],
    )
    print(f"  masks cached → {npy_path}")
    return npy_path


def load_masks(stem: str) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(str(masks_meta_path(stem)))
    masks = np.load(str(masks_path(stem)), mmap_mode="r")
    meta = {
        "w": int(data["w"]), "h": int(data["h"]),
        "fps": float(data["fps"]), "total": int(data["total"]),
    }
    return masks, data["indices"], meta


def masks_iter(masks: np.ndarray, indices: np.ndarray, total: int) -> Iterator[np.ndarray]:
    """Yield one mask per frame, holding each keyframe until the next index."""
    K = len(indices)
    cur = 0
    for i in range(total):
        while cur + 1 < K and indices[cur + 1] <= i:
            cur += 1
        yield masks[cur]


# ── Render stage ───────────────────────────────────────────────────────────

def render_video(
    input_path: str,
    output_path: str,
    poses: list[Pose],
    strings: list[list[int]],
    meta: dict,
    masks: tuple[np.ndarray, np.ndarray] | None = None,
    fast: bool = False,
    include_audio: bool = True,
    draw_axes: bool = True,
) -> None:
    w, h, fps, total = meta["w"], meta["h"], meta["fps"], meta["total"]
    K = camera_matrix(w, h)
    dist = np.zeros(5, dtype=np.float64)
    axis_len = SQUARE_SIZE * 3

    cap = cv2.VideoCapture(input_path)
    frame_stream = _stream_frames(cap, total)
    if masks is not None:
        mask_stream = masks_iter(masks[0], masks[1], total)
    else:
        empty = np.zeros((h, w), dtype=np.uint8)
        mask_stream = (empty for _ in range(total))

    last_active: dict[int, int] = {}

    encode = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if fast:
        encode = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p"]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
        "-i", "pipe:0",
    ]
    if include_audio:
        cmd += ["-i", input_path, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy", "-shortest"]
    cmd += [*encode, output_path]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i, (frame, mask) in enumerate(zip(frame_stream, mask_stream)):
            if i >= len(poses):
                break
            rvec, tvec = poses[i]
            original = frame.copy() if mask.any() else None
            if draw_axes and rvec is not None:
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, axis_len)
            draw_strings_frame(frame, i, rvec, tvec, strings[i], last_active, K, fps)
            if original is not None:
                frame[mask.astype(bool)] = original[mask.astype(bool)]
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    cap.release()
    stderr = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed:\n{stderr}")


# ── Top-level orchestration ────────────────────────────────────────────────

def process_video(
    input_path: str,
    output_path: str,
    frames: int | None = None,
    disable_masking: bool = False,
    random_strings: bool = False,
    debug_masks: bool = False,
    only_poses: bool = False,
    only_masks: bool = False,
    use_cached_poses: bool = False,
    use_cached_masks: bool = False,
) -> None:
    stem = Path(input_path).stem

    if debug_masks:
        debug_out = str(Path(output_path).with_suffix("").with_suffix("")) + ".debug.mp4"
        cap = cv2.VideoCapture(input_path)
        meta = _video_meta(cap, frames)
        cap.release()
        total = meta["total"]
        debug_writer = cv2.VideoWriter(
            debug_out, cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["w"], meta["h"])
        )
        try:
            if use_cached_masks and masks_cached(stem):
                print(f"  using cached masks ← {masks_path(stem)}")
                masks_arr, indices, _ = load_masks(stem)
                cap2 = cv2.VideoCapture(input_path)
                for frame, mask in zip(_stream_frames(cap2, total), masks_iter(masks_arr, indices, total)):
                    vis = frame.copy()
                    overlay = vis.copy()
                    overlay[mask.astype(bool)] = (0, 0, 200)
                    cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
                    debug_writer.write(vis)
                cap2.release()
            else:
                compute_masks(input_path, frames=frames, debug_writer=debug_writer)
        finally:
            debug_writer.release()
        print(f"Done.  Debug video → {debug_out}")
        return

    only_mode = only_poses or only_masks

    poses: list[Pose] | None = None
    meta: dict | None = None

    # Pose stage: always run unless --use-cached-poses (and not --masks-only)
    if only_poses or (not only_mode and not use_cached_poses):
        poses, meta = compute_poses(input_path, frames)
        save_poses(stem, poses, meta)
    elif not only_mode and use_cached_poses:
        if not poses_path(stem).exists():
            raise FileNotFoundError(f"No cached poses at {poses_path(stem)}")
        print(f"  using cached poses ← {poses_path(stem)}")
        poses, meta = load_poses(stem)
        if frames is not None:
            meta = {**meta, "total": min(meta["total"], frames)}
            poses = poses[:meta["total"]]

    # --masks: just compute & cache masks
    if only_masks:
        masks_arr, indices, mmeta = compute_masks(input_path, stem=stem, frames=frames)
        save_masks(stem, indices, mmeta)

    if only_mode:
        return

    total, fps = meta["total"], meta["fps"]

    npy_path = Path(input_path).with_suffix(".npy")
    if random_strings or not npy_path.exists():
        if not random_strings:
            print(f"  no tab data at {npy_path}, using random strings")
        strings = get_random_strings(total, fps)
    else:
        strings = get_strings_to_highlight(input_path, total, fps)

    masks_data = None
    if not disable_masking:
        if use_cached_masks:
            if not masks_cached(stem):
                raise FileNotFoundError(f"No cached masks at {masks_path(stem)}")
            print(f"  using cached masks ← {masks_path(stem)}")
            masks_arr, indices, _ = load_masks(stem)
        else:
            masks_arr, indices, mmeta = compute_masks(input_path, stem=stem, frames=frames)
            save_masks(stem, indices, mmeta)
        masks_data = (masks_arr, indices)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    render_video(input_path, output_path, poses, strings, meta, masks=masks_data)
    detected = sum(1 for r, _ in poses if r is not None)
    print(f"Done.  Board pose found in {detected}/{total} frames ({100*detected//total}%).")
