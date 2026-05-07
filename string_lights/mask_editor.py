import ast
import bisect
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .pipeline import masks_path, masks_meta_path, save_masks

WIN = "Mask Editor"
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".m4v"]
INPUT_DIR = Path("data/input")


def _resolve_input(stem: str) -> Path | None:
    for ext in VIDEO_EXTS:
        p = INPUT_DIR / (stem + ext)
        if p.exists():
            return p
    return None


def _load_meta(stem: str) -> tuple[np.ndarray, dict]:
    data = np.load(str(masks_meta_path(stem)))
    meta = {"w": int(data["w"]), "h": int(data["h"]),
            "fps": float(data["fps"]), "total": int(data["total"])}
    return data["indices"].copy(), meta


def _parse_npy_header(f):
    assert f.read(6) == b'\x93NUMPY', "not a .npy file"
    major = int.from_bytes(f.read(1), 'little')
    int.from_bytes(f.read(1), 'little')  # minor
    hlen = int.from_bytes(f.read(2 if major == 1 else 4), 'little')
    d = ast.literal_eval(f.read(hlen).decode('latin1').strip())
    return tuple(d['shape']), np.dtype(d['descr'])


def _extract_to_mmap(npz_path: Path, tmp_path: Path) -> np.ndarray:
    """Stream-decompress masks from npz to a temp mmap, keeping RAM use low."""
    with zipfile.ZipFile(str(npz_path)) as zf:
        with zf.open('masks.npy') as f:
            shape, dtype = _parse_npy_header(f)
            N, H, W = shape
            mmap = np.lib.format.open_memmap(str(tmp_path), mode='w+', dtype=dtype, shape=shape)
            chunk = max(1, 64 * 1024 * 1024 // (H * W))  # ~64 MB of frames per read
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                raw = f.read((end - start) * H * W)
                mmap[start:end] = np.frombuffer(raw, dtype=dtype).reshape(end - start, H, W)
                print(f"\r  decompressing {end}/{N} frames...", end='', flush=True)
    print()
    mmap.flush()
    return mmap


def _ki(indices: np.ndarray, frame_idx: int) -> int:
    return max(0, bisect.bisect_right(indices, frame_idx) - 1)


def _overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return frame
    out = frame.copy()
    ov = out.copy()
    ov[mask.astype(bool)] = (0, 0, 200)
    cv2.addWeighted(ov, 0.4, out, 0.6, 0, out)
    return out


def _get_frame(cap: cv2.VideoCapture, idx: int, h: int, w: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else np.zeros((h, w, 3), np.uint8)


def run_mask_editor(stem: str) -> None:
    video_path = _resolve_input(stem)
    if video_path is None:
        print(f"No input video found for '{stem}' in {INPUT_DIR}")
        return
    npz = masks_path(stem)
    if not npz.exists():
        print(f"No cached masks for '{stem}'. Run: uv run sl run {stem} --masks")
        return

    indices, meta = _load_meta(stem)
    N, total = len(indices), meta["total"]
    h, w = meta["h"], meta["w"]
    fps = meta["fps"]

    tmp = Path(tempfile.mktemp(suffix='.npy'))
    print(f"Decompressing {N} masks ({N * h * w / 1e9:.1f} GB uncompressed)...")
    try:
        masks = _extract_to_mmap(npz, tmp)
        _run_gui(stem, masks, indices, meta, video_path)
    finally:
        tmp.unlink(missing_ok=True)


def _run_gui(stem: str, masks: np.ndarray, indices: np.ndarray, meta: dict, video_path: Path) -> None:
    N, total = len(indices), meta["total"]
    h, w = meta["h"], meta["w"]
    fps = meta["fps"]

    cap = cv2.VideoCapture(str(video_path))
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    fi = 0
    playing = False
    show_mask = True
    clipboard: np.ndarray | None = None
    dirty: set[int] = set()

    # Trackbar: use a flag to suppress feedback when we set it from code.
    _tb_dragged = [False]

    def _on_trackbar(val: int) -> None:
        _tb_dragged[0] = True

    cv2.createTrackbar('time', WIN, 0, total - 1, _on_trackbar)

    while True:
        # If user dragged the trackbar, adopt its value.
        if _tb_dragged[0]:
            _tb_dragged[0] = False
            fi = cv2.getTrackbarPos('time', WIN)
            playing = False

        ki = _ki(indices, fi)
        frame = _get_frame(cap, fi, h, w)
        out = _overlay(frame, masks[ki]) if show_mask else frame.copy()

        secs = fi / fps
        label = f"{int(secs // 60):02d}:{secs % 60:05.2f}  frame {fi}/{total - 1}  kf {ki}/{N - 1}{'  [modified]' if ki in dirty else ''}"
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if ki in dirty else (0, 255, 0), 2)
        cv2.putText(out, "c:copy  v:paste  m:mask  s:save  spc:play  ,/.:prev/next-kf  ←→:frame  ESC:quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        if clipboard is not None:
            cv2.putText(out, "[clipboard]", (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow(WIN, out)

        # Keep trackbar in sync when navigating via keys or playback.
        cv2.setTrackbarPos('time', WIN, fi)

        delay = max(1, int(1000 / fps)) if playing else 30
        key = cv2.waitKeyEx(delay)

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

        k = key & 0xFF
        if key == 27 or k == 27:  # ESC
            break
        elif k == ord(' '):
            playing = not playing
        elif k == ord('m'):
            show_mask = not show_mask
        elif k == ord('c'):
            clipboard = masks[ki].copy()
            print(f"copied keyframe {ki} (video frame {indices[ki]})")
        elif k == ord('v') and clipboard is not None:
            masks[ki] = clipboard
            dirty.add(ki)
            print(f"pasted to keyframe {ki} (video frame {indices[ki]})")
        elif k == ord('s'):
            print("saving...", end=' ', flush=True)
            save_masks(stem, masks, indices, meta)
            dirty.clear()
            print("done.")
        elif k == ord(','):  # prev keyframe
            if ki > 0:
                fi = int(indices[ki - 1])
            playing = False
        elif k == ord('.'):  # next keyframe
            if ki + 1 < N:
                fi = int(indices[ki + 1])
            playing = False
        elif key in (65361, 0xFF51):  # left arrow — one frame back
            fi = max(0, fi - 1)
            playing = False
        elif key in (65363, 0xFF53):  # right arrow — one frame forward
            fi = min(total - 1, fi + 1)
            playing = False

        if playing:
            fi += 1
            if fi >= total:
                fi = total - 1
                playing = False

    cap.release()
    cv2.destroyAllWindows()

    if dirty:
        ans = input(f"\n{len(dirty)} unsaved change(s). Save? [y/N] ")
        if ans.strip().lower() == 'y':
            print("saving...", end=' ', flush=True)
            save_masks(stem, masks, indices, meta)
            print("done.")
