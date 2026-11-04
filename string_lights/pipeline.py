import cv2
import json
import numpy as np
import subprocess
import sys
import threading
from contextlib import closing
from pathlib import Path
from typing import Generator, Iterator

from tqdm import tqdm

from .config import MASK_MODEL_ID, MASK_CHECKPOINT, MASK_ENCODING
from .masking import video_masks, read_seed, seed_mask_path, blend_mask, mask_overlay
from .mask_stabilization import stabilize_video_masks
from .audio import get_strings_to_highlight, get_random_strings
from .strings import draw_projected_strings_frame

from .paths import artifact_path, project_dir, artifact_overrides


def masks_path(stem: str) -> Path:
    return artifact_path(stem, "masks")


def _masks_tmp_path(stem: str) -> Path:
    if "masks" in artifact_overrides.get():
        return masks_path(stem).with_suffix(".npy")
    return project_dir(stem) / "tmp/masks.npy"


def masks_meta_path(stem: str) -> Path:
    overrides = artifact_overrides.get()
    if "mask_metadata" not in overrides and "masks" in overrides:
        return overrides["masks"].with_suffix(".meta.npz")
    return artifact_path(stem, "mask_metadata")


def masks_cached(stem: str) -> bool:
    return masks_path(stem).exists() and masks_meta_path(stem).exists()


def _ffprobe_meta(path: str) -> dict | None:
    """Probe accurate avg_frame_rate, nb_frames, duration via ffprobe."""
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,nb_frames,duration",
            "-of", "json", path,
        ], stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    streams = json.loads(out).get("streams") or []
    if not streams:
        return None
    s = streams[0]
    fps = None
    fr = s.get("avg_frame_rate", "0/0")
    if "/" in fr:
        num, den = fr.split("/")
        if float(den):
            fps = float(num) / float(den)
    nb = s.get("nb_frames")
    nb_frames = int(nb) if isinstance(nb, str) and nb.isdigit() else None
    dur = s.get("duration")
    duration = float(dur) if isinstance(dur, str) else None
    return {"fps": fps, "nb_frames": nb_frames, "duration": duration}


def _video_meta(cap: cv2.VideoCapture, frames: int | None = None, path: str | None = None) -> dict:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if path is not None:
        probe = _ffprobe_meta(path)
        if probe is not None:
            if probe["fps"]:
                fps = probe["fps"]
            if probe["nb_frames"]:
                total = probe["nb_frames"]
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


# ── Mask stage ─────────────────────────────────────────────────────────────

def compute_masks(
    input_path: str,
    stem: str | None = None,
    frames: int | None = None,
    debug_writer: cv2.VideoWriter | None = None,
    frame_skip: int = 1,
    seed_mask: str | Path | None = None,
    stabilize: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Track hands on every frame with SAM2Matting; sample the output cache if requested."""
    if frame_skip < 1:
        raise ValueError("frame_skip must be positive")
    cap = cv2.VideoCapture(input_path)
    try:
        meta = _video_meta(cap, frames, path=input_path)
    finally:
        cap.release()
    total, h, w = meta["total"], meta["h"], meta["w"]
    if total < 1:
        raise ValueError("Input video has no frames")
    seed_path = Path(seed_mask) if seed_mask else seed_mask_path(stem or str(project_dir(input_path)))
    seed = read_seed(seed_path, h, w)
    meta.update(model=MASK_MODEL_ID, checkpoint=MASK_CHECKPOINT, encoding=MASK_ENCODING,
                seed_mask=str(seed_path), frame_skip=frame_skip,
                stabilization="motion_median_3" if stabilize else "none")
    n_max = -(-total // frame_skip)
    if stem is not None:
        out_path = _masks_tmp_path(stem)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        buf = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.uint8, shape=(n_max, h, w))
    else:
        buf = np.zeros((n_max, h, w), dtype=np.uint8)

    indices_list: list[int] = []
    current = np.zeros((h, w), dtype=np.uint8)
    debug_cap = cv2.VideoCapture(input_path) if debug_writer is not None else None
    seen = 0
    detected = 0
    try:
        stream = video_masks(input_path, total, h, w, seed)
        if stabilize:
            stream = stabilize_video_masks(input_path, stream)
        with closing(stream) as masks:
            for i, mask in masks:
                if i != seen:
                    raise ValueError(f"Expected mask for frame {seen}, received {i}")
                seen += 1
                if i % frame_skip == 0:
                    current = mask
                    buf[len(indices_list)] = current
                    indices_list.append(i)
                    detected += int(current.any())
                if debug_writer is not None:
                    ret, vis = debug_cap.read()
                    if not ret:
                        raise ValueError(f"Could not read debug frame {i}")
                    debug_writer.write(mask_overlay(vis, current))
                if seen % 30 == 0 or seen == total:
                    print(f"  masks: {seen}/{total} frames, {detected}/{len(indices_list)} cached masks had detections")
        if seen != total:
            raise ValueError(f"SAM2Matting returned {seen}/{total} frames; refusing incomplete cache")
    finally:
        if debug_cap is not None:
            debug_cap.release()
        if isinstance(buf, np.memmap):
            buf.flush()

    return buf, np.array(indices_list, dtype=np.int32), meta


def save_masks(stem: str, masks: np.ndarray, indices: np.ndarray, meta: dict) -> Path:
    npz_path = masks_path(stem)
    meta_path = masks_meta_path(stem)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(npz_path), masks=masks)
    np.savez_compressed(
        str(meta_path), indices=indices,
        **meta,
    )
    tmp = _masks_tmp_path(stem)
    if tmp.exists():
        tmp.unlink()
    print(f"  masks cached → {npz_path}")
    return npz_path


def load_masks(stem: str) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(masks_meta_path(stem)) as data:
        if "encoding" not in data or data["encoding"].item() != MASK_ENCODING:
            raise ValueError(f"Old binary mask cache for {stem}; regenerate with string-lights generate-masks {stem} --force")
        meta = {key: data[key].item() for key in data.files if key != "indices"}
        indices = data["indices"].copy()
    with np.load(masks_path(stem)) as data:
        masks = data["masks"]
    return masks, indices, meta


def masks_iter(
    masks: np.ndarray, indices: np.ndarray, total: int, *, valid_frames: int | None = None,
) -> Iterator[np.ndarray]:
    """Hold sampled masks within cache coverage; yield empty masks beyond it."""
    K = len(indices)
    cur = 0
    empty = np.zeros(masks.shape[1:], dtype=masks.dtype)
    for i in range(total):
        if not K or (valid_frames is not None and i >= valid_frames):
            yield empty
            continue
        while cur + 1 < K and indices[cur + 1] <= i:
            cur += 1
        yield masks[cur]


# ── Render stage ───────────────────────────────────────────────────────────

def render_video(
    input_path: str,
    output_path: str,
    projections: np.ndarray,
    strings: list[list[int]],
    meta: dict,
    masks: tuple[np.ndarray, np.ndarray] | None = None,
    fast: bool = False,
    include_audio: bool = True,
    truncate_audio: bool = False,
    mask_frames: int | None = None,
) -> None:
    w, h, fps, total = meta["w"], meta["h"], meta["fps"], meta["total"]
    cap = cv2.VideoCapture(input_path)
    frame_stream = _stream_frames(cap, total)
    if masks is not None:
        mask_stream = masks_iter(masks[0], masks[1], total, valid_frames=mask_frames)
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
        cmd += ["-i", input_path, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"]
        if truncate_audio:
            duration = total / fps
            cmd += ["-t", f"{duration:.6f}"]
    cmd += [*encode, output_path]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_chunks: list[bytes] = []
    stderr_thread = threading.Thread(
        target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True
    )
    stderr_thread.start()
    render_count = min(total, len(projections), len(strings))
    pbar = tqdm(total=render_count, desc="render", unit="f")
    try:
        for i, (frame, mask) in enumerate(zip(frame_stream, mask_stream)):
            if i >= render_count:
                break
            original = frame.copy() if mask.any() else None
            draw_projected_strings_frame(
                frame, i, projections[i], strings[i], last_active, fps
            )
            if original is not None:
                frame = blend_mask(original, frame, mask)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            pbar.update(1)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    finally:
        pbar.close()
    cap.release()
    rc = proc.wait()
    stderr_thread.join()
    stderr = b"".join(stderr_chunks).decode(errors="replace")
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed:\n{stderr}")


# ── Top-level orchestration ────────────────────────────────────────────────

def process_video(
    input_path: str,
    output_path: str,
    frames: int | None = None,
    disable_masking: bool = False,
    random_strings: bool = False,
    debug_masks: bool = False,
) -> None:
    stem = str(project_dir(input_path))

    cap = cv2.VideoCapture(input_path)
    video_meta = _video_meta(cap, frames, path=input_path)
    cap.release()

    def validate_cache(cache_meta: dict, name: str) -> None:
        if (cache_meta["w"], cache_meta["h"]) != (video_meta["w"], video_meta["h"]):
            raise ValueError(f"cached {name} dimensions do not match the input video")

    masks_data = None
    mask_frames = None
    if debug_masks or not disable_masking:
        if not masks_cached(stem):
            raise FileNotFoundError(f"No cached masks at {masks_path(stem)}. Generate them with: uv run string-lights generate-masks {input_path}")
        print(f"  using cached masks ← {masks_path(stem)}")
        masks_arr, indices, mask_meta = load_masks(stem)
        validate_cache(mask_meta, "masks")
        mask_frames = mask_meta["total"]
        if mask_frames < video_meta["total"]:
            print(f"Warning: mask cache covers {mask_frames}/{video_meta['total']} frames; using no mask for the remaining {video_meta['total'] - mask_frames} frames", file=sys.stderr)
        masks_data = (masks_arr, indices)

    if debug_masks:
        debug_out = output_path
        meta = video_meta
        total = meta["total"]
        debug_writer = cv2.VideoWriter(
            debug_out, cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["w"], meta["h"])
        )
        try:
            cap2 = cv2.VideoCapture(input_path)
            try:
                for frame, mask in zip(_stream_frames(cap2, total), masks_iter(masks_arr, indices, total, valid_frames=mask_frames)):
                    vis = frame.copy()
                    debug_writer.write(mask_overlay(vis, mask))
            finally:
                cap2.release()
        finally:
            debug_writer.release()
        print(f"Done.  Debug video → {debug_out}")
        return

    from .planar import load_planar_track
    from .planar_io import projection_path

    path = projection_path(stem)
    if not path.exists():
        raise FileNotFoundError(f"No cached projections at {path}. Generate them with: uv run string-lights generate-poses {input_path}")
    print(f"  using cached planar projections ← {path}")
    track, meta = load_planar_track(path)
    projections = track.projected_lines

    validate_cache(meta, "projections")
    if len(projections) < video_meta["total"]:
        valid = np.flatnonzero(np.isfinite(projections).all(axis=(1, 2, 3)))
        if not len(valid):
            raise ValueError("cached projections end early and contain no valid projection to hold; regenerate with generate-poses")
        missing = video_meta["total"] - len(projections)
        last_valid = int(valid[-1])
        print(f"Warning: projection cache covers {len(projections)}/{video_meta['total']} frames; holding the last valid projection (frame index {last_valid}) for the remaining {missing} frames", file=sys.stderr)
        projections = np.concatenate((projections, np.repeat(projections[last_valid:last_valid + 1], missing, axis=0)))
    meta = {**meta, "fps": video_meta["fps"], "total": video_meta["total"]}
    projections = projections[:meta["total"]]

    total, fps = meta["total"], meta["fps"]

    npy_path = artifact_path(input_path, "transcription")
    if "transcription" in artifact_overrides.get() and not npy_path.is_file() and not random_strings:
        raise FileNotFoundError(f"Tab data not found: {npy_path}")
    if random_strings or not npy_path.exists():
        if not random_strings:
            print(f"  no tab data at {npy_path}, using random strings")
        strings = get_random_strings(total, fps)
    else:
        strings = get_strings_to_highlight(input_path, total, fps, transcription=npy_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    render_video(
        input_path,
        output_path,
        projections,
        strings,
        meta,
        masks=masks_data,
        mask_frames=mask_frames,
        truncate_audio=frames is not None,
    )
    observed = int(np.isfinite(projections).all(axis=(1, 2, 3)).sum())
    print(
        f"Done.  Planar projection found in {observed}/{total} frames "
        f"({100 * observed // total}%)."
    )
