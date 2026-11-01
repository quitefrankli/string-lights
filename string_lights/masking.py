from collections.abc import Iterator
from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import MASK_CHECKPOINT, MASK_CONFIG, MASK_MODEL_ID


def seed_mask_path(stem: str) -> Path:
    return Path("data/components/mask_seeds") / f"{stem}.png"


def read_seed(path: str | Path, height: int, width: int) -> np.ndarray:
    seed = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if seed is None:
        raise ValueError(f"Hand seed mask not found: {path}. Supply --seed-mask with a first-frame grayscale PNG (white hands, black background).")
    if seed.shape != (height, width) or seed.dtype != np.uint8:
        raise ValueError(f"Seed mask must be an 8-bit grayscale image of size {width}x{height}")
    seed = (seed > 127).astype(np.uint8)
    if not seed.any():
        raise ValueError("Seed mask contains no foreground hands")
    return seed


def load_predictor():
    if not torch.cuda.is_available():
        raise RuntimeError("SAM2Matting mask generation requires a CUDA GPU.")
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2matting_video_predictor

    print("  loading SAM2Matting Base+...")
    checkpoint = hf_hub_download(MASK_MODEL_ID, MASK_CHECKPOINT)
    predictor = build_sam2matting_video_predictor(MASK_CONFIG, checkpoint, device="cuda")
    # The optional CUDA connected-components extension is not needed for matting.
    predictor.fill_hole_area = 0
    return predictor


def matte_to_uint8(alpha: torch.Tensor, height: int, width: int) -> np.ndarray:
    matte = alpha.detach().float().cpu().squeeze().numpy()
    if matte.shape != (height, width) or not np.isfinite(matte).all():
        raise ValueError(f"Invalid SAM2Matting alpha output: {matte.shape}")
    return np.rint(matte.clip(0, 1) * 255).astype(np.uint8)


def blend_mask(original: np.ndarray, foreground: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Composite the original hand over the effect using uint8 opacity."""
    weight = mask.astype(np.float32)[..., None] / 255
    return np.rint(original * weight + foreground * (1 - weight)).clip(0, 255).astype(np.uint8)


def mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.empty_like(frame)
    color[:] = (0, 0, 200)
    return blend_mask(color, frame, np.rint(mask * 0.4).astype(np.uint8))


def video_masks(input_path: str, total: int, height: int, width: int,
                seed: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    predictor = load_predictor()
    state = None
    try:
        with tempfile.TemporaryDirectory(prefix="sam2matting-") as frame_dir:
            cap = cv2.VideoCapture(input_path)
            try:
                for i in range(total):
                    ret, frame = cap.read()
                    if not ret:
                        raise ValueError(f"Could not decode frame {i}")
                    target = str(Path(frame_dir) / f"{i:05d}.jpg")
                    if not cv2.imwrite(target, frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
                        raise OSError(f"Could not write temporary frame {i}")
            finally:
                cap.release()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(video_path=frame_dir, offload_video_to_cpu=True,
                                             offload_state_to_cpu=True)
                mask = torch.from_numpy(seed.astype(np.float32) * 20 - 10)[None, None]
                mask = F.interpolate(mask, size=(256, 256), mode="bilinear", align_corners=False)
                predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask.cuda())
                stream = predictor.propagate_in_video(state, start_frame_idx=0,
                                                     max_frame_num_to_track=total - 1)
                try:
                    for index, _, _, alpha, _ in stream:
                        if not 0 <= index < total:
                            raise ValueError(f"SAM2Matting returned frame {index} outside requested range")
                        yield index, matte_to_uint8(alpha, height, width)
                finally:
                    stream.close()
    finally:
        if state is not None:
            predictor.reset_state(state)
        del state, predictor
        torch.cuda.empty_cache()
