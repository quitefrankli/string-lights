from collections.abc import Iterator
from contextlib import closing

import cv2
import numpy as np


def temporal_median(current: np.ndarray, previous: np.ndarray, following: np.ndarray) -> np.ndarray:
    return np.median(np.stack((previous, current, following)), axis=0).astype(np.uint8)


def align_mask(current_frame: np.ndarray, neighbor_frame: np.ndarray,
               neighbor_mask: np.ndarray, current_mask: np.ndarray) -> np.ndarray:
    h, w = current_mask.shape
    scale = min(1.0, 768 / max(h, w))
    size = (max(16, round(w * scale)), max(16, round(h * scale)))
    current_gray = cv2.resize(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY), size)
    neighbor_gray = cv2.resize(cv2.cvtColor(neighbor_frame, cv2.COLOR_BGR2GRAY), size)
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow = estimator.calc(current_gray, neighbor_gray, None)
    flow = cv2.resize(flow, (w, h))
    flow[..., 0] *= w / size[0]
    flow[..., 1] *= h / size[1]
    y, x = np.mgrid[:h, :w].astype(np.float32)
    map_x, map_y = x + flow[..., 0], y + flow[..., 1]
    warped = cv2.remap(neighbor_mask, map_x, map_y, cv2.INTER_LINEAR)
    matched_frame = cv2.remap(neighbor_frame, map_x, map_y, cv2.INTER_LINEAR)
    difference = np.mean(np.abs(matched_frame.astype(np.float32) - current_frame), axis=2)
    valid = ((map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
             & (difference <= 25))
    # Where the appearance changed or flow leaves the frame, trust the current matte.
    return np.where(valid, warped, current_mask).astype(np.uint8)


def stabilize_video_masks(input_path: str, masks: Iterator[tuple[int, np.ndarray]]) -> Iterator[tuple[int, np.ndarray]]:
    """Remove isolated alpha changes using motion-aligned adjacent frames."""
    cap = cv2.VideoCapture(input_path)
    previous = None
    current = None
    try:
        with closing(masks):
            for index, mask in masks:
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Could not decode stabilization frame {index}")
                following = (index, mask, frame)
                if current is not None:
                    if previous is None:
                        yield current[0], current[1]
                    else:
                        before = align_mask(current[2], previous[2], previous[1], current[1])
                        after = align_mask(current[2], following[2], following[1], current[1])
                        yield current[0], temporal_median(current[1], before, after)
                previous, current = current, following
            if current is not None:
                yield current[0], current[1]
    finally:
        cap.release()
