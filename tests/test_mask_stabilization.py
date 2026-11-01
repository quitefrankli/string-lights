import cv2
import numpy as np

from string_lights.mask_stabilization import stabilize_video_masks, temporal_median


def test_temporal_median_removes_flash_but_preserves_real_gap_and_hand():
    # Columns: stable hand, stable gap, one-frame false foreground, newly opened gap.
    previous = np.array([[255, 0, 0, 255]], dtype=np.uint8)
    current = np.array([[255, 0, 255, 0]], dtype=np.uint8)
    following = np.array([[255, 0, 0, 0]], dtype=np.uint8)
    assert temporal_median(current, previous, following).tolist() == [[255, 0, 0, 0]]


def test_stabilization_keeps_frame_order_and_removes_opacity_flash(tmp_path):
    video = str(tmp_path / "static.avi")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 64))
    frame = np.random.default_rng(1).integers(20, 220, (64, 64, 3), dtype=np.uint8)
    for _ in range(5):
        writer.write(frame)
    writer.release()
    masks = np.zeros((5, 64, 64), dtype=np.uint8)
    masks[:, 10:30, 10:30] = 255
    masks[2, 18:22, 18:22] = 0

    def stream():
        for i, mask in enumerate(masks):
            yield i, mask

    result = list(stabilize_video_masks(video, stream()))
    assert [i for i, _ in result] == list(range(5))
    assert all(mask[20, 20] == 255 for _, mask in result)
    assert all(mask[40, 40] == 0 for _, mask in result)
