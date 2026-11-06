from contextlib import nullcontext

import numpy as np
import pytest

from string_lights import masking, pipeline


def test_matte_encoding_and_soft_compositing():
    alpha = masking.torch.tensor([[[0.0, 0.5, 1.0], [-0.1, 1.2, 0.25]]])
    mask = masking.matte_to_uint8(alpha, 2, 3)
    assert mask.tolist() == [[0, 128, 255], [0, 255, 64]]
    original = np.full((2, 3, 3), 200, dtype=np.uint8)
    effect = np.zeros_like(original)
    result = masking.blend_mask(original, effect, mask)
    assert result[0, :, 0].tolist() == [0, 100, 200]
    assert result[1, 2, 0] == 50


def test_seed_requires_foreground_and_correct_dimensions(tmp_path):
    path = tmp_path / "seed.png"
    masking.cv2.imwrite(str(path), np.zeros((4, 6), dtype=np.uint8))
    with pytest.raises(ValueError, match="no foreground"):
        masking.read_seed(path, 4, 6)
    with pytest.raises(ValueError, match="size"):
        masking.read_seed(path, 6, 4)


def test_rejects_binary_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = pipeline.masks_meta_path("old")
    path.parent.mkdir(parents=True)
    np.savez(path, w=6, h=4, fps=30, total=5, indices=np.arange(5))
    with pytest.raises(ValueError, match="Old binary mask cache"):
        pipeline.load_masks("old")


@pytest.mark.parametrize("returned", [5, 3])
def test_mask_cache_sampling_and_incomplete_video(tmp_path, monkeypatch, returned):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "_video_meta", lambda *a, **kw: dict(total=5, h=4, w=6, fps=30))

    monkeypatch.setattr(pipeline, "read_seed", lambda *a: np.ones((4, 6), dtype=np.uint8))

    def stream(*args):
        for i in range(returned):
            yield i, np.full((4, 6), i % 2, dtype=np.uint8)

    monkeypatch.setattr(pipeline, "video_masks", stream)
    if returned < 5:
        with pytest.raises(ValueError, match="refusing incomplete cache"):
            pipeline.compute_masks("missing.mp4", frames=5, frame_skip=2, stabilize=False)
        return
    masks, indices, meta = pipeline.compute_masks("missing.mp4", stem="clip", frames=5, frame_skip=2, stabilize=False)
    assert masks.shape == (3, 4, 6)
    assert indices.tolist() == [0, 2, 4]
    assert not masks.any()
    pipeline.save_masks("clip", masks, indices, meta)
    cached, cached_indices, cached_meta = pipeline.load_masks("clip")
    np.testing.assert_array_equal(np.stack(list(cached.frames())), masks)
    np.testing.assert_array_equal(cached_indices, indices)
    assert cached_meta["total"] == 5
    with np.load(pipeline.masks_meta_path("clip")) as metadata:
        assert metadata["model"].item() == "FudanCVL/SAM2Matting"
        assert metadata["frame_skip"].item() == 2
