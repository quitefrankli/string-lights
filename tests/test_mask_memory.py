import numpy as np
from PIL import Image
import pytest
import torch


def test_jpeg_loader_decodes_on_demand_with_bounded_cache(tmp_path, monkeypatch):
    from sam2.utils import misc
    for i in range(40):
        Image.fromarray(np.full((12, 18, 3), i * 5, dtype=np.uint8)).save(tmp_path / f"{i:05d}.jpg")
    calls = []
    original = misc._load_img_as_tensor

    def decode(path, size):
        calls.append(path)
        return original(path, size)

    monkeypatch.setattr(misc, "_load_img_as_tensor", decode)
    images, h, w = misc.load_video_frames_from_jpg_images(str(tmp_path), 32, True)
    assert calls == [], "Initialization must not decode the full clip"
    assert (len(images), h, w) == (40, 12, 18)
    first = images[0].clone()
    decoded, _, _ = original(str(tmp_path / "00000.jpg"), 32)
    expected = decoded.float()
    expected -= torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    expected /= torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    assert images[0].dtype == torch.float32
    assert len(calls) == 1
    for i in range(40):
        images[i]
        assert len(images.cache) <= 2
    torch.testing.assert_close(images[0], first)
    with pytest.raises(IndexError):
        images[40]


def test_forward_state_pruning_preserves_memory_horizon_and_seed():
    from string_lights.masking import prune_tracking_state
    from types import SimpleNamespace
    seed = {"alpha": "seed alpha"}
    outputs = {i: {"alpha": "full resolution", "obj_ptr": i, "maskmem_features": i} for i in range(1, 100)}
    state = {"output_dict_per_obj": {0: {"cond_frame_outputs": {0: seed}, "non_cond_frame_outputs": outputs}}}
    predictor = SimpleNamespace(num_maskmem=7, memory_temporal_stride_for_eval=1, max_obj_ptrs_in_encoder=16)
    prune_tracking_state(predictor, state, 99)
    assert sorted(outputs) == list(range(84, 100))
    assert state["output_dict_per_obj"][0]["cond_frame_outputs"][0] is seed
    assert seed["alpha"] == "seed alpha"
    assert all("alpha" not in output for output in outputs.values())
    assert outputs[99]["obj_ptr"] == 99
