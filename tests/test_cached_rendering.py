import numpy as np
import pytest

from string_lights import pipeline


@pytest.mark.parametrize("tracker", ["aruco", "planar"])
def test_render_uses_only_caches(tmp_path, monkeypatch, tracker):
    from string_lights import planar, planar_io

    monkeypatch.chdir(tmp_path)
    meta = {"w": 16, "h": 12, "fps": 30.0, "total": 3}
    monkeypatch.setattr(pipeline, "_video_meta", lambda *a, **kw: meta)
    monkeypatch.setattr(pipeline, "SMOOTH_POSES", False)
    for name in ("compute_masks", "compute_poses"):
        monkeypatch.setattr(pipeline, name, lambda *a, **kw: pytest.fail("Rendering must not compute caches"))
    monkeypatch.setattr(pipeline, "masks_cached", lambda stem: True)
    masks = np.zeros((3, 12, 16), dtype=np.uint8)
    monkeypatch.setattr(pipeline, "load_masks", lambda stem: (masks, np.arange(3), meta))
    path = pipeline.poses_path("clip") if tracker == "aruco" else planar_io.projection_path("clip")
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(pipeline, "load_poses", lambda stem: ([(None, None)] * 3, meta))
    class Track:
        projected_lines = np.zeros((3, 6, 2, 2))
    monkeypatch.setattr(planar, "load_planar_track", lambda path: (Track(), meta))
    received = {}
    monkeypatch.setattr(pipeline, "render_video", lambda *args, **kwargs: received.update(kwargs))

    pipeline.process_video("clip.mp4", "output.mp4", frames=3, random_strings=True, pose_tracker=tracker)

    assert received["masks"][0] is masks
    assert (received["projections"] is not None) == (tracker == "planar")


def test_missing_mask_cache_never_runs_sam(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "_video_meta", lambda *a, **kw: {"w": 16, "h": 12, "fps": 30., "total": 3})
    monkeypatch.setattr(pipeline, "compute_masks", lambda *a, **kw: pytest.fail("Must not run SAM"))
    with pytest.raises(FileNotFoundError, match="string-lights masks"):
        pipeline.process_video("clip.mp4", "output.mp4")
