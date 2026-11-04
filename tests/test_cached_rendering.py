import numpy as np
import pytest

from string_lights import pipeline


@pytest.mark.parametrize("projection_case", ["complete", "short", "invalid_tail", "no_valid"])
@pytest.mark.parametrize("mask_total", [3, 2])
def test_render_uses_only_caches(tmp_path, monkeypatch, capsys, projection_case, mask_total):
    from string_lights import planar, planar_io

    monkeypatch.chdir(tmp_path)
    meta = {"w": 16, "h": 12, "fps": 30.0, "total": 3}
    monkeypatch.setattr(pipeline, "_video_meta", lambda *a, **kw: meta)
    for name in ("compute_masks",):
        monkeypatch.setattr(pipeline, name, lambda *a, **kw: pytest.fail("Rendering must not compute caches"))
    monkeypatch.setattr(pipeline, "masks_cached", lambda stem: True)
    masks = np.full((mask_total, 12, 16), 128, dtype=np.uint8)
    monkeypatch.setattr(pipeline, "load_masks", lambda stem: (masks, np.arange(mask_total), {**meta, "total": mask_total}))
    path = planar_io.projection_path("clip")
    path.parent.mkdir(parents=True)
    path.touch()
    projection_total = 3 if projection_case == "complete" else 2
    lines = np.full((projection_total, 6, 2, 2), 7.0)
    if projection_case == "invalid_tail":
        lines[-1] = np.nan
    elif projection_case == "no_valid":
        lines[:] = np.nan
    class Track:
        projected_lines = lines
    monkeypatch.setattr(planar, "load_planar_track", lambda path: (Track(), {**meta, "total": projection_total}))
    received = {}
    monkeypatch.setattr(pipeline, "render_video", lambda *args, **kwargs: received.update(kwargs, projections=args[2]))

    if projection_case == "no_valid":
        with pytest.raises(ValueError, match="no valid projection"):
            pipeline.process_video("clip.mp4", "output.mp4", frames=3, random_strings=True)
        assert not received
        return

    pipeline.process_video("clip.mp4", "output.mp4", frames=3, random_strings=True)

    assert received["masks"][0] is masks
    rendered_masks = list(pipeline.masks_iter(*received["masks"], 3, valid_frames=received["mask_frames"]))
    assert all(mask.any() for mask in rendered_masks[:mask_total])
    assert all(not mask.any() for mask in rendered_masks[mask_total:])
    assert received["projections"] is not None
    warnings = capsys.readouterr().err
    assert ("Warning: mask cache" in warnings) == (mask_total < 3)
    assert ("Warning: projection cache" in warnings) == (projection_total < 3)
    np.testing.assert_array_equal(received["projections"][:projection_total], lines)
    if projection_total < 3:
        np.testing.assert_array_equal(received["projections"][projection_total:], np.full((1, 6, 2, 2), 7.0))


def test_sparse_masks_hold_last_sample_only_within_cache_coverage():
    masks = np.array([[[64]], [[128]]], dtype=np.uint8)
    result = list(pipeline.masks_iter(masks, np.array([0, 2]), 6, valid_frames=4))
    assert [mask.item() for mask in result] == [64, 64, 128, 128, 0, 0]


def test_missing_mask_cache_never_runs_sam(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "_video_meta", lambda *a, **kw: {"w": 16, "h": 12, "fps": 30., "total": 3})
    monkeypatch.setattr(pipeline, "compute_masks", lambda *a, **kw: pytest.fail("Must not run SAM"))
    with pytest.raises(FileNotFoundError, match="string-lights generate-masks"):
        pipeline.process_video("clip.mp4", "output.mp4")
