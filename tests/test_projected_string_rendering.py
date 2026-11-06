import numpy as np

from string_lights.config import FADE_DURATION
from string_lights.strings import draw_projected_strings_frame


def _separated_lines() -> np.ndarray:
    return np.array(
        [
            [[35.0, 25.0], [220.0, 25.0]],
            [[40.0, 55.0], [215.0, 55.0]],
            [[45.0, 85.0], [210.0, 85.0]],
            [[50.0, 115.0], [205.0, 115.0]],
            [[55.0, 150.0], [190.0, 175.0]],
            [[60.0, 210.0], [195.0, 210.0]],
        ],
        dtype=np.float64,
    )


def test_projected_renderer_uses_the_active_strings_direct_endpoints() -> None:
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    lines = _separated_lines()

    draw_projected_strings_frame(frame, 0, lines, [4], {}, fps=30.0)

    bridge = tuple(np.rint(lines[4, 0]).astype(int))
    nut = tuple(np.rint(lines[4, 1]).astype(int))
    midpoint = tuple(np.rint(lines[4].mean(axis=0)).astype(int))
    assert frame[bridge[1], bridge[0]].max() > 0
    assert frame[nut[1], nut[0]].max() > 0
    assert frame[midpoint[1], midpoint[0]].max() > 0

    inactive_midpoint = tuple(np.rint(lines[0].mean(axis=0)).astype(int))
    assert not frame[inactive_midpoint[1], inactive_midpoint[0]].any()


def test_unresolved_projected_geometry_draws_nothing() -> None:
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    original = frame.copy()
    unresolved = np.full((6, 2, 2), np.nan, dtype=np.float64)

    draw_projected_strings_frame(frame, 12, unresolved, [0, 3], {}, fps=30.0)

    np.testing.assert_array_equal(frame, original)


def test_projected_renderer_preserves_active_fade_and_expiry_behavior() -> None:
    lines = _separated_lines()
    fps = 10.0
    last_active: dict[int, int] = {}

    active = np.zeros((240, 256, 3), dtype=np.uint8)
    draw_projected_strings_frame(active, 0, lines, [2], last_active, fps=fps)

    fading = np.zeros_like(active)
    draw_projected_strings_frame(fading, 1, lines, [], last_active, fps=fps)

    expired = np.zeros_like(active)
    first_expired_frame = int(fps * FADE_DURATION) + 1
    draw_projected_strings_frame(
        expired,
        first_expired_frame,
        lines,
        [],
        last_active,
        fps=fps,
    )

    assert last_active == {2: 0}
    assert np.count_nonzero(active) > 0
    assert 0 < int(fading.sum()) < int(active.sum())
    assert np.count_nonzero(expired) == 0


def test_hand_mask_and_core_blending_are_limited_to_effect_region(monkeypatch):
    from string_lights import strings
    from string_lights.masking import blend_mask
    rng = np.random.default_rng(8)
    original = rng.integers(0,256,(600,800,3),dtype=np.uint8)
    mask = rng.integers(0,256,(600,800),dtype=np.uint8)
    lines = np.array([[[300,260+s*5],[450,270+s*5]] for s in range(6)],float)
    unmasked = original.copy()
    strings.draw_projected_strings_frame(unmasked,2,lines,[2],{0:0},30.)
    expected = blend_mask(original,unmasked,mask)
    regions = []
    def blend(original_roi, effect_roi, mask_roi):
        regions.append(original_roi.shape)
        return blend_mask(original_roi,effect_roi,mask_roi)
    monkeypatch.setattr(strings,'blend_mask',blend)
    result = original.copy()
    strings.draw_projected_strings_frame(result,2,lines,[2],{0:0},30.,mask=mask)
    np.testing.assert_array_equal(result,expected)
    assert len(regions)==1
    assert regions[0][0]<600 and regions[0][1]<800
    np.testing.assert_array_equal(result[:150],original[:150])


def test_no_visible_strings_skips_hand_blending(monkeypatch):
    from string_lights import strings
    def unexpected(*args):raise AssertionError('Nothing to composite')
    monkeypatch.setattr(strings,'blend_mask',unexpected)
    frame=np.full((240,256,3),123,np.uint8)
    strings.draw_projected_strings_frame(frame,30,_separated_lines(),[],{},30.,mask=np.full((240,256),255,np.uint8))
    assert (frame==123).all()
