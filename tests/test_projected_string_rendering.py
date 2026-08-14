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
