import numpy as np
import pytest

from string_lights import lyrics


def render(text, **style):
    frame = np.full((240, 400, 3), 150, dtype=np.uint8)
    lyrics.draw_lyrics_frame(frame, 30, 30, [dict(
        text=text, start=0, end=3, x=.5, y=.5, size=32,
        halo_intensity=0, **style,
    )])
    return frame


@pytest.mark.parametrize("separator", ["\n", "\\n", "\r\n", "\r"])
def test_multiline_text_has_two_centered_rows(separator):
    frame = render(f"Hello{separator}Hello")
    rows, columns = np.where(np.any(frame != 150, axis=2))
    groups = np.split(np.unique(rows), np.where(np.diff(np.unique(rows)) > 1)[0] + 1)
    assert len(groups) == 2
    assert abs((rows.min() + rows.max()) / 2 - 120) < 6
    assert abs((columns.min() + columns.max()) / 2 - 200) < 6
    np.testing.assert_array_equal(frame, render("Hello\nHello"))


def test_multiline_backdrop_covers_both_lines_and_blank_lines_are_preserved():
    plain = render("Hello\nHello")
    padded = render("Hello\nHello", background_opacity=.8, background_padding=10, background_softness=0)
    rows = np.where(np.any(plain != 150, axis=2))[0]
    assert np.all(padded[rows.min() - 4, 200] < 150)
    assert np.all(padded[rows.max() + 4, 200] < 150)
    spaced = render("Hello\n\nHello")
    spaced_rows = np.where(np.any(spaced != 150, axis=2))[0]
    assert np.ptp(spaced_rows) > np.ptp(rows)
    np.testing.assert_array_equal(render("\n\\n"), np.full_like(plain, 150))
