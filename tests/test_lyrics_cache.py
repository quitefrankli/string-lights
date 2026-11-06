import numpy as np

from string_lights import lyrics


def item(**overrides):
    return dict(dict(text="Hello 日本語", start=0, end=3, x=.5, y=.5,
                     size=32, fade=0, halo_intensity=0,
                     background_opacity=.25), **overrides)


def render(values, tick=30, shape=(240, 420, 3)):
    frame = np.full(shape, 150, dtype=np.uint8)
    lyrics.draw_lyrics_frame(frame, tick, 30, values)
    return frame


def test_static_overlay_reuses_text_and_skips_invisible_blur(monkeypatch):
    lyrics._lyric_overlay.cache_clear()

    def unexpected(*args, **kwargs):
        raise AssertionError("An invisible halo must not be blurred")

    monkeypatch.setattr(lyrics.ImageFilter.GaussianBlur, "filter", unexpected)
    first = render([item()])
    assert np.any(first != 150)

    def no_redraw(*args, **kwargs):
        raise AssertionError("A cached lyric must not be redrawn")

    monkeypatch.setattr(lyrics.ImageDraw.ImageDraw, "text", no_redraw)
    np.testing.assert_array_equal(render([item()], tick=31), first)


def test_style_edits_and_canvas_dimensions_invalidate_overlay():
    lyrics._lyric_overlay.cache_clear()
    first = render([item()])
    assert np.any(render([item(text="Changed")]) != first)
    assert np.any(render([item(color=[255, 0, 0])]) != first)
    assert np.any(render([item(x=.7)]) != first)
    assert np.any(render([item(background_opacity=.8)]) != first)
    enlarged = render([item()], shape=(300, 500, 3))
    assert enlarged.shape == (300, 500, 3)
    assert lyrics._lyric_overlay.cache_info().misses == 6


def test_fades_do_not_mutate_cached_text_or_background():
    lyrics._lyric_overlay.cache_clear()
    values = [item(fade=.5, halo_intensity=.75)]
    full = render(values)
    fading = render(values, tick=3)
    assert np.any(fading != full)
    np.testing.assert_array_equal(render(values), full)
    np.testing.assert_array_equal(render(values, tick=87), fading)
    np.testing.assert_array_equal(render(values, tick=0), np.full_like(full, 150))
    assert lyrics._lyric_overlay.cache_info().misses == 1


def test_overlapping_lyrics_compose_in_input_order():
    first = item(background_opacity=.8)
    second = item(text="Second line", color=[255, 0, 0])
    combined = render([first, second])
    sequential = render([first])
    lyrics.draw_lyrics_frame(sequential, 30, 30, [second])
    np.testing.assert_array_equal(combined, sequential)
