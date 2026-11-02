import numpy as np
import pytest
from PIL import Image, ImageDraw

from string_lights import lyrics, lyrics_editor


def lyric(**kwargs):
    return dict(text="Readable lyrics", start=0, end=3, x=.5, y=.5, size=32,
                fade=0, halo_intensity=0, background_opacity=.8,
                background_padding=16, background_softness=24, **kwargs)


def test_background_has_dark_center_and_smooth_transparent_edges():
    frame = np.full((240, 640, 3), 255, dtype=np.uint8)
    item = lyric()
    lyrics.draw_lyrics_frame(frame, 1, 30, [item])
    bounds = ImageDraw.Draw(Image.new("RGB", (640, 240))).textbbox(
        (320, 120), item["text"], font=lyrics._font(32), anchor="mm")
    y = bounds[3] + 6  # Below the glyphs, within backdrop padding.
    left = bounds[0] - 16 - 24
    ramp = frame[y, left:left + 25, 0].astype(int)
    assert ramp[0] >= 250
    assert ramp[-1] <= 55
    assert np.all(np.diff(ramp) <= 0)
    assert len(np.unique(ramp)) > 15
    assert np.all(frame[0] == 255)


def test_background_can_be_disabled_and_fades_with_text():
    original = np.full((240, 640, 3), 255, dtype=np.uint8)
    disabled, enabled, fading = (original.copy() for _ in range(3))
    item = lyric()
    lyrics.draw_lyrics_frame(disabled, 1, 30, [{**item, "background_opacity": 0}])
    lyrics.draw_lyrics_frame(enabled, 1, 30, [item])
    lyrics.draw_lyrics_frame(fading, 0, 30, [{**item, "fade": .5}])
    assert np.all(disabled == original)
    assert np.any(enabled < disabled)
    np.testing.assert_array_equal(fading, original)


@pytest.mark.parametrize("x", [0, 1])
def test_background_clips_at_canvas_edges(x):
    frame = np.full((100, 160, 3), 255, dtype=np.uint8)
    lyrics.draw_lyrics_frame(frame, 1, 30, [{**lyric(), "x": x, "y": 0}])
    assert frame.shape == (100, 160, 3)
    assert frame.min() < 255


def test_background_settings_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = lyrics_editor.create_lyrics_app().test_client()
    values = [lyric()]
    assert client.post("/lyrics/clip", json=values).status_code == 200
    assert client.get("/lyrics/clip").json == values


@pytest.mark.parametrize("field,value", [
    ("background_opacity", 1.1), ("background_opacity", -1),
    ("background_padding", 121), ("background_softness", -1),
    ("background_softness", "24"), ("background_opacity", True),
])
def test_invalid_background_settings_rejected(tmp_path, monkeypatch, field, value):
    monkeypatch.chdir(tmp_path)
    client = lyrics_editor.create_lyrics_app().test_client()
    response = client.post("/lyrics/clip", json=[{**lyric(), field: value}])
    assert response.status_code == 400
    assert "background" in response.json["error"]
