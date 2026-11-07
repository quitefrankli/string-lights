import numpy as np
import pytest

from string_lights import lyrics, lyrics_editor


def test_font_round_trip_and_browser_font_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = lyrics_editor.create_lyrics_app().test_client()
    for name in lyrics.FONT_OPTIONS:
        values = [dict(text="Hello", start=0, end=3, font=name)]
        assert client.post("/lyrics/clip", json=values).status_code == 200
        assert client.get("/lyrics/clip").json == values
        path = lyrics.font_path(name)
        if path:
            assert client.get(f"/fonts/{name}").data == path.read_bytes()
    page = client.get("/").get_data(as_text=True)
    assert 'id="lyric-font"' in page
    assert "Lyric-serif" in page
    assert client.get("/fonts/unknown").status_code == 404


@pytest.mark.parametrize("font", ["unknown", "../../etc/passwd", [], None])
def test_invalid_fonts_rejected(tmp_path, monkeypatch, font):
    monkeypatch.chdir(tmp_path)
    client = lyrics_editor.create_lyrics_app().test_client()
    response = client.post("/lyrics/clip", json=[dict(text="Hello", start=0, end=3, font=font)])
    assert response.status_code == 400
    assert "font" in response.json["error"]


def test_font_changes_export_and_invalidates_cached_overlay():
    if not all(lyrics.Path(lyrics.FONT_OPTIONS[name][1][0]).is_file() for name in ("sans", "serif")):
        pytest.skip("Test fonts are not installed")
    lyrics._lyric_overlay.cache_clear()
    frames = []
    for name in ("sans", "serif", "sans"):
        frame = np.full((120, 400, 3), 150, dtype=np.uint8)
        lyrics.draw_lyrics_frame(frame, 30, 30, [dict(
            text="Different fonts", start=0, end=3, x=.5, y=.5,
            size=32, font=name, halo_size=0, halo_intensity=0,
            background_opacity=0, background_padding=0, background_softness=0,
        )])
        frames.append(frame)
    assert np.any(frames[0] != frames[1])
    np.testing.assert_array_equal(frames[0], frames[2])
    assert lyrics._lyric_overlay.cache_info().misses == 2
