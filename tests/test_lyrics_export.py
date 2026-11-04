import json
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from string_lights import lyrics_editor


@pytest.mark.parametrize("dimensions", [(1548, 1080), (1920, 1080), (720, 1920)])
def test_original_export_keeps_lyrics_anchored_to_video(dimensions):
    w, h = dimensions
    left, top, width, height = lyrics_editor.reels_video_bounds(w, h)
    # Position the lyric 62% across and 82% down the visible video in the
    # padded editing canvas. Removing padding must preserve those fractions.
    saved = {"text": "Example", "x": (left + .62 * width) / 1080,
             "y": (top + .82 * height) / 1920, "size": 48,
             "background_padding": 16, "background_softness": 24}
    converted = lyrics_editor.original_layout_lyrics([saved], w, h)[0]
    assert converted["x"] == pytest.approx(.62)
    assert converted["y"] == pytest.approx(.82)
    for key in ("size", "background_padding", "background_softness"):
        assert converted[key] / w == pytest.approx(saved[key] / width)
    assert saved["size"] == 48  # Conversion must not rewrite saved settings.


def test_lyrics_in_padding_are_not_moved_into_original_video():
    saved = {"text": "Below video", "x": .5, "y": .95}
    converted = lyrics_editor.original_layout_lyrics([saved], 1920, 1080)[0]
    assert converted["y"] > 1


@pytest.mark.parametrize("aspect, dimensions", [
    ("original", (160, 90)), ("padded", (1080, 1920)),
])
@pytest.mark.parametrize("export_format, audio_codec", [("compatible", "aac"), ("source_audio", "alac")])
def test_export_layout_and_audio(tmp_path, monkeypatch, aspect, dimensions, export_format, audio_codec):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lyrics_editor, "_render", {"state": "running"})
    inputs = tmp_path / "data/projects/clip"
    inputs.mkdir(parents=True)
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "color=c=red:s=160x90:r=10:d=0.3", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=0.3", "-c:v", "libx264", "-c:a", "alac",
        "-shortest", str(inputs / "source.mp4"),
    ], check=True)

    lyrics_editor._do_render("clip", aspect, export_format)

    assert lyrics_editor._render["state"] == "done", lyrics_editor._render
    output = "data/projects/clip/renders/strings-lyrics.mp4"
    streams = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", output,
    ]))["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    assert (video["width"], video["height"]) == dimensions
    assert int(video["nb_frames"]) == 3
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] == "yuv420p"
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")
    assert audio["codec_name"] == audio_codec
    assert float(audio["start_time"]) == 0
    assert abs(float(audio["duration"]) - .3) < .025
    cap = cv2.VideoCapture(output)
    try:
        ok, frame = cap.read()
        assert ok
        assert frame[dimensions[1] // 2, dimensions[0] // 2, 2] > 200
        if aspect == "padded":
            assert np.max(frame[0]) < 10
        else:
            assert frame[0, 0, 2] > 200
    finally:
        cap.release()


@pytest.mark.parametrize("payload, expected", [
    ({"aspect": "original"}, "original"),
    ({"aspect": "padded"}, "padded"),
    ({}, "padded"),
    ({"aspect": "invalid"}, None),
    (["original"], None),
])
def test_export_request_passes_layout_to_renderer(monkeypatch, payload, expected):
    monkeypatch.setattr(lyrics_editor, "_render", {"state": "idle"})
    calls = []
    monkeypatch.setattr(lyrics_editor.threading, "Thread", lambda **kwargs:
                        SimpleNamespace(start=lambda: calls.append(kwargs)))
    response = lyrics_editor.create_lyrics_app().test_client().post("/render/clip", json=payload)
    assert response.status_code == (200 if expected else 400)
    if expected:
        assert calls[0]["args"] == ("clip", expected, "source_audio")
    else:
        assert not calls


@pytest.mark.parametrize("export_format", ["compatible", "source_audio", "invalid", False])
def test_export_request_validates_format(monkeypatch, export_format):
    monkeypatch.setattr(lyrics_editor, "_render", {"state": "idle"})
    calls = []
    monkeypatch.setattr(lyrics_editor.threading, "Thread", lambda **kwargs:
                        SimpleNamespace(start=lambda: calls.append(kwargs)))
    response = lyrics_editor.create_lyrics_app().test_client().post(
        "/render/clip", json={"aspect": "original", "format": export_format})
    valid = export_format in ("compatible", "source_audio")
    assert response.status_code == (200 if valid else 400)
    if valid:
        assert calls[0]["args"] == ("clip", "original", export_format)
    else:
        assert not calls
