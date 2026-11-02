import json
import subprocess

import pytest

from string_lights.lyrics_editor import create_lyrics_app
from string_lights.editor import _browser_video


@pytest.mark.parametrize("video_codec", ["libx264", "libx265"])
def test_browser_preview_converts_lossless_audio_without_changing_master(tmp_path, monkeypatch, video_codec):
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / "data/input"
    inputs.mkdir(parents=True)
    source = inputs / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=160x90:r=10:d=0.3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
        "-c:v", video_codec, "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "alac", "-shortest", str(source),
    ], check=True, capture_output=True)
    original = source.read_bytes()
    client = create_lyrics_app().test_client()
    response = client.get("/video/clip")
    assert response.status_code == 200
    preview = tmp_path / "served-preview.mp4"
    preview.write_bytes(response.data)
    response.close()
    streams = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(preview),
    ]))["streams"]
    assert [s["codec_name"] for s in streams] == ["h264", "aac"]
    assert int(streams[0]["nb_frames"]) == 3
    assert source.read_bytes() == original
    cached = client.get("/video/clip")
    assert cached.data == preview.read_bytes()
    cached.close()
    uploads = inputs / "uploads"
    uploads.mkdir()
    uploaded = uploads / "clip.mp4"
    uploaded.write_bytes(original)
    assert _browser_video(uploaded) != _browser_video(source)
