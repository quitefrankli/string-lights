import numpy as np

from string_lights.editor import create_app
from string_lights.planar import (
    PlanarFrame, PlanarTrack, TrackDiagnostics, TrackState, save_planar_track,
)


def test_editor_planar_overlay_without_aruco_poses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / 'data/input'
    inputs.mkdir(parents=True)
    (inputs / 'clip.mp4').touch()
    lines = np.arange(24, dtype=float).reshape(6, 2, 2) + 0.5
    diagnostics = TrackDiagnostics(30, 25, 0.8, 0.5, 0.75)
    track = PlanarTrack((
        PlanarFrame(np.eye(3), lines, TrackState.OBSERVED, diagnostics, 0),
        PlanarFrame(None, np.full((6, 2, 2), np.nan), TrackState.UNRESOLVED, diagnostics),
    ), ('a' * 64,))
    save_planar_track('data/components/projections/clip.npz', track,
                      {'w': 320, 'h': 240, 'fps': 30.0, 'total': 2})
    client = create_app().test_client()
    response = client.get('/strings/clip')
    assert response.status_code == 200
    assert response.json == {
        'w': 320, 'h': 240, 'fps': 30.0, 'total': 2,
        'lines': [lines.reshape(6, 4).astype(int).tolist(), None],
    }
    item = client.get('/list').json['items'][0]
    assert item['has_projections'] is True
    assert item['has_poses'] is False


def test_editor_aruco_overlay_still_supported(tmp_path, monkeypatch):
    from string_lights.pipeline import save_poses

    monkeypatch.chdir(tmp_path)
    save_poses('clip', [
        (np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]])),
        (None, None),
    ], {'w': 320, 'h': 240, 'fps': 30.0, 'total': 2})
    response = create_app().test_client().get('/strings/clip')
    assert response.status_code == 200
    assert len(response.json['lines'][0]) == 6
    assert all(len(line) == 4 for line in response.json['lines'][0])
    assert response.json['lines'][1] is None


def test_editor_missing_tracking_explains_both_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    response = create_app().test_client().get('/strings/missing')
    assert response.status_code == 400
    assert 'poses missing' in response.text
    assert 'planar-track missing' in response.text


def test_browser_video_converts_and_reuses_preview(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from string_lights.editor import _browser_video

    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'clip.mp4'
    source.write_bytes(b'hevc source')
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == 'ffprobe':
            return SimpleNamespace(stdout=json.dumps({'streams': [
                {'codec_name': 'hevc', 'pix_fmt': 'yuv420p10le'},
            ]}))
        from pathlib import Path
        Path(command[-1]).write_bytes(b'h264 preview')

    monkeypatch.setattr('string_lights.editor.subprocess.run', run)
    preview = _browser_video(source)
    assert preview.read_bytes() == b'h264 preview'
    assert source.read_bytes() == b'hevc source'
    assert _browser_video(source) == preview
    assert len(calls) == 2


def test_export_saves_and_requires_confirmation_before_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = create_app().test_client()
    payload = {'filename': 'clip', 'frames': [[0] * 6, [1] * 6]}
    response = client.post('/export', json=payload)
    assert response.status_code == 200
    target = tmp_path / 'data/input/clip.npy'
    np.testing.assert_array_equal(np.load(target), payload['frames'])
    original = target.read_bytes()
    payload['frames'] = [[2] * 6]
    response = client.post('/export', json=payload)
    assert response.status_code == 409
    assert response.json['exists'] is True
    assert target.read_bytes() == original
    response = client.post('/export', json={**payload, 'overwrite': True})
    assert response.status_code == 200
    np.testing.assert_array_equal(np.load(target), payload['frames'])


def test_export_rejects_path_traversal_and_invalid_frames(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = create_app().test_client()
    for name in ['../escape', '/tmp/escape', 'a/b', 'a\\b', '', '..']:
        assert client.post('/export', json={'filename': name, 'frames': [[0] * 6]}).status_code == 400
    assert client.post('/export', json={'filename': 'clip', 'frames': [[1]]}).status_code == 400
    assert not (tmp_path / 'data/input/clip.npy').exists()
