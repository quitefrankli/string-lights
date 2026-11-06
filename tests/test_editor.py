import numpy as np

from string_lights.editor import create_app
from string_lights.planar import (
    PlanarFrame, PlanarTrack, TrackDiagnostics, TrackState, save_planar_track,
)


def test_editor_planar_overlay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / 'data/projects/clip'
    inputs.mkdir(parents=True)
    (inputs / 'source.mp4').touch()
    lines = np.arange(24, dtype=float).reshape(6, 2, 2) + 0.5
    diagnostics = TrackDiagnostics(30, 25, 0.8, 0.5, 0.75)
    track = PlanarTrack((
        PlanarFrame(np.eye(3), lines, TrackState.OBSERVED, diagnostics, 0),
        PlanarFrame(None, np.full((6, 2, 2), np.nan), TrackState.UNRESOLVED, diagnostics),
    ), ('a' * 64,))
    save_planar_track('data/projects/clip/tracking/projections.npz', track,
                      {'w': 320, 'h': 240, 'fps': 30.0, 'total': 2})
    client = create_app().test_client()
    response = client.get('/strings/clip')
    assert response.status_code == 200
    assert response.json == {
        'w': 320, 'h': 240, 'fps': 30.0, 'total': 2,
        'lines': [lines.reshape(6, 4).astype(int).tolist(), None],
    }
    assert create_app('clip').test_client().get('/').status_code == 200


def test_editor_missing_tracking_explains_planar_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    response = create_app().test_client().get('/strings/missing')
    assert response.status_code == 400
    assert 'generate-poses missing' in response.text


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
    target = tmp_path / 'data/projects/clip/transcription.npy'
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
    assert not (tmp_path / 'data/projects/clip/transcription.npy').exists()


def test_editor_loads_transcription_with_different_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / 'data/projects/recording'
    inputs.mkdir(parents=True)
    (inputs / 'source.mp4').touch()
    other = tmp_path / 'data/projects/different take'
    other.mkdir()
    frames = np.zeros((44, 6), dtype=np.int64)
    frames[6:8, 2] = 1
    np.save(other / 'transcription.npy', frames)
    np.save(inputs / 'transcription.npy', np.zeros((44, 6), dtype=np.int64))
    client = create_app().test_client()
    response = client.get('/load/different%20take')
    assert response.status_code == 200
    assert response.json['frames'] == frames.tolist()


def test_editor_rejects_invalid_transcription_without_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from string_lights.paths import artifact_path
    for stem in ['wrong-shape', 'nonfinite', 'broken', 'empty']:
        artifact_path(stem, 'transcription').parent.mkdir(parents=True)
    np.save(artifact_path('wrong-shape', 'transcription'), np.zeros((10, 3)))
    np.save(artifact_path('nonfinite', 'transcription'), np.full((10, 6), np.nan))
    artifact_path('broken', 'transcription').write_bytes(b'not a numpy file')
    artifact_path('empty', 'transcription').touch()
    client = create_app().test_client()
    for stem in ['wrong-shape', 'nonfinite', 'broken', 'empty']:
        response = client.get('/load/' + stem)
        assert response.status_code == 400
        assert 'six string values' in response.json['error']
    assert client.get('/load/missing').status_code == 404


def test_project_editor_scopes_listing_loading_and_saving(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ('cupid', 'gunjou'):
        directory = tmp_path / 'data/projects' / name
        directory.mkdir(parents=True)
        (directory / 'source.mp4').touch()
        np.save(directory / 'transcription.npy', np.zeros((2, 6), dtype=np.int64))
    client = create_app('cupid').test_client()
    assert client.get('/list').status_code == 404
    assert client.get('/load/cupid').status_code == 200
    assert client.get('/load/gunjou').status_code == 404
    assert client.get('/video/gunjou').status_code == 404
    assert client.get('/strings/gunjou').status_code == 404
    assert client.post('/export', json={'filename': 'gunjou', 'frames': [[1] * 6],
                                       'overwrite': True}).status_code == 400
    response = client.post('/export', json={'filename': 'cupid', 'frames': [[1] * 6],
                                           'overwrite': True})
    assert response.status_code == 200
    np.testing.assert_array_equal(np.load('data/projects/cupid/transcription.npy'), [[1] * 6])
    np.testing.assert_array_equal(np.load('data/projects/gunjou/transcription.npy'), np.zeros((2, 6)))
    from string_lights import paths
    assert paths.project_override.get() is None


def test_tabs_cli_transcription_override_survives_request_context(tmp_path, monkeypatch):
    from string_lights import paths

    monkeypatch.chdir(tmp_path)
    directory = tmp_path / 'data/projects/clip'
    directory.mkdir(parents=True)
    (directory / 'source.mp4').touch()
    custom = tmp_path / 'custom.npy'
    np.save(custom, [[3] * 6])
    token = paths.artifact_overrides.set({'transcription': custom})
    try:
        app = create_app('clip')
    finally:
        paths.artifact_overrides.reset(token)
    client = app.test_client()
    page = client.get('/').text
    assert 'id="filename"' not in page
    assert 'id="transcription"' not in page
    assert client.get('/load/clip').json['frames'] == [[3] * 6]
    assert client.post('/export', json={
        'filename': 'clip', 'frames': [[2] * 6], 'overwrite': True,
    }).status_code == 200
    np.testing.assert_array_equal(np.load(custom), [[2] * 6])
    assert not (directory / 'transcription.npy').exists()
