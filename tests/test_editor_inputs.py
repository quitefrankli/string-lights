import json
from types import SimpleNamespace

from click.testing import CliRunner

from string_lights import lyrics_editor, paths
from string_lights.__main__ import main


def test_lyrics_selected_video_and_json_apply_to_requests_and_render(tmp_path, monkeypatch):
    source = tmp_path / 'alternate.mp4'
    source.touch()
    custom = tmp_path / 'custom.json'
    custom.write_text('[]')
    directory = tmp_path / 'project'
    project_token = paths.project_override.set(directory)
    artifact_token = paths.artifact_overrides.set({'lyrics': custom})
    try:
        app = lyrics_editor.create_lyrics_app(str(source))
    finally:
        paths.project_override.reset(project_token)
        paths.artifact_overrides.reset(artifact_token)
    client = app.test_client()
    page = client.get('/').text
    for control in ('filename', 'video-file', 'json-file'):
        assert f'id="{control}"' not in page
    assert client.post('/upload').status_code == 404
    assert client.get('/lyrics/other').status_code == 404
    assert client.get('/lyrics/project').json == []
    values = [{'text': 'hello', 'start': 0, 'end': 1}]
    assert client.post('/lyrics/project', json=values).status_code == 200
    assert json.loads(custom.read_text()) == values
    assert not (directory / 'lyrics.json').exists()

    monkeypatch.setattr('string_lights.editor._browser_video', lambda p: p)
    response = client.get('/video/project')
    assert response.status_code == 200
    response.close()
    threads = []
    rendered = []
    monkeypatch.setattr(lyrics_editor, '_render', {'state': 'idle'})
    monkeypatch.setattr(lyrics_editor.threading, 'Thread', lambda **kw:
                        SimpleNamespace(start=lambda: threads.append(kw)))
    monkeypatch.setattr(lyrics_editor, '_do_render', lambda *args: rendered.append(
        (args, paths.project_override.get(), paths.artifact_overrides.get())))
    assert client.post('/render/project', json={}).status_code == 200
    threads[0]['target'](*threads[0]['args'])
    assert rendered == [((str(source), 'padded', 'source_audio'), directory, {'lyrics': custom})]
    assert paths.project_override.get() is None
    assert paths.artifact_overrides.get() == {}


def test_lyrics_cli_requires_input_and_passes_overrides(tmp_path, monkeypatch):
    runner = CliRunner()
    assert runner.invoke(main, ['lyrics-editor']).exit_code != 0
    source = tmp_path / 'source.mp4'
    source.touch()
    custom = tmp_path / 'custom.json'
    calls = []
    monkeypatch.setattr('webbrowser.open', lambda url: None)
    monkeypatch.setattr(lyrics_editor, 'create_lyrics_app', lambda filename:
                        calls.append((filename, paths.artifact_overrides.get())) or
                        SimpleNamespace(run=lambda **kw: calls.append(kw)))
    result = runner.invoke(main, ['lyrics-editor', str(source), '--lyrics', str(custom)])
    assert result.exit_code == 0, result.output
    assert calls[0] == (str(source), {'lyrics': custom})
    assert calls[1]['port'] == 8081


def test_lyrics_project_uses_strings_render_for_preview_and_export(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'PROJECTS_DIR', tmp_path)
    directory = tmp_path / 'gunjou_planar'
    directory.mkdir()
    source = directory / 'source.mov'
    source.write_bytes(b'original footage')
    rendered = directory / 'renders' / 'strings.mp4'
    rendered.parent.mkdir()
    rendered.write_bytes(b'string highlights')
    monkeypatch.setattr('string_lights.editor._browser_video', lambda p: p)
    client = lyrics_editor.create_lyrics_app('gunjou_planar').test_client()
    response = client.get('/video/gunjou_planar')
    assert response.data == b'string highlights'
    response.close()
    assert lyrics_editor._base_video('gunjou_planar') == rendered
    assert lyrics_editor._base_video(str(source)) == source
    threads, inputs = [], []
    monkeypatch.setattr(lyrics_editor, '_render', {'state': 'idle'})
    monkeypatch.setattr(lyrics_editor.threading, 'Thread', lambda **kw:
                        SimpleNamespace(start=lambda: threads.append(kw)))
    monkeypatch.setattr(lyrics_editor, '_do_render', lambda *args: inputs.append(args[0]))
    assert client.post('/render/gunjou_planar', json={}).status_code == 200
    threads[0]['target'](*threads[0]['args'])
    assert inputs == [str(rendered.resolve())]
    rendered.unlink()
    assert lyrics_editor._base_video('gunjou_planar') == source


def test_lyrics_cli_keeps_project_name_for_render_selection(monkeypatch):
    calls = []
    monkeypatch.setattr('webbrowser.open', lambda url: None)
    monkeypatch.setattr(lyrics_editor, 'create_lyrics_app', lambda filename:
                        calls.append(filename) or SimpleNamespace(run=lambda **kw: None))
    result = CliRunner().invoke(main, ['lyrics-editor', 'gunjou_planar'])
    assert result.exit_code == 0, result.output
    assert calls == ['gunjou_planar']
