import numpy as np
from click.testing import CliRunner

from string_lights import paths, transcription
from string_lights.__main__ import main


def test_transcription_uses_shared_project_root(tmp_path, monkeypatch):
    project = tmp_path / 'song'
    project.mkdir()
    (project / 'source.mov').touch()
    original_root = paths.PROJECTS_DIR

    def generate(video, **options):
        assert video == project / 'source.mov'
        assert options['capo'] == 2
        assert options['tuning'] == [40, 45, 50, 55, 59, 64]
        return np.zeros((44, 6)), {'events': [], 'unassigned_count': 0}

    monkeypatch.setattr(transcription, 'generate', generate)
    result = CliRunner().invoke(main, [
        '--projects-dir', str(tmp_path), 'generate-transcription', 'song',
        '--capo', '2', '--tuning', '40', '45', '50', '55', '59', '64',
    ])
    assert result.exit_code == 0, result.output
    assert (project / 'transcription.npy').exists()
    assert (project / 'transcription.json').exists()
    assert paths.PROJECTS_DIR == original_root


def test_subcommand_help_and_validation():
    runner = CliRunner()
    for command in ['generate-transcription', 'refine-transcription']:
        result = runner.invoke(main, [command, '--help'], prog_name='string-lights')
        assert result.exit_code == 0, result.output
        assert f'string-lights {command}' in result.output
    result = runner.invoke(main, ['generate-transcription', 'song', '--capo', '20'])
    assert result.exit_code == 2
    assert 'Capo must be' in result.output
