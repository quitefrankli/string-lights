from pathlib import Path

import cv2
import numpy as np
from click.testing import CliRunner

from string_lights.__main__ import main
from string_lights import paths, pipeline
from string_lights.config import MASK_ENCODING


def test_project_resolution_and_source_preference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = Path('data/projects/cupid')
    project.mkdir(parents=True)
    (project / 'source.mov').touch()
    assert paths.resolve_video('cupid') == project / 'source.mov'
    (project / 'source.mp4').touch()
    assert paths.resolve_video('cupid') == project / 'source.mp4'
    assert paths.resolve_video(project) == project / 'source.mp4'
    assert paths.artifact_path(project / 'tmp/custom.mov', 'profile') == project / 'profile.npz'


def test_custom_cache_and_render_paths_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'arbitrary-name.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    for _ in range(3):
        writer.write(np.full((48, 64, 3), 100, dtype=np.uint8))
    writer.release()
    project = Path('data/projects/cupid')
    experiment = project / 'tmp'
    cache = experiment / 'custom-projections.npz'
    output = experiment / 'custom.render.mp4'
    from string_lights.planar import PlanarFrame, PlanarTrack, TrackDiagnostics, TrackState, save_planar_track
    lines = np.array([[[10, 10 + i * 4], [50, 10 + i * 4]] for i in range(6)], dtype=float)
    frames = tuple(PlanarFrame(np.eye(3), lines, TrackState.OBSERVED,
                              TrackDiagnostics(20, 20, 1.0, 0.0, 1.0), 0) for _ in range(3))
    save_planar_track(cache, PlanarTrack(frames, ("a" * 64,)), dict(w=64, h=48, fps=10.0, total=3))
    runner = CliRunner()
    assert cache.is_file()
    reference = experiment / 'reference.npz'
    save_planar_track(reference, PlanarTrack(frames, ("a" * 64,)), dict(w=64, h=48, fps=10.0, total=3))
    validation = runner.invoke(main, ['planar-validate', 'cupid', '--projections', str(cache),
                                      '--reference', str(reference)])
    assert validation.exit_code == 0, validation.output
    result = runner.invoke(main, ['render-strings', str(source), '--project-dir', str(project),
                                 '--projections', str(cache), '--output', str(output),
                                 '--disable-masking', '--random-strings'])
    assert result.exit_code == 0, result.output
    cap = cv2.VideoCapture(str(output))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        cap.release()
    assert not (project / 'tracking/projections.npz').exists()
    assert not (project / 'renders').exists()
    assert paths.artifact_overrides.get() == {}
    assert paths.project_override.get() is None


def test_custom_mask_sidecar_round_trip_and_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'clip.mov'
    source.touch()
    output = Path('data/projects/clip/tmp/trial.npz')
    masks = np.ones((2, 4, 6), dtype=np.uint8)
    metadata = dict(w=6, h=4, total=2, fps=30.0, encoding=MASK_ENCODING)
    monkeypatch.setattr(pipeline, 'compute_masks', lambda *a, **kw: (masks, np.arange(2), metadata))
    arguments = ['generate-masks', str(source), '--output', str(output)]
    runner = CliRunner()
    assert runner.invoke(main, arguments).exit_code == 0
    assert output.with_suffix('.meta.npz').is_file()
    token = paths.artifact_overrides.set({'masks': output})
    try:
        actual, indices, meta = pipeline.load_masks('clip')
    finally:
        paths.artifact_overrides.reset(token)
    np.testing.assert_array_equal(actual, masks)
    assert meta == metadata
    refused = runner.invoke(main, arguments)
    assert refused.exit_code != 0
    assert 'use --force' in refused.output
    assert runner.invoke(main, arguments + ['--force']).exit_code == 0
    assert not Path('data/projects/clip/masks').exists()


def test_custom_mask_metadata_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('clip.mov').touch()
    monkeypatch.setattr(pipeline, 'compute_masks', lambda *a, **kw: (
        np.zeros((1, 2, 2), dtype=np.uint8), np.array([0]), dict(encoding=MASK_ENCODING)))
    result = CliRunner().invoke(main, ['generate-masks', './clip.mov', '--output', 'tmp/cache/m.npz',
                                      '--metadata-output', 'tmp/metadata/custom.npz'])
    assert result.exit_code == 0, result.output
    assert Path('tmp/cache/m.npz').is_file()
    assert Path('tmp/metadata/custom.npz').is_file()


def test_collection_override_and_default_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path('custom-projects/cupid/source.mp4')
    source.parent.mkdir(parents=True)
    source.touch()
    received = []
    monkeypatch.setattr('string_lights.__main__.process_video', lambda *a, **kw: received.append(a))
    result = CliRunner().invoke(main, ['--projects-dir', 'custom-projects', 'render-strings', 'cupid'])
    assert result.exit_code == 0, result.output
    assert received == [('custom-projects/cupid/source.mp4', 'custom-projects/cupid/renders/strings.mp4')]
    assert paths.PROJECTS_DIR == Path('data/projects')
