import pytest
from click.testing import CliRunner

from string_lights.__main__ import main


def test_cache_command_runs_only_requested_stage(tmp_path, monkeypatch):
    import string_lights.pipeline as pipeline

    video = tmp_path / "clip.mp4"
    video.touch()
    monkeypatch.chdir(tmp_path)
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("Unrelated processing stage invoked")

    monkeypatch.setattr(pipeline, "process_video", forbidden)

    def compute(path, **kwargs):
        calls.append((path, kwargs))
        return ("values", "indices", "metadata")

    monkeypatch.setattr(pipeline, "compute_masks", compute)
    monkeypatch.setattr(pipeline, "save_masks", lambda *args: calls.append(args))

    result = CliRunner().invoke(main, ["generate-masks", str(video), "--frames", "500"])

    assert result.exit_code == 0, result.output
    expected = dict(frames=500, stem="clip", frame_skip=1, seed_mask=None, stabilize=True)
    assert calls[0] == (str(video), expected)
    assert calls[1] == ("clip", "values", "indices", "metadata")


def test_cache_command_requires_force_for_existing_artifacts(tmp_path, monkeypatch):
    import string_lights.pipeline as pipeline

    monkeypatch.chdir(tmp_path)
    video = tmp_path / "clip.mp4"
    video.touch()
    target = pipeline.masks_path("clip")
    target.parent.mkdir(parents=True)
    target.touch()
    monkeypatch.setattr(pipeline, "compute_masks", lambda *a, **kw: pytest.fail("must refuse before computing"))

    result = CliRunner().invoke(main, ["generate-masks", str(video)])

    assert result.exit_code != 0
    assert "use --force" in result.output


@pytest.mark.parametrize("flag", ["--masks", "--projections"])
def test_run_cache_overrides_require_paths(flag):
    result = CliRunner().invoke(main, ["render-strings", "missing", flag])
    assert result.exit_code != 0
    assert "requires an argument" in result.output
