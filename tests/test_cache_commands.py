import pytest
from click.testing import CliRunner

from string_lights.__main__ import main


@pytest.mark.parametrize("command", ["poses", "masks"])
def test_cache_command_runs_only_requested_stage(tmp_path, monkeypatch, command):
    import string_lights.pipeline as pipeline

    video = tmp_path / "clip.mp4"
    video.touch()
    monkeypatch.chdir(tmp_path)
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("Unrelated processing stage invoked")

    monkeypatch.setattr(pipeline, "process_video", forbidden)
    other = "masks" if command == "poses" else "poses"
    monkeypatch.setattr(pipeline, f"compute_{other}", forbidden)

    def compute(path, **kwargs):
        calls.append((path, kwargs))
        return ("values", "indices", "metadata") if command == "masks" else ("values", "metadata")

    monkeypatch.setattr(pipeline, f"compute_{command}", compute)
    monkeypatch.setattr(pipeline, f"save_{command}", lambda *args: calls.append(args))

    result = CliRunner().invoke(main, [command, str(video), "--frames", "500"])

    assert result.exit_code == 0, result.output
    expected = {"frames": 500}
    if command == "masks":
        expected["stem"] = "clip"
        expected["frame_skip"] = 1
    assert calls[0] == (str(video), expected)
    assert calls[1] == (("clip", "values", "indices", "metadata") if command == "masks" else ("clip", "values", "metadata"))


@pytest.mark.parametrize("command", ["poses", "masks"])
def test_cache_command_requires_force_for_existing_artifacts(tmp_path, monkeypatch, command):
    import string_lights.pipeline as pipeline

    monkeypatch.chdir(tmp_path)
    video = tmp_path / "clip.mp4"
    video.touch()
    target = getattr(pipeline, f"{command}_path")("clip")
    target.parent.mkdir(parents=True)
    target.touch()
    monkeypatch.setattr(pipeline, f"compute_{command}", lambda *a, **kw: pytest.fail("must refuse before computing"))

    result = CliRunner().invoke(main, [command, str(video)])

    assert result.exit_code != 0
    assert "use --force" in result.output


@pytest.mark.parametrize("flag", ["--masks", "--poses"])
def test_run_no_longer_accepts_cache_generation_flags(flag):
    result = CliRunner().invoke(main, ["run", "missing", flag])
    assert result.exit_code != 0
    assert "No such option" in result.output
