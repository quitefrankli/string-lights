from pathlib import Path

import numpy as np
from click.testing import CliRunner

from string_lights.__main__ import main
from string_lights.planar import (
    PlanarFrame,
    PlanarTrack,
    TrackDiagnostics,
    TrackState,
    ValidationReport,
)


def _artifact(path: Path) -> Path:
    path.write_bytes(b"fixture")
    return path


def _report(**overrides) -> ValidationReport:
    values = {
        "total_frames": 100,
        "evaluated_frames": 100,
        "coverage": 1.0,
        "median_perpendicular_error_px": 2.0,
        "p95_perpendicular_error_px": 5.0,
        "max_perpendicular_error_px": 8.0,
        "median_endpoint_error_px": 2.5,
        "p95_endpoint_error_px": 8.0,
        "max_endpoint_error_px": 12.0,
        "p95_bridge_endpoint_error_px": 3.0,
        "p95_nut_endpoint_error_px": 9.0,
    }
    values.update(overrides)
    return ValidationReport(**values)


def test_run_forwards_cached_planar_projections(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "data/input"
    input_dir.mkdir(parents=True)
    _artifact(input_dir / "clip.mp4")
    monkeypatch.chdir(tmp_path)
    received = {}

    import string_lights.__main__ as cli

    monkeypatch.setattr(cli, "process_video", lambda *args, **kwargs: received.update(kwargs))

    result = CliRunner().invoke(main, ["run", "clip", "--use-cached-projections"])

    assert result.exit_code == 0, result.output
    assert received["use_cached_projections"] is True
    assert received["use_cached_poses"] is False


def test_run_rejects_conflicting_pose_sources_before_processing(monkeypatch) -> None:
    import string_lights.__main__ as cli

    monkeypatch.setattr(
        cli,
        "process_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not process")),
    )
    conflicts = (
        ["run", "missing", "--poses", "--use-cached-projections"],
        [
            "run",
            "missing",
            "--use-cached-poses",
            "--use-cached-projections",
        ],
    )

    for arguments in conflicts:
        result = CliRunner().invoke(main, arguments)

        assert result.exit_code != 0
        assert "cannot be combined" in result.output
        assert "Input file not found" not in result.output


def test_planar_profile_refuses_to_overwrite_without_force(tmp_path) -> None:
    video = _artifact(tmp_path / "clip.mp4")
    annotation = _artifact(tmp_path / "profile.json")
    output = _artifact(tmp_path / "profile.npz")

    result = CliRunner().invoke(
        main,
        ["planar-profile", str(video), str(annotation), "--output", str(output)],
    )

    assert result.exit_code != 0
    assert "Refusing to overwrite" in result.output


def test_planar_track_does_not_save_an_entirely_unresolved_track(
    tmp_path, monkeypatch
) -> None:
    video = _artifact(tmp_path / "clip.mp4")
    profile = _artifact(tmp_path / "profile.npz")
    output = tmp_path / "track.npz"
    unresolved = PlanarFrame(
        homography=None,
        projected_lines=np.full((6, 2, 2), np.nan),
        state=TrackState.UNRESOLVED,
        diagnostics=TrackDiagnostics(0, 0, 0.0, float("nan"), 0.0),
    )

    import string_lights.planar as planar
    import string_lights.planar_io as planar_io

    monkeypatch.setattr(planar, "load_planar_profile", lambda _: object())
    monkeypatch.setattr(
        planar_io,
        "track_planar_video",
        lambda *_: (PlanarTrack((unresolved,)), {"w": 10, "h": 10, "fps": 30.0, "total": 1}),
    )

    result = CliRunner().invoke(
        main,
        [
            "planar-track",
            str(video),
            "--profile",
            str(profile),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "could not resolve any frame" in result.output
    assert not output.exists()


def test_planar_validate_passes_and_enforces_accuracy_gates(
    tmp_path, monkeypatch
) -> None:
    projections = _artifact(tmp_path / "track.npz")
    poses = _artifact(tmp_path / "poses.npz")
    metadata = {"w": 1920, "h": 1080, "fps": 30.0, "total": 100}

    import string_lights.planar as planar
    import string_lights.planar_io as planar_io

    monkeypatch.setattr(planar, "load_planar_track", lambda _: (object(), metadata))
    monkeypatch.setattr(
        planar_io,
        "projected_strings_from_pose_cache",
        lambda _: (np.zeros((100, 6, 2, 2)), metadata),
    )
    monkeypatch.setattr(planar, "validate_projection", lambda *_: _report())

    arguments = [
        "planar-validate",
        "clip",
        "--projections",
        str(projections),
        "--poses",
        str(poses),
    ]
    passed = CliRunner().invoke(main, arguments)

    assert passed.exit_code == 0, passed.output
    assert "Planar validation passed" in passed.output

    monkeypatch.setattr(
        planar,
        "validate_projection",
        lambda *_: _report(p95_nut_endpoint_error_px=10.51),
    )
    failed = CliRunner().invoke(main, arguments)

    assert failed.exit_code != 0
    assert "p95 nut endpoint error" in failed.output


def test_planar_validate_rejects_misaligned_artifacts(tmp_path, monkeypatch) -> None:
    projections = _artifact(tmp_path / "track.npz")
    poses = _artifact(tmp_path / "poses.npz")

    import string_lights.planar as planar
    import string_lights.planar_io as planar_io

    monkeypatch.setattr(
        planar,
        "load_planar_track",
        lambda _: (object(), {"w": 1920, "h": 1080, "fps": 30.0, "total": 100}),
    )
    monkeypatch.setattr(
        planar_io,
        "projected_strings_from_pose_cache",
        lambda _: (
            np.zeros((99, 6, 2, 2)),
            {"w": 1920, "h": 1080, "fps": 30.0, "total": 99},
        ),
    )

    result = CliRunner().invoke(
        main,
        [
            "planar-validate",
            "clip",
            "--projections",
            str(projections),
            "--poses",
            str(poses),
        ],
    )

    assert result.exit_code != 0
    assert "metadata differ" in result.output
