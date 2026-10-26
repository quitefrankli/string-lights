import click
import math
from pathlib import Path

from .pipeline import process_video


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".m4v")

PLANAR_MIN_COVERAGE = 0.95
PLANAR_MAX_MEDIAN_ERROR_PX = 3.0
PLANAR_MAX_P95_ERROR_PX = 6.5
PLANAR_MAX_P95_BRIDGE_ERROR_PX = 4.0
PLANAR_MAX_P95_NUT_ERROR_PX = 10.5


def _resolve_video(filename: str) -> Path:
    supplied = Path(filename)
    base = supplied if supplied.is_absolute() or len(supplied.parts) > 1 else Path("data/input") / supplied
    candidates = [base] if base.suffix else [base.with_suffix(ext) for ext in VIDEO_EXTENSIONS]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise click.ClickException(f"Input file not found: {base}")


def _require_new_artifact(path: Path, force: bool) -> None:
    if path.suffix != ".npz":
        raise click.ClickException(f"Artifact path must end in .npz: {path}")
    if path.exists() and not force:
        raise click.ClickException(f"Refusing to overwrite existing artifact: {path} (use --force)")


def _resolve_profile_artifact(value: str | None, stem: str) -> Path:
    from .planar_io import profile_path

    if value is None:
        return profile_path(stem)
    supplied = Path(value)
    if supplied.is_absolute() or len(supplied.parts) > 1:
        return supplied
    return profile_path(supplied.stem if supplied.suffix == ".npz" else value)


def _incompatible_planar_metadata(
    track_meta: dict[str, int | float],
    pose_meta: dict[str, int | float],
) -> str | None:
    mismatches = [
        f"{field} ({track_meta[field]} != {pose_meta[field]})"
        for field in ("w", "h", "total")
        if int(track_meta[field]) != int(pose_meta[field])
    ]
    return ", ".join(mismatches) or None


def _fails_maximum(value: float, maximum: float) -> bool:
    return not math.isfinite(value) or value > maximum


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("filename")
@click.option("--frames", type=int, default=None, help="Max number of frames to process.")
@click.option("--disable-masking", is_flag=True, default=False, help="Skip hand masking (faster for dev testing).")
@click.option("--random-strings", is_flag=True, default=False, help="Generate random string highlights instead of reading from .npy.")
@click.option("--debug-masks", is_flag=True, default=False, help="Write debug video with SAM boxes and mask overlay; skips normal output.")
@click.option("--poses", "only_poses", is_flag=True, default=False, help="Only compute and cache pose data; no render.")
@click.option("--masks", "only_masks", is_flag=True, default=False, help="Only compute and cache mask data; no render.")
@click.option("--use-cached-poses", is_flag=True, default=False, help="Load poses from data/components/poses/ instead of recomputing.")
@click.option("--use-cached-projections", is_flag=True, default=False, help="Load planar string projections from data/components/projections/ instead of board poses.")
@click.option("--use-cached-masks", is_flag=True, default=False, help="Load masks from data/components/masks/ instead of recomputing.")
def run(filename: str, frames: int | None, disable_masking: bool, random_strings: bool, debug_masks: bool,
        only_poses: bool, only_masks: bool, use_cached_poses: bool,
        use_cached_projections: bool, use_cached_masks: bool) -> None:
    if only_poses and use_cached_projections:
        raise click.UsageError("--poses cannot be combined with --use-cached-projections")
    if use_cached_poses and use_cached_projections:
        raise click.UsageError(
            "--use-cached-poses cannot be combined with --use-cached-projections"
        )

    input_dir = Path("data/input")

    if Path(filename).suffix:
        input_path = input_dir / filename
    else:
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".m4v"]:
            candidate = input_dir / (filename + ext)
            if candidate.exists():
                input_path = candidate
                break
        else:
            input_path = input_dir / filename

    if not input_path.exists():
        raise click.ClickException(f"Input file not found: {input_path}")

    stem = input_path.stem
    output_path = Path("data/output") / f"{stem}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_video(
        str(input_path), str(output_path),
        frames=frames,
        disable_masking=disable_masking,
        random_strings=random_strings,
        debug_masks=debug_masks,
        only_poses=only_poses,
        only_masks=only_masks,
        use_cached_poses=use_cached_poses,
        use_cached_projections=use_cached_projections,
        use_cached_masks=use_cached_masks,
    )


@main.command()
@click.option("--port", type=int, default=8080, show_default=True)
def editor(port: int) -> None:
    import webbrowser
    from .editor import create_app

    import os
    app = create_app()
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open(f"http://localhost:{port}")
    app.run(host="localhost", port=port, debug=True)


@main.command()
@click.argument("filename")
@click.option("--aruco", is_flag=True, help="Tune board-based ArUco coordinates.")
@click.option("--no-aruco", is_flag=True, help="Tune markerless planar string lines.")
def tune(filename: str, aruco: bool, no_aruco: bool) -> None:
    from .tuner import run_planar_tuner, run_tuner

    if aruco == no_aruco:
        raise click.UsageError("exactly one of --aruco or --no-aruco is required")

    input_dir = Path("data/input")
    if Path(filename).suffix:
        input_path = input_dir / filename
    else:
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".flv", ".m4v"]:
            candidate = input_dir / (filename + ext)
            if candidate.exists():
                input_path = candidate
                break
        else:
            input_path = input_dir / filename

    if not input_path.exists():
        raise click.ClickException(f"Input file not found: {input_path}")

    if aruco:
        run_tuner(str(input_path))
        return

    annotation = Path("data/components/profiles") / f"{input_path.stem}.json"
    if not annotation.is_file():
        raise click.ClickException(
            f"Planar annotation not found: {annotation}. "
            "Create an annotation before using --no-aruco."
        )
    try:
        run_planar_tuner(str(input_path), annotation)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error


@main.command("mask-edit")
@click.argument("filename")
def mask_edit(filename: str) -> None:
    from .mask_editor import run_mask_editor

    stem = Path(filename).stem if Path(filename).suffix else filename
    run_mask_editor(stem)


@main.command("planar-profile")
@click.argument("filename")
@click.argument(
    "annotation",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--name", help="Profile name under data/components/profiles/.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), help="Override the profile artifact path.")
@click.option("--force", is_flag=True, help="Overwrite an existing profile artifact.")
def planar_profile(
    filename: str,
    annotation: Path,
    name: str | None,
    output: Path | None,
    force: bool,
) -> None:
    """Create a planar guitar PROFILE from an annotated reference frame."""
    from .planar import save_planar_profile
    from .planar_io import create_planar_profile_from_annotation, profile_path

    video_path = _resolve_video(filename)
    target = output or profile_path(name or video_path.stem)
    _require_new_artifact(target, force)
    try:
        profile = create_planar_profile_from_annotation(video_path, annotation)
        save_planar_profile(target, profile)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Planar profile saved -> {target}")


@main.command("planar-track")
@click.argument("filename")
@click.option("--frames", type=click.IntRange(min=1), help="Max number of frames to track.")
@click.option(
    "--profile",
    "profile_values",
    multiple=True,
    help="Profile name or explicit .npz path. Repeat for multiple views; defaults to the video stem.",
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), help="Override the projection artifact path.")
@click.option("--force", is_flag=True, help="Overwrite an existing projection artifact.")
def planar_track(
    filename: str,
    frames: int | None,
    profile_values: tuple[str, ...],
    output: Path | None,
    force: bool,
) -> None:
    """Track a planar guitar through a VIDEO and cache projected strings."""
    from .planar import TrackState, load_planar_profile, save_planar_track
    from .planar_io import projection_path, track_planar_video

    video_path = _resolve_video(filename)
    sources = tuple(
        _resolve_profile_artifact(value, video_path.stem)
        for value in (profile_values or (None,))
    )
    resolved_sources = tuple(source.resolve() for source in sources)
    duplicate = next(
        (
            source
            for index, source in enumerate(resolved_sources)
            if source in resolved_sources[:index]
        ),
        None,
    )
    if duplicate is not None:
        raise click.ClickException(f"Duplicate planar profile: {duplicate}")
    missing = next((source for source in sources if not source.is_file()), None)
    if missing is not None:
        raise click.ClickException(f"Planar profile not found: {missing}")
    target = output or projection_path(video_path.stem)
    _require_new_artifact(target, force)

    try:
        profiles = tuple(load_planar_profile(source) for source in sources)
        tracker_profiles = profiles[0] if len(profiles) == 1 else profiles
        if frames is None:
            track, metadata = track_planar_video(video_path, tracker_profiles)
        else:
            track, metadata = track_planar_video(
                video_path, tracker_profiles, max_frames=frames
            )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    total = len(track.frames)
    observed = sum(frame.state == TrackState.OBSERVED for frame in track.frames)
    if not total:
        raise click.ClickException("Planar tracking produced no frames")
    if not observed:
        raise click.ClickException("Planar tracking could not resolve any frame; no artifact was saved")
    try:
        save_planar_track(target, track, metadata)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(f"Planar projections saved -> {target}")
    click.echo(f"Observed {observed}/{total} frames ({100.0 * observed / total:.1f}% coverage)")


@main.command("planar-validate")
@click.argument("filename")
@click.option("--projections", type=click.Path(dir_okay=False, path_type=Path), help="Override the planar projection artifact.")
@click.option("--poses", type=click.Path(dir_okay=False, path_type=Path), help="Override the board-pose oracle artifact.")
@click.option(
    "--min-coverage",
    type=click.FloatRange(min=0.0, max=1.0),
    default=PLANAR_MIN_COVERAGE,
    show_default=True,
)
@click.option("--max-median-error", type=click.FloatRange(min=0.0), default=PLANAR_MAX_MEDIAN_ERROR_PX, show_default=True)
@click.option("--max-p95-error", type=click.FloatRange(min=0.0), default=PLANAR_MAX_P95_ERROR_PX, show_default=True)
@click.option("--max-p95-bridge-error", type=click.FloatRange(min=0.0), default=PLANAR_MAX_P95_BRIDGE_ERROR_PX, show_default=True)
@click.option("--max-p95-nut-error", type=click.FloatRange(min=0.0), default=PLANAR_MAX_P95_NUT_ERROR_PX, show_default=True)
def planar_validate(
    filename: str,
    projections: Path | None,
    poses: Path | None,
    min_coverage: float,
    max_median_error: float,
    max_p95_error: float,
    max_p95_bridge_error: float,
    max_p95_nut_error: float,
) -> None:
    """Validate cached planar projections against cached board poses."""
    from .pipeline import poses_path
    from .planar import load_planar_track, validate_projection
    from .planar_io import projected_strings_from_pose_cache, projection_path

    stem = Path(filename).stem
    projection_source = projections or projection_path(stem)
    pose_source = poses or poses_path(stem)
    if not projection_source.is_file():
        raise click.ClickException(f"Planar projections not found: {projection_source}")
    if not pose_source.is_file():
        raise click.ClickException(f"Board poses not found: {pose_source}")

    try:
        track, track_meta = load_planar_track(projection_source)
        expected, pose_meta = projected_strings_from_pose_cache(pose_source)
        mismatch = _incompatible_planar_metadata(track_meta, pose_meta)
        if mismatch:
            raise ValueError(f"projection and pose metadata differ: {mismatch}")
        report = validate_projection(track, expected)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(
        f"Coverage: {100.0 * report.coverage:.1f}% "
        f"({report.evaluated_frames}/{report.total_frames} frames)"
    )
    click.echo(
        "Perpendicular error: "
        f"median {report.median_perpendicular_error_px:.3f}px, "
        f"p95 {report.p95_perpendicular_error_px:.3f}px, "
        f"max {report.max_perpendicular_error_px:.3f}px"
    )
    click.echo(
        "Endpoint error: "
        f"median {report.median_endpoint_error_px:.3f}px, "
        f"p95 {report.p95_endpoint_error_px:.3f}px, "
        f"max {report.max_endpoint_error_px:.3f}px"
    )
    click.echo(
        "Endpoint p95 by end: "
        f"bridge {report.p95_bridge_endpoint_error_px:.3f}px, "
        f"nut {report.p95_nut_endpoint_error_px:.3f}px"
    )

    failures = []
    if not math.isfinite(report.coverage) or report.coverage < min_coverage:
        failures.append(f"coverage {report.coverage:.3f} < {min_coverage:.3f}")
    for label, value, maximum in (
        ("median perpendicular error", report.median_perpendicular_error_px, max_median_error),
        ("p95 perpendicular error", report.p95_perpendicular_error_px, max_p95_error),
        ("p95 bridge endpoint error", report.p95_bridge_endpoint_error_px, max_p95_bridge_error),
        ("p95 nut endpoint error", report.p95_nut_endpoint_error_px, max_p95_nut_error),
    ):
        if _fails_maximum(value, maximum):
            failures.append(f"{label} {value:.3f}px > {maximum:.3f}px")
    if failures:
        raise click.ClickException("Planar validation failed: " + "; ".join(failures))
    click.echo("Planar validation passed")


if __name__ == "__main__":
    main()
