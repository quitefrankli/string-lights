import click
import math
from functools import wraps

from pathlib import Path

from . import paths

from .pipeline import process_video


PLANAR_MIN_COVERAGE = 0.95
PLANAR_MAX_MEDIAN_ERROR_PX = 3.0
PLANAR_MAX_P95_ERROR_PX = 6.5
PLANAR_MAX_P95_BRIDGE_ERROR_PX = 4.0
PLANAR_MAX_P95_NUT_ERROR_PX = 10.5


def _resolve_video(filename: str) -> Path:
    try:
        return paths.resolve_video(filename)
    except FileNotFoundError as error:
        raise click.ClickException(str(error)) from error


def path_options(**artifacts):
    """Scope path choices to one command, including downstream cache helpers."""
    def decorate(function):
        @wraps(function)
        def invoke(*args, project_dir=None, **kwargs):
            overrides = {kind: kwargs.pop(option) for option, kind in artifacts.items()}
            overrides = {kind: value for kind, value in overrides.items() if value is not None}
            filename = kwargs.get("filename")
            directory = project_dir
            if directory is None and filename:
                try:
                    directory = paths.project_dir(paths.resolve_video(filename))
                except FileNotFoundError:
                    directory = paths.project_dir(filename)
            project_token = paths.project_override.set(directory)
            artifacts_token = paths.artifact_overrides.set(overrides)
            try:
                return function(*args, **kwargs)
            finally:
                paths.artifact_overrides.reset(artifacts_token)
                paths.project_override.reset(project_token)
        for option, kind in artifacts.items():
            invoke = click.option(
                "--" + option.replace("_", "-"),
                type=click.Path(dir_okay=False, path_type=Path),
                help=f"Override the {kind.replace('_', ' ')} file path.",
            )(invoke)
        return click.option("--project-dir", type=click.Path(file_okay=False, path_type=Path),
                            help="Override the project directory for default artifacts.")(invoke)
    return decorate


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
    if supplied.suffix == ".npz" or supplied.is_absolute() or len(supplied.parts) > 1:
        return supplied
    return paths.project_dir(stem) / (supplied.name if supplied.suffix else f"{value}.npz")


def _incompatible_planar_metadata(
    track_meta: dict[str, int | float],
    reference_meta: dict[str, int | float],
) -> str | None:
    mismatches = [
        f"{field} ({track_meta[field]} != {reference_meta[field]})"
        for field in ("w", "h", "total")
        if int(track_meta[field]) != int(reference_meta[field])
    ]
    return ", ".join(mismatches) or None


def _fails_maximum(value: float, maximum: float) -> bool:
    return not math.isfinite(value) or value > maximum


@click.group()
@click.option("--projects-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("data/projects"), show_default=True)
@click.pass_context
def main(ctx, projects_dir: Path) -> None:
    previous = paths.PROJECTS_DIR
    paths.PROJECTS_DIR = projects_dir
    ctx.call_on_close(lambda: setattr(paths, "PROJECTS_DIR", previous))


@main.command("render-strings")
@click.argument("filename")
@click.option("--frames", type=int, default=None, help="Max number of frames to process.")
@click.option("--disable-masking", is_flag=True, default=False, help="Skip hand masking (faster for dev testing).")
@click.option("--random-strings", is_flag=True, default=False, help="Generate random string highlights instead of reading from .npy.")
@click.option("--debug-masks", is_flag=True, default=False, help="Write debug video with soft hand-mask overlay; skips normal output.")
@path_options(output="render", projections="projections", masks="masks", mask_metadata="mask_metadata", transcription="transcription")
def run(filename: str, frames: int | None, disable_masking: bool, random_strings: bool, debug_masks: bool) -> None:
    input_path = _resolve_video(filename)

    output_path = paths.artifact_path(input_path, "render")
    if debug_masks and "render" not in paths.artifact_overrides.get():
        output_path = paths.project_dir(input_path) / "tmp/masks-debug.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        process_video(
            str(input_path), str(output_path),
            frames=frames,
            disable_masking=disable_masking,
            random_strings=random_strings,
            debug_masks=debug_masks,
        )
    except (FileNotFoundError, ValueError) as error:
        raise click.ClickException(str(error)) from error


@main.command("generate-masks")
@click.argument("filename")
@click.option("--frames", type=click.IntRange(min=1), help="Max number of frames to process.")
@click.option("--frame-skip", type=click.IntRange(min=1), default=1, show_default=True, help="Cache a mask every N frames; SAM2Matting still tracks every frame.")
@click.option("--force", is_flag=True, help="Overwrite an existing mask cache.")
@click.option("--seed-mask", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="First-frame hand mask PNG; defaults to PROJECT/masks/seed.png.")
@click.option("--stabilize/--no-stabilize", default=True, show_default=True, help="Reduce flicker with a motion-aligned three-frame median.")
@path_options(output="masks", metadata_output="mask_metadata")
def masks(filename: str, frames: int | None, frame_skip: int, force: bool, seed_mask: Path | None, stabilize: bool) -> None:
    """Track hands with SAM2Matting and cache masks without rendering."""
    from .pipeline import compute_masks, masks_meta_path, masks_path, save_masks

    video = _resolve_video(filename)
    for target in (masks_path(video.stem), masks_meta_path(video.stem)):
        _require_new_artifact(target, force)
    try:
        values, indices, metadata = compute_masks(str(video), stem=video.stem, frames=frames, frame_skip=frame_skip, seed_mask=seed_mask, stabilize=stabilize)
        save_masks(video.stem, values, indices, metadata)
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error


@main.command("tabs-editor")
@click.argument("project")
@click.option("--port", type=int, default=8080, show_default=True)
def tabs_editor(project: str, port: int) -> None:
    import webbrowser
    from .editor import create_app

    import os
    if project in (".", "..") or Path(project).name != project or "\\" in project:
        raise click.BadParameter("Expected a project name.", param_hint="project")
    try:
        app = create_app(project)
    except FileNotFoundError as error:
        raise click.ClickException(str(error)) from error
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open(f"http://localhost:{port}")
    app.run(host="localhost", port=port, debug=True)


@main.command("lyrics-editor")
@click.option("--port", type=int, default=8081, show_default=True)
def lyrics_editor(port: int) -> None:
    import webbrowser
    from .lyrics_editor import create_lyrics_app

    webbrowser.open(f"http://localhost:{port}")
    create_lyrics_app().run(host="localhost", port=port)


@main.command("tune-guitar-plane")
@click.argument("filename")
@path_options(annotation="annotation")
def tune(filename: str) -> None:
    from .planar_tuner import run_browser_tuner

    input_path = _resolve_video(filename)
    annotation = paths.artifact_path(input_path, "annotation")
    if not annotation.is_file():
        raise click.ClickException(
            f"Planar annotation not found: {annotation}. "
            "Create a profile.json annotation before tuning."
        )
    try:
        run_browser_tuner(str(input_path), annotation)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error


@main.command("mask-edit")
@click.argument("filename")
@path_options(masks="masks", mask_metadata="mask_metadata", output="mask_output")
def mask_edit(filename: str) -> None:
    from .mask_editor import run_mask_editor

    run_mask_editor(str(_resolve_video(filename)))


@main.command("planar-profile")
@click.argument("filename")
@click.argument(
    "annotation",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--name", help="Alternate profile filename within the project.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), help="Override the profile artifact path.")
@click.option("--force", is_flag=True, help="Overwrite an existing profile artifact.")
@path_options()
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
    target = output or (paths.project_dir(video_path) / f"{name}.npz" if name else profile_path(str(video_path)))
    _require_new_artifact(target, force)
    try:
        profile = create_planar_profile_from_annotation(video_path, annotation)
        save_planar_profile(target, profile)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Planar profile saved -> {target}")


@main.command("generate-poses")
@click.argument("filename")
@click.option("--frames", type=click.IntRange(min=1), help="Max number of frames to track.")
@click.option(
    "--profile",
    "profile_values",
    multiple=True,
    help="Profile name or explicit .npz path. Repeat for multiple views; defaults to PROJECT/profile.npz.",
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), help="Override the projection artifact path.")
@click.option("--force", is_flag=True, help="Overwrite an existing projection artifact.")
@path_options()
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
        _resolve_profile_artifact(value, str(video_path))
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
    target = output or projection_path(str(video_path))
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
@click.option("--reference", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Reference planar projection artifact to compare against.")
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
@path_options()
def planar_validate(
    filename: str,
    projections: Path | None,
    reference: Path,
    min_coverage: float,
    max_median_error: float,
    max_p95_error: float,
    max_p95_bridge_error: float,
    max_p95_nut_error: float,
) -> None:
    """Validate cached planar projections against reference planar projections."""
    from .planar import load_planar_track, validate_projection
    from .planar_io import projection_path

    stem = Path(filename).stem
    projection_source = projections or projection_path(stem)
    reference_source = reference
    if not projection_source.is_file():
        raise click.ClickException(f"Planar projections not found: {projection_source}")

    try:
        track, track_meta = load_planar_track(projection_source)
        reference_track, reference_meta = load_planar_track(reference_source)
        mismatch = _incompatible_planar_metadata(track_meta, reference_meta)
        if mismatch:
            raise ValueError(f"projection and reference metadata differ: {mismatch}")
        report = validate_projection(track, reference_track.projected_lines)
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
