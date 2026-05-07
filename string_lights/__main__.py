import click
from pathlib import Path

from .pipeline import process_video


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
@click.option("--use-cached-masks", is_flag=True, default=False, help="Load masks from data/components/masks/ instead of recomputing.")
def run(filename: str, frames: int | None, disable_masking: bool, random_strings: bool, debug_masks: bool,
        only_poses: bool, only_masks: bool, use_cached_poses: bool, use_cached_masks: bool) -> None:
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
def tune(filename: str) -> None:
    from .tuner import run_tuner

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

    run_tuner(str(input_path))


@main.command("mask-edit")
@click.argument("filename")
def mask_edit(filename: str) -> None:
    from .mask_editor import run_mask_editor

    stem = Path(filename).stem if Path(filename).suffix else filename
    run_mask_editor(stem)


if __name__ == "__main__":
    main()
