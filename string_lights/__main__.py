import click
from pathlib import Path

from .pipeline import process_video


@click.command()
@click.argument("filename")
@click.option("--frames", type=int, default=None, help="Max number of frames to process.")
def main(filename: str, frames: int | None) -> None:
    input_dir = Path("data/input")

    if Path(filename).suffix:
        input_path = input_dir / filename
    else:
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".flv"]:
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
    process_video(str(input_path), str(output_path), frames=frames)


if __name__ == "__main__":
    main()
