"""Project layout and explicit artifact overrides for CLI and editors."""
from contextvars import ContextVar
from pathlib import Path

PROJECTS_DIR = Path("data/projects")
project_override: ContextVar[Path | None] = ContextVar("project_dir", default=None)
artifact_overrides: ContextVar[dict[str, Path]] = ContextVar("artifacts", default={})
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".m4v")


def project_dir(value: str | Path) -> Path:
    if project_override.get() is not None:
        return project_override.get()
    path = Path(value)
    if path.is_dir():
        return path
    if len(path.parts) > 1:
        # A source or experimental clip inside a project shares that project's data.
        for parent in path.parents:
            if parent.parent.resolve() == PROJECTS_DIR.resolve():
                return parent
        if path.stem == "source":
            return path.parent
    return PROJECTS_DIR / (path.stem if path.suffix.lower() in VIDEO_EXTENSIONS else path.name)


ARTIFACTS = {
    "profile": "profile.npz", "annotation": "profile.json",
    "projections": "tracking/projections.npz",
    "masks": "masks/masks.npz", "mask_metadata": "masks/metadata.npz",
    "seed_mask": "masks/seed.png", "lyrics": "lyrics.json",
    "transcription": "transcription.npy", "render": "renders/strings.mp4",
    "lyrics_render": "renders/strings-lyrics.mp4",
}


def artifact_path(value: str | Path, kind: str) -> Path:
    return artifact_overrides.get().get(kind, project_dir(value) / ARTIFACTS[kind])


def resolve_video(value: str | Path) -> Path:
    supplied = Path(value)
    if supplied.is_file():
        return supplied
    base = project_dir(value) / "source" if supplied.is_dir() or len(supplied.parts) == 1 else supplied
    candidates = [base] if base.suffix else [base.with_suffix(ext) for ext in VIDEO_EXTENSIONS]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Input file not found: {base}")


def list_projects() -> list[str]:
    result = []
    for directory in sorted(PROJECTS_DIR.iterdir()) if PROJECTS_DIR.exists() else []:
        if directory.is_dir():
            try:
                resolve_video(directory)
            except FileNotFoundError:
                continue
            result.append(directory.name)
    return result
