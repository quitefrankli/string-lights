import io
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from flask import Flask, g, jsonify, request, send_from_directory, render_template

from . import paths
from .paths import artifact_path, project_dir, resolve_video


def _browser_video(source: Path) -> Path:
    source_key = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
    preview = project_dir(source) / "tmp" / f"preview-{source_key}.mp4"
    if preview.exists() and preview.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return preview
    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,pix_fmt", "-of", "json", str(source),
    ], check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout)["streams"]
    stream = next((s for s in streams if s.get("codec_type") == "video"), streams[0])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if (source.suffix.lower() == ".mp4" and stream.get("codec_name") == "h264"
            and stream.get("pix_fmt") == "yuv420p"
            and (audio is None or audio.get("codec_name") == "aac")):
        return source
    preview.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=preview.parent) as temporary:
        output = Path(temporary) / "preview.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:a", "aac", "-movflags", "+faststart", str(output),
        ], check=True, capture_output=True)
        output.replace(preview)
    return preview


def _resolve_input(stem: str) -> Path | None:
    try:
        return resolve_video(stem)
    except FileNotFoundError:
        return None


def create_app(project: str | None = None) -> Flask:
    app = Flask(__name__)
    overrides = paths.artifact_overrides.get().copy()
    directory = None
    if project is not None:
        directory = (paths.project_override.get() or paths.PROJECTS_DIR / project).resolve()
        resolve_video(directory)

    @app.before_request
    def select_project():
        if project is None:
            return
        stem = (request.view_args or {}).get("stem")
        if stem is not None and stem != project:
            return "Not found", 404
        g.project_token = paths.project_override.set(directory)
        g.artifacts_token = paths.artifact_overrides.set(overrides)

    @app.teardown_request
    def release_project(error):
        artifact_token = g.pop("artifacts_token", None)
        if artifact_token is not None:
            paths.artifact_overrides.reset(artifact_token)
        token = g.pop("project_token", None)
        if token is not None:
            paths.project_override.reset(token)

    @app.route("/")
    def index():
        from .pipeline import masks_path
        from .planar_io import projection_path
        return render_template("editor.html", config={
            "stem": project,
            "has_transcription": artifact_path(project, "transcription").is_file(),
            "has_projections": projection_path(project).exists(),
            "has_masks": masks_path(project).exists(),
        })

    @app.route("/video/<stem>")
    def video(stem):
        p = _resolve_input(stem)
        if p is None:
            return "Not found", 404
        p = _browser_video(p)
        return send_from_directory(p.parent.resolve(), p.name)

    @app.route("/export", methods=["POST"])
    def export():
        data = request.get_json()
        filename = data.get("filename")
        if (not isinstance(filename, str) or not filename or filename in (".", "..")
                or "/" in filename or "\\" in filename or "\0" in filename):
            return jsonify({"error": "Invalid transcription name."}), 400
        if project is not None and filename != project:
            return jsonify({"error": "Choose the active project."}), 400
        try:
            arr = np.array(data["frames"], dtype=np.int64)
            if arr.ndim != 2 or arr.shape[1] != 6:
                raise ValueError
        except (KeyError, ValueError, TypeError, OverflowError):
            return jsonify({"error": "Expected an array with six string values per frame."}), 400
        path = artifact_path(filename, "transcription")
        buf = io.BytesIO()
        np.save(buf, arr)
        path.parent.mkdir(parents=True, exist_ok=True)
        if data.get("overwrite") is True:
            with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
                output = Path(temporary) / "transcription.npy"
                output.write_bytes(buf.getvalue())
                output.replace(path)
        else:
            try:
                with path.open("xb") as output:
                    output.write(buf.getvalue())
            except FileExistsError:
                return jsonify({"error": "Transcription already exists.", "exists": True}), 409
        return jsonify({"saved": str(path)})

    @app.route("/load/<stem>")
    def load_npy(stem):
        p = artifact_path(stem, "transcription")
        if not p.exists():
            return "Not found", 404
        try:
            arr = np.load(p, allow_pickle=False)
            if (arr.ndim != 2 or arr.shape[1] != 6
                    or arr.dtype.kind not in "biuf"
                    or not np.isfinite(arr).all() or (arr < 0).any()):
                raise ValueError
        except (ValueError, OSError, TypeError, EOFError):
            return jsonify({"error": "Expected a numeric .npy array with six string values per frame."}), 400
        return jsonify({"frames": arr.tolist()})

    @app.route("/strings/<stem>")
    def string_projections(stem):
        from .config import NUM_STRINGS
        from .planar import load_planar_track
        from .planar_io import projection_path

        if projection_path(stem).exists():
            track, meta = load_planar_track(projection_path(stem))
            lines = [
                frame.projected_lines.reshape(NUM_STRINGS, 4).astype(int).tolist()
                if np.isfinite(frame.projected_lines).all() else None
                for frame in track.frames
            ]
            return jsonify({**meta, "lines": lines})

        return (f"No cached planar projections for '{stem}'. "
                f"Run: uv run string-lights generate-poses {stem}", 400)

    return app
