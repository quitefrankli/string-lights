import io
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory, render_template

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".m4v"]
INPUT_DIR = Path("data/input")


def _resolve_input(stem: str) -> Path | None:
    for ext in VIDEO_EXTS:
        p = INPUT_DIR / (stem + ext)
        if p.exists():
            return p
    return None


def _list_inputs() -> list[dict]:
    from .pipeline import poses_path, masks_path

    seen: dict[str, Path] = {}
    for ext in VIDEO_EXTS:
        for p in INPUT_DIR.glob(f"*{ext}"):
            # skip output-suffixed files like clip0.output.mp4
            if any(part for part in p.stem.split(".")[1:]):
                continue
            seen.setdefault(p.stem, p)
    items = []
    for stem in sorted(seen):
        items.append({
            "stem": stem,
            "has_poses": poses_path(stem).exists(),
            "has_masks": masks_path(stem).exists(),
        })
    return items


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("editor.html")

    @app.route("/list")
    def list_videos():
        return jsonify({"items": _list_inputs()})

    @app.route("/video/<stem>")
    def video(stem):
        p = _resolve_input(stem)
        if p is None:
            return "Not found", 404
        return send_from_directory(INPUT_DIR.resolve(), p.name)

    @app.route("/export", methods=["POST"])
    def export():
        data = request.get_json()
        arr = np.array(data["frames"], dtype=np.int64)  # (N, 6)
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        filename = data.get("filename", "clip1")
        return send_file(buf, mimetype="application/octet-stream",
                         as_attachment=True, download_name=f"{filename}.npy")

    @app.route("/strings/<stem>")
    def string_projections(stem):
        from .pipeline import load_poses, poses_path
        from .board import camera_matrix
        from .strings import _project_string
        from .config import NUM_STRINGS

        if not poses_path(stem).exists():
            return (f"No cached poses for '{stem}'. Run: uv run sl run {stem} --poses", 400)

        poses, meta = load_poses(stem)
        K = camera_matrix(meta["w"], meta["h"])

        lines_per_frame: list[list[list[int]] | None] = []
        for rvec, tvec in poses:
            if rvec is None:
                lines_per_frame.append(None)
                continue
            lines = []
            for i in range(NUM_STRINGS):
                a, b = _project_string(i, rvec, tvec, K)
                lines.append([int(a[0]), int(a[1]), int(b[0]), int(b[1])])
            lines_per_frame.append(lines)

        return jsonify({
            "fps": meta["fps"],
            "total": meta["total"],
            "w": meta["w"],
            "h": meta["h"],
            "lines": lines_per_frame,
        })

    return app
