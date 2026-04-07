import io
from pathlib import Path

import numpy as np
from flask import Flask, request, send_file, send_from_directory, render_template

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".flv"]


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("editor.html")

    @app.route("/video/<stem>")
    def video(stem):
        input_dir = Path("data/input")
        for ext in VIDEO_EXTS:
            p = input_dir / (stem + ext)
            if p.exists():
                return send_from_directory(input_dir.resolve(), p.name)
        return "Not found", 404

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

    return app
