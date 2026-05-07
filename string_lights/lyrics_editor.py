import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, render_template

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".m4v"]
INPUT_DIR  = Path("data/input")
OUTPUT_DIR = Path("data/output")

# Single-slot render status (one render at a time is fine for local use)
_render: dict = {"state": "idle", "stem": None, "progress": 0.0, "error": None}
_render_lock = threading.Lock()


def _resolve_input(stem: str) -> Path | None:
    for ext in VIDEO_EXTS:
        p = INPUT_DIR / (stem + ext)
        if p.exists():
            return p
    return None


def _base_video(stem: str) -> Path | None:
    """Prefer the already-processed output; fall back to raw input."""
    processed = OUTPUT_DIR / f"{stem}.mp4"
    if processed.exists():
        return processed
    return _resolve_input(stem)


def _do_render(stem: str) -> None:
    from .lyrics import load_lyrics, draw_lyrics_frame

    try:
        base = _base_video(stem)
        if base is None:
            raise FileNotFoundError(f"No video found for '{stem}'")

        lyrics = load_lyrics(stem)
        out = OUTPUT_DIR / f"{stem}.lyrics.mp4"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(base))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
            "-i", "pipe:0",
            "-i", str(base),
            "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            str(out),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            draw_lyrics_frame(frame, i, fps, lyrics)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            with _render_lock:
                _render["progress"] = (i + 1) / total

        cap.release()
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with code {rc}")

        with _render_lock:
            _render.update(state="done", progress=1.0, error=None)

    except Exception as exc:
        with _render_lock:
            _render.update(state="error", error=str(exc))


def create_lyrics_app() -> Flask:
    from .lyrics import load_lyrics, save_lyrics

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("lyrics_editor.html")

    @app.route("/list")
    def list_videos():
        seen: dict[str, Path] = {}
        for ext in VIDEO_EXTS:
            for p in INPUT_DIR.glob(f"*{ext}"):
                if any(part for part in p.stem.split(".")[1:]):
                    continue
                seen.setdefault(p.stem, p)
        return jsonify({"items": [{"stem": s} for s in sorted(seen)]})

    @app.route("/video/<stem>")
    def video(stem):
        p = _resolve_input(stem)
        if p is None:
            return "Not found", 404
        return send_from_directory(INPUT_DIR.resolve(), p.name)

    @app.route("/lyrics/<stem>", methods=["GET"])
    def get_lyrics(stem):
        return jsonify(load_lyrics(stem))

    @app.route("/lyrics/<stem>", methods=["POST"])
    def post_lyrics(stem):
        save_lyrics(stem, request.get_json())
        return jsonify({"ok": True})

    @app.route("/render/<stem>", methods=["POST"])
    def start_render(stem):
        with _render_lock:
            if _render["state"] == "running":
                return jsonify({"error": "render already in progress"}), 409
            _render.update(state="running", stem=stem, progress=0.0, error=None)
        threading.Thread(target=_do_render, args=(stem,), daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/render-status")
    def render_status():
        with _render_lock:
            return jsonify(dict(_render))

    return app
