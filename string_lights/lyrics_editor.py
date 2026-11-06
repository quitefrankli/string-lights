import subprocess
import threading
from contextvars import copy_context
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, g, jsonify, request, send_from_directory, render_template

from . import paths
from .paths import artifact_path, resolve_video

# Single-slot render status (one render at a time is fine for local use)
_render: dict = {"state": "idle", "stem": None, "progress": 0.0, "error": None}
_render_lock = threading.Lock()


def _resolve_input(stem: str) -> Path | None:
    try:
        return resolve_video(stem)
    except FileNotFoundError:
        return None


def _base_video(stem: str) -> Path | None:
    return _resolve_input(stem)


def reels_video_bounds(w: int, h: int) -> tuple[int, int, int, int]:
    scale = min(1080 / w, 1920 / h)
    width, height = min(1080, round(w * scale)), min(1920, round(h * scale))
    return (1080 - width) // 2, (1920 - height) // 2, width, height


def original_layout_lyrics(lyrics: list[dict], w: int, h: int) -> list[dict]:
    """Map the saved portrait-canvas layout into the source video's rectangle."""
    left, top, width, height = reels_video_bounds(w, h)
    scale = w / width
    result = []
    for lyric in lyrics:
        item = {**lyric,
                "x": (lyric.get("x", .5) * 1080 - left) / width,
                "y": (lyric.get("y", .85) * 1920 - top) / height}
        for key, default in (("size", 48), ("halo_size", 4),
                             ("background_padding", 16), ("background_softness", 24)):
            item[key] = lyric.get(key, default) * scale
        result.append(item)
    return result


def fit_reels_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, width, height = reels_video_bounds(w, h)
    scale = min(1080 / w, 1920 / h)
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((1920, 1080, 3), dtype=np.uint8)
    canvas[y:y + height, x:x + width] = resized
    return canvas


def _do_render(stem: str, aspect: str = "padded", export_format: str = "source_audio") -> None:
    from .lyrics import load_lyrics, draw_lyrics_frame

    try:
        base = _base_video(stem)
        if base is None:
            raise FileNotFoundError(f"No video found for '{stem}'")

        lyrics = load_lyrics(stem)
        out = artifact_path(stem, "lyrics_render")
        out.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(base))
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_w, out_h = (1080, 1920) if aspect == "padded" else (w, h)
        if aspect == "original":
            lyrics = original_layout_lyrics(lyrics, w, h)
        audio_options = ["-c:a", "aac", "-b:a", "320k"] if export_format == "compatible" else ["-c:a", "copy"]

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps:.6f}",
            "-i", "pipe:0",
            "-i", str(base),
            "-map", "0:v:0", "-map", "1:a?", *audio_options,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if aspect == "padded":
                frame = fit_reels_frame(frame)
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
            _render.update(state="done", progress=1.0, error=None, output=str(out))

    except Exception as exc:
        with _render_lock:
            _render.update(state="error", error=str(exc))


def create_lyrics_app(filename: str | None = None) -> Flask:
    from .lyrics import load_lyrics, save_lyrics

    app = Flask(__name__)

    source = resolve_video(filename).resolve() if filename is not None else None
    stem_name = paths.project_dir(source).name if source is not None else None
    directory = paths.project_dir(source).resolve() if source is not None else None
    overrides = paths.artifact_overrides.get().copy()

    @app.before_request
    def select_project():
        if source is None:
            return
        stem = (request.view_args or {}).get("stem")
        if stem is not None and stem != stem_name:
            return "Not found", 404
        g.project_token = paths.project_override.set(directory)
        g.artifacts_token = paths.artifact_overrides.set(overrides)

    @app.teardown_request
    def release_project(error):
        for key, variable in (("project_token", paths.project_override),
                              ("artifacts_token", paths.artifact_overrides)):
            token = g.pop(key, None)
            if token is not None:
                variable.reset(token)

    @app.get("/download/<stem>")
    def download(stem):
        target = artifact_path(stem, "lyrics_render")
        return send_from_directory(target.parent.resolve(), target.name, as_attachment=True)


    @app.route("/")
    def index():
        return render_template("lyrics_editor.html", config={"stem": stem_name})

    @app.route("/video/<stem>")
    def video(stem):
        from .editor import _browser_video

        p = source if source is not None else _resolve_input(stem)
        if p is None:
            return "Not found", 404
        p = _browser_video(p)
        return send_from_directory(p.parent.resolve(), p.name)

    @app.route("/lyrics/<stem>", methods=["GET"])
    def get_lyrics(stem):
        return jsonify(load_lyrics(stem))

    @app.route("/lyrics/<stem>", methods=["POST"])
    def post_lyrics(stem):
        values = request.get_json()
        if not isinstance(values, list):
            return jsonify(error="Expected a lyric array"), 400
        for item in values:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                return jsonify(error="Invalid lyric text"), 400
            for key in ("start", "end", "x", "y", "size", "fade"):
                value = item.get(key, {"x": .5, "y": .85, "size": 48, "fade": .25}.get(key))
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    return jsonify(error="Invalid lyric number"), 400
            if item["start"] < 0 or item["end"] <= item["start"] or item.get("fade", .25) < 0 or not 10 <= item.get("size", 48) <= 300:
                return jsonify(error="Invalid lyric timing or size"), 400
            for key, default, maximum in (("halo_size", 4, 30), ("halo_intensity", .75, 1)):
                value = item.get(key, default)
                if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= maximum:
                    return jsonify(error="Invalid halo setting"), 400
            for key, default, maximum in (("background_opacity", 0, 1), ("background_padding", 16, 120), ("background_softness", 24, 120)):
                value = item.get(key, default)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= maximum:
                    return jsonify(error="Invalid lyric background setting"), 400
            color = item.get("color", [255,255,255])
            if not isinstance(color, list) or len(color) != 3 or any(not isinstance(c, int) or not 0 <= c <= 255 for c in color):
                return jsonify(error="Invalid lyric color"), 400
        save_lyrics(stem, values)
        return jsonify({"ok": True})

    @app.route("/render/<stem>", methods=["POST"])
    def start_render(stem):
        options = request.get_json(silent=True)
        if options is None:
            options = {}
        if not isinstance(options, dict) or options.get("aspect", "padded") not in ("original", "padded"):
            return jsonify(error="Choose original or padded aspect ratio"), 400
        aspect = options.get("aspect", "padded")
        export_format = options.get("format", "source_audio")
        if export_format not in ("compatible", "source_audio"):
            return jsonify(error="Choose compatible or source_audio export format"), 400
        with _render_lock:
            if _render["state"] == "running":
                return jsonify({"error": "render already in progress"}), 409
            _render.update(state="running", stem=stem, progress=0.0, error=None)
        context = copy_context()
        threading.Thread(
            target=lambda *args: context.run(_do_render, *args),
            args=(str(source) if source else stem, aspect, export_format), daemon=True,
        ).start()
        return jsonify({"ok": True})

    @app.route("/render-status")
    def render_status():
        with _render_lock:
            return jsonify(dict(_render))

    return app
