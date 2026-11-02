import subprocess
import threading
import hashlib
from werkzeug.utils import secure_filename
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
    for directory in (INPUT_DIR / "uploads", INPUT_DIR):
        for ext in VIDEO_EXTS:
            p = directory / (stem + ext)
            if p.exists():
                return p
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
        out = OUTPUT_DIR / f"{stem}.lyrics.mp4"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
            _render.update(state="done", progress=1.0, error=None)

    except Exception as exc:
        with _render_lock:
            _render.update(state="error", error=str(exc))


def create_lyrics_app() -> Flask:
    from .lyrics import load_lyrics, save_lyrics

    app = Flask(__name__)

    @app.post("/upload")
    def upload():
        file = request.files["video"]
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_EXTS:
            return jsonify(error="Unsupported video format"), 400
        name = secure_filename(file.filename or "")
        base = Path(name).stem or "video"
        directory = INPUT_DIR / "uploads"
        directory.mkdir(parents=True, exist_ok=True)
        content = file.read()
        digest = hashlib.sha256(content).digest()
        stem = base
        suffix = 2
        while any((directory / (stem + extension)).exists() for extension in VIDEO_EXTS):
            target = directory / (stem + ext)
            if target.exists() and hashlib.sha256(target.read_bytes()).digest() == digest:
                return jsonify(stem=stem)
            stem = f"{base}_{suffix}"
            suffix += 1
        (directory / (stem + ext)).write_bytes(content)
        return jsonify(stem=stem)

    @app.get("/download/<stem>")
    def download(stem):
        return send_from_directory(OUTPUT_DIR.resolve(), stem + ".lyrics.mp4", as_attachment=True)


    @app.route("/")
    def index():
        return render_template("lyrics_editor.html")

    @app.route("/list")
    def list_videos():
        seen: dict[str, Path] = {}
        for directory in (INPUT_DIR / "uploads", INPUT_DIR):
            for ext in VIDEO_EXTS:
                for p in directory.glob(f"*{ext}"):
                    seen.setdefault(p.stem, p)
        return jsonify({"items": [{"stem": s} for s in sorted(seen)]})

    @app.route("/video/<stem>")
    def video(stem):
        from .editor import _browser_video

        p = _resolve_input(stem)
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
        threading.Thread(target=_do_render, args=(stem, aspect, export_format), daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/render-status")
    def render_status():
        with _render_lock:
            return jsonify(dict(_render))

    return app
