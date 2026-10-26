"""Local browser interface for editing a video's planar reference annotation."""

import io
import json
import secrets
import tempfile
import webbrowser
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_file

from .planar import save_planar_profile
from .planar_io import _read_video_frame, create_planar_profile_from_annotation


def create_tuner_app(video_path: str | Path, annotation_path: str | Path) -> Flask:
    video_path = Path(video_path).resolve()
    annotation_path = Path(annotation_path).resolve()
    app = Flask(__name__)
    token = secrets.token_urlsafe(32)

    @app.get("/")
    def index():
        return render_template("planar_tuner.html", title=video_path.stem, token=token)

    @app.get("/annotation")
    def annotation():
        cap = cv2.VideoCapture(str(video_path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()
        return jsonify(annotation=json.loads(annotation_path.read_text()), total=total, fps=fps)

    @app.get("/frame/<int:index>")
    def frame(index):
        try:
            pixels = _read_video_frame(video_path, index)
            ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise ValueError("Could not encode frame")
            return send_file(io.BytesIO(encoded.tobytes()), mimetype="image/jpeg")
        except ValueError as error:
            return jsonify(error=str(error)), 400

    @app.post("/save")
    def save():
        if request.headers.get("X-Tuner-Token") != token:
            return jsonify(error="Invalid session token. Reload the tuner."), 403
        temporary = None
        try:
            data = request.get_json()
            # Validate and build the profile before touching the existing annotation.
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as output:
                temporary = Path(output.name)
                json.dump(data, output, allow_nan=False)
            profile = create_planar_profile_from_annotation(video_path, temporary)
            target = annotation_path.with_suffix(".npz")
            save_planar_profile(target, profile)
            annotation_path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
            return jsonify(message="Annotation and planar profile saved. Re-run planar-track to update cached projections.")
        except (KeyError, OSError, TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    return app


def run_browser_tuner(video_path: str | Path, annotation_path: str | Path) -> None:
    from werkzeug.serving import make_server

    app = create_tuner_app(video_path, annotation_path)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Planar tuner: {url}\nPress Ctrl+C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
