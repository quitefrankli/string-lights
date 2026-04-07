import io
import numpy as np
from flask import Flask, request, send_file, render_template


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("editor.html")

    @app.route("/export", methods=["POST"])
    def export():
        data = request.get_json()
        # frames: list of [s0, s1, s2, s3, s4, s5] per frame
        arr = np.array(data["frames"], dtype=np.int64)  # (N, 6)
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        filename = data.get("filename", "clip1")
        return send_file(buf, mimetype="application/octet-stream",
                         as_attachment=True, download_name=f"{filename}.npy")

    return app
