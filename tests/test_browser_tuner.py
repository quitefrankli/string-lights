import json
import re
from pathlib import Path

import pytest

from string_lights.planar_tuner import create_tuner_app


VIDEO = Path(__file__).resolve().parents[1] / "data/input/littleroot_town.mp4"
ANNOTATION = VIDEO.parents[1] / "components/profiles/littleroot_town.json"


def test_browser_tuner_validates_before_overwriting(tmp_path):
    if not VIDEO.exists() or not ANNOTATION.exists():
        pytest.skip("local video and annotation required")
    target = tmp_path / "profile.json"
    original = ANNOTATION.read_text()
    target.write_text(original)
    client = create_tuner_app(VIDEO, target).test_client()
    page = client.get("/").get_data(as_text=True)
    token = json.loads(re.search(r"window.tunerToken = (.+);", page).group(1))
    assert client.get("/annotation").json["total"] > 150
    assert client.get("/frame/0").mimetype == "image/jpeg"
    assert client.post("/save", json={}).status_code == 403
    invalid = json.loads(original)
    invalid["string_lines"] = []
    response = client.post("/save", json=invalid, headers={"X-Tuner-Token": token})
    assert response.status_code == 400
    assert target.read_text() == original
    assert not target.with_suffix(".npz").exists()
    response = client.post("/save", json=json.loads(original), headers={"X-Tuner-Token": token})
    assert response.status_code == 200
    assert target.with_suffix(".npz").is_file()
