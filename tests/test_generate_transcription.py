import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.generate_transcription import generate


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/generate_transcription.py"


@pytest.mark.parametrize("include_percussion", [False, True])
def test_song_independent_bar_timing_and_string_columns(include_percussion):
    frames, metadata, duration = generate(
        {"bars": {"5": [[0, "A0 e3"], [1.5, "EX DX B1"]], "6": [[0, "G2"]]}},
        bpm=120, first_bar=None, last_bar=None, beats_per_bar=3,
        offset=.5, pulse=.09, include_percussion=include_percussion,
    )
    assert duration == 3.5
    assert frames.shape == (154, 6)
    np.testing.assert_array_equal(np.flatnonzero(frames[22]), [1, 5])
    np.testing.assert_array_equal(np.flatnonzero(frames[55]), [0, 2, 4] if include_percussion else [4])
    np.testing.assert_array_equal(np.flatnonzero(frames[88]), [3])
    assert not frames[:22].any()
    assert [event["seconds"] for event in metadata["events"]] == [.5, 1.25, 2]


def test_cli_selects_bar_range_and_writes_metadata(tmp_path):
    source = tmp_path / "score.json"
    source.write_text(json.dumps({
        "bpm": 90, "capo": 3,
        "bars": {"8": [[0, "E0"]], "9": [[0, "e5", "slide", .5]]},
    }))
    output = tmp_path / "result.npy"
    subprocess.run([
        sys.executable, str(SCRIPT), str(source), "--output", str(output),
        "--first-bar", "9", "--last-bar", "9", "--bpm", "120",
    ], check=True, capture_output=True)
    frames = np.load(output)
    assert frames.shape == (88, 6)
    assert frames[0, 5] == 1
    assert frames[:11, 5].all() and not frames[11:, 5].any()
    assert not frames[:, 0].any()
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["bpm"] == 120
    assert metadata["capo"] == 3
    assert metadata["events"][0]["technique"] == "slide"
    assert metadata["events"][0]["duration_seconds"] == .25


@pytest.mark.parametrize("entries", [[[3, "E0"]], [[0, "H2"]], [[-1, "e0"]], [[0, ""]]])
def test_invalid_annotation_rejected(entries):
    with pytest.raises(ValueError):
        generate({"bars": {"1": entries}}, bpm=120, first_bar=None,
                 last_bar=None, beats_per_bar=3, offset=0, pulse=.09,
                 include_percussion=False)


def test_independent_durations_cross_bars_and_clip_at_selected_end():
    frames, metadata, _ = generate(
        {"bars": {"1": [[0, "A0", "pluck", .5], [0, "e3", "pluck", 2],
                         [2.5, "D0", "pluck", 1]],
                  "2": [[2.5, "G0", "pluck", 2], [0, "B0"]]}},
        bpm=120, first_bar=1, last_bar=2, beats_per_bar=3,
        offset=.5, pulse=.09, include_percussion=False,
    )
    assert frames[22:33, 1].all() and not frames[33:, 1].any()
    assert frames[22:66, 5].all() and not frames[66:, 5].any()
    assert frames[77:99, 2].all()  # continues into bar 2
    assert frames[143:, 3].all()  # clipped to output end
    assert frames[88:92, 4].all() and not frames[92:, 4].any()
    assert metadata["events"][0]["duration_seconds"] == .25
    assert metadata["events"][3]["end_seconds"] == 3.5


@pytest.mark.parametrize("duration", [0, -1, True, None, "1", float("nan"), float("inf")])
def test_invalid_duration_rejected(duration):
    with pytest.raises(ValueError, match="duration_beats"):
        generate({"bars": {"1": [[0, "E0", "pluck", duration]]}},
                 bpm=120, first_bar=None, last_bar=None, beats_per_bar=4,
                 offset=0, pulse=.09, include_percussion=False)
