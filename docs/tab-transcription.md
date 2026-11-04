# Generating a transcription from annotated tablature

`scripts/generate_transcription.py` converts a manually annotated score JSON into
the editor's six-column `.npy` highlight format and a JSON timing/provenance
sidecar. It does not automatically read notes or rhythm from a PDF.

Create a score file, for example `data/projects/example/tmp/score.json`:

```json
{
  "source": "Manually reviewed tablature",
  "bpm": 120,
  "beats_per_bar": 3,
  "first_bar": 5,
  "capo": 0,
  "bars": {
    "5": [[0, "A0 D2 G2", "strum"], [1.5, "e3"], [2, "EX AX", "mute"]],
    "6": [[0, "E3"], [1, "B0"], [2, "G0"]]
  }
}
```

Each entry is `[zero_based_beat, notes, optional_technique]`. Use quarter-note
beats and quarter-note BPM; `beats_per_bar` defaults to 4. Notes combine a
case-sensitive string name (`E A D G B e`, lowest to highest) and a nonnegative
fret or `X` for a muted strike. Frets and technique descriptions are retained in
the sidecar; the highlight array records string activation, not pitch.

```bash
uv run python scripts/generate_transcription.py data/projects/example/tmp/score.json \
  --output data/projects/example/transcription.npy
```

Options:

- `--bpm` and `--beats-per-bar`: override score timing.
- `--first-bar` and `--last-bar`: select the included range. The default first
  bar is the score's `first_bar`, or its earliest annotated bar; the default last
  bar is its latest annotated bar. Unannotated bars within the range remain silent.
- `--offset`: seconds of silence before the first included bar, default 0.
- `--pulse`: highlight duration in seconds, default 0.09.
- `--include-percussion`: include muted strikes. Otherwise they are omitted while
  retaining any simultaneous pitched notes.

Output uses the application's existing 44 Hz transcription grid, independent of
the video's frame rate. A strum activates its strings simultaneously. Slides,
hammer-ons and pull-offs described in technique text do not create extra onsets;
add separate beat entries when they should retrigger a highlight.

Review inferred rhythm and align the result with the recording in the editor.
Keep score JSON and generated output stems different: output writes both `.npy`
and `.json` and replaces existing output files.
