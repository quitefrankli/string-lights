# Annotated tablature format for duration experiments

This records the manually annotated score format used by historical duration
experiments. The standalone score-to-transcription generator has been removed;
new transcriptions use the audio generation and optional PDF refinement commands.

For estimating hold lengths from PDF note groups and spacing while preserving
an existing recording's note onsets, follow the
[note-duration workflow](note-duration-workflow.md). It includes the preferred
`cupid_light_duration` experiment, alignment rules, and reproduction commands.

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

To control note length, add a fourth field in quarter-note beats:
`[beat, notes, technique, duration_beats]`. For example,
`[0, "A0", "pluck", 1.5]` sustains A for 1.5 beats, while
`[0, "e3", "pluck", 0.5]` lights high e for half a beat at the same onset.
Use separate entries at the same beat for strings with different lengths.
Durations must be finite and positive; entries without a duration retain the
experiment’s fallback pulse duration. Durations can cross bar boundaries and are clipped at the end
of the selected output range. Overlapping activations on the same string merge;
leave a gap when repeated notes should visibly retrigger. Durations and clipped,
quantized end times are included in the JSON sidecar.

The renderer holds the light throughout each activation, then adds its existing
0.5-second fade. The duration field controls the hold, not that fade.

The application's transcription grid uses 44 Hz, independent of video frame rate.
Review inferred rhythm against the recording. See the duration workflow for
project-specific alignment that preserves existing note onsets.
