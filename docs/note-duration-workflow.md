# Estimating string-light holds from tablature

This workflow produced the preferred `cupid_light_duration` trial. It uses
manually reviewed PDF note groups and spacing to estimate rhythmic lengths,
then changes hold lengths while preserving the existing transcription's onsets.
The PDF is the reference; the values remain estimates where rhythm is unwritten.

## 1. Preserve a baseline and the reference

Create a separate project with independent copies of the source video, profile,
transcription, lyrics, masks, and planar tracking. Include any compiled profile,
mask metadata, and seed that exist. Keep the reference PDF at project level.
For this experiment, identical PDFs live at:

- `data/projects/cupid/Cupid.pdf`
- `data/projects/cupid_light_duration/Cupid.pdf`

Use the prepared `source.mp4` when both MP4 and MOV exist. Reuse the baseline's
masks and tracking so duration is the only visual change. Put all trial scripts,
annotations, transcriptions, reports, page images, and renders under the new
project's `tmp/`. Use explicit CLI artifact overrides; leave the baseline
transcription and finished renders intact.

## 2. Review the PDF and annotate rhythm

Render every relevant page and inspect the notation visually. Text extraction
alone misses tablature that is embedded as images. For Cupid:

```bash
pdftoppm -scale-to 2200 -png data/projects/cupid_light_duration/Cupid.pdf \
  data/projects/cupid_light_duration/tmp/pdf-page
```

Prefer explicit written note values when present. Cupid's PDF has fret numbers,
techniques, barlines, and some rests, but no duration stems or beams. For this
trial, infer likely positions on a quarter/eighth/sixteenth-note grid using
group counts and relative spacing within an assumed 4/4 bar:

- Count vertically stacked chord notes as one rhythmic group.
- Four evenly spaced groups suggest one beat each; eight suggest half a beat.
- Use wider gaps to infer longer values, checking that the bar totals four beats.
- Include visible rests and muted strikes as rhythm boundaries even when they
  do not create a highlight.
- Review slides, hammer-ons, and pull-offs as connected phrases. Do not add a
  new light onset solely because a legato destination appears in the PDF.

Use the gap to the next score group, rest, mute, or barline as the initial
rhythmic hold estimate. Connected legato can span a destination. This estimates
rhythmic occupancy, not how long the acoustic string actually rings. Spacing is
not guaranteed proportional to time; record ambiguous interpretations.

Store entries as `[beat, notes, technique, duration_beats]`, as described in
[the transcription format](tab-transcription.md). The Cupid trial builds on
`tmp/cupid_score.json`, the prior manual PDF onset annotation, and records
additional rest/mute boundaries and legato destinations in its preparation
script. The emitted `pdf_spacing_score.json` contains the inferred durations.

## 3. Align the score to the existing transcription

Identify recording bar starts from recognizable bass notes and chord patterns.
Use a following bar-start anchor to define the preceding bar's duration:

```text
seconds_per_beat = (next_bar_start - bar_start) / beats_per_bar
expected_onset = bar_start + score_beat * seconds_per_beat
hold_seconds = duration_beats * seconds_per_beat
```

This accommodates timing differences across bars instead of imposing the PDF's
printed tempo on the whole recording. The final covered bar also needs a next
boundary estimate. Anchors in this trial were inferred from the existing
transcription, without audio pitch verification.

Check string-column order against recognizable notes before matching. The
existing Cupid baseline is stored high e to low E, while the generic score
generator writes low E to high e. The project-specific preparation script
handles this difference; do not reverse the baseline or change the generic
generator for this experiment.

Match score events to baseline activations in order, separately for each string.
The Cupid script uses monotonic sequence matching with a maximum 0.28-second
timing discrepancy. Its match cost is `timing_error / 0.16`; skipping either a
baseline event or a score event costs 1. Unmatched baseline notes retain their
original holds. Unmatched score events do not create extra lights.

Keep every matched note's original start frame. Convert its inferred hold to
the existing 44 Hz grid, with at least one active frame. Clip the hold before
the next activation on the same string, leaving at least one inactive grid
frame so the original activation boundaries remain distinguishable.

The renderer keeps a string lit during the hold, then applies its existing
0.5-second fade. That tail is additional to the inferred hold and can blend
nearby strikes. Keep it unchanged for a comparison of hold lengths.

## 4. Generate and validate the trial

Run from the repository root:

```bash
uv run python data/projects/cupid_light_duration/tmp/prepare_pdf_spacing_trial.py
```

This is a project-specific experiment script, not an automatic PDF reader or a
general CLI command. Its manual inputs are `cupid_score.json`, `ANCHORS`,
`EXTRA_BOUNDARIES`, and `LEGATO`. Adapting it to another recording requires
reviewing those inputs and the string order again.

It writes three files under the same `tmp/` directory:

| File | Purpose |
| --- | --- |
| `pdf_spacing_trial.npy` | Baseline-aligned highlight grid used for rendering |
| `pdf_spacing_trial.json` | Per-note alignment, holds, confidence, clipping, unmatched events, and source checksums |
| `pdf_spacing_score.json` | Nominal inferred score durations before recording alignment |

Use the aligned `.npy` for this trial. Passing the nominal score through the
generic generator alone would lose the bar-local timing and original onsets.

Validate that array shape and all per-string start frames match the baseline,
holds are positive, and neighboring activations remain separate. The script
asserts start-frame equality. Compare source/cache checksums to confirm the
experiment has not changed the baseline assets.

Review the JSON's unmatched events and timing errors. Its `medium` label means
alignment error at most 0.15 seconds; `low` means a larger error up to 0.28
seconds. These labels describe alignment, not certainty about musical duration.

## 5. Render and compare

Render the full trial with the unchanged project masks and tracking:

```bash
uv run string-lights render-strings cupid_light_duration \
  --transcription data/projects/cupid_light_duration/tmp/pdf_spacing_trial.npy \
  --output data/projects/cupid_light_duration/tmp/pdf-spacing-strings.mp4
```

For a short comparison, render the same first 360 frames twice: once with
`--transcription data/projects/cupid_light_duration/transcription.npy` and once
with the trial path above. Give both outputs explicit paths under `tmp/`.
Use identical frame limits, source, masking, tracking, and fade settings.

The reviewed outputs from this experiment are:

- `tmp/pdf-spacing-strings.mp4`: full 57.7-second trial.
- `tmp/pdf-spacing-comparison.mp4`: first 12 seconds side by side, original holds
  on the left and PDF-spacing estimates on the right.
- `tmp/pdf-spacing-experiment.md`: project-specific notes and results.

Check short plucks, long gaps, chord strums, legato, and muted passages. Watch
for merged pulses from the fade and overly long holds caused by ambiguous
spacing. Refine the score interpretation or alignment anchors and regenerate;
keep the original baseline available for comparison.

## Cupid result

The preferred trial matched 250 of 258 activations, changed 247 hold lengths,
and retained the original holds for eight unmatched notes. Of the matches,
235 have timing errors at most 0.15 seconds and 15 have larger errors up to
0.28 seconds. All original onsets and columns are preserved. Inferred values
span 0.25, 0.5, 0.75, 1, and 1.5 beats; the longest quantized holds are about
0.84 seconds in the sparse opening.

This replaces the earlier experiment that used 80% of the time until the next
strike on each string, capped at 1.5 beats. Use the PDF-based workflow for
further duration work. The accepted visual result remains in `tmp/`; baseline
transcriptions and finished exports have not been replaced.
