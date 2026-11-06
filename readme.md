# string-lights

Track guitar strings with planar image features and render string highlights and lyrics.

## Usage

`uv run string-lights render-strings cupid`

## Project files and custom paths

Each clip lives in `data/projects/<project>/`. A project name resolves to
`source.mp4` (preferred), `source.mov`, or another supported source extension.
Profiles (`profile.json`, `profile.npz`), `transcription.npy`, and `lyrics.json`
live at project level. Tracking caches live in `tracking/`, hand masks in
`masks/`, and final exports in `renders/`. Put previews and experiments in `tmp/`.
See [AGENTS.md](AGENTS.md) for the complete layout and agent guidance.

```bash
uv run string-lights render-strings cupid
uv run string-lights render-strings cupid --output data/projects/cupid/tmp/test.mp4
uv run string-lights generate-masks cupid --output data/projects/cupid/tmp/trial.npz
uv run string-lights render-strings cupid --masks data/projects/cupid/tmp/trial.npz \
  --output data/projects/cupid/tmp/trial-render.mp4
uv run string-lights render-strings /path/to/clip.mov --project-dir data/projects/cupid \
  --projections /path/to/custom-tracking.npz \
  --transcription /path/to/custom-tabs.npy --output /path/to/result.mp4
uv run string-lights --projects-dir /path/to/projects tabs-editor cupid
```

File commands accept `--project-dir` and explicit artifact overrides (see each
command's `--help`). `generate-masks`, `planar-profile`, and `generate-poses` accept
`--output`. `render-strings` accepts `--output`, `--projections`, `--masks`,
`--mask-metadata`, and `--transcription`. `tune-guitar-plane` accepts `--annotation`;
`mask-edit` accepts `--masks`, `--mask-metadata`, and `--output`.
A custom mask file defaults to a sibling `<stem>.meta.npz` metadata file;
`generate-masks --metadata-output` or `render-strings --mask-metadata` can override it.
`--debug-masks` writes `tmp/masks-debug.mp4` unless `--output` is supplied.

Launch the tabs editor for one project with `uv run string-lights tabs-editor cupid`. It loads `transcription.npy` by default; use `--transcription /path/to/take.npy` to load and save a different file.
A project name is required; the editor reads and saves that project's files.

## Useful Commands

### Generate caches without rendering

```bash
uv run string-lights generate-poses cupid --frames 500
uv run string-lights generate-masks cupid --frames 500
```

Mask generation uses SAM2Matting Base+ video tracking, with soft opacity caches
(0–255) for natural hand boundaries. Public weights download on first use; CUDA is required.
Frames are decoded from disk on demand with a two-frame cache; forward tracking
retains only its required recent memory plus the seed frame.
Supply a first-frame grayscale hand mask (white hands, black background):

```bash
uv run string-lights generate-masks VIDEO --seed-mask seed.png
```

The default seed path is `data/projects/VIDEO/masks/seed.png`.
A motion-aligned three-frame median reduces isolated opacity flicker by default;
use `--no-stabilize` to retain raw model output. Stable gaps can remain open,
but this filter does not correct persistent segmentation errors.
Tracking processes every frame; `--frame-skip N` samples the masks saved to cache
(default: 1). Skipped frames hold the latest cached mask. Use `--force` to regenerate
an existing cache. Old binary caches must be regenerated.
The vendored SAM2Matting code and weights use CC BY-NC-SA 4.0 for non-commercial research.

`generate-poses` generates markerless projections. Both tracking and masking
commands accept `--force` to replace existing caches.

Render with the projection and mask caches:

```bash
uv run string-lights render-strings cupid --frames 500
```

Rendering uses four frame-compositing threads by default. Use `--workers 1`
for serial rendering or `--workers N` to choose another count. Frames are
encoded in order, with at most twice the worker count queued. Fade timing,
masking and encoding settings are preserved. String-core compositing and hand
mask blending are limited to the region affected by the lights. OpenCV's internal processing
threads are limited while the worker pool runs to avoid nested thread pools.
The debug-mask preview remains serial.

`render-strings` always reads existing planar projection caches. Masks are streamed
sequentially from the compressed cache unless `--disable-masking` is supplied.
Only the current mask and a small decompression buffer are held in memory;
short previews stop reading masks at their frame limit. No expanded mask file is
written to disk. If a mask cache ends before
the video, remaining frames render without hand masking. If planar projections
end early, remaining frames hold the last valid cached projection. Both fallbacks
emit warnings and leave caches unchanged. Missing caches and projections with no
valid frame to hold must be generated separately; rendering never runs SAM or tracking.

### Convert .mov to .mp4 (with audio)
```bash
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
```

### Clip .mp4 by start and end time
```bash
ffmpeg -i input.mp4 -ss 00:00:10 -to 00:00:30 -c copy output_clipped.mp4
```
Replace `00:00:10` and `00:00:30` with your desired start and end timestamps (`HH:MM:SS`).

## Configuration

Generate highlights from audio with `uv run string-lights generate-transcription`,
then optionally refine string choices with `uv run string-lights refine-transcription`.
For variable string-light hold lengths, see the
[PDF-based note-duration workflow](docs/note-duration-workflow.md).

### Lyrics export

Launch the lyrics editor with `uv run string-lights lyrics-editor cupid` (or an explicit video path). A project name uses `renders/strings.mp4` for preview and export when available, falling back to the project source video. An explicit video path uses that file. It loads and saves the project’s `lyrics.json` by default; use `--lyrics /path/to/lyrics.json` to override it. Input files are selected through the CLI in both editors.

In the lyrics editor, click **Export** to choose a layout:

- **Recommended + padding** (default): 1080 × 1920 portrait, fitting the full video
  with black padding.
- **Original aspect ratio**: source dimensions, without resizing or added padding.

The preview follows the selected layout. Lyric coordinates are stored relative
to the padded 1080 × 1920 editing canvas; original-aspect exports remove that
padding while preserving text position and relative size within the video.
Lyrics placed in the padding are outside the original-aspect output.
Confirm **Export** in the popup to save lyrics and render, or cancel
to continue editing. Independently choose the playback format:

- **H.264 + AAC — broad playback support** (selected by default in the popup):
  encodes audio as AAC at 320 kbps for sharing. This recompresses the audio.
- **H.264 + original audio**: copies the source audio without quality loss;
  compatibility depends on its codec.

Both choices render video as H.264 in MP4 and support either layout. The source
file is never modified.

For readability on bright footage, select a lyric and adjust **Black backdrop**
to control its opacity (0 = off), **Padding (px)** for space around the text, and
**Softness (px)** for the smooth fade into the video. These settings are saved per
lyric and used in both preview and export. The backdrop fades with the lyric.
Existing lyrics default to no backdrop; new lyrics start at 65% opacity.

The lyrics editor creates cached H.264/AAC previews when the source codecs are
not browser compatible, including ALAC audio. Previews do not modify the source;
exports still read the original video and use the selected playback format.

### Planar tuner

```bash
uv run string-lights tune-guitar-plane littleroot_town
```

Opens a local browser editor using the video's existing JSON annotation in
`data/projects/VIDEO/profile.json`. Wheel to zoom, right-drag or Space-drag to pan,
and left-drag string endpoints or region vertices. Select a drawing tool and
click at least three vertices, then finish the polygon. Undo reverses edits.
The reference-frame slider keeps existing coordinates, so check their alignment
after switching frames. Save profile writes the JSON and `.npz` profile.
Re-run `generate-poses` afterward to refresh cached projections.

### Compare planar tracking results

Compare an experimental projection cache with a frame-aligned reference cache:

```bash
uv run string-lights planar-validate cupid \
  --projections data/projects/cupid/tmp/trial.npz \
  --reference data/projects/cupid/tracking/projections.npz
```

The command checks coverage and projection errors against configurable limits.
The reference must use the same dimensions and frame count as the trial cache.

### Generate transcription from audio

```bash
uv run string-lights generate-transcription cupid --capo 2
uv run string-lights generate-transcription /path/to/video.mp4 --project-dir data/projects/my-project --capo 2
```

Generates `transcription.npy` and `transcription.json` in the project directory.
The array uses 44 Hz, with columns low E, A, D, G, B, high E. The JSON companion
records detected pitches/times, string/fret assignments, unassigned events, capo,
tuning, model revisions/checksums and generation settings. The renderer only
needs the array. The companion describes the generation run; later edits in the
tabs editor do not update it. Its array checksum identifies that original export.

`--capo` defaults to 0. `--tuning` accepts six open-string MIDI pitches before the
capo, in low-to-high string order (default `40 45 50 55 59 64`). Choose `--model fl`
for François Leduc instead of the default GAPS pitch model. `--device auto`
selects CUDA when accessible, otherwise CPU; `--device cuda` requires GPU access.
FFmpeg and ffprobe must be installed. Pretrained weights download on first run
and are checksum-verified; `--model-cache` overrides the shared default
`data/tmp/transcription-models` cache. Models are estimates: review strings in
the tabs editor before final rendering. PDF-assisted refinement is described below.

For experiments, give an explicit output path and optionally limit duration:

```bash
uv run string-lights generate-transcription cupid --capo 2 --duration 15 \
  --output data/projects/cupid-onset-benchmark/tmp/generated.npy
```

`--output` also determines the JSON companion path. Existing array or JSON files
are preserved unless `--force` is supplied. `--projects-dir` selects a different
project collection. `--pulse` sets light hold length (default .09 seconds);
`--evidence-window` sets post-attack string evidence duration (default .08).
The former `generate-timings` command has been replaced; timing detection is
part of transcription generation.

### Refine string choices using a tab PDF

```bash
uv run string-lights refine-transcription my-project --tabs /path/to/tabs.pdf --force
```

Requires `codex` installed and authenticated, plus Poppler's `pdftoppm` and
`pdfinfo`. The command renders PDF pages, calls non-interactive `codex exec`
with image inputs, a read-only sandbox and a JSON output schema, then validates
and caches the extracted string/fret sequence. This sends the rendered tab pages
to the configured Codex provider and uses the account's normal Codex usage.

Python aligns tab pitches to the generated audio-note sequence, accounting for
capo and tuning. Matched notes receive the tab's string choice; audio pitches
and timestamps stay fixed. Uncertain events, unmatched notes and ambiguous
unisons retain their audio assignments and are listed in the JSON companion.
The aligner can skip missing/extra notes; it is a heuristic and still needs review.
It does not add PDF-only attacks or fix incorrect audio pitches.

The input must have its matching generation JSON with an intact array checksum.
Arrays edited separately from their companion are rejected. PDF/transcription
capo conflicts are also rejected. By default, at least half the audio notes must
align; inspect the extraction before lowering `--min-coverage`.

All PDF pages are read by default. Use `--pages 1`, `--first-bar 2`, `--last-bar 8`
to limit extraction. For a score that only covers the opening passage, use
`--end-time SECONDS` to bound audio alignment and preserve later note assignments.
Coverage is measured over notes within that time range. Extraction explicitly uses `gpt-6.1-sol` with `low` reasoning
effort, independent of local Codex defaults; `--codex-model` overrides the model.
Extracted `tabs.json`,
page images, prompt/schema and Codex log are cached under the project's
`tmp/tab-refinement/`. Cache identity includes the PDF checksum, page/bar range,
prompt/schema version, CLI version, requested model and reasoning effort. A matching extraction is
reused without another inference call. `--work-dir` overrides that location.
The default extraction timeout is 300 seconds (`--timeout` overrides it).

Use explicit paths for experiments:

```bash
uv run string-lights refine-transcription cupid --tabs data/projects/cupid/Cupid.pdf \
  --transcription data/projects/cupid-onset-benchmark/tmp/command-transcription.npy \
  --output data/projects/cupid-onset-benchmark/tmp/pdf-refined.npy \
  --work-dir data/projects/cupid-onset-benchmark/tmp/pdf-refinement \
  --pages 1 --first-bar 2 --last-bar 8
```

Existing outputs require `--force`. Without `--output`, refinement updates the
project's `transcription.npy` and companion. `--transcription` selects an input
array; `--projects-dir` and `--project-dir` work as in generation. The companion
records extraction provenance, note-level matches, changed strings, unresolved
notes and a new output checksum. Use the tabs editor to review before rendering.
