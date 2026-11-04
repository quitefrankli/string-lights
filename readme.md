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

Launch the tabs editor for one project with `uv run string-lights tabs-editor cupid`.
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

`render-strings` always reads existing planar projection caches. Masks are loaded
from cache unless `--disable-masking` is supplied. If a mask cache ends before
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

To generate highlight data from a manually annotated tablature score, use
`scripts/generate_transcription.py`. See [the score format and workflow](docs/tab-transcription.md).

### Lyrics export

In `uv run string-lights lyrics-editor`, click **Export** to choose a layout:

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
