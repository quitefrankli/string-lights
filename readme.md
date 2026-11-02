# string-lights

A tool for processing video frames of a guitar with an ArUco marker board to detect board pose and apply visual overlays.

## Usage

`uv run string-lights $INPUT_MP4`

## Useful Commands

### Generate caches without rendering

```bash
uv run string-lights planar-track cupid.mp4 --frames 500
uv run string-lights masks cupid.mp4 --frames 500
uv run string-lights poses VIDEO.mp4 --frames 500
```

Mask generation uses SAM2Matting Base+ video tracking, with soft opacity caches
(0–255) for natural hand boundaries. Public weights download on first use; CUDA is required.
Frames are decoded from disk on demand with a two-frame cache; forward tracking
retains only its required recent memory plus the seed frame.
Supply a first-frame grayscale hand mask (white hands, black background):

```bash
uv run string-lights masks VIDEO.mp4 --seed-mask seed.png
```

The default seed path is `data/components/mask_seeds/VIDEO.png`.
A motion-aligned three-frame median reduces isolated opacity flicker by default;
use `--no-stabilize` to retain raw model output. Stable gaps can remain open,
but this filter does not correct persistent segmentation errors.
Tracking processes every frame; `--frame-skip N` samples the masks saved to cache
(default: 1). Skipped frames hold the latest cached mask. Use `--force` to regenerate
an existing cache. Old binary caches must be regenerated.
The vendored SAM2Matting code and weights use CC BY-NC-SA 4.0 for non-commercial research.

`poses` estimates ArUco board poses; `planar-track` generates markerless
projections. All three commands accept `--force` to replace existing caches.
The standalone `masks` and `poses` commands replace `run --masks` and `run --poses`.

Render with the projection and mask caches:

```bash
uv run string-lights run cupid.mp4 --frames 500 --pose-tracker planar
```

`run` always reads existing caches. Choose `--pose-tracker aruco` (the default)
for board poses or `--pose-tracker planar` for projections. Masks are loaded
from cache unless `--disable-masking` is supplied. If a mask cache ends before
the video, remaining frames render without hand masking. If planar projections
end early, remaining frames hold the last valid cached projection. Both fallbacks
emit warnings and leave caches unchanged. Missing caches, projections with no
valid frame to hold, and insufficient ArUco pose coverage must be generated
separately; rendering never runs SAM or tracking.

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
uv run string-lights tune littleroot_town.mp4 --no-aruco
```

Opens a local browser editor using the video's existing JSON annotation in
`data/components/profiles/`. Wheel to zoom, right-drag or Space-drag to pan,
and left-drag string endpoints or region vertices. Select a drawing tool and
click at least three vertices, then finish the polygon. Undo reverses edits.
The reference-frame slider keeps existing coordinates, so check their alignment
after switching frames. Save profile writes the JSON and `.npz` profile.
Re-run `planar-track` afterward to refresh cached projections.

Use `tune VIDEO --aruco` for the original board-based calibration window.

- **Board**: 6×6 ChArUco board with 12 mm squares and 9 mm markers
- **Camera**: iPhone 15 main camera (1800 px focal length at 1920 px width)
