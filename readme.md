# string-lights

A tool for processing video frames of a guitar with an ArUco marker board to detect board pose and apply visual overlays.

## Usage

`uv run string-lights $INPUT_MP4`

## Useful Commands

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
