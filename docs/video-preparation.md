# Preparing HDR video with replacement audio

## Investigation and findings

The investigated source was an iPhone HEVC Main 10 MOV: 10-bit YUV 4:2:0,
BT.2020 primaries, HLG (`arib-std-b67`) transfer, and Dolby Vision profile 8
metadata with an HLG-compatible base layer. It also carried a display rotation
matrix. Its video timestamps were variable despite a nominal rate of 30 fps.

A direct SDR frame decode looked muted and washed out. Comparing the same frame
with an explicit HDR-to-SDR conversion produced stronger color and contrast.
This supports missing tone mapping as the cause for the observed SDR appearance;
it does not establish how every HDR-aware player displays the original.

The initial H.265 conversion preserved HDR tags but did not tone-map the image.
Preserving those tags alone does not fix playback or processing paths that treat
the decoded image as SDR. The overlay renderer decodes through OpenCV to BGR and
does not explicitly tone-map HDR, so preparing an SDR input is appropriate.

The replacement WAV was 24-bit stereo PCM at 44.1 kHz and longer than the video.
The user confirmed that both recordings already aligned at time zero. No audio
offset, resampling, normalization, or speed change was needed; only its extra tail
was trimmed.

## Quality choices

- Rebuild from the raw MOV and WAV, avoiding repeated compression of an MP4.
- Convert the HLG base layer to linear light, map BT.2020 to BT.709, apply Mobius
  tone mapping, and encode standard BT.709 SDR. This workflow does not use Dolby
  Vision dynamic metadata to reproduce a specific display's rendering.
- Keep 10-bit video and use H.265 CRF 12 with the slow preset as a high-quality,
  visually lossless target. This is still lossy compression; neither CRF nor a
  preset guarantees perceptual equivalence for every scene.
- Encode WAV samples as ALAC in MP4 to preserve audio losslessly. AAC would be
  another lossy generation. ALAC playback support should be checked in the target
  player.
- Preserve frame order and source presentation timestamps. A previous conversion
  rounded timestamps to 30 fps, moving individual frames by up to about 11.7 ms.
  An explicit encoder time base eliminated that rounding in the final input.

Tone mapping necessarily changes the original HDR image. Keep the raw MOV and
WAV as masters. Lossless video encoding after tone mapping would preserve the
converted pixels exactly, but would create a substantially larger file.

## Workflow

### 1. Inspect the sources

```bash
ffprobe -v error -show_streams -show_format -of json raw/source.mov
ffprobe -v error -show_streams -show_format -of json raw/replacement.wav
```

Check dimensions, pixel format, color primaries, transfer, matrix, rotation,
frame count, time base, stream start times, durations, and audio sample format.
Confirm that the WAV aligns with the picture at the start and near the end.
Starting both streams at zero preserves an existing alignment; it cannot establish
that two independently recorded sources are aligned.

### 2. Compare a representative frame

```bash
ffmpeg -n -ss 10 -i raw/source.mov -frames:v 1 before.png

ffmpeg -n -ss 10 -i raw/source.mov \
  -vf 'zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius:desat=2:peak=10,zscale=t=bt709:m=bt709:r=tv,format=yuv420p' \
  -frames:v 1 after.png
```

The first image approximates an ordinary SDR decode without explicit tone mapping;
it is not a reference rendering from an HDR display. Inspect skin, dark details,
saturated objects, and bright windows. Review several scenes when exposure varies.
The peak and nominal luminance settings here were used for this HLG source, not
established as universal values for all HDR footage.

### 3. Encode to a temporary MP4

The following example assumes an HLG/BT.2020 source with time base `1/600` and
already aligned audio. Replace `VIDEO_DURATION` with the source video duration
in seconds, and use the source's actual time base. Leave autorotation enabled so
the display rotation is baked into the output pixels.

```bash
ffmpeg -n -i raw/source.mov -i raw/replacement.wav \
  -map 0:v:0 -map 1:a:0 \
  -vf 'zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius:desat=2:peak=10,zscale=t=bt709:m=bt709:r=tv:d=error_diffusion,format=yuv420p10le,sidedata=mode=delete' \
  -c:v libx265 -preset slow -crf 12 -tag:v hvc1 \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
  -fps_mode passthrough -enc_time_base:v 1:600 \
  -c:a alac -af 'atrim=duration=VIDEO_DURATION,asetpts=PTS-STARTPTS' \
  -map_metadata -1 -map_chapters -1 -movflags +faststart \
  prepared.mp4
```

The explicit stream maps replace camera audio with the WAV and omit data tracks.
Clearing side data and specifying BT.709 avoids retaining HDR descriptions on SDR
pixels. Disabling metadata and chapter copying also avoids unrelated source tags
and a dangling QuickTime chapter reference encountered during preparation.
`-n` prevents accidental overwrites.

`asetpts` starts the trimmed WAV at zero; use this only when that is the intended
alignment. Do not stretch audio or force video to a new frame rate to conceal a
sync problem. This command trims a longer WAV; it does not pad a shorter one.

### 4. Verify before replacing the working input

Inspect output metadata and decode the entire file:

```bash
ffprobe -v error -show_streams -show_format -of json prepared.mp4
ffmpeg -v error -i prepared.mp4 -f null -
```

Require the expected dimensions and frame count, BT.709 color tags, and the
expected ALAC sample rate, channel count, and bit depth. Compare sorted video
presentation timestamps from source and output; sorting is necessary because
HEVC packet decode order can differ from display order:

```bash
ffprobe -v error -select_streams v:0 -show_entries packet=pts_time \
  -of csv=p=0 raw/source.mov | sort -n > source-timestamps.txt
ffprobe -v error -select_streams v:0 -show_entries packet=pts_time \
  -of csv=p=0 prepared.mp4 | sort -n > output-timestamps.txt
diff -u source-timestamps.txt output-timestamps.txt
```

All presentation timestamps matched exactly in the final prepared file.

Verify lossless audio by decoding both sources to the same PCM format and
comparing hashes over the retained duration:

```bash
ffmpeg -v error -i raw/replacement.wav -t VIDEO_DURATION \
  -map 0:a:0 -c:a pcm_s24le -f hash -hash sha256 -
ffmpeg -v error -i prepared.mp4 \
  -map 0:a:0 -c:a pcm_s24le -f hash -hash sha256 -
```

The hashes matched in this investigation. This proves that the retained WAV samples
survived encoding exactly; it does not independently prove musical or lip sync.
Check playback near the beginning and end as well. Back up the working input,
then replace it with the verified file.

## Reusing tracking caches

The application stores artifacts in the selected project directory:

- `data/projects/VIDEO/tracking/projections.npz`: planar projections.
- `data/projects/VIDEO/masks/masks.npz` and `metadata.npz`: hand masks and metadata.
- `data/projects/VIDEO/transcription.npy`: string highlight data.
- `data/projects/VIDEO/profile.json`: profile annotation; a compiled profile
  may also exist as `profile.npz` in that directory.

For a new project, copy the matching artifacts and preserve existing destinations in `tmp/`.
Do not leave an unrelated compiled profile paired with a newly copied annotation.
Cached rendering requires projections and masks; the profile is useful for future
tracking or editing. `render-strings` reads caches without regenerating them.

Reuse is valid only when frame content, orientation, geometry, and frame order
still match. The application checks dimensions and coverage, but those checks
cannot prove alignment. Tone mapping can change segmentation or feature detection
results if tracking is recomputed; copying caches deliberately preserves the old
results instead.

When caches cover fewer frames than the new source, rendering warns and uses
these fallbacks without changing the cache files:

- Masks: frames beyond the recorded coverage render without hand masking,
  including in mask debug output. Sampled masks still hold their values within
  the cache's recorded coverage.
- Planar projections: extra frames repeat the last fully finite cached projection,
  searching backward past unresolved frames if needed. Existing cached frames
  remain unchanged. If no valid projection exists, rendering fails with a clear
  error. A held projection does not track any new motion in the extra frames.

To explicitly limit a render to a cached segment instead:

```bash
uv run string-lights render-strings --frames CACHED_FRAME_COUNT VIDEO.mp4
```

## Limits of the overlay renderer

The prepared input and rendered output have different guarantees. Currently,
`render_video` in `string_lights/pipeline.py`:

- Reads BGR frames through OpenCV and encodes 8-bit H.264 using FFmpeg defaults
  for normal rendering. The input's 10-bit precision and CRF 12 quality setting
  are not preserved through that additional encoding.
- Copies the audio stream without recompressing it.
- Uses the input's average frame rate to assign evenly spaced output timestamps,
  rather than carrying each source frame's timestamp through the pipeline.
- Trims audio to the requested frame count divided by that frame rate when a
  frame limit is supplied.

For the investigated cached segment, the maximum difference between source
presentation times and the renderer's constant-rate schedule was approximately
11.7 ms, less than one 30 fps frame. This is a measurement for this source, not a
general guarantee against drift in other variable-rate recordings. Exact timing
preservation and comparable video quality in the final overlay output would
require changes to the renderer; preparing a better input alone does not provide
those guarantees.

## Reference

[FFmpeg filter documentation](https://ffmpeg.org/ffmpeg-filters.html) describes
`zscale` and the linear-light floating-point input expected by `tonemap`.
