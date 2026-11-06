# Audio onset prototype

This is a historical research report. The standalone onset comparison utility
has been removed. Note timing detection remains part of
`uv run string-lights generate-transcription`.

The outputs below describe the original experiment artifacts.

Outputs:

- `onsets.json`: unassigned attack times in video seconds, extraction offset and detector settings. No string identity or calibrated confidence is inferred.
- `comparison.json`: one-to-one precision, recall, F1 and timing errors at 25/50 ms, including matched event indices.
- `review.wav`: original mono audio in both channels; detected clicks on the left, reference clicks on the right. Clicks follow the extracted audio's timeline.
- `timeline.html`: standalone visual comparison with hover timestamps; keep it beside the WAV.

The script uses librosa 0.11 spectral flux, a roughly 10 ms hop, default peak threshold 0.07 and minimum spacing 30 ms. Tune with `--delta` and `--min-gap`. It does not fit an offset to the reference, infer note durations, export fake string assignments, or modify the editor yet.

Reference events are positive rising edges on each of six strings, deduplicated at identical timestamps. A pluck can start while another string is ringing. Contiguous positive runs cannot reveal retriggers. Grouped metrics apply a fixed 100 ms window from each group's first attack to both sequences; this is a separate musical interpretation, not a replacement for strict metrics.

## Cupid baseline, 2026-11-04

Cupid’s transcription is a roughly aligned, handcrafted reference, not exact ground truth. The user clarified that notes were manually adjusted and remain approximate. All metrics below measure agreement with that annotation, not verified audio detection accuracy. Timing differences, unmatched predictions and unmatched reference events can reflect errors on either side. Review against the audio before labelling them false attacks or missed attacks.

Source duration: 57.687 s. Reference: 197 distinct start times. Predictions: 162. No threshold tuning was performed against Cupid.

| Comparison | Matches | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Strict, ±25 ms | 42 | 25.9% | 21.3% | 23.4% |
| Strict, ±50 ms | 68 | 42.0% | 34.5% | 37.9% |
| Grouped, ±25 ms | 34 | 21.4% | 20.2% | 20.8% |
| Grouped, ±50 ms | 61 | 38.4% | 36.3% | 37.3% |

Strict 50 ms matching yields 94 unmatched detections and 129 unmatched reference events. Median absolute error among matched events is 23.2 ms; this excludes misses and is not an overall accuracy score.

A separate diagnostic scanned global shifts from -1 to +1 seconds in 10 ms steps using only the first 29 seconds. Its best F1 selected zero shift; second-half F1 remained 35.1%. This does not explain the poor result as a simple global synchronization error. The diagnostic is saved as `offset-diagnostic.json`; it was not applied to output events.

These agreement scores alone cannot establish whether the detector is ready to replace manual timing. Next, listen to the review and carefully annotate a small set of attack times directly from the audio, resolving disagreements on both sides, then evaluate alternate spectral bands, peak parameters, and note-model candidates using a fixed tuning/evaluation split. Cupid is the only evaluated recording, so these results do not establish generalization. The editor integration should wait until candidates meaningfully reduce corrections.

## Follow-up detector comparison

Compared 40 configurations: default spectral flux, frequency maximum filtering, median aggregation, and maximum-filtered flux below 4 kHz; each used five thresholds and two minimum gaps. Maximum filtering is inspired by vibrato suppression, not a complete implementation of the SuperFlux paper. Selected by annotation agreement on the first 29 seconds only. The remainder was used for validation within this same recording, not as an independent recording benchmark.

The selected setting retained default spectral flux with `--delta .05 --min-gap .07`. At ±50 ms on the second half, F1 rose from 35.1% to 37.5%, precision from 40.7% to 42.4%, and recall from 30.8% to 33.6%. This modest improvement is agreement with approximate annotations, not demonstrated perceptual improvement. Defaults remain unchanged.

Artifacts in the benchmark project's `tmp/`: `compare_detectors.py` reproduces the sweep; `detector-comparison.json` records all settings; `improved-onsets.json` and `improved-review.wav` contain the selected candidate. In `ab-review.wav`, baseline clicks are left and selected-candidate clicks right, both over the original audio (click pitches differ to distinguish channels).

### Next experiment for string identity

Evaluate a pretrained guitar tablature model before training anything custom. [TabCNN](https://github.com/andywiggins/tab-cnn) estimates string/fret activity directly; [SynthTab](https://github.com/yongyizang/SynthTab) supplies related models and synthetic-data training resources. [TART](https://arxiv.org/abs/2609.11904) is a newer modular audio-to-tablature research system; availability and runtime compatibility need checking before integration. None has been run on Cupid here.

For this application, keep per-string activity and new-attack evidence separate: sustained model activity must not retrigger lights every frame. Combine pitch/string hypotheses with detected attacks and retain uncertain alternatives. Known tuning and capo constrain possibilities; sequence-level fingering constraints improve plausibility without proving which string was played. For example, standard-tuned E4 is both open high E and B-string fret 5.

When matching tabs exist, aligning their string assignments to audio remains the most direct route. A longer-term audio/video approach could use visible fretting or picking positions to resolve alternatives, but occlusion and hand localization require a new experiment; existing planar tracking alone does not identify played strings.
