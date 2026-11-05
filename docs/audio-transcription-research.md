# Audio-assisted string transcription

Research checked 2026-11-04. These are recommendations, not measured results on the project's clips.

## Recommended first milestone

Detect **attack times** from the video's audio, retain them as unassigned events, and let the user assign strings or align supplied tabs. This directly addresses the useful partial solution without pretending timing alone identifies a string.

The current renderer samples `transcription.npy` at 44 Hz; its columns represent E2, A2, D3, G3, B3, E4. It consumes positive activations directly (`_clean_tab` is commented out). Preserve event times in seconds until export to avoid unnecessary quantization. A timing-only result belongs in an event JSON / unassigned editor lane, not in an arbitrary string column.

## Candidate methods

| Method | Useful output | Practical role |
| --- | --- | --- |
| Spectral-flux onset detection | Candidate attack times | First baseline: inexpensive, inspectable, no model training |
| Basic Pitch | Polyphonic note pitches, starts, ends, bends | Optional second source of onset/pitch evidence |
| Known tabs aligned to audio | String assignments with performance timing | Recommended route to automatic six-string output when tabs exist |
| Guitar-specific tablature model | Estimated string/fret activity | Research candidate requiring local validation |

Librosa's `onset_strength` measures positive spectral changes, using a log-power Mel spectrogram by default; `onset_detect` selects peaks. Its parameters permit tuning the peak threshold and minimum spacing. Backtracking moves events to preceding energy minima and is intended primarily for segmentation: compare it against unbacktracked times instead of assuming it improves visual synchronization. Sources: [onset strength](https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_strength.html), [onset detection](https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_detect.html).

Spotify Basic Pitch supports polyphonic transcription and exports MIDI and note events. Its authors say it works best with one instrument at a time. It does not provide guitar string identities. The repository's stated Python compatibility stops at 3.11, whereas this project requires 3.12+, so verify runtime compatibility or evaluate it in an isolated environment before adopting it. Source: [official Basic Pitch repository](https://github.com/spotify/basic-pitch).

TabCNN explicitly estimates tablature, but its published implementation targets Python 2.7 and documents training/testing on GuitarSet; it is not a drop-in modern dependency. GuitarSet includes microphone recordings and time-aligned string/fret annotations, making it useful for evaluation as well as training. Sources: [TabCNN authors' implementation](https://github.com/andywiggins/tab-cnn), [GuitarSet authors' dataset page](https://guitarset.weebly.com/).

## Proposed pipeline

1. Extract audio with its offset relative to the video preserved. Retain full-precision timestamps and record extraction settings.
2. Compute spectral-flux peaks at approximately 5–10 ms analysis steps. Tune threshold and minimum spacing on reviewed examples; neither setting should be tied to the 44 Hz export grid.
3. Write candidate events with timestamps, detector scores, and provenance. Scores are ranking evidence, not calibrated probabilities. Add click-track listening and an editor lane for moving, adding, or deleting events.
4. With no tabs, assign strings manually at the candidate times. This still removes much of the timing work.
5. With tabs, align the known musical sequence to audio; use pitch and rhythm evidence and allow skipped/extra events. Do not simply pair the nth tab event with the nth detected attack, since one missed attack would shift the remainder.
6. Export reviewed string activations through the existing 44 Hz convention, with configurable pulse lengths. Keep hold duration separate from onset timing and the renderer's fade.

For timed tabs, use score pitches to construct a reference feature sequence (or synthesize reference audio), align it to recording features, then refine predicted attack times locally. Dynamic time warping can accommodate tempo variation: librosa demonstrates chroma-based alignment between recordings with different tempi. Applying it to a synthesized score is a proposed extension here, not a capability tested in this repo. Repeats, cuts and repeated chords need constraints or manual anchors. Source: [librosa music synchronization example](https://librosa.org/doc/0.11.0/auto_examples/plot_music_sync.html).

Untimed tabs still supply valuable string order, but alignment is more ambiguous. Tuning, capo, arrangement and any omitted ornaments must match the recording. Audio pitch alone cannot uniquely recover the played string where different fret/string positions share a pitch; fingering constraints can suggest a plausible assignment, not prove the actual one.

## Reliability and acceptance

Expected failure cases to measure include quiet plucks under ringing notes, hammer-ons/pull-offs, handling noise and body percussion, strums containing several nearby attacks, and backing instruments. A global onset detector cannot identify which instrument caused a transient. Agree on whether a strum is one timing event or multiple string attacks before annotating evaluation data.

Review representative passages from multiple local projects, reserving some for evaluation rather than parameter tuning. Existing hand-authored arrays can seed review but should not automatically count as accurate ground truth: held activations may hide repeated attacks.

Measure one-to-one onset precision, recall and F1 at ±50 ms, and also ±25 ms for visual timing; report median/95th-percentile timing error on matches and manual corrections per minute. `mir_eval.onset` supplies precision/recall/F1 with a default 50 ms tolerance. These thresholds are evaluation choices, not a claimed accuracy level. Source: [mir_eval onset evaluation](https://mir-eval.readthedocs.io/latest/api/onset.html).

First compare spectral flux with Basic Pitch-derived onset candidates on the same reviewed passages. Adopt a more complex model only if it reduces correction effort on held-out clips. Keep all trial audio, event files, arrays and diagnostic renders under `data/projects/<project>/tmp/`, using explicit output overrides; leave canonical transcriptions and finished renders untouched.
