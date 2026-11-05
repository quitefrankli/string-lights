# Cupid pretrained-model trials

Ran on 2026-11-04 using the first 15 seconds of Cupid's source audio. All inputs in `data/projects/cupid/` were read-only; trial code, downloaded repositories/weights and exports are under `data/projects/cupid-onset-benchmark/tmp/`.

CUDA was verified outside the sandbox on an RTX 5070 Ti. Trial dependencies were installed in the existing `.venv`; they are experimental and not added to the project's dependency manifest. The earlier plan to use an isolated environment was not followed. `uv sync` may remove these experimental packages; record/install their dependencies before rerunning adapters.

## Outputs and reproducibility

- `run_guitar_models.py`: GAPS and François Leduc inference, with explicit Hugging Face downloads and CUDA. Outputs `gaps.mid`, `fl.mid`.
- `review_models.py`: reads MIDI events, writes timing arrays and stereo review WAV files (original left, synthesis right). Synthesis was rendered using FluidSynth with `/usr/share/sounds/sf2/TimGM6mb.sf2`.
- `run_string_model.py`: fold-1 instrument-informed multi-task inference. Uses the authors' HCQT/Mel feature extractors with original settings, 19-frame centered windows, zero padding, and the published config. Package stubs bypass unused training/dataset imports; model and feature computation are unchanged.
- `string-predictions.npz`: timestamps, six-string activity probabilities and per-string fret/silence probabilities. Native frame rate is 22050/512 Hz, not 44 Hz.
- `summarize_trials.py`: same-excerpt benchmark, activity review and experimental export.
- `model-review.html`: review page, stereo audio players and string probability plot.
- `trial-summary.json`: note events and agreement metrics.
- `experimental-strings.npy`: 15-second, 44 Hz six-column lighting candidate. It applies 4-frame pulses at GAPS attacks to all strings with predicted sounding probability >= .5 at the following model frame. This can retrigger ringing strings and is not established string-onset transcription. It is for inspection only.

## Results and limitations

At ±50 ms, the original spectral-flux detector has annotation-agreement F1 **56.4%** on this excerpt. Both guitar transcription models have F1 **56.2%**, each outputting 49 notes. Their note lists differ despite matching counts/scores. Thus these trials do not demonstrate improved timing. They do add pitch estimates for listening and subsequent string inference. Scores use approximate handcrafted annotations, not verified audio ground truth; only one short excerpt was tested.

The string activity model at threshold .5 predicts on average **3.43 sounding strings per frame**. Against existing light-hold frames, precision is 6.2% and recall 69.9%; this is not a valid accuracy claim because sounding-string duration and handcrafted light pulses represent different things. Assess new attacks separately from sustained activity before choosing thresholds or comparing string assignments. Fold 1 only was evaluated, without local calibration or fine-tuning. No claim of audible improvement is made: review files are provided for listening.

Checkpoint SHA256:

- GAPS: `65483e7c0e340a90415b15b520687587698c8c728f5fa470a205f13ee45c6513`
- François Leduc: `50d93dba89bdd3401849bc735614478e83d9f46d21fa3f71d8aca5acc0a52028`

Next useful work is to review the same passages audibly and visually, then decode new per-string attacks using pitch changes, activity transitions and repeated-note evidence. Sounding-string probability alone is insufficient for repeated plucks. Any expanded comparison should retain a reviewed evaluation passage and report changes in correction effort, not just agreement with rough labels.

## Pitch-aware assignment candidate

`string_lights.assign_strings` now performs a beam search over legal pitch-to-string choices, using the pretrained activity probabilities, fret-span and position-change costs, and a penalty for replacing a still-sounding different note. Notes starting within 40 ms of the group's first note cannot share a string. Audio offsets can contain ringing, so occupation is a soft penalty rather than a hard prohibition. Unassigned notes carry an explicit cost and remain in the sidecar. Costs are heuristics and have not been tuned against Cupid's annotations.

Reproduce the 15-second candidate:

```bash
.venv/bin/python -m string_lights.assign_strings \
  data/projects/cupid-onset-benchmark/tmp/guitar-comparison.json \
  data/projects/cupid-onset-benchmark/tmp/string-predictions.npz \
  --output data/projects/cupid-onset-benchmark/tmp/pitch-aware-transcription.npy \
  --duration 15
```

Use `--model fl` to use FL events, `--tuning` for six open-string MIDI pitches, and `--capo` for a fret offset. Current run assumes standard tuning/no capo, allows frets through physical fret 19, and assigns 45 of 49 GAPS notes. `assignments.html` lists each event; the JSON sidecar retains every pitch and assignment, including unassigned notes. `pitch-aware-transcription.npy` is a renderer-compatible `(660,6)` integer grid at 44 Hz, with 90 ms pulses. Overlapping pulses merge, so very close repeated attacks on the same string may not retrigger separately. This is a 15-second candidate, not a full-length replacement. It has not been verified against played strings in the video.

## Reference-guided string disagreements

`tmp/analyze_string_disagreements.py` compares rising-edge string sets, grouped within 40 ms of the first attack and matched one-to-one by time. The handcrafted reference's historical ordering is important: at ±50 ms, exact string-set agreement is 1/21 matched groups as stored, versus 11/21 when its six columns are reversed in memory. This supports the compensating-flip diagnosis but is not independent proof of actual played strings. The source file is untouched; both orientations are reported.

With the reversed reference, conditional string precision/recall are 58.1%/72.0% and F1 is 64.3% over those 21 matches; ten groups disagree. At ±100 ms there are 26 matches, 13 exact sets and 61.8% string F1. These are rough-reference agreement metrics, exclude unmatched attacks and do not measure pitch correctness. Grouping can also combine separate fast notes or split rolled chords. The report `tmp/string-disagreements.html` lists all disagreements with reference time, generated time, both string sets and generated pitch/fret candidates. `tmp/string-disagreements.json` preserves the full data for both tolerances and orientations.

Use these ten close-timing disagreements as diagnostic examples before modifying assignment scoring. Because the reference lacks pitch and may be approximate, a difference can represent wrong assignment, wrong detected pitch, omitted chord tones or annotation error. Keep some examples held out when tuning, and preserve the detected attack times.

## Assignment evidence comparison after reference-order correction

The canonical Cupid reference now has low-to-high columns; no in-memory reversal was applied in this experiment. `evaluate_assignment_evidence.py` compares four predetermined variants with identical GAPS notes and timestamps: baseline activity at the attack; mean activity over 80 ms after the attack; a 50/50 log-cost blend of specific fret probability and activity; and that blend with the same window. It uses the existing 40 ms grouping and ±50 ms timing matching. Unmatched attacks are excluded, so this is conditional string agreement, not overall transcription accuracy. None of the local reference labels are passed into the assignment algorithm.

| Variant | Exact sets / 21 matched groups | String F1 | First-half F1 | Second-half F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 11 | 64.3% | 57.1% | 71.4% |
| Activity window, 80 ms | 12 | 67.9% | 64.3% | 71.4% |
| Specific fret evidence | 10 | 60.7% | 57.1% | 64.3% |
| Fret evidence + window | 10 | 60.7% | 57.1% | 64.3% |

Window-only is the best first-half candidate and ties baseline in the second half; there is no demonstrated second-half improvement. This remains a small, approximate-reference experiment, not independently verified accuracy. Specific fret evidence hurt in this trial and is not recommended as default. Both halves were reported together; do not treat the second half as a pristine holdout for subsequent tuning.

The assignment function now accepts optional tablature probabilities, an evidence window, and a fret weight; CLI flags are `--evidence-window` and `--fret-weight`. Existing defaults preserve the earlier behavior. Tests verify pitch/fret disambiguation, post-attack evidence, unchanged onset time, capo offsets, and same-chord string exclusivity.

`assignment-window.npy` is the new 15-second candidate, and `assignment-window.json` preserves its full note assignments. `assignment-evidence-comparison.json` contains all metrics and matched groups. Previously exported MP4 comparisons still show the baseline until rerendered. Run with `--evidence-window .08` and zero fret weight to reproduce the candidate through the CLI.

## Capo-2 correction

The PDF specifies capo 2; the earlier assignment assumed zero. Re-ran with `--capo 2 --evidence-window .08` and unchanged GAPS pitches/timestamps. Correct pitch-to-fret relation is `fret_relative_to_capo = pitch - open_string_pitch - 2`. This correction affects candidate fingering, not audio pitch or attack timing. Nine of the 49 events change assigned string; 45 remain assigned and four MIDI-108 detections remain outside range.

Against the corrected canonical Cupid reference, using the same 40 ms grouping and ±50 ms matching:

| Candidate | Exact sets / 21 matches | Conditional string F1 | First-half F1 | Second-half F1 |
| --- | ---: | ---: | ---: | ---: |
| No capo, instantaneous activity | 11 | 64.3% | 57.1% | 71.4% |
| No capo, 80 ms activity window | 12 | 67.9% | 64.3% | 71.4% |
| Capo 2, 80 ms activity window | 14 | 75.0% | 64.3% | 85.7% |

This is agreement over timing-matched groups on a previously inspected excerpt, not independently verified accuracy or a new held-out result. Unmatched attacks are excluded. `tmp/evaluate_capo.py` reproduces analysis; `tmp/capo-comparison.json` has per-group results. `tmp/assignment-capo2.npy` and `.json` are the rendering candidate and provenance. `tmp/strings-side-by-side-capo2.mp4` shows this candidate on the left against the existing handcrafted Cupid render on the right. All trial outputs remain outside Cupid's project directory.

## Non-interactive Codex PDF refinement smoke test

Implemented `uv run refine-transcription` with PDF page rasterization, Codex read-only image inference constrained by a JSON schema, validated/cached extracted events, and Python sequence alignment. See the README for commands and prerequisites. No PDF file is edited. Input arrays require a matching generation companion checksum.

The test used PDF page 1, bars 2–8, capo 2, and the 15-second `command-transcription.npy` produced by the packaged generation command. It matched 42 of 51 audio notes, changed four string assignments and retained nine unresolved notes. All audio pitches, starts and ends were asserted unchanged. The reference NPY was not supplied to Codex or the aligner; it was used only for evaluation.

Against corrected Cupid annotations at ±50 ms over 21 matched attack groups, conditional string F1 rose from 75.0% to 78.6%, with exact sets increasing from 14 to 15. This is a quick, previously inspected excerpt comparison, not verified general accuracy. PDF extraction and alignment both remain potential sources of error. The aligner uses 80 ms attack grouping, exact pitch matches and insertion/deletion penalties; ambiguity and unmatched notes are retained rather than fabricated. It does not recover missing audio notes, correct detected pitches or solve every split/rolled chord.

Artifacts: `tmp/pdf-refined.npy`, `.json`, `tmp/pdf-refinement/tabs.json`, extraction prompt/schema/page images/log, and `tmp/pdf-refinement-comparison.json`. The source generation and canonical Cupid files remain unchanged.
