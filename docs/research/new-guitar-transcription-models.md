# Newer audio-to-guitar transcription candidates

Checked 2026-11-04 against authors' papers, repositories and model cards. No models below have been run on Cupid. Its handcrafted transcription is approximate; local agreement scores are not verified transcription accuracy. Cupid's directory must remain untouched.

## Recommendation

Continue investigating audio transcription. A bad TabCNN reconstruction does not establish a ceiling for newer approaches. Separate two experiments: (1) reliable pitches and attack times, assessed by a simple resynthesis; (2) actual string identity, assessed against reviewed video/reference events. A musically plausible fingering can still light the wrong string in a performance video.

Start with the released GAPS/François Leduc guitar transcription models for pitch and onset quality. Then try a direct string model, preferably the released instrument-informed multi-task model, with Noise2Fret as a distinct research candidate. Do not train a new model before testing released checkpoints. This order is an engineering recommendation based on accessibility and task fit, not a measured ranking on this project's recordings.

## Guitar-adapted audio-to-MIDI: practical first test

[Xavier Riley's inference package](https://github.com/xavriley/hf_midi_transcription) exposes pretrained guitar transcription models through a CLI. The [authors' model repository](https://huggingface.co/xavriley/midi-transcription-models) links the 2024 high-resolution guitar domain-adaptation and GAPS papers and provides `guitar-gaps.pth` (99.2 MB); its model card specifies MIT. This is a concrete alternative to generic peak detection and the old TabCNN path. Its [instrument configuration](https://github.com/xavriley/hf_midi_transcription/blob/main/instruments.json) maps `guitar` to GAPS and `guitar_fl` to François Leduc. It produces note timing/pitch, not string identity. The package README's monophonic wording is confusing relative to the polyphonic guitar research; verify actual checkpoint behavior on overlapping notes in the first smoke test.

[MuScriptor](https://github.com/muscriptor/muscriptor), released July 2026, provides `uvx muscriptor transcribe` and a modern audio-to-score alternative. Its [project paper/demo](https://muscriptor.github.io/) compares with YourMT3+. Checkpoint access requires Hugging Face login and acceptance of noncommercial terms (CC BY-NC). Its [medium model card](https://huggingface.co/MuScriptor/muscriptor-medium) supports acoustic-guitar conditioning and note-event outputs, but cannot represent simultaneous notes of the same pitch and instrument—relevant to unisons on different strings. It does not solve guitar string identity directly. Consider it as a second pitch/onset candidate, subject to those access conditions.

## Direct string/fret models

### Instrument-informed multi-task model: clearest available checkpoint path

The [authors' repository](https://github.com/ErenReyhanlioglu/Guitar-Transcription) combines harmonic CQT and Mel features, supervising fret region, six-string activity, pitch class and multi-pitch estimates. It reports GuitarSet leave-one-player-out tab F1 0.776 and TDR 0.943; the auxiliary string-activity head reports F1 0.836. TDR is conditional on correctly detected pitches, not overall transcription accuracy. The authors explicitly say total F1 does not beat their strongest cited baseline. The paper is under review.

The [weights repository](https://github.com/ErenReyhanlioglu/Guitar-Transcription-Weights) contains six fold-specific `model_best.pt` state dictionaries, exact configuration and evaluation reports, with MIT licensing. Its published inference loading path is concrete. Current setup uses Python 3.12/3.13 for the model and a separate Python 3.11 feature-extraction environment because older dependencies require it. A standalone arbitrary-WAV adapter is still needed; the documented workflow is dataset-oriented.

For string-lights, the six-string activity output is especially relevant: evaluate that directly rather than requiring every fret prediction to be right. It represents sounding activity, however, not necessarily a fresh pluck. Attack decoding must distinguish sustained notes from retriggers.

### Noise2Fret: meaningful architectural change, more integration work

The [August 2026 paper](https://arxiv.org/html/2608.30854v1) uses diffusion plus physical/musical constraints rather than independent frame classification. It reports GuitarSet tablature F1 0.845 versus 0.717 for its TabCNN comparison, and TDR 0.988. These are the paper's task/protocol results, not a comparison with our timing-only metric; they do not predict performance on Cupid. Its GOAT baselines were reimplemented/adapted, another reason not to combine tables across papers into a league table.

The [official repository](https://github.com/RiccardoVib/Noise2Fret) now distinguishes the published auxiliary-loss model and a newer event-based variant under review. The [published-model README](https://github.com/RiccardoVib/Noise2Fret/blob/main/Noise2Fret_Aux_Losses/README.md) links pretrained weights at [Zenodo record 23361924](https://zenodo.org/records/23361924) and documents inference through dataset-oriented training scripts with training disabled. The weight link was found, but the record contents/download could not be verified in this research pass. Do not describe the weights as successfully downloaded or tested.

The [event-based variant](https://github.com/RiccardoVib/Noise2Fret/blob/main/Noise2Fret_and_Onsets/README.md) adds event-count and onset heads, directly relevant to repeated plucks and timing. Its README lists a weights directory, but no verified downloadable checkpoint was established here. This is promising research rather than a ready CLI recommendation.

### TART: promising strategy, release availability unresolved

The [September 2026 TART paper](https://arxiv.org/html/2609.11904v1) separates guitar-adapted high-resolution note transcription, technique classification, audio-conditioned T5 string/fret assignment and score generation. Importantly, the string stage retains acoustic evidence instead of assigning a fingering from MIDI alone. Across four clean/noisy benchmarks, authors report 81.35% audio-to-MIDI F50, 71.8% string/fret Tab F1 and 54.08% end-to-end Tab F1; the intermediate and final metrics must not be conflated. I did not locate an official downloadable code/checkpoint release through the paper or targeted search. Keep it on the research shortlist, not the immediate executable plan.

## What to measure locally

Use short passages covering single notes, ringing polyphony, repeated plucks and strums. Keep a fixed small manually reviewed audio/video subset, and use the rest of Cupid only for approximate-reference comparisons. Evaluate pitch/onset correctness separately from string assignment. Retain probabilities when available, inspect per-string activation plots, and listen to resynthesis; audio playback alone cannot reveal a same-pitch wrong-string assignment. Record onset decoding and suppression settings so frame flicker is not mistaken for new notes.

All exports, downloaded trial assets and adapters used for Cupid experiments belong in the separate benchmark project's `tmp/`, with explicit paths. Do not replace the project's existing `transcription.npy`.
