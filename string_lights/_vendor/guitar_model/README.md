# Instrument-informed guitar inference architecture

`cnn_mtl.py` and `base_model.py` (trailing whitespace normalized) from
https://github.com/ErenReyhanlioglu/Guitar-Transcription
revision ec63f36b3329020211665ae4be18a949647edf8f.
Copyright (c) 2026 M. Eren Reyhanlıoğlu; MIT license reproduced in LICENSE.

config.json is the published YAML config converted to JSON from
https://github.com/ErenReyhanlioglu/Guitar-Transcription-Weights
revision a03b4470a69bffa7799b1926da8eea21b5d3c204,
cnn_mtl_20260929_191206/config.yaml. Only fold 1 is used.

Feature formulas in transcription_models.py reproduce the HCQT/Mel operations
in Frank Cwitkowitz's amt-tools (MIT), revision
7822dc253912b7d514143d91c15eaae0e3ce9d56, without its dataset/UI dependencies.
Weights download separately with pinned revisions and SHA256 verification.
