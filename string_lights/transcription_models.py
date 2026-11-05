"""Pinned pretrained guitar inference, independent of experiment directories."""
import hashlib
import json
from pathlib import Path

import numpy as np

PITCH_REVISION = 'b7bec65a2b860aca72856b0feef58b5df407b777'
PITCH_HASHES = {
    'gaps': '65483e7c0e340a90415b15b520687587698c8c728f5fa470a205f13ee45c6513',
    'fl': '50d93dba89bdd3401849bc735614478e83d9f46d21fa3f71d8aca5acc0a52028',
}
STRING_REVISION = 'a03b4470a69bffa7799b1926da8eea21b5d3c204'
STRING_HASH = '3562b9fae650a407e8889d1bf860556dc82a7920e3b66fbc15d47c2cbcb5270e'
STRING_URL = ('https://raw.githubusercontent.com/ErenReyhanlioglu/'
              f'Guitar-Transcription-Weights/{STRING_REVISION}/'
              'cnn_mtl_20260929_191206/fold_1/model_best.pt')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'Checkpoint checksum mismatch: {path}')
    return Path(path)


def string_features(audio, sample_rate=22050):
    """Reproduce amt-tools 7822dc2 HCQT/Mel preprocessing, without its dataset stack."""
    import librosa
    harmonics = []
    for harmonic in [.5, 1, 2, 3, 4, 5]:
        magnitude = np.abs(librosa.vqt(
            y=audio, sr=sample_rate, hop_length=512,
            fmin=librosa.note_to_hz('E2') * harmonic,
            n_bins=144, bins_per_octave=36, gamma=0))
        harmonics.append(librosa.amplitude_to_db(magnitude, ref=np.max) / 80 + 1)
    frames = min(v.shape[-1] for v in harmonics)
    hcqt = np.stack([v[:, :frames] for v in harmonics])
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, hop_length=512, n_mels=256, n_fft=2048)
    mel = (librosa.power_to_db(mel, ref=np.max) / 80 + 1)[None]
    frames = min(frames, mel.shape[-1])
    return {'hcqt': hcqt[..., :frames], 'mel': mel[..., :frames]}


def infer(audio, sample_rate, *, model, device, cache_dir):
    import librosa
    import torch
    from huggingface_hub import hf_hub_download
    from piano_transcription_inference import PianoTranscription
    from ._vendor.guitar_model.cnn_mtl import cnn_mtl

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not accessible to this process; use --device cpu or enable GPU access')
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = hf_hub_download(
        'xavriley/midi-transcription-models', f'guitar-{model}.pth',
        revision=PITCH_REVISION, cache_dir=cache_dir / 'huggingface')
    verified(checkpoint, PITCH_HASHES[model])
    string_checkpoint = cache_dir / 'guitar-activity-fold1.pt'
    if not string_checkpoint.exists():
        temporary = cache_dir / 'guitar-activity-fold1.download'
        try:
            torch.hub.download_url_to_file(STRING_URL, str(temporary))
            verified(temporary, STRING_HASH)
            temporary.replace(string_checkpoint)
        finally:
            temporary.unlink(missing_ok=True)
    verified(string_checkpoint, STRING_HASH)

    print(f'Transcribing pitches with {model} on {device}…', flush=True)
    pitch_audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    transcriber = PianoTranscription(
        model_type='Regress_onset_offset_frame_velocity_CRNN',
        checkpoint_path=str(checkpoint), device=device, batch_size=8)
    result = transcriber.transcribe(pitch_audio, None)
    notes = [dict(start=float(n['onset_time']), end=float(n['offset_time']),
                  pitch=int(n['midi_note']), velocity=int(n['velocity']))
             for n in result['est_note_events']]
    del transcriber, result

    print('Estimating string activity…', flush=True)
    feature_audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=22050)
    features = string_features(feature_audio)
    config_path = Path(__file__).parent / '_vendor/guitar_model/config.json'
    config = json.loads(config_path.read_text())
    network = cnn_mtl(config).to(device).eval()
    network.load_state_dict(torch.load(string_checkpoint, map_location=device, weights_only=True))
    frames = min(v.shape[-1] for v in features.values())
    windows = {
        key: torch.from_numpy(np.pad(value[..., :frames], ((0,0),(0,0),(9,9))))
        .float().unfold(-1, 19, 1).permute(2,0,1,3)
        for key, value in features.items()
    }
    activity = []
    with torch.inference_mode():
        for start in range(0, frames, 64):
            prediction = network({k: v[start:start+64].to(device) for k,v in windows.items()})
            activity.append(prediction['activity_logits'].sigmoid().cpu().numpy())
    provenance = dict(
        pitch_model=model, pitch_repository='xavriley/midi-transcription-models',
        pitch_revision=PITCH_REVISION, pitch_sha256=PITCH_HASHES[model],
        string_model='instrument-informed-multitask', string_fold=1,
        string_revision=STRING_REVISION, string_sha256=STRING_HASH,
        architecture_revision='ec63f36b3329020211665ae4be18a949647edf8f',
        feature_implementation='amt-tools-7822dc2-equivalent', device=device,
        pitch_inference_revision='7568dc7f78b625e40cf9776e2806d164006610e3',
        onset_threshold=.3, offset_threshold=.3, frame_threshold=.1,
        string_config_sha256=sha256(config_path),
        torch_version=torch.__version__, librosa_version=librosa.__version__,
    )
    return notes, np.arange(frames) * 512 / 22050, np.concatenate(activity), provenance
