"""Spectral-flux timing baseline retained for diagnostic scripts."""
import json
import subprocess

import numpy as np



def detect(video, *, delta=.07, min_gap=.03):
    """Return attack seconds, decoded audio, sample rate and audio/video offset."""
    if not all(np.isfinite(v) and v >= 0 for v in (delta, min_gap)):
        raise ValueError('Threshold and minimum gap must be finite and nonnegative')
    import librosa
    sr, hop = 22050, 220
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)]))
    audio_stream = next((s for s in probe['streams'] if s['codec_type']=='audio'), None)
    video_stream = next((s for s in probe['streams'] if s['codec_type']=='video'), None)
    if audio_stream is None or video_stream is None:
        raise ValueError('Input must contain both an audio and a video stream')
    offset = float(audio_stream.get('start_time',0))-float(video_stream.get('start_time',0))
    raw = subprocess.check_output(['ffmpeg','-v','error','-i',str(video),'-map','0:a:0','-ac','1','-ar',str(sr),'-f','f32le','-'])
    audio = np.frombuffer(raw,dtype='<f4').copy()
    envelope = librosa.onset.onset_strength(y=audio,sr=sr,hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=envelope,sr=sr,hop_length=hop,
        backtrack=False,delta=delta,wait=max(0,round(min_gap*sr/hop)))
    times = librosa.frames_to_time(frames,sr=sr,hop_length=hop)+offset
    times = times[times>=0]
    return times.astype(np.float64), audio, sr, offset
