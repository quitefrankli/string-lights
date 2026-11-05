"""Generate renderer-ready string activations and a reproducible JSON companion."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from . import paths
from .assign_strings import assign, export
from .transcription_models import infer, sha256


def extract(video, duration=None):
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)]))
    audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    if audio_stream is None or video_stream is None:
        raise ValueError('Input must contain audio and video streams')
    offset = float(audio_stream.get('start_time', 0)) - float(video_stream.get('start_time', 0))
    length = float(video_stream.get('duration', probe['format'].get('duration', 0)))
    length = min(length, duration) if duration is not None else length
    if not np.isfinite([offset, length]).all() or length <= 0:
        raise ValueError('Could not determine a positive video duration')
    cmd = ['ffmpeg', '-v', 'error', '-i', str(video), '-map', '0:a:0', '-ac', '1', '-ar', '22050']
    cmd += ['-t', str(max(.001, length - offset)), '-f', 'f32le', '-']
    audio = np.frombuffer(subprocess.check_output(cmd), dtype='<f4').copy()
    if len(audio) < 2048:
        raise ValueError('Audio is too short for transcription')
    return audio, 22050, offset, length


def generate(video, *, capo, tuning, model, device, cache_dir, duration, pulse, evidence_window):
    audio, sr, offset, length = extract(video, duration)
    notes, seconds, activity, provenance = infer(
        audio, sr, model=model, device=device, cache_dir=cache_dir)
    # Audio times become video-relative times once; trim at video/excerpt boundaries.
    notes = [dict(n, start=max(0., n['start'] + offset), end=min(length, n['end'] + offset))
             for n in notes if n['start'] + offset < length and n['end'] + offset > 0]
    notes = [n for n in notes if n['end'] > n['start']]
    events = assign(notes, seconds + offset, activity, capo=capo,
                    tuning=tuning, evidence_window=evidence_window)
    frames = export(events, length, pulse)
    metadata = dict(
        schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
        source=dict(path=str(video.resolve()), sha256=sha256(video)),
        duration_seconds=length, audio_offset_seconds=offset,
        transcription_fps=44, columns=['E2', 'A2', 'D3', 'G3', 'B3', 'E4'],
        column_order='low_to_high', tuning_midi=list(tuning), capo=capo,
        pulse_seconds=pulse, evidence_window_seconds=evidence_window,
        max_physical_fret=19, beam_size=128, models=provenance,
        string_lights_version=importlib.metadata.version('string-lights'),
        events=events, unassigned_count=sum(e['string'] is None for e in events),
        caveats=['String/fret assignments are estimates; review in the tabs editor',
                 'Fret numbers are relative to the capo',
                 'Overlapping pulses on the same string merge',
                 'PDF tab refinement has not been applied',
                 'String model trained on standard-tuned guitar; alternate tunings may reduce accuracy'],
    )
    return frames, metadata


def write_outputs(output, frames, metadata, force=False):
    sidecar = output.with_suffix('.json')
    # Recheck after inference, before modifying either artifact.
    if not force and (output.exists() or sidecar.exists()):
        raise ValueError('Output or JSON companion already exists; use --force to replace both')
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for target in (output, sidecar):
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
                staged.append((Path(stream.name), target))
                if target == output:
                    np.save(stream, frames, allow_pickle=False)
                else:
                    payload = dict(metadata, array_sha256=sha256(staged[0][0]))
                    stream.write((json.dumps(payload, indent=2, allow_nan=False) + '\n').encode())
        for temporary, target in staged:
            if force:
                os.replace(temporary, target)
            else:
                # Exclusive creation, including races with another generator.
                os.link(temporary, target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Project name or explicit video path')
    parser.add_argument('--capo', type=int, default=0, help='Capo fret (default: 0)')
    parser.add_argument('--tuning', nargs=6, type=int, default=[40,45,50,55,59,64],
                        metavar='MIDI', help='Six open-string MIDI pitches, low to high, before capo')
    parser.add_argument('--model', choices=['gaps', 'fl'], default='gaps')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--projects-dir', type=Path, default=Path('data/projects'))
    parser.add_argument('--project-dir', type=Path)
    parser.add_argument('--output', type=Path, help='Default: project transcription.npy; also writes same-stem JSON')
    parser.add_argument('--model-cache', type=Path, default=Path('data/tmp/transcription-models'))
    parser.add_argument('--duration', type=float, help='Analyze only the first N seconds')
    parser.add_argument('--pulse', type=float, default=.09, help='Light hold in seconds')
    parser.add_argument('--evidence-window', type=float, default=.08)
    parser.add_argument('--force', action='store_true', help='Replace existing output and companion JSON')
    args = parser.parse_args()
    if not 0 <= args.capo <= 19 or any(not 0 <= p <= 127 for p in args.tuning):
        parser.error('Capo must be 0..19 and tuning pitches must be 0..127')
    if args.pulse <= 0 or not np.isfinite(args.pulse) or args.evidence_window < 0 or not np.isfinite(args.evidence_window):
        parser.error('Use a finite positive pulse and finite nonnegative evidence window')
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error('Duration must be finite and positive')
    previous_root = paths.PROJECTS_DIR
    paths.PROJECTS_DIR = args.projects_dir
    token = paths.project_override.set(args.project_dir)
    try:
        video = paths.resolve_video(args.input)
        output = args.output if args.output else paths.project_dir(args.input) / 'transcription.npy'
        if output.suffix != '.npy':
            raise ValueError('Output must end in .npy')
        if video.resolve() in (output.resolve(), output.with_suffix('.json').resolve()):
            raise ValueError('Output must not overwrite the input')
        if not args.force and (output.exists() or output.with_suffix('.json').exists()):
            raise ValueError(f'Output or JSON companion exists: {output}; use --force to replace both')
        frames, metadata = generate(
            video, capo=args.capo, tuning=args.tuning, model=args.model,
            device=args.device, cache_dir=args.model_cache, duration=args.duration,
            pulse=args.pulse, evidence_window=args.evidence_window)
        write_outputs(output, frames, metadata, args.force)
        print(f'Saved {output} and {output.with_suffix(".json")} '
              f'({len(metadata["events"])} detected notes, {metadata["unassigned_count"]} unassigned)')
    except (ValueError, OSError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    finally:
        paths.project_override.reset(token)
        paths.PROJECTS_DIR = previous_root


if __name__ == '__main__':
    main()
