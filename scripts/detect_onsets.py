"""Detect unassigned audio attacks and optionally benchmark a 44 Hz transcription.

Run with the audio optional dependency installed. All outputs require an explicit
output directory; input video and reference transcription are opened read-only.
"""
import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np


def reference_onsets(tab, fps=44):
    tab = np.asarray(tab)
    if tab.ndim != 2 or tab.shape[1] != 6 or not np.isfinite(tab).all():
        raise ValueError('Reference must be a finite (N, 6) array')
    active = tab > 0
    previous = np.vstack([np.zeros((1, 6), dtype=bool), active])[:-1]
    return np.flatnonzero((active & ~previous).any(axis=1)) / fps


def group_onsets(times, window):
    """Group within a fixed window of the first attack, without chaining."""
    grouped = []
    for time in times:
        if not grouped or time - grouped[-1] > window:
            grouped.append(float(time))
    return np.array(grouped)


def compare(reference, predicted, tolerance):
    """Chronological one-to-one matching, maximizing match count in tolerance."""
    i = j = 0
    errors, matches = [], []
    while i < len(reference) and j < len(predicted):
        difference = predicted[j] - reference[i]
        if difference < -tolerance:
            j += 1
        elif difference > tolerance:
            i += 1
        else:
            errors.append(float(difference))
            matches.append([i, j])
            i += 1
            j += 1
    hits = len(errors)
    precision = hits / len(predicted) if len(predicted) else 0
    recall = hits / len(reference) if len(reference) else 0
    return dict(reference_count=len(reference), predicted_count=len(predicted),
                matches=hits, false_positives=len(predicted)-hits,
                misses=len(reference)-hits, precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                median_absolute_error_ms=float(np.median(np.abs(errors))*1000) if errors else None,
                p95_absolute_error_ms=float(np.percentile(np.abs(errors),95)*1000) if errors else None,
                matched_indices=matches)


def write_review(path, audio, sr, predicted, reference):
    # Stereo: detected clicks left, reference clicks right; original in both.
    mix = np.column_stack([audio, audio]) * .65
    for channel, times, frequency in [(0, predicted, 1600), (1, reference, 1000)]:
        n = int(.025 * sr)
        click = .3*np.sin(2*np.pi*frequency*np.arange(n)/sr)*np.exp(-np.arange(n)/(sr*.006))
        for time in times:
            start = round(time*sr)
            stop = min(len(audio), start+n)
            if 0 <= start < stop:
                mix[start:stop,channel] += click[:stop-start]
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(2); output.setsampwidth(2); output.setframerate(sr)
        output.writeframes((np.clip(mix,-1,1)*32767).astype('<i2').tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--delta', default=.07, type=float, help='Normalized peak threshold')
    parser.add_argument('--min-gap', default=.03, type=float)
    parser.add_argument('--group-window', default=.10, type=float)
    args = parser.parse_args()
    if not all(np.isfinite(v) and v >= 0 for v in (args.delta,args.min_gap,args.group_window)):
        parser.error('Thresholds and windows must be finite and nonnegative')
    import librosa
    from string_lights.timings import detect
    times, audio, sr, offset = detect(args.video, delta=args.delta, min_gap=args.min_gap)
    hop = 220
    reference = reference_onsets(np.load(args.reference,allow_pickle=False)) if args.reference else np.array([])
    output = args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    metadata = dict(video=str(args.video),sample_rate=sr,hop_length=hop,audio_offset_seconds=offset,
        detector='librosa spectral flux / onset_detect',librosa_version=librosa.__version__,
        delta=args.delta,min_gap_seconds=args.min_gap,
        events=[dict(seconds=float(t),strings=None) for t in times])
    (output/'onsets.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if args.reference:
        report = dict(reference=str(args.reference),reference_fps=44,group_window_seconds=args.group_window,
            note='Rising edges per string, simultaneous starts deduplicated. Grouped metrics group both sequences. No offset fitted.',
            metrics={})
        for label,ref,pred in [('strict',reference,times),('grouped',group_onsets(reference,args.group_window),group_onsets(times,args.group_window))]:
            for tolerance in [.025,.05]:
                report['metrics'][f'{label}_{round(tolerance*1000)}ms'] = compare(ref,pred,tolerance)
        (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({key:{k:v for k,v in result.items() if k!='matched_indices'} for key,result in report['metrics'].items()},indent=2))
    write_review(output/'review.wav',audio,sr,times-offset,reference-offset)
    # Standalone timeline: audio player and SVG markers, no external dependencies.
    duration = len(audio)/sr
    rows = []
    for label,events,color,y in [('Reference',reference,'#157',35),('Detected',times,'#d62',80)]:
        marks = ''.join(f'<path d="M {t/duration*1400:.2f} {y} v 25"><title>{t:.3f}s</title></path>' for t in events)
        rows.append(f'<text x="0" y="{y-5}">{label}</text><g stroke="{color}">{marks}</g>')
    (output/'timeline.html').write_text('<!doctype html><meta charset="utf-8"><title>Onset comparison</title><h1>Onset comparison</h1><p>Left: detected clicks. Right: reference clicks. Hover markers for seconds.</p><audio controls src="review.wav"></audio><div style="overflow:auto"><svg xmlns="http://www.w3.org/2000/svg" width="2800" viewBox="0 0 1400 115">'+''.join(rows)+'</svg></div>')
    print(f'Wrote {len(times)} attacks and review artifacts to {output}')


if __name__ == '__main__':
    main()
