"""Generate string-highlight frames from manually annotated guitar tablature JSON.

This does not parse PDFs. See docs/tab-transcription.md for the input format.
Column order is low E, A, D, G, B, high e; timing uses quarter-note beats.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


TRANSCRIPTION_FPS = 44


def generate(score, *, bpm, first_bar, last_bar, beats_per_bar, offset, pulse,
             include_percussion):
    bars = {int(bar): entries for bar, entries in score["bars"].items()}
    if not bars or min(bars) < 1:
        raise ValueError("Provide at least one positive bar number")
    if not (np.isfinite([bpm, beats_per_bar, offset, pulse]).all()
            and bpm > 0 and beats_per_bar > 0 and offset >= 0 and pulse > 0):
        raise ValueError("Use finite positive BPM, beats per bar and pulse, and a nonnegative offset")
    first_bar = min(bars) if first_bar is None else first_bar
    last_bar = max(bars) if last_bar is None else last_bar
    if not 1 <= first_bar <= last_bar:
        raise ValueError("Use a positive first bar and a last bar at or after it")
    duration = offset + (last_bar - first_bar + 1) * beats_per_bar * 60 / bpm
    frames = np.zeros((int(np.ceil(duration * TRANSCRIPTION_FPS)), 6), dtype=np.int64)
    events = []
    for bar, entries in sorted(bars.items()):
        if not isinstance(entries, list):
            raise ValueError(f"Bar {bar}: expected an array of entries")
        for entry in entries:
            if not isinstance(entry, list) or len(entry) not in (2, 3):
                raise ValueError(f"Bar {bar}: expected [beat, notes, optional technique]")
            beat, notes, *technique = entry
            if (isinstance(beat, bool) or not isinstance(beat, (int, float))
                    or not np.isfinite(beat) or not 0 <= beat < beats_per_bar):
                raise ValueError(f"Bar {bar}: beat must be within the bar")
            if (not isinstance(notes, str) or not notes.split()
                    or any(not re.fullmatch(r"[EADGBe](?:\d+|X)", note) for note in notes.split())):
                raise ValueError(f"Bar {bar}: use notes such as E3, A0, e5 or EX")
            if technique and not isinstance(technique[0], str):
                raise ValueError(f"Bar {bar}: technique must be text")
            if not first_bar <= bar <= last_bar:
                continue
            if not include_percussion:
                notes = " ".join(note for note in notes.split() if not note.endswith("X"))
            if not notes:
                continue
            start = offset + ((bar - first_bar) * beats_per_bar + beat) * 60 / bpm
            columns = sorted({"EADGBe".index(note[0]) for note in notes.split()})
            first = round(start * TRANSCRIPTION_FPS)
            stop = min(len(frames), first + max(1, round(pulse * TRANSCRIPTION_FPS)))
            frames[first:stop, columns] = 1
            events.append(dict(bar=bar, beat=beat, seconds=start, notes=notes,
                               columns=columns, technique=technique[0] if technique else "pluck"))
    metadata = dict(
        source=score.get("source", "Manually annotated tablature"), bpm=bpm,
        first_bar=first_bar, last_bar=last_bar, beats_per_bar=beats_per_bar,
        include_percussion=include_percussion, transcription_fps=TRANSCRIPTION_FPS,
        offset_seconds=offset, pulse_seconds=pulse,
        columns=["E2", "A2", "D3", "G3", "B3", "E4"],
        caveats=["Timing follows supplied beat annotations; review against the recording",
                 "Bars before first_bar omitted; first_bar begins at offset_seconds",
                 "Strums simultaneous; technique text does not create extra onsets",
                 "Muted strikes included" if include_percussion else "Muted/percussive strikes omitted; coincident pitched notes retained",
                 "No automatic alignment to recorded audio"], events=events,
    )
    if "capo" in score:
        metadata["capo"] = score["capo"]
    return frames, metadata, duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score", type=Path, help="Manually annotated score JSON")
    parser.add_argument("--bpm", type=float, help="Override the score's quarter-note BPM")
    parser.add_argument("--beats-per-bar", type=float, help="Override quarter-note beats per bar (default: 4)")
    parser.add_argument("--first-bar", type=int, help="First included bar (default: score first_bar or earliest bar)")
    parser.add_argument("--last-bar", type=int, help="Last included bar (default: latest bar)")
    parser.add_argument("--offset", type=float, default=0, help="Seconds before the first included bar")
    parser.add_argument("--pulse", type=float, default=.09, help="Highlight duration in seconds")
    parser.add_argument("--include-percussion", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy path; also writes a .json sidecar")
    args = parser.parse_args()
    if args.output.suffix != ".npy":
        parser.error("Output must end in .npy")
    if args.output.with_suffix(".json").resolve() == args.score.resolve():
        parser.error("Output sidecar must not overwrite the input score")
    try:
        score = json.loads(args.score.read_text())
        frames, metadata, duration = generate(
            score, bpm=args.bpm if args.bpm is not None else score["bpm"],
            beats_per_bar=args.beats_per_bar if args.beats_per_bar is not None else score.get("beats_per_bar", 4),
            first_bar=args.first_bar if args.first_bar is not None else score.get("first_bar"),
            last_bar=args.last_bar, offset=args.offset, pulse=args.pulse,
            include_percussion=args.include_percussion,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.error(f"Invalid score or options: {exc}")
    metadata["score_file"] = str(args.score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, frames)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output}: {frames.shape}, {duration:.3f}s, {len(metadata['events'])} events")


if __name__ == "__main__":
    main()
