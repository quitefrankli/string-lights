// Existing .npy files use eleven rows per quarter second.
const TRANSCRIPTION_FPS = 44;

function notesFromFrames(frames) {
  const notes = [];
  for (let string = 0; string < 6; string++) {
    let start = null;
    let value = 0;
    for (let f = 0; f <= frames.length; f++) {
      const next = f < frames.length ? Number(frames[f][string]) : 0;
      if (next === value) continue;
      if (start !== null) notes.push({ string, start: start / TRANSCRIPTION_FPS,
        end: f / TRANSCRIPTION_FPS, value });
      start = next > 0 ? f : null;
      value = next;
    }
  }
  return notes;
}

function notesToFrames(notes, duration) {
  const end = Math.max(duration, ...notes.map(note => note.end), 0);
  const frames = Array.from({ length: Math.ceil(end * TRANSCRIPTION_FPS) }, () => Array(6).fill(0));
  for (let string = 0; string < 6; string++) {
    const ordered = notes.filter(note => note.string === string).sort((a, b) => a.start - b.start);
    ordered.forEach((note, index) => {
      const start = Math.max(0, Math.round(note.start * TRANSCRIPTION_FPS));
      let stop = Math.min(frames.length, Math.max(start + 1, Math.round(note.end * TRANSCRIPTION_FPS)));
      const next = ordered[index + 1];
      if (next) {
        // A silent frame separates repeated plucks with the same stored value.
        const nextStart = Math.round(next.start * TRANSCRIPTION_FPS);
        stop = Math.min(stop, nextStart - (next.value === note.value ? 1 : 0));
      }
      for (let f = start; f < stop; f++) frames[f][string] = note.value;
    });
  }
  return frames;
}

function scaleNotes(notes, ratio) {
  return notes.map(note => ({ ...note, start: note.start * ratio, end: note.end * ratio }));
}

function snapTime(time, interval, enabled) {
  return Math.max(0, enabled ? Math.round(time / interval) * interval : time);
}
