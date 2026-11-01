const STRINGS = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4'];
let notes = [];
let gridInterval = 0.25;
let pixelsPerSecond = 80;
let transcriptionDuration = 0;
let loadGeneration = 0;
const history = [];
const video = document.getElementById('video');
const playhead = document.getElementById('playhead');
const gridWrap = document.getElementById('grid-wrap');
const labelsEl = document.getElementById('labels');
const rowsEl = document.getElementById('rows');
const selEl = document.getElementById('selection');
let selection = null;
let copyBuffer = null;
let gesture = null;
let reverseStrings = false;

function assignedString(row) {
  return reverseStrings ? STRINGS.length - 1 - row : row;
}

function noteTitle(note) {
  return `${STRINGS[assignedString(note.string)]} at ${note.start.toFixed(3)}s — drag to move; Alt = free timing; right-click = remove`;
}

function timelineDuration() {
  return Math.max(10, Number.isFinite(video.duration) ? video.duration : 0,
    transcriptionDuration, ...notes.map(note => note.end));
}

function snapshot() {
  return { notes: notes.map(note => ({ ...note })), gridInterval, transcriptionDuration };
}

function saveUndo() { loadGeneration++; history.push(snapshot()); }

function renderTimeline() {
  const duration = timelineDuration();
  const width = Math.ceil(duration * pixelsPerSecond);
  labelsEl.innerHTML = '';
  rowsEl.innerHTML = '';
  const timeEl = document.getElementById('time-numbers');
  timeEl.innerHTML = '';
  document.getElementById('grid-inner').style.width = width + 'px';
  // Thin grid lines stay visible when zoomed out without changing snap timing.
  const tickStep = gridInterval * Math.max(1, Math.ceil(Math.max(12 / pixelsPerSecond, duration / 2000) / gridInterval));
  for (let t = 0; t < duration; t += tickStep) {
    const tick = document.createElement('div');
    tick.className = 'hcell time-tick';
    tick.style.width = tickStep * pixelsPerSecond + 'px';
    tick.textContent = t.toFixed(2) + 's';
    timeEl.appendChild(tick);
  }
  for (let si = STRINGS.length - 1; si >= 0; si--) {
    const label = document.createElement('div');
    label.className = 'string-label';
    label.textContent = STRINGS[assignedString(si)];
    label.dataset.s = si;
    labelsEl.appendChild(label);
    const row = document.createElement('div');
    row.className = 'row note-row';
    row.dataset.s = si;
    row.style.width = width + 'px';
    row.style.backgroundSize = tickStep * pixelsPerSecond + 'px 100%';
    notes.forEach((note, index) => {
      if (note.string !== si) return;
      const marker = document.createElement('div');
      marker.className = 'note';
      marker.dataset.note = index;
      marker.style.left = note.start * pixelsPerSecond + 'px';
      marker.style.width = Math.max(6, (note.end - note.start) * pixelsPerSecond) + 'px';
      marker.title = noteTitle(note);
      row.appendChild(marker);
    });
    rowsEl.appendChild(row);
  }
  updateSelectionEl();
  updatePlayhead();
}

function updatePlayhead() {
  const x = video.currentTime * pixelsPerSecond;
  playhead.style.left = x + 'px';
  if (!video.paused) {
    const width = gridWrap.offsetWidth - labelsEl.offsetWidth;
    if (x < gridWrap.scrollLeft || x > gridWrap.scrollLeft + width - 80) {
      gridWrap.scrollLeft = Math.max(0, x - width / 3);
    }
  }
}
video.addEventListener('timeupdate', updatePlayhead);
video.addEventListener('seeked', updatePlayhead);

function rawTime(clientX) {
  const rect = document.getElementById('grid-inner').getBoundingClientRect();
  return Math.max(0, Math.min(timelineDuration(), (clientX - rect.left) / pixelsPerSecond));
}
function editingTime(time, altKey) {
  return snapTime(time, gridInterval, document.getElementById('snap').checked && !altKey);
}
function pointerTime(event) { return editingTime(rawTime(event.clientX), event.altKey); }

function updateSelectionEl() {
  selEl.style.display = selection ? 'block' : 'none';
  if (!selection) return;
  selEl.style.left = Math.min(...selection) * pixelsPerSecond + 'px';
  selEl.style.width = Math.abs(selection[1] - selection[0]) * pixelsPerSecond + 'px';
}

playhead.addEventListener('mousedown', event => {
  event.preventDefault();
  event.stopPropagation();
  document.activeElement?.blur();
  gesture = { type: 'playhead' };
});
document.getElementById('time-numbers').addEventListener('mousedown', event => {
  if (event.button !== 0) return;
  event.preventDefault();
  document.activeElement?.blur();
  selection = [pointerTime(event), pointerTime(event)];
  gesture = { type: 'selection', clientX: event.clientX };
  updateSelectionEl();
});

rowsEl.addEventListener('mousedown', event => {
  if (event.button !== 0 && event.button !== 2) return;
  const row = event.target.closest('.note-row');
  if (!row) return;
  event.preventDefault();
  document.activeElement?.blur();
  const marker = event.target.closest('.note');
  if (marker) {
    const index = Number(marker.dataset.note);
    if (event.button === 2) {
      saveUndo();
      notes.splice(index, 1);
      renderTimeline();
    } else {
      gesture = { type: 'note', index, clientX: event.clientX,
        original: { ...notes[index] }, before: snapshot(), moved: false };
    }
    return;
  }
  if (event.button === 2) return;
  saveUndo();
  const start = pointerTime(event);
  notes.push({ string: Number(row.dataset.s), start, end: start + gridInterval, value: 1 });
  gesture = { type: 'paint', string: Number(row.dataset.s), last: start };
  renderTimeline();
});
rowsEl.addEventListener('contextmenu', event => event.preventDefault());

document.addEventListener('mousemove', event => {
  if (!gesture) return;
  if (gesture.type === 'playhead') video.currentTime = Math.min(rawTime(event.clientX), video.duration || Infinity);
  if (gesture.type === 'selection') {
    selection[1] = pointerTime(event);
    updateSelectionEl();
  }
  if (gesture.type === 'note' && (gesture.moved || Math.abs(event.clientX - gesture.clientX) > 3)) {
    const start = editingTime(gesture.original.start + (event.clientX - gesture.clientX) / pixelsPerSecond, event.altKey);
    notes[gesture.index] = { ...gesture.original, start,
      end: start + gesture.original.end - gesture.original.start };
    gesture.moved = true;
    renderTimeline();
  }
  if (gesture.type === 'paint' && document.getElementById('snap').checked && !event.altKey) {
    const row = event.target.closest?.('.note-row');
    if (!row) return;
    const start = pointerTime(event);
    const string = Number(row.dataset.s);
    if (start === gesture.last && string === gesture.string) return;
    if (!notes.some(note => note.string === string && Math.abs(note.start - start) < 1e-8)) {
      notes.push({ string, start, end: start + gridInterval, value: 1 });
      renderTimeline();
    }
    gesture.last = start;
    gesture.string = string;
  }
});
document.addEventListener('mouseup', event => {
  if (gesture?.type === 'note') {
    if (gesture.moved) history.push(gesture.before);
    else { history.push(gesture.before); notes.splice(gesture.index, 1); renderTimeline(); }
  }
  if (gesture?.type === 'selection' && Math.abs(event.clientX - gesture.clientX) < 3) {
    video.currentTime = Math.min(pointerTime(event), video.duration || Infinity);
    selection = null;
    updateSelectionEl();
  }
  gesture = null;
});

const STRING_KEYS = { q: 5, w: 4, e: 3, r: 2, t: 1, y: 0 };
function isInputFocused() {
  return ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
}
document.addEventListener('keydown', event => {
  if (isInputFocused()) return;
  if ((event.ctrlKey || event.metaKey) && event.key === 'z') {
    if (!history.length) return;
    event.preventDefault();
    const previous = history.pop();
    notes = previous.notes;
    gridInterval = previous.gridInterval;
    transcriptionDuration = previous.transcriptionDuration;
    document.getElementById('grid-interval').value = gridInterval * 1000;
    renderTimeline();
    return;
  }
  const interval = (Number(document.getElementById('interval').value) || 250) / 1000;
  if (event.key === ' ') {
    event.preventDefault();
    if (video.paused) video.play(); else video.pause();
  } else if (event.key === 'Enter') {
    event.preventDefault(); video.currentTime = 0;
  } else if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
    event.preventDefault();
    video.currentTime = Math.max(0, Math.min(video.duration || Infinity,
      video.currentTime + (event.key === 'ArrowRight' ? interval : -interval)));
  } else if (event.key === 'c' && selection) {
    const start = Math.min(...selection), end = Math.max(...selection);
    copyBuffer = { duration: end - start,
      notes: notes.filter(note => note.start >= start && note.start < end)
        .map(note => ({ ...note, start: note.start - start, end: note.end - start })) };
  } else if (event.key === 'v' && copyBuffer && video.paused) {
    saveUndo();
    const start = editingTime(video.currentTime, event.altKey);
    notes = notes.filter(note => note.start < start || note.start >= start + copyBuffer.duration);
    notes.push(...copyBuffer.notes.map(note => ({ ...note, start: note.start + start, end: note.end + start })));
    renderTimeline();
  } else if ((event.key === 'a' || event.key === 'd') && video.paused) {
    saveUndo();
    const start = editingTime(video.currentTime, event.altKey);
    if (event.key === 'a') notes = notes.filter(note => note.start < start || note.start >= start + gridInterval);
    notes = notes.map(note => note.start >= start ? { ...note,
      start: Math.max(0, note.start + (event.key === 'd' ? gridInterval : -gridInterval)),
      end: Math.max(0, note.end + (event.key === 'd' ? gridInterval : -gridInterval)) } : note);
    renderTimeline();
  } else if (event.key in STRING_KEYS && video.paused) {
    saveUndo();
    const string = STRING_KEYS[event.key], start = editingTime(video.currentTime, event.altKey);
    const index = notes.findIndex(note => note.string === string && Math.abs(note.start - start) < 1e-8);
    if (index >= 0) notes.splice(index, 1);
    else notes.push({ string, start, end: start + gridInterval, value: 1 });
    renderTimeline();
  }
});

async function loadNpy(stem, generation, undoable = false) {
  try {
    const response = await fetch('/load/' + encodeURIComponent(stem));
    if (generation !== loadGeneration) return;
    if (!response.ok) throw new Error(response.status === 404 ? 'File not found.' : (await response.json()).error);
    const { frames } = await response.json();
    if (generation !== loadGeneration) return;
    if (undoable) saveUndo();
    notes = notesFromFrames(frames).map(note => ({ ...note, string: assignedString(note.string) }));
    transcriptionDuration = frames.length / TRANSCRIPTION_FPS;
    selection = null; copyBuffer = null; gesture = null;
    renderTimeline();
    setStatus(`Loaded ${stem}.npy. Export saves as ${currentStem()}.npy.`, 'ok');
  } catch (error) {
    if (generation === loadGeneration) setStatus('Load failed: ' + error.message, 'err');
  }
}

// ── Video loading ──────────────────────────────────────────────────────────

let availableItems = [];
let availableTranscriptions = [];

function populateTranscriptions(selected = '') {
  const select = document.getElementById('transcription');
  select.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = 'Choose .npy…';
  select.appendChild(placeholder);
  for (const stem of availableTranscriptions) {
    const option = document.createElement('option');
    option.value = stem;
    option.textContent = stem + '.npy';
    select.appendChild(option);
  }
  select.value = availableTranscriptions.includes(selected) ? selected : '';
}

function currentStem() {
  return document.getElementById('filename').value;
}

function setStatus(msg, kind = '') {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status ' + kind;
}

function loadVideo() {
  const stem = currentStem();
  if (!stem) return;
  const generation = ++loadGeneration;
  notes = []; history.length = 0; selection = null; copyBuffer = null; gesture = null;
  transcriptionDuration = 0; gridInterval = 0.25;
  document.getElementById('grid-interval').value = 250;
  renderTimeline();
  populateTranscriptions(stem);
  if (availableTranscriptions.includes(stem)) loadNpy(stem, generation);
  video.src = '/video/' + stem;
  video.load();
  loadProjections(stem);
  const item = availableItems.find(i => i.stem === stem);
  if (item) {
    const tags = [];
    if (item.has_projections) tags.push('planar projections ✓');
    else if (item.has_poses) tags.push('poses ✓'); else tags.push('tracking ✗');
    if (item.has_masks) tags.push('masks ✓'); else tags.push('masks ✗');
    setStatus(tags.join('  '));
  }
}

async function refreshFileList() {
  const resp = await fetch('/list');
  const data = await resp.json();
  availableItems = data.items || [];
  availableTranscriptions = data.transcriptions || [];
  const sel = document.getElementById('filename');
  const prev = sel.value;
  sel.innerHTML = '';
  for (const item of availableItems) {
    const opt = document.createElement('option');
    opt.value = item.stem;
    let label = item.stem;
    const flags = [];
    if (item.has_poses) flags.push('P');
    if (item.has_projections) flags.push('H');
    if (item.has_masks) flags.push('M');
    if (flags.length) label += '  [' + flags.join('') + ']';
    opt.textContent = label;
    sel.appendChild(opt);
  }
  if (availableItems.length) {
    sel.value = availableItems.find(i => i.stem === prev) ? prev : availableItems[0].stem;
    loadVideo();
  }
}

document.getElementById('filename').addEventListener('change', loadVideo);
document.getElementById('load-transcription-btn').addEventListener('click', () => {
  const stem = document.getElementById('transcription').value;
  if (!currentStem() || !stem) { setStatus('Choose a video and transcription to load.', 'err'); return; }
  video.pause();
  loadNpy(stem, ++loadGeneration, true);
});
video.addEventListener('loadedmetadata', renderTimeline);

document.getElementById('clear-btn').addEventListener('click', () => {
  saveUndo(); notes = []; renderTimeline();
});
document.getElementById('reverse-strings').addEventListener('change', event => {
  reverseStrings = event.target.checked;
  labelsEl.querySelectorAll('.string-label').forEach(label => {
    label.textContent = STRINGS[assignedString(Number(label.dataset.s))];
  });
  rowsEl.querySelectorAll('.note').forEach(marker => {
    marker.title = noteTitle(notes[Number(marker.dataset.note)]);
  });
  drawOverlay();
});
document.getElementById('grid-interval').addEventListener('change', event => {
  const value = Number(event.target.value) / 1000;
  if (Number.isFinite(value) && value >= 0.001 && value <= 10) {
    gridInterval = value; renderTimeline();
  } else event.target.value = gridInterval * 1000;
});
document.getElementById('zoom').addEventListener('input', event => {
  const centerTime = (gridWrap.scrollLeft + (gridWrap.offsetWidth - labelsEl.offsetWidth) / 2) / pixelsPerSecond;
  pixelsPerSecond = Number(event.target.value);
  renderTimeline();
  gridWrap.scrollLeft = Math.max(0, centerTime * pixelsPerSecond - (gridWrap.offsetWidth - labelsEl.offsetWidth) / 2);
});
document.getElementById('convert-tempo-btn').addEventListener('click', () => {
  const source = Number(document.getElementById('source-bpm').value);
  const target = Number(document.getElementById('target-bpm').value);
  const ratio = source / target;
  if (!Number.isFinite(ratio) || source <= 0 || target <= 0 ||
      !Number.isFinite(timelineDuration() * ratio) || gridInterval * ratio < 0.001 ||
      timelineDuration() * ratio > 86400) {
    setStatus('Enter positive BPM values; converted duration must be under 24 hours and grid spacing at least 1 ms.', 'err');
    return;
  }
  saveUndo(); video.pause();
  notes = scaleNotes(notes, ratio);
  transcriptionDuration *= ratio;
  gridInterval *= ratio;
  document.getElementById('grid-interval').value = gridInterval * 1000;
  document.getElementById('source-bpm').value = target;
  renderTimeline();
  setStatus(`Tempo converted: ${source} → ${target} BPM. Undo: Ctrl/Cmd+Z.`, 'ok');
});
document.getElementById('export-btn').addEventListener('click', async () => {
  const filename = currentStem();
  if (!filename) { setStatus('Choose a recording before exporting.', 'err'); return; }
  const button = document.getElementById('export-btn');
  button.disabled = true;
  const outputNotes = notes.map(note => ({ ...note, string: assignedString(note.string) }));
  const payload = { filename, frames: notesToFrames(outputNotes, Math.max(transcriptionDuration, video.duration || 0)) };
  const save = overwrite => fetch('/export', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...payload, overwrite }),
  });
  try {
    let response = await save(false);
    if (response.status === 409) {
      if (!window.confirm(`data/input/${filename}.npy already exists. Overwrite it with the current transcription?`)) {
        setStatus('Export cancelled.');
        return;
      }
      response = await save(true);
    }
    if (!response.ok) throw new Error('Could not save transcription.');
    const result = await response.json();
    if (!availableTranscriptions.includes(filename)) {
      availableTranscriptions.push(filename);
      availableTranscriptions.sort();
      populateTranscriptions(document.getElementById('transcription').value || filename);
    }
    setStatus(`Saved ${result.saved}`, 'ok');
  } catch (error) {
    setStatus('Export failed: ' + error.message, 'err');
  } finally { button.disabled = false; }
});

// ── Live string overlay ────────────────────────────────────────────────────

const HOLD_DURATION = 0.25;
const STRING_COLOR = '#00ff00';
const STRING_WIDTH = 4;

const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
let projData = null;
const projCache = {};

async function loadProjections(stem) {
  projData = null;
  if (projCache[stem] !== undefined) {
    projData = projCache[stem];
    if (projData) {
      overlay.width = projData.w;
      overlay.height = projData.h;
    }
    return;
  }
  try {
    const r = await fetch('/strings/' + stem);
    if (!r.ok) {
      const txt = await r.text();
      setStatus('overlay disabled — ' + txt, 'err');
      projCache[stem] = null;
      return;
    }
    const data = await r.json();
    projCache[stem] = data;
    if (stem !== currentStem()) return;
    projData = data;
    overlay.width = data.w;
    overlay.height = data.h;
  } catch (e) {
    setStatus('overlay error: ' + e.message, 'err');
    projCache[stem] = null;
  }
}

function isLit(si, t) {
  return notes.some(note => note.string === si && t >= note.start &&
    t < Math.max(note.end, note.start + HOLD_DURATION));
}

function drawOverlay() {
  if (!overlay.width) return;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!projData) return;
  const t = video.currentTime;
  const fi = Math.min(projData.lines.length - 1, Math.max(0, Math.floor(t * projData.fps)));
  const lines = projData.lines[fi];
  if (!lines) return;
  octx.strokeStyle = STRING_COLOR;
  octx.lineWidth = STRING_WIDTH;
  octx.lineCap = 'round';
  for (let si = 0; si < 6; si++) {
    if (!isLit(si, t)) continue;
    const [x0, y0, x1, y1] = lines[STRINGS.length - 1 - assignedString(si)];
    octx.beginPath();
    octx.moveTo(x0, y0);
    octx.lineTo(x1, y1);
    octx.stroke();
  }
}

function rafLoop() {
  drawOverlay();
  requestAnimationFrame(rafLoop);
}
rafLoop();

// ── Init ───────────────────────────────────────────────────────────────────

renderTimeline();
refreshFileList();
