const STRINGS = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4'];
const AUDIO_FPS = 22050 / 512;  // ~43.066 audio frames per second
const COL_DUR = 0.25;           // seconds per grid column
const FRAMES_PER_COL = Math.round(COL_DUR * AUDIO_FPS); // ~11 audio frames per col
const CELL_W = 20;              // px per column

let numCols = 0;
let grid = []; // grid[stringIdx][colIdx] = 0 | 1

const video = document.getElementById('video');
const playhead = document.getElementById('playhead');
const gridWrap = document.getElementById('grid-wrap');
const labelsEl = document.getElementById('labels');

function initGrid(n) {
  grid = STRINGS.map(() => new Array(n).fill(0));
}

function buildUI() {
  const secs = Math.max(COL_DUR, (isFinite(video.duration) ? video.duration : null) || 10);
  numCols = Math.ceil(secs / COL_DUR);
  initGrid(numCols);

  // String labels (high → low)
  labelsEl.innerHTML = '';
  for (let si = STRINGS.length - 1; si >= 0; si--) {
    const d = document.createElement('div');
    d.className = 'string-label';
    d.textContent = STRINGS[si];
    labelsEl.appendChild(d);
  }

  // Time ruler: tick every col (0.25s), label every other col (0.5s)
  const timeEl = document.getElementById('time-numbers');
  timeEl.innerHTML = '';
  for (let c = 0; c < numCols; c++) {
    const d = document.createElement('div');
    d.style.width = CELL_W + 'px';
    d.style.flexShrink = '0';
    d.className = 'hcell time-tick';
    if (c % 2 === 0) d.textContent = (c * COL_DUR).toFixed(1) + 's';
    timeEl.appendChild(d);
  }

  // Grid rows (high → low)
  const rowsEl = document.getElementById('rows');
  rowsEl.innerHTML = '';
  for (let si = STRINGS.length - 1; si >= 0; si--) {
    const row = document.createElement('div');
    row.className = 'row';
    for (let c = 0; c < numCols; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.s = si;
      cell.dataset.f = c;
      row.appendChild(cell);
    }
    rowsEl.appendChild(row);
  }

  renderGrid();
  updatePlayhead();
}

// ── Rendering ──────────────────────────────────────────────────────────────

function cellEl(si, c) {
  return document.querySelector(`.cell[data-s="${si}"][data-f="${c}"]`);
}

function renderCell(si, c) {
  const el = cellEl(si, c);
  if (el) el.classList.toggle('active', grid[si][c] === 1);
}

function renderGrid() {
  STRINGS.forEach((_, si) => {
    for (let c = 0; c < numCols; c++) renderCell(si, c);
  });
}

// ── Playhead ───────────────────────────────────────────────────────────────

function updatePlayhead() {
  const x = (video.currentTime / COL_DUR) * CELL_W;
  playhead.style.left = x + 'px';

  if (!video.paused) {
    const labelsW = labelsEl.offsetWidth;
    const viewW = gridWrap.offsetWidth - labelsW;
    const scrollX = gridWrap.scrollLeft;
    if (x < scrollX || x > scrollX + viewW - CELL_W * 4) {
      gridWrap.scrollLeft = Math.max(0, x - viewW / 3);
    }
  }
}

video.addEventListener('timeupdate', updatePlayhead);
video.addEventListener('seeked', updatePlayhead);

// ── Playhead dragging ──────────────────────────────────────────────────────

function clientXToTime(clientX) {
  const wrapRect = gridWrap.getBoundingClientRect();
  const x = clientX - wrapRect.left - labelsEl.offsetWidth + gridWrap.scrollLeft;
  const col = Math.max(0, Math.floor(x / CELL_W));
  return col * COL_DUR;
}

let draggingPlayhead = false;

playhead.addEventListener('mousedown', e => {
  e.preventDefault();
  e.stopPropagation();
  draggingPlayhead = true;
});

document.addEventListener('mousemove', e => {
  if (!draggingPlayhead) return;
  video.currentTime = clientXToTime(e.clientX);
});

// ── Seek by clicking time ruler ────────────────────────────────────────────

document.getElementById('time-numbers').addEventListener('click', e => {
  if (draggingPlayhead) return;
  video.currentTime = clientXToTime(e.clientX);
});

// ── Cell painting ──────────────────────────────────────────────────────────

let painting = false;
let paintValue = 0;
const history = [];
let snapshotBeforePaint = null;

function snapshot() {
  return grid.map(row => row.slice());
}

document.getElementById('rows').addEventListener('mousedown', e => {
  const cell = e.target.closest('.cell');
  if (!cell) return;
  e.preventDefault();
  const si = parseInt(cell.dataset.s);
  const c = parseInt(cell.dataset.f);
  paintValue = e.button === 2 ? 0 : (grid[si][c] === 1 ? 0 : 1);
  snapshotBeforePaint = snapshot();
  painting = true;
  grid[si][c] = paintValue;
  renderCell(si, c);
});

document.getElementById('rows').addEventListener('mouseover', e => {
  if (!painting) return;
  const cell = e.target.closest('.cell');
  if (!cell) return;
  const si = parseInt(cell.dataset.s);
  const c = parseInt(cell.dataset.f);
  grid[si][c] = paintValue;
  renderCell(si, c);
});

document.addEventListener('mouseup', () => {
  if (painting && snapshotBeforePaint) history.push(snapshotBeforePaint);
  snapshotBeforePaint = null;
  painting = false;
  draggingPlayhead = false;
});

const STRING_KEYS = { q: 0, w: 1, e: 2, a: 3, s: 4, d: 5 };

function isInputFocused() {
  const t = document.activeElement?.tagName;
  return t === 'INPUT' || t === 'TEXTAREA';
}

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
    e.preventDefault();
    if (!history.length) return;
    grid = history.pop();
    renderGrid();
    return;
  }

  if (isInputFocused()) return;

  const intervalSec = (parseFloat(document.getElementById('interval').value) || 250) / 1000;

  if (e.key === ' ') {
    e.preventDefault();
    if (video.paused) video.play(); else video.pause();
    return;
  }

  if (e.key === 'Enter') {
    e.preventDefault();
    video.currentTime = 0;
    return;
  }

  if (e.key === 'ArrowRight') {
    e.preventDefault();
    video.currentTime = Math.min(video.currentTime + intervalSec, video.duration || 1e9);
    return;
  }

  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    video.currentTime = Math.max(video.currentTime - intervalSec, 0);
    return;
  }

  if (e.key in STRING_KEYS && video.paused) {
    const si = STRING_KEYS[e.key];
    const c = Math.floor(video.currentTime / COL_DUR);
    if (c < numCols) {
      history.push(snapshot());
      grid[si][c] ^= 1;
      renderCell(si, c);
    }
  }
});
document.getElementById('rows').addEventListener('contextmenu', e => e.preventDefault());

// ── Video loading ──────────────────────────────────────────────────────────

let availableItems = [];

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
  video.src = '/video/' + stem;
  video.load();
  loadProjections(stem);
  const item = availableItems.find(i => i.stem === stem);
  if (item) {
    const tags = [];
    if (item.has_poses) tags.push('poses ✓'); else tags.push('poses ✗');
    if (item.has_masks) tags.push('masks ✓'); else tags.push('masks ✗');
    setStatus(tags.join('  '));
  }
}

async function refreshFileList() {
  const resp = await fetch('/list');
  const data = await resp.json();
  availableItems = data.items || [];
  const sel = document.getElementById('filename');
  const prev = sel.value;
  sel.innerHTML = '';
  for (const item of availableItems) {
    const opt = document.createElement('option');
    opt.value = item.stem;
    let label = item.stem;
    const flags = [];
    if (item.has_poses) flags.push('P');
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
video.addEventListener('loadedmetadata', buildUI);

// ── Toolbar actions ────────────────────────────────────────────────────────

document.getElementById('clear-btn').addEventListener('click', () => {
  initGrid(numCols);
  renderGrid();
});

function gridToAudioFrames() {
  const totalAudioFrames = numCols * FRAMES_PER_COL;
  const frames = [];
  for (let f = 0; f < totalAudioFrames; f++) {
    const col = Math.floor(f / FRAMES_PER_COL);
    frames.push(STRINGS.map((_, si) => grid[si][col]));
  }
  return frames;
}

document.getElementById('export-btn').addEventListener('click', async () => {
  const filename = currentStem() || 'clip1';
  const frames = gridToAudioFrames();
  const resp = await fetch('/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frames, filename }),
  });
  if (!resp.ok) { alert('Export failed'); return; }
  const blob = await resp.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename + '.npy';
  a.click();
});

// ── Live string overlay ────────────────────────────────────────────────────

const HOLD_DURATION = 0.8;
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
    projData = data;
    overlay.width = data.w;
    overlay.height = data.h;
  } catch (e) {
    setStatus('overlay error: ' + e.message, 'err');
    projCache[stem] = null;
  }
}

function isLit(si, t) {
  const c = Math.floor(t / COL_DUR);
  if (c < 0 || !grid[si]) return false;
  for (let cc = c; cc >= 0; cc--) {
    if (grid[si][cc] === 1) {
      return (t - cc * COL_DUR) <= HOLD_DURATION;
    }
  }
  return false;
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
    const [x0, y0, x1, y1] = lines[si];
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

buildUI();
refreshFileList();
