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
  const secs = Math.max(COL_DUR, parseFloat(document.getElementById('seconds').value) || 10);
  numCols = Math.ceil(secs / COL_DUR);
  initGrid(numCols);

  // String labels
  labelsEl.innerHTML = '';
  STRINGS.forEach(s => {
    const d = document.createElement('div');
    d.className = 'string-label';
    d.textContent = s;
    labelsEl.appendChild(d);
  });

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

  // Grid rows
  const rowsEl = document.getElementById('rows');
  rowsEl.innerHTML = '';
  STRINGS.forEach((_, si) => {
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
  });

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

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
    e.preventDefault();
    if (!history.length) return;
    grid = history.pop();
    renderGrid();
  }
});
document.getElementById('rows').addEventListener('contextmenu', e => e.preventDefault());

// ── Video loading ──────────────────────────────────────────────────────────

function loadVideo() {
  const stem = document.getElementById('filename').value.trim() || 'clip1';
  video.src = '/video/' + stem;
  video.load();
}

document.getElementById('filename').addEventListener('change', loadVideo);

// ── Toolbar actions ────────────────────────────────────────────────────────

document.getElementById('seconds').addEventListener('change', buildUI);

document.getElementById('clear-btn').addEventListener('click', () => {
  initGrid(numCols);
  renderGrid();
});

document.getElementById('export-btn').addEventListener('click', async () => {
  const filename = document.getElementById('filename').value.trim() || 'clip1';

  // Expand each 0.25s column to its audio frames
  const totalAudioFrames = numCols * FRAMES_PER_COL;
  const frames = [];
  for (let f = 0; f < totalAudioFrames; f++) {
    const col = Math.floor(f / FRAMES_PER_COL);
    frames.push(STRINGS.map((_, si) => grid[si][col]));
  }

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

// ── Init ───────────────────────────────────────────────────────────────────

buildUI();
loadVideo();
