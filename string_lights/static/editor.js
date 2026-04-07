const STRINGS = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4'];
const AUDIO_FPS = 22050 / 512; // ~43.066
const CELL_W = 20;

let numFrames = 0;
let grid = []; // grid[stringIdx][frameIdx] = 0 | 1

function initGrid(n) {
  grid = STRINGS.map(() => new Array(n).fill(0));
}

function secondsToFrames(s) {
  return Math.max(1, Math.round(parseFloat(s) * AUDIO_FPS));
}

function buildUI() {
  numFrames = secondsToFrames(document.getElementById('seconds').value);
  initGrid(numFrames);

  // String labels
  const labelsEl = document.getElementById('labels');
  labelsEl.innerHTML = '';
  STRINGS.forEach(s => {
    const d = document.createElement('div');
    d.className = 'string-label';
    d.textContent = s;
    labelsEl.appendChild(d);
  });

  // Frame index header (tick every 10 frames)
  const fnEl = document.getElementById('frame-numbers');
  fnEl.innerHTML = '';
  for (let f = 0; f < numFrames; f++) {
    const d = document.createElement('div');
    d.className = 'hcell' + (f % 10 === 0 ? ' tick' : '');
    d.style.width = CELL_W + 'px';
    if (f % 10 === 0) d.textContent = f;
    fnEl.appendChild(d);
  }

  // Time header: tick + label every 0.5s
  const timeEl = document.getElementById('time-numbers');
  timeEl.innerHTML = '';
  const tickEvery = Math.round(AUDIO_FPS / 2);
  for (let f = 0; f < numFrames; f++) {
    const d = document.createElement('div');
    d.style.width = CELL_W + 'px';
    d.style.flexShrink = '0';
    if (f % tickEvery === 0) {
      d.className = 'hcell time-tick';
      d.textContent = (f / AUDIO_FPS).toFixed(1) + 's';
    } else {
      d.className = 'hcell';
    }
    timeEl.appendChild(d);
  }

  // Grid rows
  const rowsEl = document.getElementById('rows');
  rowsEl.innerHTML = '';
  STRINGS.forEach((_, si) => {
    const row = document.createElement('div');
    row.className = 'row';
    for (let f = 0; f < numFrames; f++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.s = si;
      cell.dataset.f = f;
      row.appendChild(cell);
    }
    rowsEl.appendChild(row);
  });

  renderGrid();
}

function cellEl(si, f) {
  return document.querySelector(`.cell[data-s="${si}"][data-f="${f}"]`);
}

function renderCell(si, f) {
  const el = cellEl(si, f);
  if (!el) return;
  el.classList.toggle('active', grid[si][f] === 1);
}

function renderGrid() {
  STRINGS.forEach((_, si) => {
    for (let f = 0; f < numFrames; f++) renderCell(si, f);
  });
}

let painting = false;
let paintValue = 0;

document.getElementById('rows').addEventListener('mousedown', e => {
  const cell = e.target.closest('.cell');
  if (!cell) return;
  e.preventDefault();
  const si = parseInt(cell.dataset.s);
  const f = parseInt(cell.dataset.f);
  paintValue = e.button === 2 ? 0 : (grid[si][f] === 1 ? 0 : 1);
  painting = true;
  grid[si][f] = paintValue;
  renderCell(si, f);
});

document.getElementById('rows').addEventListener('mouseover', e => {
  if (!painting) return;
  const cell = e.target.closest('.cell');
  if (!cell) return;
  const si = parseInt(cell.dataset.s);
  const f = parseInt(cell.dataset.f);
  grid[si][f] = paintValue;
  renderCell(si, f);
});

document.addEventListener('mouseup', () => { painting = false; });
document.getElementById('rows').addEventListener('contextmenu', e => e.preventDefault());

document.getElementById('seconds').addEventListener('change', buildUI);

document.getElementById('clear-btn').addEventListener('click', () => {
  initGrid(numFrames);
  renderGrid();
});

document.getElementById('export-btn').addEventListener('click', async () => {
  const filename = document.getElementById('filename').value.trim() || 'clip1';
  const resp = await fetch('/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      frames: grid[0].map((_, f) => STRINGS.map((_, si) => grid[si][f])),
      filename,
    }),
  });
  if (!resp.ok) { alert('Export failed'); return; }
  const blob = await resp.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename + '.npy';
  a.click();
});

buildUI();
