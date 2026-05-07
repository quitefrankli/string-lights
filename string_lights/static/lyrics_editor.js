// ── State ──────────────────────────────────────────────────────────────────

let lyrics = [];       // [{id, text, start, end, x, y, size, color:[r,g,b]}, ...]
let selectedId = null;
let currentStem = '';
let jitterTick = 0;
let lastJitterUpdate = 0;  // timestamp of last jitter step

const PX_PER_SEC = 80;  // timeline pixels per second

// ── DOM refs ───────────────────────────────────────────────────────────────

const video      = document.getElementById('video');
const overlay    = document.getElementById('overlay');
const ctx        = overlay.getContext('2d');
const lyricForm  = document.getElementById('lyric-form');
const lyricsList = document.getElementById('lyrics-list');
const tlRuler    = document.getElementById('tl-ruler');
const tlTrack    = document.getElementById('tl-track');
const tlPlayhead = document.getElementById('tl-playhead');
const tlInner    = document.getElementById('timeline-inner');
const tlWrap     = document.getElementById('timeline-wrap');
const statusEl   = document.getElementById('status');
const posHint    = document.getElementById('pos-hint');

// ── Utilities ──────────────────────────────────────────────────────────────

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

function setStatus(msg, kind = '') {
  statusEl.textContent = msg;
  statusEl.className = 'status ' + kind;
}

function rgbToHex([r, g, b]) {
  return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// Simple seeded pseudo-random (deterministic per frame+char)
function seededRng(seed) {
  let s = seed | 0;
  return () => {
    s = (s ^ (s << 13)) >>> 0;
    s = (s ^ (s >> 7))  >>> 0;
    s = (s ^ (s << 17)) >>> 0;
    return (s >>> 0) / 4294967296;
  };
}

// ── Lyrics CRUD ─────────────────────────────────────────────────────────────

function getLyric(id) {
  return lyrics.find(l => l.id === id);
}

function addLyric() {
  const t = video.currentTime || 0;
  const lyric = {
    id: uid(), text: '', start: parseFloat(t.toFixed(2)),
    end: parseFloat((t + 3).toFixed(2)),
    x: 0.5, y: 0.5, size: 72, color: [255, 255, 255],
  };
  lyrics.push(lyric);
  renderList();
  renderTimeline();
  selectLyric(lyric.id);
  document.getElementById('lyric-text').focus();
}

function deleteLyric(id) {
  lyrics = lyrics.filter(l => l.id !== id);
  if (selectedId === id) selectLyric(null);
  renderList();
  renderTimeline();
}

function selectLyric(id) {
  selectedId = id;
  document.querySelectorAll('.lyric-item').forEach(el => el.classList.toggle('selected', el.dataset.id === id));
  document.querySelectorAll('.tl-block').forEach(el => el.classList.toggle('selected', el.dataset.id === id));
  const lyric = id ? getLyric(id) : null;
  lyricForm.classList.toggle('disabled', !lyric);
  if (lyric) populateForm(lyric);
}

function populateForm(lyric) {
  document.getElementById('lyric-text').value   = lyric.text;
  document.getElementById('lyric-start').value  = lyric.start;
  document.getElementById('lyric-end').value    = lyric.end;
  document.getElementById('lyric-x').value      = lyric.x;
  document.getElementById('lyric-y').value      = lyric.y;
  document.getElementById('lyric-size').value   = lyric.size;
  document.getElementById('lyric-color').value  = rgbToHex(lyric.color);
  const jv = lyric.jitter ?? 1.0;
  document.getElementById('lyric-jitter').value = jv;
  document.getElementById('jitter-val').textContent = jv.toFixed(1);
}

function readForm() {
  return {
    text:  document.getElementById('lyric-text').value,
    start: parseFloat(document.getElementById('lyric-start').value) || 0,
    end:   parseFloat(document.getElementById('lyric-end').value)   || 0,
    x:     parseFloat(document.getElementById('lyric-x').value)     || 0,
    y:     parseFloat(document.getElementById('lyric-y').value)     || 0,
    size:   parseInt(document.getElementById('lyric-size').value)    || 72,
    color:  hexToRgb(document.getElementById('lyric-color').value),
    jitter: parseFloat(document.getElementById('lyric-jitter').value) ?? 1.0,
  };
}

function commitForm() {
  if (!selectedId) return;
  const lyric = getLyric(selectedId);
  if (!lyric) return;
  Object.assign(lyric, readForm());
  renderList();
  renderTimeline();
}

// Live-update from form inputs
['lyric-text','lyric-start','lyric-end','lyric-x','lyric-y','lyric-size','lyric-color','lyric-jitter'].forEach(id => {
  document.getElementById(id).addEventListener('input', commitForm);
});
document.getElementById('lyric-jitter').addEventListener('input', e => {
  document.getElementById('jitter-val').textContent = parseFloat(e.target.value).toFixed(1);
});

// ── Network ────────────────────────────────────────────────────────────────

async function loadLyrics(stem) {
  try {
    const r = await fetch('/lyrics/' + stem);
    lyrics = r.ok ? await r.json() : [];
    // ensure every lyric has an id
    lyrics.forEach(l => { if (!l.id) l.id = uid(); });
    renderList();
    renderTimeline();
    selectLyric(null);
  } catch (e) {
    setStatus('load error: ' + e.message, 'err');
  }
}

async function saveLyrics() {
  try {
    setStatus('saving…', 'busy');
    await fetch('/lyrics/' + currentStem, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(lyrics),
    });
    setStatus('saved ✓', 'ok');
    setTimeout(() => setStatus(''), 2000);
  } catch (e) {
    setStatus('save error: ' + e.message, 'err');
  }
}

async function refreshFileList() {
  const r = await fetch('/list');
  const { items } = await r.json();
  const sel = document.getElementById('filename');
  const prev = sel.value;
  sel.innerHTML = '';
  for (const item of items) {
    const opt = document.createElement('option');
    opt.value = item.stem;
    opt.textContent = item.stem;
    sel.appendChild(opt);
  }
  if (items.length) {
    sel.value = items.find(i => i.stem === prev) ? prev : items[0].stem;
    loadVideo();
  }
}

function loadVideo() {
  currentStem = document.getElementById('filename').value;
  if (!currentStem) return;
  video.src = '/video/' + currentStem;
  video.load();
  loadLyrics(currentStem);
}

// ── Render list ────────────────────────────────────────────────────────────

function renderList() {
  lyricsList.innerHTML = '';
  const sorted = [...lyrics].sort((a, b) => a.start - b.start);
  for (const lyric of sorted) {
    const el = document.createElement('div');
    el.className = 'lyric-item' + (lyric.id === selectedId ? ' selected' : '');
    el.dataset.id = lyric.id;
    el.innerHTML = `
      <div class="lyric-text">
        <span class="lyric-swatch" style="background:${rgbToHex(lyric.color)}"></span>
        ${lyric.text || '<em style="color:#555">empty</em>'}
      </div>
      <div class="lyric-time">${lyric.start.toFixed(2)}s → ${lyric.end.toFixed(2)}s</div>
    `;
    el.addEventListener('click', () => {
      selectLyric(lyric.id);
      video.currentTime = lyric.start;
    });
    lyricsList.appendChild(el);
  }
}

// ── Timeline ───────────────────────────────────────────────────────────────

function buildTimeline() {
  const dur = video.duration || 0;
  const totalW = Math.max(dur * PX_PER_SEC, tlWrap.offsetWidth);
  tlInner.style.width = totalW + 'px';

  // Ruler
  tlRuler.innerHTML = '';
  for (let s = 0; s <= Math.ceil(dur); s++) {
    const tick = document.createElement('div');
    tick.className = 'tl-tick';
    tick.style.left = (s * PX_PER_SEC) + 'px';
    tick.textContent = s + 's';
    tlRuler.appendChild(tick);
  }

  renderTimeline();
}

function renderTimeline() {
  tlTrack.innerHTML = '';
  const COLORS = ['#4466cc','#cc4466','#44aa66','#cc8844','#9944cc','#44aacc'];
  lyrics.forEach((lyric, i) => {
    const block = document.createElement('div');
    block.className = 'tl-block' + (lyric.id === selectedId ? ' selected' : '');
    block.dataset.id = lyric.id;
    block.style.left  = (lyric.start * PX_PER_SEC) + 'px';
    block.style.width = Math.max(4, (lyric.end - lyric.start) * PX_PER_SEC) + 'px';
    block.style.background = rgbToHex(lyric.color) + 'bb';
    block.textContent = lyric.text || '…';
    block.addEventListener('mousedown', e => {
      e.preventDefault();
      e.stopPropagation();
      selectLyric(lyric.id);
      const blockRect = block.getBoundingClientRect();
      const distLeft  = e.clientX - blockRect.left;
      const distRight = blockRect.right - e.clientX;
      const edgeZone  = Math.min(8, blockRect.width * 0.25);
      const mode = distLeft < edgeZone ? 'start' : distRight < edgeZone ? 'end' : 'move';
      tlDrag = { id: lyric.id, mode, startX: e.clientX, origStart: lyric.start, origEnd: lyric.end };
      if (mode === 'move') video.currentTime = lyric.start;
    });
    block.addEventListener('mousemove', e => {
      if (tlDrag) return;
      const blockRect = block.getBoundingClientRect();
      const distLeft  = e.clientX - blockRect.left;
      const distRight = blockRect.right - e.clientX;
      const edgeZone  = Math.min(8, blockRect.width * 0.25);
      block.style.cursor = (distLeft < edgeZone || distRight < edgeZone) ? 'ew-resize' : 'grab';
    });
    tlTrack.appendChild(block);
  });
}

function updatePlayhead() {
  const x = (video.currentTime || 0) * PX_PER_SEC;
  tlPlayhead.style.left = x + 'px';

  // Auto-scroll timeline
  if (!video.paused) {
    const scrollX = tlWrap.scrollLeft;
    const viewW = tlWrap.offsetWidth;
    if (x < scrollX || x > scrollX + viewW - 40) {
      tlWrap.scrollLeft = Math.max(0, x - viewW / 3);
    }
  }
}

tlTrack.addEventListener('mousedown', e => {
  if (!e.target.closest('.tl-block')) selectLyric(null);
});

// Seek by clicking ruler
tlRuler.addEventListener('click', e => {
  const rect = tlRuler.getBoundingClientRect();
  const x = e.clientX - rect.left + tlWrap.scrollLeft;
  video.currentTime = x / PX_PER_SEC;
});

video.addEventListener('timeupdate', updatePlayhead);
video.addEventListener('seeked', updatePlayhead);
video.addEventListener('loadedmetadata', buildTimeline);

// ── Canvas overlay ─────────────────────────────────────────────────────────

// Update canvas size to match natural video resolution
video.addEventListener('loadedmetadata', () => {
  overlay.width  = video.videoWidth  || 1280;
  overlay.height = video.videoHeight || 720;
});

function drawScribbleText(x, y, text, size, color, tick, jitter = 1.0) {
  if (!text) return;

  // Scale size to canvas resolution
  const scale = overlay.height / (video.videoHeight || overlay.height);
  const scaledSize = Math.round(size * scale);

  ctx.save();
  ctx.font = `bold ${scaledSize}px 'Quicksand', 'DejaVu Sans', sans-serif`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  ctx.lineJoin = 'round';

  const chars = [...text];
  const charWidths = chars.map(c => ctx.measureText(c).width);
  const totalWidth = charWidths.reduce((a, b) => a + b, 0);
  let cx = x - totalWidth / 2;

  const strokeW = Math.max(2, scaledSize / 18);
  const rng = seededRng(tick * 9973);

  for (let i = 0; i < chars.length; i++) {
    const jx = (rng() - 0.5) * 2 * jitter * scale;
    const jy = (rng() - 0.5) * 3 * jitter * scale;

    // Outline
    ctx.strokeStyle = 'rgba(0,0,0,0.82)';
    ctx.lineWidth = strokeW;
    ctx.strokeText(chars[i], cx + jx, y + jy);

    // Fill
    ctx.fillStyle = color;
    ctx.fillText(chars[i], cx + jx, y + jy);

    cx += charWidths[i] || scaledSize * 0.33;
  }
  ctx.restore();
}

function drawOverlay() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const t = video.currentTime;
  const w = overlay.width;
  const h = overlay.height;

  for (const lyric of lyrics) {
    if (t < lyric.start || t >= lyric.end) continue;
    drawScribbleText(
      lyric.x * w, lyric.y * h,
      lyric.text, lyric.size,
      rgbToHex(lyric.color),
      jitterTick, lyric.jitter ?? 1.0,
    );
  }

  // Crosshair on selected lyric's position (when paused or no lyric active)
  const sel = selectedId ? getLyric(selectedId) : null;
  if (sel && (video.paused || !(sel.start <= t && t < sel.end))) {
    const px = sel.x * w;
    const py = sel.y * h;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(px - 12, py); ctx.lineTo(px + 12, py); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px, py - 12); ctx.lineTo(px, py + 12); ctx.stroke();
    ctx.restore();

    // Preview text (at configured position)
    if (sel.text) {
      drawScribbleText(px, py, sel.text, sel.size, rgbToHex(sel.color), jitterTick, sel.jitter ?? 1.0);
    }
  }

  // Advance jitter at ~10hz to match Python's per-3-frame seed stepping
  const now = performance.now();
  if (now - lastJitterUpdate > 100) { jitterTick++; lastJitterUpdate = now; }
}

function rafLoop() {
  drawOverlay();
  requestAnimationFrame(rafLoop);
}
rafLoop();

// ── Spatial drag on canvas ─────────────────────────────────────────────────

let canvasDrag = null; // {id, ox, oy} — offset in normalised units at drag start
let tlDrag     = null; // {id, mode, startX, origStart, origEnd}

function findLyricAt(nx, ny) {
  const vw = overlay.width  || 1280;
  const vh = overlay.height || 720;
  const t = video.currentTime;
  for (let i = lyrics.length - 1; i >= 0; i--) {
    const l = lyrics[i];
    if (!l.text || t < l.start || t >= l.end) continue;
    const halfW = (l.text.length * l.size * 0.55 / 2 + 10) / vw;
    const halfH = (l.size * 0.75 + 8) / vh;
    if (Math.abs(nx - l.x) < halfW && Math.abs(ny - l.y) < halfH) return l;
  }
  return null;
}

overlay.addEventListener('mousedown', e => {
  e.preventDefault();
  const rect = overlay.getBoundingClientRect();
  const nx = (e.clientX - rect.left) / rect.width;
  const ny = (e.clientY - rect.top)  / rect.height;
  const hit = findLyricAt(nx, ny);
  if (!hit) { selectLyric(null); return; }
  selectLyric(hit.id);
  canvasDrag = { id: hit.id, ox: nx - hit.x, oy: ny - hit.y };
  overlay.style.cursor = 'grabbing';
});

overlay.addEventListener('mousemove', e => {
  const rect = overlay.getBoundingClientRect();
  const nx = (e.clientX - rect.left) / rect.width;
  const ny = (e.clientY - rect.top)  / rect.height;
  posHint.textContent = `x=${nx.toFixed(2)} y=${ny.toFixed(2)}`;

  if (canvasDrag) {
    const lyric = getLyric(canvasDrag.id);
    if (lyric) {
      lyric.x = parseFloat(Math.max(0, Math.min(1, nx - canvasDrag.ox)).toFixed(3));
      lyric.y = parseFloat(Math.max(0, Math.min(1, ny - canvasDrag.oy)).toFixed(3));
      document.getElementById('lyric-x').value = lyric.x;
      document.getElementById('lyric-y').value = lyric.y;
    }
    return;
  }

  overlay.style.cursor = findLyricAt(nx, ny) ? 'grab' : 'default';
});

overlay.addEventListener('mouseleave', () => {
  posHint.textContent = '';
  if (!canvasDrag) overlay.style.cursor = 'default';
});

// ── Global drag end + timeline drag movement ───────────────────────────────

document.addEventListener('mousemove', e => {
  if (!tlDrag) return;
  const lyric = getLyric(tlDrag.id);
  if (!lyric) return;
  const dt = (e.clientX - tlDrag.startX) / PX_PER_SEC;
  if (tlDrag.mode === 'move') {
    const dur = tlDrag.origEnd - tlDrag.origStart;
    lyric.start = parseFloat(Math.max(0, tlDrag.origStart + dt).toFixed(2));
    lyric.end   = parseFloat((lyric.start + dur).toFixed(2));
  } else if (tlDrag.mode === 'start') {
    lyric.start = parseFloat(Math.max(0, Math.min(lyric.end - 0.1, tlDrag.origStart + dt)).toFixed(2));
  } else {
    lyric.end = parseFloat(Math.max(lyric.start + 0.1, tlDrag.origEnd + dt).toFixed(2));
  }
  if (selectedId === tlDrag.id) populateForm(lyric);
  renderTimeline();
});

document.addEventListener('mouseup', () => {
  if (canvasDrag) {
    renderList();
    canvasDrag = null;
    overlay.style.cursor = 'default';
  }
  if (tlDrag) {
    renderList();
    tlDrag = null;
  }
});

// ── Keyboard shortcuts ─────────────────────────────────────────────────────

function isInputFocused() {
  const t = document.activeElement?.tagName;
  return t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT';
}

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    saveLyrics();
    return;
  }

  if (isInputFocused()) return;

  if (e.key === ' ') {
    e.preventDefault();
    video.paused ? video.play() : video.pause();
    return;
  }

  if (e.key === '[' || e.key === 'BracketLeft') {
    e.preventDefault();
    if (!selectedId) return;
    const lyric = getLyric(selectedId);
    lyric.start = parseFloat(video.currentTime.toFixed(2));
    if (lyric.end <= lyric.start) lyric.end = parseFloat((lyric.start + 3).toFixed(2));
    populateForm(lyric); renderList(); renderTimeline();
    return;
  }

  if (e.key === ']' || e.key === 'BracketRight') {
    e.preventDefault();
    if (!selectedId) return;
    const lyric = getLyric(selectedId);
    lyric.end = parseFloat(video.currentTime.toFixed(2));
    if (lyric.end <= lyric.start) lyric.start = Math.max(0, parseFloat((lyric.end - 1).toFixed(2)));
    populateForm(lyric); renderList(); renderTimeline();
    return;
  }

  if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) {
    e.preventDefault();
    deleteLyric(selectedId);
    return;
  }

  if (e.key === 'ArrowRight') {
    e.preventDefault();
    video.currentTime = Math.min(video.currentTime + 0.5, video.duration || 1e9);
    return;
  }

  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    video.currentTime = Math.max(video.currentTime - 0.5, 0);
    return;
  }

  if (e.key === 'Escape') { selectLyric(null); }
});

// ── Toolbar actions ────────────────────────────────────────────────────────

document.getElementById('add-btn').addEventListener('click', addLyric);
document.getElementById('save-btn').addEventListener('click', saveLyrics);
document.getElementById('set-start-btn').addEventListener('click', () => {
  document.dispatchEvent(new KeyboardEvent('keydown', { key: '[' }));
});
document.getElementById('set-end-btn').addEventListener('click', () => {
  document.dispatchEvent(new KeyboardEvent('keydown', { key: ']' }));
});
document.getElementById('delete-btn').addEventListener('click', () => {
  if (selectedId) deleteLyric(selectedId);
});
document.getElementById('filename').addEventListener('change', loadVideo);

// ── Render ─────────────────────────────────────────────────────────────────

let _pollInterval = null;

async function startRender() {
  const renderBtn = document.getElementById('render-btn');
  // Save first, then render
  await saveLyrics();
  renderBtn.disabled = true;
  setStatus('starting render…', 'busy');

  const r = await fetch('/render/' + currentStem, { method: 'POST' });
  if (!r.ok) {
    const { error } = await r.json().catch(() => ({ error: r.statusText }));
    setStatus('render error: ' + error, 'err');
    renderBtn.disabled = false;
    return;
  }

  _pollInterval = setInterval(pollRender, 800);
}

async function pollRender() {
  const renderBtn = document.getElementById('render-btn');
  try {
    const r = await fetch('/render-status');
    const { state, progress, error, stem } = await r.json();

    if (state === 'running') {
      const pct = Math.round((progress || 0) * 100);
      setStatus(`rendering ${stem}… ${pct}%`, 'busy');
    } else {
      clearInterval(_pollInterval);
      _pollInterval = null;
      renderBtn.disabled = false;
      if (state === 'done') {
        setStatus(`done → data/output/${stem}.lyrics.mp4 ✓`, 'ok');
      } else if (state === 'error') {
        setStatus('render failed: ' + error, 'err');
      }
    }
  } catch (e) {
    clearInterval(_pollInterval);
    _pollInterval = null;
    renderBtn.disabled = false;
    setStatus('poll error: ' + e.message, 'err');
  }
}

document.getElementById('render-btn').addEventListener('click', startRender);

// ── Init ───────────────────────────────────────────────────────────────────

refreshFileList();
