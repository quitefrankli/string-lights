// ── State ──────────────────────────────────────────────────────────────────

let lyrics = [];       // [{id, text, start, end, x, y, size, color:[r,g,b]}, ...]
let selectedId = null;
const currentStem = JSON.parse(document.getElementById('editor-config').textContent).stem;
let exportAspect = 'padded';
let exportFormat = 'compatible';
let jitterTick = 0;
let lastJitterUpdate = 0;  // timestamp of last jitter step
let canvasDrag = null; // {id, ox, oy, snapX, snapY}

const PX_PER_SEC = 80;  // timeline pixels per second
const LANE_HEIGHT = 44;
let timelineLanes = new Map();
const undoStack = [], redoStack = [];
let formEdit = null;

function snapshot() {
  return {lyrics: JSON.stringify(lyrics), selectedId};
}

function rememberEdit(before) {
  if (before.lyrics === JSON.stringify(lyrics)) return;
  undoStack.push(before);
  if (undoStack.length > 100) undoStack.shift();
  redoStack.length = 0;
}

function finishFormEdit() {
  if (!formEdit) return;
  rememberEdit(formEdit);
  formEdit = null;
}

function restoreHistory(redo = false) {
  if (canvasDrag || tlDrag) return;
  finishFormEdit();
  const from = redo ? redoStack : undoStack;
  const to = redo ? undoStack : redoStack;
  if (!from.length) return;
  to.push(snapshot());
  const state = from.pop();
  lyrics = JSON.parse(state.lyrics);
  selectedId = state.selectedId;
  renderList(); renderTimeline(); selectLyric(selectedId);
}

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
  finishFormEdit();
  const before = snapshot();
  const t = video.currentTime || 0;
  const lyric = {
    id: uid(), text: '', start: parseFloat(t.toFixed(2)),
    end: parseFloat((t + 3).toFixed(2)),
    x: 0.5, y: 0.85, size: 48, color: [255, 255, 255],
    font: 'auto', halo_size: 0, halo_intensity: 0,
    background_opacity: 0, background_padding: 0, background_softness: 0,
  };
  lyrics.push(lyric);
  renderList();
  renderTimeline();
  selectLyric(lyric.id);
  rememberEdit(before);
  document.getElementById('lyric-text').focus();
}

function deleteLyric(id) {
  finishFormEdit();
  const before = snapshot();
  lyrics = lyrics.filter(l => l.id !== id);
  if (selectedId === id) selectLyric(null);
  renderList();
  renderTimeline();
  rememberEdit(before);
}

function duplicateLyric() {
  finishFormEdit();
  const original = getLyric(selectedId);
  if (!original) return;
  const before = snapshot();
  const copy = JSON.parse(JSON.stringify(original));
  copy.id = uid();
  delete copy.lane;
  lyrics.push(copy);
  renderList(); renderTimeline(); selectLyric(copy.id);
  rememberEdit(before);
}

function selectLyric(id) {
  finishFormEdit();
  selectedId = id;
  document.querySelectorAll('.lyric-item').forEach(el => el.classList.toggle('selected', el.dataset.id === id));
  document.querySelectorAll('.tl-block').forEach(el => el.classList.toggle('selected', el.dataset.id === id));
  const lyric = id ? getLyric(id) : null;
  lyricForm.classList.toggle('disabled', !lyric);
  if (lyric) populateForm(lyric);
}

function populateForm(lyric) {
  document.getElementById('lyric-font').value = lyric.font ?? 'auto';
  document.getElementById('lyric-background-opacity').value = lyric.background_opacity ?? 0;
  document.getElementById('lyric-background-padding').value = lyric.background_padding ?? 16;
  document.getElementById('lyric-background-softness').value = lyric.background_softness ?? 24;
  const opacity = lyric.background_opacity ?? 0;
  document.getElementById('background-opacity-val').textContent = opacity ? Math.round(opacity * 100) + '%' : 'Off';
  document.getElementById('lyric-halo-size').value = lyric.halo_size ?? 4;
  document.getElementById('lyric-halo-intensity').value = lyric.halo_intensity ?? .75;
  document.getElementById('halo-intensity-val').textContent = Math.round((lyric.halo_intensity ?? .75) * 100) + '%';
  document.getElementById('lyric-text').value   = lyric.text;
  document.getElementById('lyric-start').value  = lyric.start;
  document.getElementById('lyric-end').value    = lyric.end;
  document.getElementById('lyric-x').value      = lyric.x;
  document.getElementById('lyric-y').value      = lyric.y;
  document.getElementById('lyric-size').value   = lyric.size;
  document.getElementById('lyric-color').value  = rgbToHex(lyric.color);
  const jv = lyric.fade ?? 0.25;
  document.getElementById('lyric-jitter').value = jv;
  document.getElementById('jitter-val').textContent = jv.toFixed(1);
}

function readForm() {
  return {
    font: document.getElementById('lyric-font').value,
    background_opacity: Number(document.getElementById('lyric-background-opacity').value),
    background_padding: Number(document.getElementById('lyric-background-padding').value),
    background_softness: Number(document.getElementById('lyric-background-softness').value),
    halo_size: Number(document.getElementById('lyric-halo-size').value),
    halo_intensity: Number(document.getElementById('lyric-halo-intensity').value),
    text:  document.getElementById('lyric-text').value,
    start: parseFloat(document.getElementById('lyric-start').value) || 0,
    end:   parseFloat(document.getElementById('lyric-end').value)   || 0,
    x:     parseFloat(document.getElementById('lyric-x').value)     || 0,
    y:     parseFloat(document.getElementById('lyric-y').value)     || 0,
    size:   parseInt(document.getElementById('lyric-size').value)    || 72,
    color:  hexToRgb(document.getElementById('lyric-color').value),
    fade: parseFloat(document.getElementById('lyric-jitter').value) ?? 0.25,
  };
}

function commitForm() {
  if (!selectedId) return;
  const lyric = getLyric(selectedId);
  if (!lyric) return;
  if (!formEdit) formEdit = snapshot();
  Object.assign(lyric, readForm());
  document.getElementById('background-opacity-val').textContent = lyric.background_opacity ? Math.round(lyric.background_opacity * 100) + '%' : 'Off';
  document.getElementById('halo-intensity-val').textContent = Math.round(lyric.halo_intensity * 100) + '%';
  renderList();
  renderTimeline();
}

// Live-update from form inputs
lyricForm.addEventListener('change', finishFormEdit);
lyricForm.addEventListener('focusout', finishFormEdit);
['lyric-background-opacity','lyric-background-padding','lyric-background-softness','lyric-halo-size','lyric-halo-intensity','lyric-text','lyric-start','lyric-end','lyric-x','lyric-y','lyric-size','lyric-color','lyric-jitter'].forEach(id => {
  document.getElementById(id).addEventListener('input', commitForm);
});
document.getElementById('lyric-jitter').addEventListener('input', e => {
  document.getElementById('jitter-val').textContent = parseFloat(e.target.value).toFixed(1);
});

// ── Network ────────────────────────────────────────────────────────────────
document.getElementById('lyric-font').addEventListener('change', commitForm);

function lyricFont(size, text, name = 'auto') {
  const family = /[\u3040-\u30ff\u3400-\u9fff]/.test(text) ? 'Japanese' : name;
  const font = `${size}px 'Lyric-${family}'`;
  // Canvas does not initiate font downloads itself. Load on demand; the animation
  // loop repaints with the selected font as soon as it becomes available.
  if (!document.fonts.check(font, text || ' ')) document.fonts.load(font, text || ' ');
  return font;
}

async function loadLyrics(stem) {
  try {
    const r = await fetch('/lyrics/' + stem);
    lyrics = r.ok ? await r.json() : [];
    // ensure every lyric has an id
    lyrics.forEach(l => { if (!l.id) l.id = uid(); });
    undoStack.length = 0; redoStack.length = 0; formEdit = null;
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
    const response = await fetch('/lyrics/' + currentStem, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(lyrics),
    });
    if (!response.ok) throw new Error('Save failed');
    setStatus('saved ✓', 'ok');
    return true;
    setTimeout(() => setStatus(''), 2000);
  } catch (e) {
    setStatus('save error: ' + e.message, 'err');
  }
}

function loadVideo() {
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
        ${escapeText(lyric.text) || '<em style="color:#555">empty</em>'}
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
  // Freeze the layout during a gesture; repack only when the mouse is released.
  if (!tlDrag) {
    timelineLanes = new Map();
    const occupied = [];
    const hasLane = l => Number.isInteger(l.lane) && l.lane >= 0 && l.lane <= 100;
    for (const lyric of [...lyrics].sort((a, b) => Number(hasLane(b)) - Number(hasLane(a)) || a.start - b.start || a.end - b.end)) {
      const left = lyric.start * PX_PER_SEC;
      const right = left + Math.max(16, (lyric.end - lyric.start) * PX_PER_SEC);
      const fits = lane => !(occupied[lane] || []).some(([a, b]) => left < b + 4 && right + 4 > a);
      let lane = hasLane(lyric) && fits(lyric.lane) ? lyric.lane : 0;
      while (!fits(lane)) lane++;
      (occupied[lane] ||= []).push([left, right]);
      timelineLanes.set(lyric.id, lane);
    }
  }
  const laneCount = Math.max(1, ...Array.from(timelineLanes.values(), lane => lane + 1));
  const trackHeight = (laneCount + 1) * LANE_HEIGHT + 12;
  tlTrack.style.height = trackHeight + 'px';
  tlInner.style.height = (trackHeight + 24) + 'px';
  tlWrap.style.height = (trackHeight + 24 + 16) + 'px';
  lyrics.forEach(lyric => {
    const block = document.createElement('div');
    block.className = 'tl-block' + (lyric.id === selectedId ? ' selected' : '');
    block.dataset.id = lyric.id;
    block.style.top = (6 + (timelineLanes.get(lyric.id) || 0) * LANE_HEIGHT) + 'px';
    block.style.left  = (lyric.start * PX_PER_SEC) + 'px';
    block.style.width = Math.max(16, (lyric.end - lyric.start) * PX_PER_SEC) + 'px';
    block.style.background = rgbToHex(lyric.color) + 'bb';
    block.textContent = lyric.text || '…';
    block.title = `${lyric.text || 'Empty lyric'} (${lyric.start.toFixed(2)}s – ${lyric.end.toFixed(2)}s)`;
    block.addEventListener('mousedown', e => {
      e.preventDefault();
      e.stopPropagation();
      selectLyric(lyric.id);
      const before = snapshot();
      const blockRect = block.getBoundingClientRect();
      const distLeft  = e.clientX - blockRect.left;
      const distRight = blockRect.right - e.clientX;
      const edgeZone  = Math.min(8, blockRect.width * 0.25);
      const mode = distLeft < edgeZone ? 'start' : distRight < edgeZone ? 'end' : 'move';
      tlDrag = { id: lyric.id, mode, startX: e.clientX, startY: e.clientY,
        origLane: timelineLanes.get(lyric.id) || 0, origStart: lyric.start, origEnd: lyric.end, before };
      if (mode === 'move') {
        // The fade is invisible at the start; preview the selected lyric at its midpoint.
        video.currentTime = (lyric.start + lyric.end) / 2;
        updatePlayhead();
      }
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
  if (!video.paused && !scrubbing) {
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

let scrubbing = false;

function seekTimeline(clientX) {
  const rect = tlInner.getBoundingClientRect();
  const time = (clientX - rect.left) / PX_PER_SEC;
  if (!Number.isFinite(video.duration)) return;
  video.currentTime = Math.max(0, Math.min(video.duration, time));
  updatePlayhead();
}

function startScrubbing(e) {
  if (e.button !== 0) return;
  e.preventDefault();
  e.stopPropagation();
  scrubbing = true;
  video.pause();
  e.currentTarget.setPointerCapture(e.pointerId);
  seekTimeline(e.clientX);
}

for (const target of [tlRuler, tlPlayhead, tlTrack]) {
  target.addEventListener('pointerdown', e => {
    if (e.target.closest('.tl-block')) return;
    startScrubbing(e);
  });
  target.addEventListener('pointermove', e => {
    if (scrubbing) seekTimeline(e.clientX);
  });
  target.addEventListener('pointerup', e => {
    if (!scrubbing) return;
    seekTimeline(e.clientX);
    scrubbing = false;
  });
  target.addEventListener('pointercancel', () => { scrubbing = false; });
  target.addEventListener('lostpointercapture', () => { scrubbing = false; });
}

video.addEventListener('timeupdate', updatePlayhead);
video.addEventListener('seeked', updatePlayhead);
video.addEventListener('loadedmetadata', buildTimeline);

// ── Canvas overlay ─────────────────────────────────────────────────────────

// Update canvas size to match natural video resolution
function reelsVideoBounds() {
  const w = video.videoWidth || 1080, h = video.videoHeight || 1920;
  const scale = Math.min(1080 / w, 1920 / h);
  const width = Math.min(1080, Math.round(w * scale));
  const height = Math.min(1920, Math.round(h * scale));
  return {left: Math.floor((1080 - width) / 2), top: Math.floor((1920 - height) / 2), width, height};
}

function lyricPointer(e) {
  const rect = overlay.getBoundingClientRect();
  let nx = (e.clientX - rect.left) / rect.width;
  let ny = (e.clientY - rect.top) / rect.height;
  if (exportAspect === 'original') {
    const bounds = reelsVideoBounds();
    nx = (bounds.left + nx * bounds.width) / 1080;
    ny = (bounds.top + ny * bounds.height) / 1920;
  }
  return {nx, ny};
}

function updateExportPreview() {
  const original = exportAspect === 'original';
  const w = original ? video.videoWidth : 1080;
  const h = original ? video.videoHeight : 1920;
  if (!w || !h) return;
  overlay.width = w;
  overlay.height = h;
  const stage = document.getElementById('video-stage');
  stage.style.aspectRatio = `${w} / ${h}`;
  stage.style.width = `min(100%, calc((100vh - 240px) * ${w / h}))`;
}
video.addEventListener('loadedmetadata', updateExportPreview);

function lyricLines(text) {
  return text.replace(/\\n/g, '\n').replace(/\r\n?/g, '\n').split('\n');
}

// Shared geometry for text, its backdrop, and picking any line while dragging.
function lyricTextLayout(text, size, x, y) {
  const lines = lyricLines(text);
  const positioned = lines.map((line, i) => ({text: line, x, y: y + (i - (lines.length - 1) / 2) * size * 1.2}));
  const boxes = positioned.filter(line => line.text).map(line => {
    const m = ctx.measureText(line.text);
    return {left: x - m.actualBoundingBoxLeft, right: x + m.actualBoundingBoxRight,
      top: line.y - m.actualBoundingBoxAscent, bottom: line.y + m.actualBoundingBoxDescent};
  });
  return {lines: positioned, left: Math.min(x, ...boxes.map(b => b.left)),
    right: Math.max(x, ...boxes.map(b => b.right)), top: Math.min(y, ...boxes.map(b => b.top)),
    bottom: Math.max(y, ...boxes.map(b => b.bottom))};
}

function drawScribbleText(x, y, text, size, color, tick, fade, haloSize = 4, haloIntensity = .75, font = 'auto') {
  ctx.save();
  ctx.font = lyricFont(size, text, font);
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillStyle = color;
  ctx.shadowColor = `rgba(0,0,0,${haloIntensity})`;
  ctx.shadowBlur = haloSize;
  ctx.shadowOffsetY = Math.max(1, size / 24);
  ctx.strokeStyle = `rgba(0,0,0,${haloIntensity})`;
  ctx.lineWidth = 2 * Math.max(1, Math.round(size / 36));
  ctx.lineJoin = 'round';
  const layout = lyricTextLayout(text, size, x, y);
  if (haloIntensity > 0) {
    for (const line of layout.lines) ctx.strokeText(line.text, line.x, line.y);
  }
  ctx.shadowColor = 'transparent';
  for (const line of layout.lines) ctx.fillText(line.text, line.x, line.y);
  ctx.restore();
}

function drawLyricBackground(lyric, x, y) {
  const opacity = lyric.background_opacity ?? 0;
  if (!opacity || !lyric.text) return;
  const padding = lyric.background_padding ?? 16;
  const softness = lyric.background_softness ?? 24;
  ctx.save();
  ctx.font = lyricFont(lyric.size, lyric.text, lyric.font);
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  const layout = lyricTextLayout(lyric.text, lyric.size, x, y);
  const left = layout.left - padding - softness;
  const top = layout.top - padding - softness;
  const width = layout.right - layout.left + 2 * (padding + softness);
  const height = layout.bottom - layout.top + 2 * (padding + softness);
  const backdrop = document.createElement('canvas');
  backdrop.width = Math.ceil(width); backdrop.height = Math.ceil(height);
  const bg = backdrop.getContext('2d');
  function gradient(length, horizontal) {
    if (!softness) return '#000';
    const g = bg.createLinearGradient(0, 0, horizontal ? length : 0, horizontal ? 0 : length);
    const edge = Math.min(.5, softness / length);
    for (let i = 0; i <= 16; i++) {
      const t = i / 16;
      g.addColorStop(t * edge, `rgba(0,0,0,${t * t * (3 - 2 * t)})`);
    }
    for (let i = 0; i <= 16; i++) {
      const t = i / 16;
      g.addColorStop(1 - edge + t * edge, `rgba(0,0,0,${1 - t * t * (3 - 2 * t)})`);
    }
    return g;
  }
  bg.fillStyle = gradient(width, true);
  bg.fillRect(0, 0, width, height);
  bg.globalCompositeOperation = 'destination-in';
  bg.fillStyle = gradient(height, false);
  bg.fillRect(0, 0, width, height);
  ctx.globalAlpha *= opacity;
  ctx.drawImage(backdrop, left, top);
  ctx.restore();
}

function drawReelsGuides(w, h) {
  if (!document.getElementById('reels-guides').checked || !w || !h) return;
  // The 9:16 boundary coincides with the export canvas for padded footage.
  const cropW = Math.min(w, h * 9 / 16);
  const cropH = Math.min(h, w * 16 / 9);
  const left = (w - cropW) / 2, top = (h - cropH) / 2;
  ctx.save();
  ctx.fillStyle = 'rgba(235,55,65,.22)';
  ctx.fillRect(0, 0, w, top);
  ctx.fillRect(0, top + cropH, w, top);
  ctx.fillRect(0, top, left, cropH);
  ctx.fillRect(left + cropW, top, left, cropH);
  const x = left + cropW * .06, y = top + cropH * .14;
  const safeW = cropW * .88, safeH = cropH * .51;
  ctx.fillStyle = 'rgba(255,160,40,.16)';
  ctx.fillRect(left, top, cropW, cropH * .14);
  ctx.fillRect(left, y + safeH, cropW, cropH * .35);
  ctx.fillRect(left, y, cropW * .06, safeH);
  ctx.fillRect(x + safeW, y, cropW * .06, safeH);
  ctx.strokeStyle = 'rgba(255,190,80,.9)';
  ctx.lineWidth = Math.max(1, h / 720);
  ctx.setLineDash([h / 100, h / 150]);
  ctx.strokeRect(x, y, safeW, safeH);
  ctx.font = `${Math.max(12, h / 55)}px sans-serif`;
  ctx.fillStyle = '#fff';
  ctx.shadowColor = '#000'; ctx.shadowBlur = 4;
  ctx.fillStyle = '#ffbe50';
  ctx.fillText('Text clearance • approximate', x + 8, y + Math.max(18, h / 40));
  const inset = Math.max(2, h / 480);
  ctx.setLineDash([]);
  ctx.strokeStyle = '#55d9ff';
  ctx.strokeRect(left + inset, top + inset, cropW - inset * 2, cropH - inset * 2);
  ctx.fillStyle = '#55d9ff';
  ctx.fillText('9:16 crop / export boundary', left + inset + 12, top + Math.max(24, h / 35));
  ctx.restore();
}

function drawOverlay() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const t = video.currentTime;
  drawReelsGuides(overlay.width, overlay.height);
  // Saved lyric coordinates always refer to the 1080x1920 editing canvas.
  // Original-layout output removes padding without moving text within the video.
  const w = 1080, h = 1920;
  ctx.save();
  if (exportAspect === 'original') {
    const bounds = reelsVideoBounds();
    const sx = overlay.width / bounds.width, sy = overlay.height / bounds.height;
    ctx.setTransform(sx, 0, 0, sy, -bounds.left * sx, -bounds.top * sy);
  }

  for (const lyric of lyrics) {
    if (t < lyric.start || t >= lyric.end) continue;
    const fade = Math.min(lyric.fade ?? .25, (lyric.end - lyric.start) / 2);
    ctx.globalAlpha = fade ? Math.min(1, (t-lyric.start)/fade, (lyric.end-t)/fade) : 1;
    drawLyricBackground(lyric, lyric.x * w, lyric.y * h);
    drawScribbleText(
      lyric.x * w, lyric.y * h,
      lyric.text, lyric.size,
      rgbToHex(lyric.color),
      jitterTick, lyric.fade ?? 0.25, lyric.halo_size ?? 4, lyric.halo_intensity ?? .75,
      lyric.font ?? 'auto',
    );
  }

  ctx.restore();
  ctx.globalAlpha = 1;
  drawCenterGuides();
  // Advance jitter at ~10hz to match Python's per-3-frame seed stepping
  const now = performance.now();
  if (now - lastJitterUpdate > 100) { jitterTick++; lastJitterUpdate = now; }
}

function drawCenterGuides() {
  if (!canvasDrag) return;
  const rect = overlay.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const w = overlay.width, h = overlay.height;
  const scale = w / rect.width;
  ctx.save();
  ctx.lineWidth = scale;
  for (const axis of ['x', 'y']) {
    const snapped = canvasDrag[axis === 'x' ? 'snapX' : 'snapY'];
    ctx.strokeStyle = snapped ? '#55d9ff' : 'rgba(255,255,255,.45)';
    ctx.setLineDash(snapped ? [] : [6 * scale, 6 * scale]);
    ctx.beginPath();
    ctx.moveTo(axis === 'x' ? w / 2 : 0, axis === 'y' ? h / 2 : 0);
    ctx.lineTo(axis === 'x' ? w / 2 : w, axis === 'y' ? h / 2 : h);
    ctx.stroke();
  }
  ctx.restore();
}

// Use screen pixels so snapping feels the same at every preview size.
function snapToCenter(value, pixelsPerUnit, disabled) {
  return !disabled && Math.abs(value - 0.5) * pixelsPerUnit <= 8 ? 0.5 : value;
}

function rafLoop() {
  drawOverlay();
  requestAnimationFrame(rafLoop);
}
rafLoop();

// ── Spatial drag on canvas ─────────────────────────────────────────────────

let tlDrag     = null; // {id, mode, startX, origStart, origEnd}

function findLyricAt(nx, ny) {
  const vw = 1080;
  const vh = 1920;
  const t = video.currentTime;
  for (let i = lyrics.length - 1; i >= 0; i--) {
    const l = lyrics[i];
    if (!l.text || t < l.start || t >= l.end) continue;
    ctx.save();
    ctx.font = lyricFont(l.size, l.text, l.font);
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const box = lyricTextLayout(l.text, l.size, l.x * vw, l.y * vh);
    ctx.restore();
    if (nx * vw >= box.left - 10 && nx * vw <= box.right + 10 &&
        ny * vh >= box.top - 8 && ny * vh <= box.bottom + 8) return l;
  }
  return null;
}

overlay.addEventListener('mousedown', e => {
  const {nx, ny} = lyricPointer(e);
  const hit = findLyricAt(nx, ny);
  if (!hit) { selectLyric(null); return; }
  e.preventDefault();
  selectLyric(hit.id);
  canvasDrag = { id: hit.id, ox: nx - hit.x, oy: ny - hit.y, before: snapshot() };
  overlay.style.cursor = 'grabbing';
});

document.addEventListener('mousemove', e => {
  const {nx, ny} = lyricPointer(e);
  if (canvasDrag) {
    const lyric = getLyric(canvasDrag.id);
    if (!lyric) return;
    const rect = overlay.getBoundingClientRect();
    const bounds = exportAspect === 'original' ? reelsVideoBounds() : {width: 1080, height: 1920};
    lyric.x = snapToCenter(Math.max(0, Math.min(1, nx - canvasDrag.ox)), rect.width * 1080 / bounds.width, e.altKey);
    lyric.y = snapToCenter(Math.max(0, Math.min(1, ny - canvasDrag.oy)), rect.height * 1920 / bounds.height, e.altKey);
    canvasDrag.snapX = lyric.x === 0.5;
    canvasDrag.snapY = lyric.y === 0.5;
    populateForm(lyric);
  } else {
    overlay.style.cursor = findLyricAt(nx, ny) ? 'grab' : 'default';
  }
});

// ── Global drag end + timeline drag movement ───────────────────────────────

document.addEventListener('mousemove', e => {
  if (!tlDrag) return;
  const lyric = getLyric(tlDrag.id);
  if (!lyric) return;
  const dt = (e.clientX - tlDrag.startX) / PX_PER_SEC;
  if (tlDrag.mode === 'move') {
    const laneDelta = Math.round((e.clientY - tlDrag.startY) / LANE_HEIGHT);
    if (laneDelta !== 0 || tlDrag.laneChanged) {
      lyric.lane = Math.max(0, Math.min(100, tlDrag.origLane + laneDelta));
      timelineLanes.set(lyric.id, lyric.lane);
      tlDrag.laneChanged = true;
    }
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
    rememberEdit(canvasDrag.before);
    renderList();
    canvasDrag = null;
    overlay.style.cursor = 'default';
  }
  if (tlDrag) {
    const moved = getLyric(tlDrag.id);
    if (tlDrag.laneChanged && moved) {
      // Give the dropped bar its requested row; overlapping bars find a free row.
      for (const other of lyrics) {
        if (other.id !== moved.id && other.lane === moved.lane &&
            other.start * PX_PER_SEC < moved.start * PX_PER_SEC + Math.max(16, (moved.end - moved.start) * PX_PER_SEC) + 4 &&
            moved.start * PX_PER_SEC < other.start * PX_PER_SEC + Math.max(16, (other.end - other.start) * PX_PER_SEC) + 4) delete other.lane;
      }
    }
    rememberEdit(tlDrag.before);
    renderList();
    tlDrag = null;
    renderTimeline();
  }
});

// ── Keyboard shortcuts ─────────────────────────────────────────────────────

function isInputFocused() {
  const t = document.activeElement?.tagName;
  return t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT';
}

document.addEventListener('keydown', e => {
  if (document.getElementById('export-dialog').open) return;
  if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'z' || e.key.toLowerCase() === 'y')) {
    e.preventDefault();
    restoreHistory(e.shiftKey || e.key.toLowerCase() === 'y');
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    saveLyrics();
    return;
  }

  if (isInputFocused()) return;

  if (e.key === 'Enter') {
    e.preventDefault();
    video.currentTime = 0;
    return;
  }

  if (e.key === ' ') {
    e.preventDefault();
    video.paused ? video.play() : video.pause();
    return;
  }

  if (e.key === '[' || e.key === 'BracketLeft') {
    e.preventDefault();
    if (!selectedId) return;
    const lyric = getLyric(selectedId);
    finishFormEdit();
    const before = snapshot();
    lyric.start = parseFloat(video.currentTime.toFixed(2));
    if (lyric.end <= lyric.start) lyric.end = parseFloat((lyric.start + 3).toFixed(2));
    populateForm(lyric); renderList(); renderTimeline();
    rememberEdit(before);
    return;
  }

  if (e.key === ']' || e.key === 'BracketRight') {
    e.preventDefault();
    if (!selectedId) return;
    const lyric = getLyric(selectedId);
    finishFormEdit();
    const before = snapshot();
    lyric.end = parseFloat(video.currentTime.toFixed(2));
    if (lyric.end <= lyric.start) lyric.start = Math.max(0, parseFloat((lyric.end - 1).toFixed(2)));
    populateForm(lyric); renderList(); renderTimeline();
    rememberEdit(before);
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
document.getElementById('duplicate-btn').addEventListener('click', duplicateLyric);
document.getElementById('undo-btn').addEventListener('click', () => restoreHistory());
document.getElementById('redo-btn').addEventListener('click', () => restoreHistory(true));
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

// ── Render ─────────────────────────────────────────────────────────────────

let _pollInterval = null;

async function startRender() {
  const renderBtn = document.getElementById('render-btn');
  // Save first, then render
  if (!currentStem || !await saveLyrics()) return;
  renderBtn.disabled = true;
  setStatus('starting render…', 'busy');

  let r;
  try {
    r = await fetch('/render/' + currentStem, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ aspect: exportAspect, format: exportFormat }),
    });
  } catch (e) {
    setStatus('render error: ' + e.message, 'err');
    renderBtn.disabled = false;
    return;
  }
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
    const { state, progress, error, stem, output } = await r.json();

    if (state === 'running') {
      const pct = Math.round((progress || 0) * 100);
      setStatus(`rendering ${stem}… ${pct}%`, 'busy');
    } else {
      clearInterval(_pollInterval);
      _pollInterval = null;
      renderBtn.disabled = false;
      if (state === 'done') {
        window.location.href = "/download/" + stem;
        setStatus(`done → ${output || stem} ✓`, 'ok');
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

const exportDialog = document.getElementById('export-dialog');
document.getElementById('render-btn').addEventListener('click', () => {
  if (!currentStem) return;
  video.pause();
  exportDialog.showModal();
});
document.getElementById('export-cancel').addEventListener('click', () => exportDialog.close());
document.getElementById('export-form').addEventListener('change', e => {
  if (e.target.name === 'format') {
    exportFormat = e.target.value;
    return;
  }
  if (e.target.name !== 'aspect') return;
  exportAspect = e.target.value;
  updateExportPreview();
});
document.getElementById('export-form').addEventListener('submit', e => {
  e.preventDefault();
  exportDialog.close();
  startRender();
});

// ── Init ───────────────────────────────────────────────────────────────────

loadVideo();

function escapeText(text) { const el = document.createElement("span"); el.textContent = text; return el.innerHTML; }

document.addEventListener('mousedown', e => {
  if (e.target.closest('#lyric-form, .lyric-item, .tl-block, #overlay, #toolbar')) return;
  selectLyric(null);
});
