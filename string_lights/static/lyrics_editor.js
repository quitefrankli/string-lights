// ── State ──────────────────────────────────────────────────────────────────

let lyrics = [];       // [{id, text, start, end, x, y, size, color:[r,g,b]}, ...]
let selectedId = null;
let currentStem = '';
let exportAspect = 'padded';
let exportFormat = 'compatible';
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
    x: 0.5, y: 0.85, size: 48, color: [255, 255, 255],
    background_opacity: .65, background_padding: 16, background_softness: 24,
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
  Object.assign(lyric, readForm());
  document.getElementById('background-opacity-val').textContent = lyric.background_opacity ? Math.round(lyric.background_opacity * 100) + '%' : 'Off';
  document.getElementById('halo-intensity-val').textContent = Math.round(lyric.halo_intensity * 100) + '%';
  renderList();
  renderTimeline();
}

// Live-update from form inputs
['lyric-background-opacity','lyric-background-padding','lyric-background-softness','lyric-halo-size','lyric-halo-intensity','lyric-text','lyric-start','lyric-end','lyric-x','lyric-y','lyric-size','lyric-color','lyric-jitter'].forEach(id => {
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

function drawScribbleText(x, y, text, size, color, tick, fade, haloSize = 4, haloIntensity = .75) {
  ctx.save();
  ctx.font = `${size}px 'DejaVu Sans', sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillStyle = color;
  ctx.shadowColor = `rgba(0,0,0,${haloIntensity})`;
  ctx.shadowBlur = haloSize;
  ctx.shadowOffsetY = Math.max(1, size / 24);
  ctx.strokeStyle = `rgba(0,0,0,${haloIntensity})`;
  ctx.lineWidth = 2 * Math.max(1, Math.round(size / 36));
  ctx.lineJoin = 'round';
  ctx.strokeText(text, x, y);
  ctx.shadowColor = 'transparent';
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawLyricBackground(lyric, x, y) {
  const opacity = lyric.background_opacity ?? 0;
  if (!opacity || !lyric.text) return;
  const padding = lyric.background_padding ?? 16;
  const softness = lyric.background_softness ?? 24;
  ctx.save();
  ctx.font = `${lyric.size}px 'DejaVu Sans', sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  const metrics = ctx.measureText(lyric.text);
  const left = x - metrics.actualBoundingBoxLeft - padding - softness;
  const top = y - metrics.actualBoundingBoxAscent - padding - softness;
  const width = metrics.actualBoundingBoxLeft + metrics.actualBoundingBoxRight + 2 * (padding + softness);
  const height = metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent + 2 * (padding + softness);
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
    );
  }

  ctx.restore();
  ctx.globalAlpha = 1;
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
  const vw = 1080;
  const vh = 1920;
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
  const {nx, ny} = lyricPointer(e);
  const hit = findLyricAt(nx, ny);
  if (!hit) { selectLyric(null); return; }
  e.preventDefault();
  selectLyric(hit.id);
  canvasDrag = { id: hit.id, ox: nx - hit.x, oy: ny - hit.y };
  overlay.style.cursor = 'grabbing';
});

document.addEventListener('mousemove', e => {
  const {nx, ny} = lyricPointer(e);
  if (canvasDrag) {
    const lyric = getLyric(canvasDrag.id);
    if (!lyric) return;
    lyric.x = Math.max(0, Math.min(1, nx - canvasDrag.ox));
    lyric.y = Math.max(0, Math.min(1, ny - canvasDrag.oy));
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
  if (document.getElementById('export-dialog').open) return;
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
    const { state, progress, error, stem } = await r.json();

    if (state === 'running') {
      const pct = Math.round((progress || 0) * 100);
      setStatus(`rendering ${stem}… ${pct}%`, 'busy');
    } else {
      clearInterval(_pollInterval);
      _pollInterval = null;
      renderBtn.disabled = false;
      if (state === 'done') {
        window.location.href = "/download/" + stem;
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

refreshFileList();

document.getElementById('video-file').addEventListener('change', async e => {
  try {
    const form = new FormData(); form.append('video', e.target.files[0]);
    setStatus('Opening video…');
    const response = await fetch('/upload', {method:'POST', body:form});
    if (!response.ok) throw new Error('Video upload failed');
    const {stem} = await response.json();
    const option = new Option(e.target.files[0].name, stem, true, true);
    document.getElementById('filename').add(option); loadVideo();
  } catch (error) { setStatus(error.message, 'err'); }
});
document.getElementById('json-file').addEventListener('change', async e => {
  try {
    const data = JSON.parse(await e.target.files[0].text());
    if (!Array.isArray(data) || data.some(l => typeof l.text !== 'string' || !Number.isFinite(l.start) || !Number.isFinite(l.end) || l.start < 0 || l.end <= l.start)) throw new Error('Expected an array of lyrics with text, start, and end seconds');
    lyrics = data.map(l => ({x:.5, y:.85, size:48, color:[255,255,255], fade:.25, ...l, id:uid()}));
    selectLyric(null); renderList(); renderTimeline(); setStatus('Lyrics imported');
  } catch (error) { setStatus(error.message, 'err'); }
});

function escapeText(text) { const el = document.createElement("span"); el.textContent = text; return el.innerHTML; }

document.addEventListener('mousedown', e => {
  if (e.target.closest('#lyric-form, .lyric-item, .tl-block, #overlay, #toolbar')) return;
  selectLyric(null);
});
