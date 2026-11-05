const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
function element() {
  return {
    value: '', checked: true, style: {}, dataset: {},
    listeners: {}, offsetWidth: 100, scrollLeft: 0,
    addEventListener(name, fn) { this.listeners[name] = fn; },
    appendChild() {}, getContext() { return {}; },
    pause() { this.paused = true; },
  };
}
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  },
  createElement: element,
  addEventListener(name, fn) { this[name] = fn; },
};
const context = vm.createContext({ document, requestAnimationFrame() {},
  fetch() { return new Promise(() => {}); } });
for (const name of ['editor_timing', 'editor']) {
  vm.runInContext(fs.readFileSync(`string_lights/static/${name}.js`, 'utf8'), context);
}
const run = code => vm.runInContext(code, context);
run('video.duration = 10; notes = [{string: 0, start: 3.123, end: 3.373, value: 1}];');
function convert(source, target) {
  document.getElementById('source-bpm').value = source;
  document.getElementById('target-bpm').value = target;
  elements.get('convert-tempo-btn').listeners.click();
}
convert(120, 90);
assert.ok(Math.abs(run('notes[0].start') - 4.164) < 1e-10);
assert.ok(Math.abs(run('notes[0].end') - 3.373 * 120 / 90) < 1e-10);
convert(90, 120);
assert.ok(Math.abs(run('notes[0].start') - 3.123) < 1e-10);
// Undo restores exact timestamps and grid spacing.
document.keydown({ ctrlKey: true, key: 'z', preventDefault() {} });
assert.ok(Math.abs(run('notes[0].start') - 4.164) < 1e-10);
const before = run('notes[0].start');
convert(0, 180);
assert.equal(run('notes[0].start'), before);
// Grid and zoom are display preferences, never a quantization operation.
elements.get('grid-interval').listeners.change({target: {value: 100}});
elements.get('zoom').listeners.input({target: {value: 400}});
assert.equal(run('notes[0].start'), before);
assert.equal(run('editingTime(0.137, true)'), 0.137);
assert.ok(Math.abs(run('editingTime(0.137, false)') - 0.1) < 1e-10);
run("document.getElementById('snap').checked = false");
assert.equal(run('editingTime(0.137, false)'), 0.137);
console.log('Tempo conversion, off-beat timestamps, undo, grid, zoom and snap passed.');

// Reversal changes string assignment without moving notes or changing edit history.
run(`notes = [{string: 0, start: .1, end: .3, value: 1}];
  const beforeReverse = JSON.stringify(notes);
  const beforeHistory = history.length;
  const testLabels = [5,4,3,2,1,0].map(s => ({dataset: {s}}));
  const testMarkers = [{dataset: {note: 0}}];
  labelsEl.querySelectorAll = () => testLabels;
  rowsEl.querySelectorAll = () => testMarkers;
  overlay.width = 320; video.currentTime = .15;
  projData = {fps: 1, lines: [Array.from({length: 6}, (_, i) => [i,0,i,1])]};
  const moves = [];
  Object.assign(octx, {clearRect() {}, beginPath() {}, moveTo(x) {moves.push(x);}, lineTo() {}, stroke() {}});
`);
// Each canonical row previews on its corresponding physical projected string.
for (let string = 0; string < 6; string++) {
  run(`notes = [{string: ${string}, start: .1, end: .3, value: 1}]; moves.length = 0; drawOverlay();`);
  assert.equal(run('moves[0]'), string);
}
run('notes = JSON.parse(beforeReverse);');
elements.get('reverse-strings').listeners.change({target: {checked: true}});
assert.equal(run('testLabels.map(label => label.textContent).join(",")'), 'E2,A2,D3,G3,B3,E4');
assert.equal(run('JSON.stringify(notes)'), run('beforeReverse'));
assert.equal(run('history.length'), run('beforeHistory'));
assert.equal(run('moves[moves.length - 1]'), 5);
assert.match(run('testMarkers[0].title'), /^E4 at /);
elements.get('filename').value = 'clip';
let saved;
context.fetch = async (url, request) => {
  saved = JSON.parse(request.body);
  return {ok: true, status: 200, json: async () => ({saved: 'data/input/clip.npy'})};
};
elements.get('export-btn').listeners.click().then(() => {
  assert.equal(saved.frames[4][5], 1);
  assert.equal(saved.frames[4][0], 0);
  elements.get('reverse-strings').listeners.change({target: {checked: false}});
  assert.equal(run('testLabels.map(label => label.textContent).join(",")'), 'E4,B3,G3,D3,A2,E2');
  assert.equal(run('moves[moves.length - 1]'), 0);
  assert.equal(run('JSON.stringify(notes)'), run('beforeReverse'));
  console.log('String reversal: labels, preview, export and unchanged notes passed.');
}).catch(error => { console.error(error); process.exitCode = 1; });
