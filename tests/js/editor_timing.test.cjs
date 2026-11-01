const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const ctx = vm.createContext({});
vm.runInContext(fs.readFileSync('string_lights/static/editor_timing.js', 'utf8'), ctx);
const run = s => vm.runInContext(s, ctx);
run(`const frames = Array.from({length: 88}, () => Array(6).fill(0));
  frames[3][0] = 1; frames[4][0] = 1;
  frames[6][0] = 1;
  frames[3][5] = 4; frames[4][5] = 5;
  frames[87][1] = 1;
  const imported = notesFromFrames(frames);`);
assert.equal(run('imported.length'), 5);
assert.equal(run('imported[0].start'), 3 / 44);
assert.equal(run('JSON.stringify(notesToFrames(imported, 2))'), run('JSON.stringify(frames)'));
assert.equal(run('notesFromFrames([]).length'), 0);
assert.equal(run('notesToFrames([], 1).length'), 44);
run('const freeNote = [{string: 2, start: 0.137, end: 0.237, value: 1}];');
assert.equal(run('notesToFrames(freeNote, 1)[6][2]'), 1);
assert.equal(run('notesToFrames(freeNote, 1)[5][2]'), 0);
assert.ok(Math.abs(run('notesFromFrames(notesToFrames(freeNote, 1))[0].start') - 0.137) < 1 / 44);
console.log('Frame-perfect import/export, short off-beat notes and value changes passed.');
// Overlapping display spans must not merge distinct same-string plucks on export.
run(`const repeated = [
  {string: 0, start: .1, end: .35, value: 1},
  {string: 0, start: .2, end: .45, value: 1},
]; const restored = notesFromFrames(notesToFrames(repeated, 1));`);
assert.equal(run('restored.length'), 2);
assert.ok(Math.abs(run('restored[1].start') - .2) < 1 / 44);
