// Run with: node tests/lyrics_editor_history.cjs
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const elements = new Map();
function element() {
  return {style:{}, dataset:{}, value:'0', children:[], listeners:{},
    classList:{toggle(){}}, focus(){}, load(){}, checked:false,
    addEventListener(type, fn){(this.listeners[type] ||= []).push(fn);},
    appendChild(child){this.children.push(child);},
    getBoundingClientRect(){return {left:0,top:0,right:240,width:240,height:400};},
    getContext(){return {clearRect(){},save(){},restore(){},measureText(){return {
      actualBoundingBoxLeft:20,actualBoundingBoxRight:20,actualBoundingBoxAscent:10,actualBoundingBoxDescent:10,
    };}};},
    set innerHTML(value){this.children=[];},
  };
}
const document = {...element(), activeElement:null,
  fonts:{check(){return true;}},
  getElementById(id){if(!elements.has(id)) elements.set(id,element()); return elements.get(id);},
  querySelectorAll(){return [];}, createElement:element,
};
document.getElementById('editor-config').textContent = '{"stem":null}';
const sandbox = vm.createContext({document, assert, console,
  requestAnimationFrame(){}, performance:{now:()=>0},
});
vm.runInContext(fs.readFileSync('string_lights/static/lyrics_editor.js','utf8'),sandbox);
vm.runInContext(`
function fire(target, type, fields={}) {
  const event={preventDefault(){},stopPropagation(){},clientX:120,clientY:20,...fields};
  for(const listener of target.listeners[type] || []) listener(event);
}
lyrics=[{id:'a',text:'Hello',start:1,end:4,x:.5,y:.5,size:48,color:[255,255,255]}];
selectLyric('a'); renderTimeline();
duplicateLyric();
assert.equal(lyrics.length,2);
assert.notEqual(lyrics[1].id,'a');
assert.equal(lyrics[1].text,'Hello');
assert.notEqual(timelineLanes.get('a'), timelineLanes.get(lyrics[1].id));
restoreHistory(); assert.equal(lyrics.length,1);
restoreHistory(true); assert.equal(lyrics.length,2);
lyrics[1].color[0]=0; assert.equal(lyrics[0].color[0],255);
restoreHistory();
renderTimeline();
fire(tlTrack.children[0],'mousedown');
fire(document,'mousemove',{clientX:200,clientY:64});
fire(document,'mousemove',{clientX:280,clientY:108});
fire(document,'mouseup');
assert.equal(lyrics[0].start,3);
assert.equal(lyrics[0].lane,2);
assert.equal(timelineLanes.get('a'),2);
assert.equal(undoStack.length,1);
restoreHistory(); assert.equal(lyrics[0].start,1); assert.equal(lyrics[0].lane,undefined);
restoreHistory(true); assert.equal(lyrics[0].lane,2);
// Form edits group until change/focusout, including undo from inside a text field.
document.getElementById('lyric-text').value='Changed'; commitForm();
document.getElementById('lyric-text').value='Changed again'; commitForm();
document.activeElement={tagName:'TEXTAREA'};
fire(document,'keydown',{ctrlKey:true,key:'z'});
assert.equal(lyrics[0].text,'Hello');
fire(document,'keydown',{ctrlKey:true,shiftKey:true,key:'z'});
assert.equal(lyrics[0].text,'Changed again');
deleteLyric('a'); assert.equal(lyrics.length,0);
restoreHistory(); assert.equal(lyrics[0].text,'Changed again');
// A new edit discards redo history.
restoreHistory(); duplicateLyric(); assert.equal(redoStack.length,0);
// Pinned bars stay accessible when another lyric is dropped on their row.
lyrics=[{id:'a',text:'A',start:1,end:4,lane:0,color:[255,255,255]},
{id:'b',text:'B',start:1,end:4,lane:1,color:[255,255,255]}];
selectedId=null; renderTimeline();
fire(tlTrack.children[0],'mousedown');
fire(document,'mousemove',{clientY:64}); fire(document,'mouseup');
assert.equal(lyrics[0].lane,1);
assert.equal(timelineLanes.get('a'),1);
assert.notEqual(timelineLanes.get('a'),timelineLanes.get('b'));
console.log('History, duplication, grouped edits, lane dragging and collisions passed.');
`,sandbox);
