const canvas = document.querySelector('#canvas');
const ctx = canvas.getContext('2d');
const tool = document.querySelector('#tool');
const status = document.querySelector('#status');
const slider = document.querySelector('#frame');
const number = document.querySelector('#frame-number');
let annotation, image, scale = 1, ox = 0, oy = 0, drag = null, selected = null;
let draft = [], history = [], space = false, dirty = false, loadVersion = 0;
const copy = value => JSON.parse(JSON.stringify(value));
const point = event => {const r = canvas.getBoundingClientRect(); return [event.clientX-r.left,event.clientY-r.top];};
const toImage = p => [(p[0]-ox)/scale,(p[1]-oy)/scale];
const toScreen = p => [p[0]*scale+ox,p[1]*scale+oy];
function message(text,error=false){status.textContent=text;status.classList.toggle('error',error);}
function checkpoint(){history.push(copy(annotation));if(history.length>100)history.shift();dirty=true;}
function fit(){if(!image)return;scale=Math.min(canvas.width/image.width,canvas.height/image.height)*.95;ox=(canvas.width-image.width*scale)/2;oy=(canvas.height-image.height*scale)/2;draw();}
function path(points,color,closed=false){if(!points.length)return;ctx.beginPath();points.forEach((p,i)=>{const q=toScreen(p);i?ctx.lineTo(...q):ctx.moveTo(...q);});if(closed)ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();}
function handle(p,color){const q=toScreen(p);ctx.beginPath();ctx.arc(...q,5,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#111';ctx.stroke();}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);if(!image||!annotation)return;ctx.drawImage(image,ox,oy,image.width*scale,image.height*scale);
  for(const key of ['tracking_polygons','exclusion_polygons'])annotation[key].forEach((poly,i)=>{const color=key==='tracking_polygons'?'#52cfff':'#ff7c82';path(poly,color,true);poly.forEach(p=>handle(p,selected?.key===key&&selected.i===i?'#ffe18c':color));});
  annotation.string_lines.forEach((line,i)=>{path(line,'#70ff9e');line.forEach(p=>handle(p,'#eaffef'));const p=toScreen(line[0]);ctx.fillStyle='#fff';ctx.font='12px sans-serif';ctx.fillText(String(i+1),p[0]+8,p[1]-8);});path(draft,'#ffe18c');draft.forEach(p=>handle(p,'#ffe18c'));
}
function hit(p){let best=null,distance=12;for(const key of ['string_lines','tracking_polygons','exclusion_polygons'])annotation[key].forEach((poly,i)=>poly.forEach((v,j)=>{const q=toScreen(v),d=Math.hypot(q[0]-p[0],q[1]-p[1]);if(d<distance){distance=d;best={key,i,j};}}));return best;}
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('wheel',e=>{e.preventDefault();if(!image)return;const p=point(e),anchor=toImage(p);scale=Math.max(.05,Math.min(30,scale*Math.exp(-e.deltaY*.0015)));ox=p[0]-anchor[0]*scale;oy=p[1]-anchor[1]*scale;draw();},{passive:false});
canvas.addEventListener('pointerdown',e=>{if(!annotation||!image)return;canvas.focus();const p=point(e);canvas.setPointerCapture(e.pointerId);
  if(e.button===2||e.button===1||space){drag={pan:p};return;}
  if(e.button!==0)return;
  if(tool.value!=='edit'){draft.push(toImage(p));draw();return;}
  selected=hit(p);if(selected){checkpoint();drag={handle:selected};}draw();
});
canvas.addEventListener('pointermove',e=>{if(!drag)return;const p=point(e);if(drag.pan){ox+=p[0]-drag.pan[0];oy+=p[1]-drag.pan[1];drag.pan=p;}else{const h=drag.handle,q=toImage(p);annotation[h.key][h.i][h.j]=[Math.max(0,Math.min(image.width-1,q[0])),Math.max(0,Math.min(image.height-1,q[1]))];}draw();});
for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{drag=null;});
function finish(){if(tool.value==='edit'||draft.length<3){message('A region needs at least three vertices.',true);return;}checkpoint();annotation[tool.value].push(draft);draft=[];draw();}
document.querySelector('#finish').onclick=finish;
document.querySelector('#remove').onclick=()=>{if(!selected||selected.key==='string_lines'){message('Select a region vertex first.',true);return;}checkpoint();annotation[selected.key].splice(selected.i,1);selected=null;draw();};
document.querySelector('#undo').onclick=()=>{if(history.length){annotation=history.pop();slider.value=number.value=annotation.frame_index;selected=null;draft=[];dirty=true;loadFrame(annotation.frame_index);}};
document.querySelector('#fit').onclick=fit;
tool.onchange=()=>{draft=[];selected=null;draw();};
window.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;if(e.code==='Space'){space=true;e.preventDefault();}if(e.key==='Enter')finish();if(e.key==='Escape'){draft=[];drag=null;draw();}});
window.addEventListener('keyup',e=>{if(e.code==='Space')space=false;});window.addEventListener('blur',()=>{space=false;drag=null;});
async function loadFrame(index){const version=++loadVersion;message('Loading frame…');const next=new Image();next.src='/frame/'+index;await new Promise((resolve,reject)=>{next.onload=resolve;next.onerror=()=>reject(new Error('Could not load reference frame'));});if(version!==loadVersion)return;const initial=!image;image=next;if(initial)fit();else draw();message('Reference frame '+index+'. Changing frames keeps the annotation coordinates; check alignment before saving.');}
async function changeFrame(value){const index=Math.max(0,Math.min(Number(slider.max),Math.round(Number(value))));if(!Number.isFinite(index))return;checkpoint();annotation.frame_index=index;slider.value=number.value=index;draft=[];selected=null;try{await loadFrame(index);}catch(e){message(e.message,true);}}
slider.onchange=()=>changeFrame(slider.value);number.onchange=()=>changeFrame(number.value);
document.querySelector('#save').onclick=async()=>{const button=document.querySelector('#save');button.disabled=true;try{if(draft.length)throw new Error('Finish or cancel the polygon before saving.');const response=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json','X-Tuner-Token':window.tunerToken},body:JSON.stringify(annotation)});const result=await response.json();if(!response.ok)throw new Error(result.error);dirty=false;message(result.message);}catch(e){message(e.message,true);}finally{button.disabled=false;}};
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
new ResizeObserver(()=>{const r=canvas.getBoundingClientRect();canvas.width=Math.round(r.width);canvas.height=Math.round(r.height);draw();}).observe(document.querySelector('#stage'));
(async()=>{try{const response=await fetch('/annotation');if(!response.ok)throw new Error('Could not load annotation');const data=await response.json();annotation=data.annotation;annotation.exclusion_polygons??=[];slider.max=number.max=data.total-1;slider.value=number.value=annotation.frame_index??0;await loadFrame(annotation.frame_index??0);}catch(e){message(e.message,true);}})();
