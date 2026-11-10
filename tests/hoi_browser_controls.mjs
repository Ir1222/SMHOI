import assert from 'node:assert/strict';
import {Clock, bindControls, mountPlayer} from '../src/supermimic/hoi/viewer/player.mjs';
const elements = Object.fromEntries(['toggle','next','previous','reset','faster','slower','frame','view','colliders','image','status','gate','error','speed'].map(x => [x, {}]));
let listener;
const document = {getElementById: key => elements[key], addEventListener: (key, fn) => { listener = fn; }};
let changes = 0;
const clock = new Clock();
bindControls(document, clock, () => changes++, () => {});
elements.next.onclick(); assert.equal(clock.frame, 1); assert.equal(clock.paused, true);
listener({key:'ArrowLeft', preventDefault() {}}); assert.equal(clock.frame, 0);
elements.toggle.onclick(); for(let i=0;i<60;i++) clock.advance(1/60); assert.equal(clock.frame,50);
elements.faster.onclick(); clock.advance(.5); assert.equal(clock.frame,100); assert.equal(clock.speed,2);
elements.slower.onclick(); assert.equal(clock.speed,1);
elements.previous.onclick(); clock.advance(10); assert.equal(clock.frame,99);
elements.frame.oninput({target:{value:'839'}}); assert.equal(clock.frame,839);
elements.next.onclick(); assert.equal(clock.frame,839);
elements.reset.onclick(); assert.equal(clock.frame,0); assert.equal(clock.paused,true);
for(const speed of [.125,.5,1,2,8]) { const c = new Clock(); c.speed=speed; c.paused=false; c.advance(16.78/speed); assert.equal(c.frame,839); assert.equal(c.paused,true); }
assert.ok(changes >= 9);
// Exercise the actual UI render loop and exact requested frame, including a failed request.
let callback, wall=0, requested, fail=false;
elements.view.value='overview'; elements.colliders.checked=false;
const mounted = mountPlayer(document, fn => {callback=fn;}, async url => {
 requested=url;
 return fail ? {ok:false, text:async()=> 'test error'} : {ok:true, json:async()=>({image:'fixture',status:'frame status',training_ready:false})};
},()=>wall);
await callback(); assert.equal(requested,'/frame?index=0&view=overview&colliders=0');
assert.equal(elements.status.textContent,'frame status'); assert.match(elements.gate.textContent,/training_ready=false/);
elements.toggle.onclick(); wall=1000; await callback(); assert.match(requested,/index=50&/);
elements.frame.oninput({target:{value:'260'}}); await callback(); assert.match(requested,/index=260&/);
fail=true; elements.next.onclick(); await callback(); assert.equal(mounted.paused,true); assert.match(elements.error.textContent,/test error/);
console.log('PASS: browser buttons, keyboard events, frame slider, speeds, terminal bounds, UI render loop and errors');
