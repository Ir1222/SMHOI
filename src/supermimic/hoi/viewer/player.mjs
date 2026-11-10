export class Clock {
  constructor() { this.frame = 0; this.paused = true; this.speed = 1; this.remainder = 0; }
  advance(seconds) {
    if (this.paused) return;
    const samples = this.remainder + seconds * 50 * this.speed;
    const steps = Math.floor(samples + 1e-10);
    this.remainder = Math.max(0, samples - steps);
    this.frame = Math.min(839, this.frame + steps);
    if (this.frame === 839) { this.paused = true; this.remainder = 0; }
  }
  command(name) {
    if (name === 'toggle') this.paused = !this.paused;
    else if (name === 'next' || name === 'previous') {
      this.frame = Math.max(0, Math.min(839, this.frame + (name === 'next' ? 1 : -1)));
      this.paused = true; this.remainder = 0;
    } else if (name === 'reset') { this.frame = 0; this.paused = true; this.remainder = 0; }
    else if (name === 'faster') this.speed = Math.min(8, this.speed * 2);
    else if (name === 'slower') this.speed = Math.max(.125, this.speed / 2);
  }
}
export const keyCommands = {' ': 'toggle', ArrowRight: 'next', ArrowLeft: 'previous',
  r: 'reset', R: 'reset', '+': 'faster', '=': 'faster', ']': 'faster', '-': 'slower', '[': 'slower'};

export function bindControls(doc, clock, changed, tick) {
  for (const name of ['toggle', 'next', 'previous', 'reset', 'faster', 'slower']) {
    doc.getElementById(name).onclick = () => { tick(); clock.command(name); changed(); };
  }
  doc.addEventListener('keydown', event => {
    if (event.target?.tagName === 'SELECT') return;
    if (keyCommands[event.key]) {
      event.preventDefault(); tick(); clock.command(keyCommands[event.key]); changed();
    }
  });
  doc.getElementById('frame').oninput = event => {
    clock.frame = Math.max(0, Math.min(839, Number(event.target.value)));
    clock.paused = true; clock.remainder = 0; tick(); changed();
  };
}

export function mountPlayer(doc, requestFrame, fetchFrame, now) {
  const clock = new Clock();
  let previous = now(), inFlight = false, renderedKey = '';
  const el = id => doc.getElementById(id);
  function tick() { const current = now(); clock.advance((current - previous)/1000); previous = current; }
  function update() {
    el('toggle').textContent = clock.paused ? 'Play' : 'Pause';
    el('speed').textContent = `${clock.speed}×`;
    el('frame').value = clock.frame;
  }
  bindControls(doc, clock, update, tick);
  async function loop() {
    tick(); update();
    const view = el('view').value, collision = el('colliders').checked;
    const key = `${clock.frame}/${view}/${collision}`;
    if (!inFlight && key !== renderedKey) {
      inFlight = true;
      try {
        const response = await fetchFrame(`/frame?index=${clock.frame}&view=${view}&colliders=${Number(collision)}`);
        if (!response.ok) throw new Error(await response.text());
        const result = await response.json();
        el('image').src = 'data:image/jpeg;base64,' + result.image;
        el('status').textContent = result.status;
        el('gate').textContent = `KINEMATIC REFERENCE · training_ready=${result.training_ready}`;
        el('error').textContent = '';
        renderedKey = key;
      } catch (error) { el('error').textContent = String(error); clock.paused = true; }
      finally { inFlight = false; }
    }
    requestFrame(loop);
  }
  requestFrame(loop);
  return clock;
}
if (typeof window !== 'undefined') {
  mountPlayer(document, callback => requestAnimationFrame(callback), url => fetch(url), () => performance.now());
}
