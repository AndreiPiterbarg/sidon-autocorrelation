// Shared animation/tween utilities.

const EASINGS = {
  linear: t => t,
  easeOut: t => 1 - (1 - t) ** 3,
  easeIn: t => t ** 3,
  easeInOut: t => t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2,
  bounce: t => {
    const n = 7.5625, d = 2.75;
    if (t < 1 / d) return n * t * t;
    if (t < 2 / d) return n * (t -= 1.5 / d) * t + 0.75;
    if (t < 2.5 / d) return n * (t -= 2.25 / d) * t + 0.9375;
    return n * (t -= 2.625 / d) * t + 0.984375;
  },
};

let _idCounter = 0;
const _active = new Map();

export function tween({ from = 0, to = 1, duration = 400, easing = 'easeOut', onUpdate, onDone }) {
  const id = ++_idCounter;
  const start = performance.now();

  function tick(now) {
    const raw = Math.min(1, (now - start) / duration);
    const t = (EASINGS[easing] || EASINGS.easeOut)(raw);
    const v = from + (to - from) * t;
    onUpdate(v, t);
    if (raw < 1) {
      _active.set(id, requestAnimationFrame(tick));
    } else {
      _active.delete(id);
      if (onDone) onDone();
    }
  }

  _active.set(id, requestAnimationFrame(tick));
  return id;
}

export function cancel(id) {
  const raf = _active.get(id);
  if (raf != null) {
    cancelAnimationFrame(raf);
    _active.delete(id);
  }
}

export function cancelAll() {
  for (const raf of _active.values()) cancelAnimationFrame(raf);
  _active.clear();
}

export function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

export async function staggered(items, intervalMs, fn) {
  for (let i = 0; i < items.length; i++) {
    fn(items[i], i);
    if (i < items.length - 1) await delay(intervalMs);
  }
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

export { EASINGS };
