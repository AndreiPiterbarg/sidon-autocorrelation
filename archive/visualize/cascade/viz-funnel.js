// Section 1: Animated Cascade Funnel
// Particles flow through chambers (L0-L4 or L0-L5), get pruned at each level.

import { CASCADE_130, CASCADE_140, fmtLarge, fmtTime } from './cascade-data.js';
import { delay } from './animate.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const W = 960, H = 420;
const CHAMBER_H = 200;
const TOP_Y = 100;
const MAX_PARTICLES = 160;
const COLORS = {
  pending: '#64b5f6',
  survivor: '#81c784',
  pruned: '#ef5350',
  chamber: '#4ea8de',
  tube: '#2a2d3a',
};

let svg, playing = false, currentStage = -1;
let currentCascade = CASCADE_130;

function chamberDims() {
  const n = currentCascade.levels.length;
  const chamberW = n <= 5 ? 120 : 100;
  const gap = n <= 5 ? 60 : 36;
  return { chamberW, gap };
}

function chamberX(i) {
  const { chamberW, gap } = chamberDims();
  const n = currentCascade.levels.length;
  const totalW = n * chamberW + (n - 1) * gap;
  const startX = (W - totalW) / 2;
  return startX + i * (chamberW + gap);
}

function createSVG(container) {
  svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', '100%');
  svg.style.maxHeight = '420px';
  container.appendChild(svg);

  const { chamberW } = chamberDims();
  const n = currentCascade.levels.length;

  // Title showing bound target and parameters
  svg.appendChild(el('text', {
    x: W / 2, y: 22,
    'text-anchor': 'middle', 'font-size': '13px', 'font-weight': '600',
    fill: '#e0e0e0',
  }, `Proving C₁ₐ ≥ ${currentCascade.c_target.toFixed(2)}  |  m = ${currentCascade.m}  |  n_half = ${currentCascade.n_half}`));

  // Tubes between chambers
  for (let i = 0; i < n - 1; i++) {
    const x1 = chamberX(i) + chamberW;
    const x2 = chamberX(i + 1);
    const cy = TOP_Y + CHAMBER_H / 2;
    const tubeH = 30 - i * 4;
    const tube = el('path', {
      d: `M${x1},${cy - tubeH} L${x2},${cy - tubeH * 0.6} L${x2},${cy + tubeH * 0.6} L${x1},${cy + tubeH} Z`,
      fill: COLORS.tube, opacity: 0.6,
    });
    svg.appendChild(tube);
  }

  const levels = currentCascade.levels;

  // Chambers and labels
  for (let i = 0; i < levels.length; i++) {
    const lv = levels[i];
    const x = chamberX(i);

    // Chamber rect
    svg.appendChild(el('rect', {
      x, y: TOP_Y, width: chamberW, height: CHAMBER_H,
      rx: 8, fill: COLORS.chamber, opacity: 0.08,
      class: 'chamber', id: `chamber-${i}`,
    }));

    // Level label
    svg.appendChild(el('text', {
      x: x + chamberW / 2, y: TOP_Y - 30,
      class: 'funnel-label',
    }, lv.label));

    // d value
    svg.appendChild(el('text', {
      x: x + chamberW / 2, y: TOP_Y - 14,
      class: 'funnel-sublabel',
    }, `d=${lv.d}`));

    // Children count
    svg.appendChild(el('text', {
      x: x + chamberW / 2, y: TOP_Y + CHAMBER_H + 20,
      class: 'funnel-count', id: `count-children-${i}`,
    }, ''));

    // Survivors count
    svg.appendChild(el('text', {
      x: x + chamberW / 2, y: TOP_Y + CHAMBER_H + 36,
      class: 'funnel-count', id: `count-survivors-${i}`,
      fill: COLORS.survivor, 'font-weight': '700',
    }, ''));

    // Time
    svg.appendChild(el('text', {
      x: x + chamberW / 2, y: TOP_Y + CHAMBER_H + 50,
      class: 'funnel-sublabel', id: `time-${i}`,
    }, ''));

    // Particle container group
    svg.appendChild(el('g', { id: `particles-${i}` }));

    // Annotation
    svg.appendChild(el('text', {
      x: x + chamberW / 2, y: TOP_Y + CHAMBER_H / 2,
      class: 'funnel-annotation', id: `annotation-${i}`,
    }, ''));
  }

  // Final proven badge (offset below annotation text to avoid overlap)
  const lastIdx = levels.length - 1;
  svg.appendChild(el('text', {
    x: chamberX(lastIdx) + chamberDims().chamberW / 2, y: TOP_Y + CHAMBER_H / 2 + 32,
    class: 'funnel-annotation', id: 'proven-badge',
    'font-size': '16px', fill: COLORS.survivor,
  }, ''));
}

function el(tag, attrs, text) {
  const e = document.createElementNS(SVG_NS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

function setCount(i, field, text) {
  const node = svg.querySelector(`#count-${field}-${i}`);
  if (node) node.textContent = text;
}

function setTime(i, text) {
  const node = svg.querySelector(`#time-${i}`);
  if (node) node.textContent = text;
}

function createParticles(chamberIdx, count, color) {
  const { chamberW } = chamberDims();
  const g = svg.querySelector(`#particles-${chamberIdx}`);
  g.innerHTML = '';
  const x0 = chamberX(chamberIdx);
  const particles = [];
  for (let i = 0; i < count; i++) {
    const cx = x0 + 10 + Math.random() * (chamberW - 20);
    const cy = TOP_Y + 10 + Math.random() * (CHAMBER_H - 20);
    const c = el('circle', {
      cx, cy, r: 3, fill: color, opacity: 0, class: 'particle',
    });
    g.appendChild(c);
    particles.push(c);
  }
  return particles;
}

async function showParticles(particles, staggerMs = 8) {
  for (const p of particles) {
    p.setAttribute('opacity', '0.8');
    if (staggerMs > 0) await delay(staggerMs);
  }
}

async function pruneParticles(particles, survivorCount) {
  const total = particles.length;
  const surviveCount = Math.min(survivorCount, total);
  const indices = Array.from({ length: total }, (_, i) => i);
  for (let i = total - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const survivors = new Set(indices.slice(0, surviveCount));

  for (const i of survivors) {
    particles[i].setAttribute('fill', COLORS.survivor);
  }
  await delay(300);

  for (let i = 0; i < total; i++) {
    if (!survivors.has(i)) {
      particles[i].setAttribute('fill', COLORS.pruned);
      particles[i].setAttribute('opacity', '0.6');
    }
  }
  await delay(400);
  for (let i = 0; i < total; i++) {
    if (!survivors.has(i)) {
      particles[i].setAttribute('opacity', '0');
    }
  }
  await delay(300);

  return survivors;
}

function setAnnotation(i, text, visible = true) {
  const node = svg.querySelector(`#annotation-${i}`);
  if (!node) return;
  node.textContent = text;
  node.setAttribute('class', 'funnel-annotation' + (visible ? ' funnel-annotation-visible' : ''));
}

function clearAnnotations() {
  for (let i = 0; i < currentCascade.levels.length; i++) setAnnotation(i, '', false);
  const badge = svg.querySelector('#proven-badge');
  if (badge) badge.setAttribute('class', 'funnel-annotation');
}

function particleCountForLevel(lv) {
  return Math.min(MAX_PARTICLES, Math.max(10, Math.round(Math.sqrt(lv.children) / 200)));
}

function survivorParticleCount(lv) {
  if (lv.survivors === 0) return 0;
  const total = particleCountForLevel(lv);
  const ratio = lv.survivors / lv.children;
  return Math.max(1, Math.round(total * Math.max(ratio, 0.05)));
}

async function playStage(stage) {
  if (!playing) return;
  const levels = currentCascade.levels;
  const lv = levels[stage];
  const stageLabel = document.getElementById('funnel-stage-label');
  const scrubber = document.getElementById('funnel-scrubber');

  currentStage = stage;
  scrubber.value = Math.round((stage / (levels.length - 1)) * 100);

  if (stage > 0) setAnnotation(stage - 1, '', false);

  stageLabel.textContent = `${lv.label}: d=${lv.d}`;

  setCount(stage, 'children', `${fmtLarge(lv.children)} tested`);

  const pCount = particleCountForLevel(lv);
  const particles = createParticles(stage, pCount, COLORS.pending);
  await showParticles(particles, stage === 0 ? 12 : 5);

  if (!playing) return;

  const survCount = survivorParticleCount(lv);
  const pruneRate = Math.floor((1 - lv.survivors / lv.children) * 1e6) / 1e4;
  if (lv.survivors === 0) {
    setAnnotation(stage, 'ALL PRUNED', true);
  } else if (pruneRate > 99) {
    setAnnotation(stage, `${pruneRate.toFixed(4)}% pruned`, true);
  } else {
    setAnnotation(stage, `${pruneRate.toFixed(1)}% pruned`, true);
  }

  await delay(600);
  if (!playing) return;

  await pruneParticles(particles, survCount);

  if (lv.survivors > 0) {
    setCount(stage, 'survivors', `${fmtLarge(lv.survivors)} survive`);
  } else {
    setCount(stage, 'survivors', '0 survive');
    const badge = svg.querySelector('#proven-badge');
    if (badge) {
      badge.textContent = 'PROVEN';
      badge.setAttribute('class', 'funnel-annotation funnel-annotation-visible');
    }
  }
  setTime(stage, fmtTime(lv.elapsed));
  await delay(800);
}

async function playAll() {
  playing = true;
  const btn = document.getElementById('funnel-play');
  btn.textContent = '⏸ Pause';

  clearAnnotations();
  const n = currentCascade.levels.length;
  for (let i = 0; i < n; i++) {
    const g = svg.querySelector(`#particles-${i}`);
    if (g) g.innerHTML = '';
    setCount(i, 'children', '');
    setCount(i, 'survivors', '');
    setTime(i, '');
  }
  const badge = svg.querySelector('#proven-badge');
  if (badge) { badge.textContent = ''; badge.setAttribute('class', 'funnel-annotation'); }

  for (let stage = 0; stage < currentCascade.levels.length; stage++) {
    if (!playing) break;
    await playStage(stage);
  }

  playing = false;
  btn.textContent = '▶ Play';
}

function reset() {
  playing = false;
  currentStage = -1;
  const btn = document.getElementById('funnel-play');
  btn.textContent = '▶ Play';
  document.getElementById('funnel-scrubber').value = 0;
  document.getElementById('funnel-stage-label').textContent = 'Ready';

  const n = currentCascade.levels.length;
  for (let i = 0; i < n; i++) {
    const g = svg.querySelector(`#particles-${i}`);
    if (g) g.innerHTML = '';
    setCount(i, 'children', '');
    setCount(i, 'survivors', '');
    setTime(i, '');
    setAnnotation(i, '', false);
  }
  const badge = svg.querySelector('#proven-badge');
  if (badge) { badge.textContent = ''; badge.setAttribute('class', 'funnel-annotation'); }
}

export function initFunnel() {
  const container = document.getElementById('funnel-container');
  createSVG(container);

  document.getElementById('funnel-play').addEventListener('click', () => {
    if (playing) {
      playing = false;
    } else {
      playAll();
    }
  });

  document.getElementById('funnel-reset').addEventListener('click', reset);

  document.getElementById('funnel-toggle').addEventListener('click', async () => {
    if (playing) { playing = false; await delay(100); }
    currentCascade = currentCascade === CASCADE_130 ? CASCADE_140 : CASCADE_130;
    document.getElementById('funnel-toggle').textContent =
      currentCascade === CASCADE_130 ? 'Show c ≥ 1.40' : 'Show c ≥ 1.30';
    const cont = document.getElementById('funnel-container');
    cont.innerHTML = '';
    createSVG(cont);
    reset();
  });

  // Scrubber
  document.getElementById('funnel-scrubber').addEventListener('input', async (e) => {
    if (playing) { playing = false; await delay(100); }
    const pct = parseInt(e.target.value);
    const stage = Math.round((pct / 100) * (currentCascade.levels.length - 1));

    const n = currentCascade.levels.length;
    for (let i = 0; i < n; i++) {
      const g = svg.querySelector(`#particles-${i}`);
      if (g) g.innerHTML = '';
      setCount(i, 'children', '');
      setCount(i, 'survivors', '');
      setTime(i, '');
      setAnnotation(i, '', false);
    }
    const badge = svg.querySelector('#proven-badge');
    if (badge) { badge.textContent = ''; badge.setAttribute('class', 'funnel-annotation'); }

    for (let s = 0; s <= stage; s++) {
      const lv = currentCascade.levels[s];
      const survCount = survivorParticleCount(lv);

      setCount(s, 'children', `${fmtLarge(lv.children)} tested`);

      const particles = createParticles(s, survCount, COLORS.survivor);
      for (const p of particles) p.setAttribute('opacity', '0.8');

      if (lv.survivors > 0) {
        setCount(s, 'survivors', `${fmtLarge(lv.survivors)} survive`);
      } else {
        setCount(s, 'survivors', '0 survive');
        if (badge) {
          badge.textContent = 'PROVEN';
          badge.setAttribute('class', 'funnel-annotation funnel-annotation-visible');
        }
      }
      setTime(s, fmtTime(lv.elapsed));
    }

    currentStage = stage;
    document.getElementById('funnel-stage-label').textContent =
      currentCascade.levels[stage].label + `: d=${currentCascade.levels[stage].d}`;
  });
}
