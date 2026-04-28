// Section 5: Bound Progress Timeline
// Animated SVG number line showing historical C₁ₐ bounds.

import { BOUND_HISTORY } from './cascade-data.js';
import { delay } from './animate.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const W = 900, H = 360;
const PAD = { left: 60, right: 60, top: 100, bottom: 100 };
const LINE_Y = H / 2;
const LO = 1.10, HI = 1.62;

const COLORS = {
  lower: '#4ea8de',
  upper: '#ef5350',
  highlight: '#81c784',
  proven: '#1a3a5c',
  unknown: '#2a2d3a',
  axis: '#999',
  tick: '#666',
};

let svg, playing = false;

function xPos(val) {
  return PAD.left + ((val - LO) / (HI - LO)) * (W - PAD.left - PAD.right);
}

function el(tag, attrs, text) {
  const e = document.createElementNS(SVG_NS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

function buildSVG(container) {
  svg = container.querySelector('#timeline-svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = '';

  const regionGroup = el('g', { id: 'tl-regions' });

  // Proven region
  regionGroup.appendChild(el('rect', {
    id: 'region-proven',
    x: xPos(LO), y: LINE_Y - 20,
    width: 0, height: 40,
    fill: COLORS.proven, rx: 4,
    class: 'tl-region',
  }));

  // Unknown gap (ends at best upper bound 1.5029)
  regionGroup.appendChild(el('rect', {
    id: 'region-unknown',
    x: xPos(1.40), y: LINE_Y - 20,
    width: xPos(1.5029) - xPos(1.40), height: 40,
    fill: COLORS.unknown, rx: 4,
    class: 'tl-region',
  }));

  svg.appendChild(regionGroup);

  // Main axis line
  svg.appendChild(el('line', {
    x1: PAD.left - 10, y1: LINE_Y,
    x2: W - PAD.right + 10, y2: LINE_Y,
    stroke: COLORS.axis, 'stroke-width': 1.5,
  }));

  // Arrow at right end
  svg.appendChild(el('polygon', {
    points: `${W - PAD.right + 10},${LINE_Y} ${W - PAD.right + 3},${LINE_Y - 4} ${W - PAD.right + 3},${LINE_Y + 4}`,
    fill: COLORS.axis,
  }));

  // Tick marks at 0.1 intervals
  for (let v = 1.1; v <= 1.6; v += 0.1) {
    const x = xPos(v);
    svg.appendChild(el('line', {
      x1: x, y1: LINE_Y - 5, x2: x, y2: LINE_Y + 5,
      stroke: COLORS.tick, 'stroke-width': 1,
    }));
    svg.appendChild(el('text', {
      x, y: LINE_Y + 20,
      'text-anchor': 'middle', 'font-size': '10px', fill: COLORS.tick,
    }, v.toFixed(1)));
  }

  // Axis label
  svg.appendChild(el('text', {
    x: W / 2, y: H - 8,
    'text-anchor': 'middle', 'font-size': '12px', fill: '#777',
  }, 'C₁ₐ'));

  svg.appendChild(el('g', { id: 'tl-bounds' }));
}

function createBoundMarker(bound, idx) {
  const g = el('g', {
    id: `bound-${idx}`,
    opacity: 0,
    transform: `translate(${xPos(bound.value)}, ${LINE_Y})`,
  });

  const isUpper = bound.type === 'upper';
  const isHighlight = bound.highlight;
  const color = isHighlight ? COLORS.highlight : (isUpper ? COLORS.upper : COLORS.lower);
  const side = isUpper ? -1 : 1;

  // Stagger labels to reduce crowding
  const stagger = (idx % 2 === 0) ? 1.0 : 1.4;
  const tickLen = side * 30 * stagger;
  const valY = side * 44 * stagger;
  const nameY = side * 58 * stagger;
  const detailY = side * 72 * stagger;

  // Dot
  g.appendChild(el('circle', {
    cx: 0, cy: 0, r: isHighlight ? 7 : 5,
    fill: color,
    stroke: isHighlight ? '#fff' : 'none',
    'stroke-width': isHighlight ? 2 : 0,
  }));

  // Tick line
  g.appendChild(el('line', {
    x1: 0, y1: 0, x2: 0, y2: tickLen,
    stroke: color, 'stroke-width': 1.2,
  }));

  // Value label
  g.appendChild(el('text', {
    x: 0, y: valY,
    'text-anchor': 'middle', 'font-size': '11px', 'font-weight': '700',
    fill: color,
  }, bound.value.toFixed(bound.value === 1.5708 ? 4 : (bound.value >= 1.3 ? 2 : 4))));

  // Name + year
  g.appendChild(el('text', {
    x: 0, y: nameY,
    'text-anchor': 'middle', 'font-size': '9px',
    fill: '#999',
  }, `${bound.label} (${bound.year})`));

  // Detail (e.g. π/2)
  if (bound.detail) {
    g.appendChild(el('text', {
      x: 0, y: detailY,
      'text-anchor': 'middle', 'font-size': '9px', 'font-style': 'italic',
      fill: '#777',
    }, bound.detail));
  }

  return g;
}

async function animateBounds() {
  playing = true;
  const btn = document.getElementById('tl-play');
  btn.textContent = '⏸ Pause';

  const boundsGroup = svg.querySelector('#tl-bounds');
  boundsGroup.innerHTML = '';

  const provenRect = svg.querySelector('#region-proven');
  const unknownRect = svg.querySelector('#region-unknown');
  provenRect.setAttribute('class', 'tl-region');
  provenRect.setAttribute('width', '0');
  unknownRect.setAttribute('class', 'tl-region');

  const sorted = [...BOUND_HISTORY].sort((a, b) => {
    if (a.year !== b.year) return a.year - b.year;
    return a.type === 'upper' ? -1 : 1;
  });

  let bestLower = LO;

  for (let i = 0; i < sorted.length; i++) {
    if (!playing) break;
    const bound = sorted[i];
    const marker = createBoundMarker(bound, i);
    boundsGroup.appendChild(marker);

    await delay(200);
    if (!playing) break;
    marker.setAttribute('opacity', '1');
    marker.style.transition = 'opacity 0.5s';

    if (bound.type === 'lower') {
      bestLower = Math.max(bestLower, bound.value);
      const provenW = xPos(bestLower) - xPos(LO);
      provenRect.setAttribute('width', String(provenW));
      provenRect.setAttribute('class', 'tl-region tl-region-visible');

      unknownRect.setAttribute('x', String(xPos(bestLower)));
      unknownRect.setAttribute('width', String(xPos(1.5029) - xPos(bestLower)));
      unknownRect.setAttribute('class', 'tl-region tl-region-visible');
    }

    if (bound.highlight) {
      const dot = marker.querySelector('circle');
      dot.setAttribute('r', '10');
      await delay(200);
      dot.setAttribute('r', '7');
      dot.style.transition = 'r 0.3s';
    }

    await delay(600);
  }

  if (playing) {
    await delay(300);
    const provenLabel = el('text', {
      x: xPos((LO + bestLower) / 2), y: LINE_Y - 28,
      'text-anchor': 'middle', 'font-size': '10px', fill: COLORS.lower,
      opacity: 0,
    }, 'Proven region');
    boundsGroup.appendChild(provenLabel);
    await delay(100);
    provenLabel.setAttribute('opacity', '1');
    provenLabel.style.transition = 'opacity 0.5s';

    const unknownLabel = el('text', {
      x: xPos((bestLower + 1.5029) / 2), y: LINE_Y + 28,
      'text-anchor': 'middle', 'font-size': '10px', fill: '#666',
      opacity: 0,
    }, 'Unknown');
    boundsGroup.appendChild(unknownLabel);
    await delay(100);
    unknownLabel.setAttribute('opacity', '1');
    unknownLabel.style.transition = 'opacity 0.5s';
  }

  playing = false;
  btn.textContent = '▶ Play';
}

function resetTimeline() {
  playing = false;
  const boundsGroup = svg.querySelector('#tl-bounds');
  if (boundsGroup) boundsGroup.innerHTML = '';
  const provenRect = svg.querySelector('#region-proven');
  if (provenRect) { provenRect.setAttribute('class', 'tl-region'); provenRect.setAttribute('width', '0'); }
  const unknownRect = svg.querySelector('#region-unknown');
  if (unknownRect) unknownRect.setAttribute('class', 'tl-region');
  const btn = document.getElementById('tl-play');
  btn.textContent = '▶ Play';
}

export function initTimeline() {
  const container = document.getElementById('timeline-container');
  buildSVG(container);

  document.getElementById('tl-play').addEventListener('click', () => {
    if (playing) {
      playing = false;
    } else {
      resetTimeline();
      animateBounds();
    }
  });

  document.getElementById('tl-reset').addEventListener('click', resetTimeline);
}
