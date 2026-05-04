// Section 4: The Lasserre Ladder
// Three rungs showing order 1 -> 2 -> 3 tightening the bound.

import { LASSERRE_RESULTS, VAL_D } from './lasserre-data.js';
import { tween, delay, lerp, clamp } from './animate.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const W = 900, H = 450;

const COL = {
  bg: '#1a1d27', border: '#2a2d3a', text: '#e0e0e0', textLight: '#999',
  blue: '#4ea8de', blueLight: '#64b5f6', green: '#81c784',
  gold: '#ffb74d', red: '#ef5350', purple: '#b39ddb',
};

let svg, currentD = 4, animatedOrder = 0;
let playing = false;

function el(tag, attrs = {}, text) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text) e.textContent = text;
  return e;
}

function getResults(d) {
  const valD = VAL_D[d] || 1.1;
  return [1, 2, 3].map(order => {
    const found = LASSERRE_RESULTS.find(r => r.d === d && r.order === order);
    if (found) return found;
    if (order === 1) return { d, order: 1, lb: 1.000, gc: 0, time: 0.1 };
    return null;
  }).filter(Boolean);
}

function buildSVG(container) {
  container.innerHTML = '';
  svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', style: 'max-height:450px;' });
  container.appendChild(svg);

  const results = getResults(currentD);
  const valD = VAL_D[currentD] || 1.1;

  // Title
  svg.appendChild(el('text', {
    x: W / 2, y: 30, 'text-anchor': 'middle',
    'font-size': '14px', 'font-weight': '700', fill: COL.text,
  }, `Lasserre Hierarchy for d = ${currentD}    |    val(${currentD}) = ${valD.toFixed(4)}`));

  const rungY = [90, 200, 310];
  const rungH = 85;

  for (let idx = 0; idx < results.length; idx++) {
    const r = results[idx];
    const y = rungY[idx];
    const visible = r.order <= animatedOrder;
    const opacity = visible ? 1 : 0.15;

    const g = el('g', { opacity });
    g.setAttribute('data-order', r.order);
    svg.appendChild(g);

    // Rung background
    g.appendChild(el('rect', {
      x: 40, y, width: W - 80, height: rungH,
      rx: 8, fill: visible ? 'rgba(78,168,222,0.06)' : 'rgba(255,255,255,0.02)',
      stroke: visible ? COL.blue : COL.border, 'stroke-width': visible ? 1.5 : 1,
    }));

    // Order badge
    const badgeColor = [COL.blue, COL.green, COL.gold][idx];
    g.appendChild(el('circle', {
      cx: 80, cy: y + rungH / 2, r: 18, fill: badgeColor,
    }));
    g.appendChild(el('text', {
      x: 80, y: y + rungH / 2 + 5, 'text-anchor': 'middle',
      'font-size': '14px', 'font-weight': '800', fill: '#fff',
    }, `${r.order}`));

    // Description
    const descs = [
      'Check: are the averages consistent?',
      'Check: are the pairwise products consistent?',
      'Check: are the triple products consistent?',
    ];
    g.appendChild(el('text', {
      x: 115, y: y + 24, 'font-size': '13px', 'font-weight': '700', fill: COL.text,
    }, `Order ${r.order}`));
    g.appendChild(el('text', {
      x: 115, y: y + 42, 'font-size': '12px', fill: COL.textLight,
    }, descs[idx]));

    // Matrix size indicator
    const matSizes = ['d × d', 'C(d+2,2) × C(d+2,2)', 'C(d+3,3) × C(d+3,3)'];
    const thumbSizes = [16, 28, 40];
    const thumbX = 420, thumbY = y + rungH / 2;
    const sz = thumbSizes[idx];

    // Matrix thumbnail (grid of cells)
    const cells = Math.min(idx + 2, 5);
    const cellSz = sz / cells;
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const bright = 0.2 + 0.6 * Math.exp(-((i - j) ** 2) / 2);
        g.appendChild(el('rect', {
          x: thumbX - sz / 2 + j * cellSz,
          y: thumbY - sz / 2 + i * cellSz,
          width: cellSz - 1, height: cellSz - 1,
          rx: 1,
          fill: `rgba(78,168,222,${bright.toFixed(2)})`,
        }));
      }
    }

    g.appendChild(el('text', {
      x: thumbX, y: thumbY + sz / 2 + 14, 'text-anchor': 'middle',
      'font-size': '10px', fill: COL.textLight,
    }, matSizes[idx]));

    // Feasible region circle (shrinks with order)
    const regionR = [50, 35, 22][idx];
    const regionX = 550;
    g.appendChild(el('circle', {
      cx: regionX, cy: y + rungH / 2, r: regionR,
      fill: 'rgba(129,199,132,0.12)', stroke: COL.green, 'stroke-width': 1.5,
    }));
    g.appendChild(el('text', {
      x: regionX, y: y + rungH / 2 + 4, 'text-anchor': 'middle',
      'font-size': '10px', fill: COL.green,
    }, regionR > 30 ? 'Feasible' : ''));

    // Bound value
    g.appendChild(el('text', {
      x: 680, y: y + 28, 'font-size': '12px', fill: COL.textLight,
    }, 'Lower bound'));
    g.appendChild(el('text', {
      x: 680, y: y + 50, 'font-size': '20px', 'font-weight': '800', fill: badgeColor,
    }, r.lb.toFixed(3)));

    // Gap closure bar
    const gcBarX = 680, gcBarY = y + 58, gcBarW = 150, gcBarH = 12;
    g.appendChild(el('rect', {
      x: gcBarX, y: gcBarY, width: gcBarW, height: gcBarH,
      rx: 3, fill: COL.border,
    }));
    g.appendChild(el('rect', {
      x: gcBarX, y: gcBarY, width: gcBarW * (r.gc / 100), height: gcBarH,
      rx: 3, fill: badgeColor, opacity: 0.8,
    }));
    g.appendChild(el('text', {
      x: gcBarX + gcBarW + 8, y: gcBarY + 10,
      'font-size': '11px', 'font-weight': '700', fill: badgeColor,
    }, `${r.gc.toFixed(1)}%`));

    // Time
    if (r.time) {
      g.appendChild(el('text', {
        x: 115, y: y + 60, 'font-size': '10px', fill: COL.textLight,
      }, `Solved in ${r.time < 1 ? r.time.toFixed(1) + 's' : r.time < 60 ? r.time.toFixed(0) + 's' : (r.time / 60).toFixed(1) + 'min'}`));
    }
  }

  // Connection arrows
  for (let i = 0; i < results.length - 1; i++) {
    const y1 = rungY[i] + rungH;
    const y2 = rungY[i + 1];
    const midY = (y1 + y2) / 2;
    svg.appendChild(el('line', {
      x1: 80, y1, x2: 80, y2,
      stroke: COL.border, 'stroke-width': 2, 'stroke-dasharray': '4 3',
    }));
    svg.appendChild(el('text', {
      x: 100, y: midY + 4, 'font-size': '14px', fill: COL.textLight,
    }, '+'));
  }
}

async function playAnimation() {
  if (playing) return;
  playing = true;
  animatedOrder = 0;
  buildSVG(document.getElementById('hier-container'));

  for (let order = 1; order <= 3; order++) {
    const results = getResults(currentD);
    if (!results.find(r => r.order === order)) break;

    animatedOrder = order;
    const g = svg.querySelector(`[data-order="${order}"]`);
    if (g) {
      await new Promise(resolve => {
        tween({
          from: 0.15, to: 1, duration: 600, easing: 'easeOut',
          onUpdate: (v) => g.setAttribute('opacity', v),
          onDone: resolve,
        });
      });
    }
    await delay(800);
  }
  playing = false;
}

function resetAnimation() {
  playing = false;
  animatedOrder = 3;
  buildSVG(document.getElementById('hier-container'));
}

export function initHierarchy() {
  animatedOrder = 3;
  buildSVG(document.getElementById('hier-container'));

  document.getElementById('hier-play').addEventListener('click', playAnimation);
  document.getElementById('hier-reset').addEventListener('click', resetAnimation);

  document.getElementById('hier-d').addEventListener('change', (e) => {
    currentD = parseInt(e.target.value);
    animatedOrder = 3;
    buildSVG(document.getElementById('hier-container'));
  });
}
