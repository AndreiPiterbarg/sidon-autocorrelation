// Section 2: What Is Relaxation?
// Triangle (simplex) morphing into ellipse (relaxed set)

import { tween, delay, lerp } from './animate.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const W = 900, H = 420;

const COL = {
  simplex: '#4ea8de',
  relaxed: '#81c784',
  relaxedLoose: '#ffb74d',
  point: '#e0e0e0',
  pointRelax: '#81c784',
  min: '#ef5350',
  text: '#e0e0e0',
  textLight: '#999',
  bg: '#1a1d27',
  grid: '#2a2d3a',
};

let svg, morphT = 0, looseMode = false;
let playing = false;

function el(tag, attrs = {}, text) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text) e.textContent = text;
  return e;
}

// Triangle vertices (centered at 225, 210)
const TRI = [
  [225, 60],   // top
  [115, 340],  // bottom-left
  [335, 340],  // bottom-right
];

// Ellipse params
const TIGHT_ELLIPSE = { cx: 225, cy: 220, rx: 140, ry: 160 };
const LOOSE_ELLIPSE = { cx: 225, cy: 220, rx: 200, ry: 210 };

// Sample points inside the triangle
const TRI_POINTS = [
  [225, 180], [180, 280], [270, 280], [225, 250], [200, 220],
  [250, 220], [215, 300], [235, 300], [225, 140], [190, 250],
];

// Extra points in the relaxed gap (outside triangle, inside ellipse)
const GAP_POINTS = [
  [130, 200], [320, 200], [100, 280], [350, 280],
  [225, 50], [160, 130], [290, 130], [370, 320],
  [80, 320], [225, 370],
];

function triPath() {
  return `M ${TRI[0][0]} ${TRI[0][1]} L ${TRI[1][0]} ${TRI[1][1]} L ${TRI[2][0]} ${TRI[2][1]} Z`;
}

function ellipsePath(ell) {
  return `M ${ell.cx - ell.rx} ${ell.cy} A ${ell.rx} ${ell.ry} 0 1 1 ${ell.cx + ell.rx} ${ell.cy} A ${ell.rx} ${ell.ry} 0 1 1 ${ell.cx - ell.rx} ${ell.cy}`;
}

function interpolatePath(t) {
  const ell = looseMode ? LOOSE_ELLIPSE : TIGHT_ELLIPSE;
  if (t <= 0) return triPath();
  if (t >= 1) return ellipsePath(ell);

  const cx = 225, cy = 220;
  const pts = [];
  const steps = 60;
  for (let i = 0; i < steps; i++) {
    const a = (i / steps) * Math.PI * 2;
    const ex = cx + ell.rx * Math.cos(a);
    const ey = cy + ell.ry * Math.sin(a);

    // Find closest point on triangle boundary
    let minD = Infinity, tx = ex, ty = ey;
    for (let s = 0; s < 3; s++) {
      const [x1, y1] = TRI[s];
      const [x2, y2] = TRI[(s + 1) % 3];
      for (let u = 0; u <= 1; u += 0.05) {
        const px = x1 + u * (x2 - x1);
        const py = y1 + u * (y2 - y1);
        const d = (px - ex) ** 2 + (py - ey) ** 2;
        if (d < minD) { minD = d; tx = px; ty = py; }
      }
    }

    pts.push([lerp(tx, ex, t), lerp(ty, ey, t)]);
  }

  let path = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
  for (let i = 1; i < pts.length; i++) {
    path += ` L ${pts[i][0].toFixed(1)} ${pts[i][1].toFixed(1)}`;
  }
  return path + ' Z';
}

let boundaryEl, pointEls = [], gapPointEls = [];
let labelLeft, labelRight, minBarTrue, minBarRelax;
let barGroup;

function buildSVG(container) {
  svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', style: 'max-height:420px;' });
  container.appendChild(svg);

  // Left side: the shape
  const leftG = el('g', { transform: 'translate(0,0)' });
  svg.appendChild(leftG);

  // Background triangle (always visible, faint)
  leftG.appendChild(el('path', {
    d: triPath(),
    fill: 'none', stroke: COL.simplex, 'stroke-width': 1.5,
    'stroke-dasharray': '4 3', opacity: 0.4,
  }));

  // Morphing boundary
  boundaryEl = el('path', {
    d: triPath(),
    fill: 'rgba(78,168,222,0.08)', stroke: COL.simplex, 'stroke-width': 2.5,
  });
  leftG.appendChild(boundaryEl);

  // Triangle points
  for (const [x, y] of TRI_POINTS) {
    const c = el('circle', { cx: x, cy: y, r: 4, fill: COL.point, opacity: 0.7 });
    pointEls.push(c);
    leftG.appendChild(c);
  }

  // Gap points (hidden initially)
  for (const [x, y] of GAP_POINTS) {
    const c = el('circle', { cx: x, cy: y, r: 4, fill: COL.pointRelax, opacity: 0 });
    gapPointEls.push(c);
    leftG.appendChild(c);
  }

  // Labels
  labelLeft = el('text', {
    x: 225, y: 395, 'text-anchor': 'middle',
    'font-size': '13px', 'font-weight': '700', fill: COL.text,
  }, 'Feasible Set');
  leftG.appendChild(labelLeft);

  // Right side: comparison bars
  barGroup = el('g', { transform: 'translate(500, 60)' });
  svg.appendChild(barGroup);

  barGroup.appendChild(el('text', {
    x: 0, y: 0, 'font-size': '13px', 'font-weight': '700', fill: COL.text,
  }, 'Minimum Value'));

  // True min bar
  barGroup.appendChild(el('text', {
    x: 0, y: 40, 'font-size': '12px', fill: COL.simplex,
  }, 'True minimum'));

  minBarTrue = el('rect', {
    x: 0, y: 50, width: 250, height: 24,
    rx: 4, fill: COL.simplex, opacity: 0.8,
  });
  barGroup.appendChild(minBarTrue);

  barGroup.appendChild(el('text', {
    x: 255, y: 67, 'font-size': '13px', 'font-weight': '700', fill: COL.simplex,
  }, '1.102'));

  // Relaxed min bar
  barGroup.appendChild(el('text', {
    x: 0, y: 110, 'font-size': '12px', fill: COL.relaxed,
  }, 'Relaxed minimum'));

  minBarRelax = el('rect', {
    x: 0, y: 120, width: 200, height: 24,
    rx: 4, fill: COL.relaxed, opacity: 0.8,
  });
  barGroup.appendChild(minBarRelax);

  const relaxLabel = el('text', {
    x: 205, y: 137, 'font-size': '13px', 'font-weight': '700', fill: COL.relaxed,
    class: 'relax-bar-label',
  }, '1.079');
  barGroup.appendChild(relaxLabel);

  // Arrow
  barGroup.appendChild(el('text', {
    x: 130, y: 180, 'font-size': '12px', fill: COL.textLight, 'text-anchor': 'middle',
  }, 'relaxed min <= true min (always)'));

  // Gap label
  const gapLabel = el('text', {
    x: 130, y: 220, 'font-size': '14px', 'font-weight': '700',
    fill: COL.relaxed, 'text-anchor': 'middle', class: 'gap-label',
  }, 'Gap = 0.023');
  barGroup.appendChild(gapLabel);

  // Explanation
  barGroup.appendChild(el('text', {
    x: 130, y: 260, 'font-size': '11px', fill: COL.textLight, 'text-anchor': 'middle',
  }, 'Smaller gap = better bound'));

  barGroup.appendChild(el('text', {
    x: 130, y: 280, 'font-size': '11px', fill: COL.textLight, 'text-anchor': 'middle',
  }, 'Tighter relaxation = closer to truth'));
}

function updateMorph(t) {
  morphT = t;
  boundaryEl.setAttribute('d', interpolatePath(t));
  boundaryEl.setAttribute('stroke', t > 0.5 ? COL.relaxed : COL.simplex);
  boundaryEl.setAttribute('fill', t > 0.5 ? 'rgba(129,199,132,0.06)' : 'rgba(78,168,222,0.08)');

  for (const c of gapPointEls) {
    c.setAttribute('opacity', Math.max(0, (t - 0.3) / 0.7).toFixed(2));
  }

  const relaxW = looseMode ? 140 : 200;
  const relaxVal = looseMode ? '0.800' : '1.079';
  const gap = looseMode ? '0.302' : '0.023';
  minBarRelax.setAttribute('width', lerp(250, relaxW, t));

  const relaxLabelEl = barGroup.querySelector('.relax-bar-label');
  if (relaxLabelEl) {
    relaxLabelEl.textContent = t > 0.3 ? relaxVal : '1.102';
    relaxLabelEl.setAttribute('x', lerp(255, relaxW + 5, t));
  }

  const gapLabelEl = barGroup.querySelector('.gap-label');
  if (gapLabelEl && t > 0.5) {
    gapLabelEl.textContent = `Gap = ${gap}`;
  }
}

async function playAnimation() {
  if (playing) return;
  playing = true;
  await new Promise(resolve => {
    tween({
      from: 0, to: 1, duration: 2000, easing: 'easeInOut',
      onUpdate: (v) => updateMorph(v),
      onDone: resolve,
    });
  });
  playing = false;
}

function resetAnimation() {
  playing = false;
  updateMorph(0);
}

export function initRelax() {
  const container = document.getElementById('relax-container');
  buildSVG(container);

  document.getElementById('relax-play').addEventListener('click', playAnimation);
  document.getElementById('relax-reset').addEventListener('click', resetAnimation);

  document.getElementById('relax-tight').addEventListener('click', () => {
    looseMode = false;
    document.getElementById('relax-tight').classList.add('active');
    document.getElementById('relax-loose').classList.remove('active');
    updateMorph(morphT);
  });

  document.getElementById('relax-loose').addEventListener('click', () => {
    looseMode = true;
    document.getElementById('relax-loose').classList.add('active');
    document.getElementById('relax-tight').classList.remove('active');
    updateMorph(morphT);
  });

  updateMorph(0);
}
