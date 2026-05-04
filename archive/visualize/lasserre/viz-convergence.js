// Section 7: Closing the Gap
// Animated number line showing proven region extending

import { LASSERRE_RESULTS, VAL_D, BOUND_HISTORY } from './lasserre-data.js';
import { tween, delay, lerp, clamp } from './animate.js';

const W = 900, H = 380;
const PAD = { left: 60, right: 40, top: 70, bottom: 60 };

const COL = {
  bg: '#1a1d27', grid: '#2a2d3a', text: '#e0e0e0', textLight: '#999',
  blue: '#4ea8de', blueLight: '#64b5f6', green: '#81c784',
  red: '#ef5350', gold: '#ffb74d', purple: '#b39ddb',
};

let canvas, ctx;
let visiblePoints = 0;
let showProjected = false;
let playing = false;

const POINTS = LASSERRE_RESULTS.map(r => ({
  label: `d=${r.d} O${r.order}`,
  lb: r.lb,
  gc: r.gc,
  time: r.time,
  d: r.d,
  order: r.order,
}));

const PROJECTED = [
  { label: 'd=16 O2', lb: 1.133, gc: 41.7, time: 2714, d: 16, order: 2, projected: true },
  { label: 'd=16 O3', lb: 1.290, gc: 91.0, time: null, d: 16, order: 3, projected: true },
];

function xForVal(v) {
  const lo = 0.95, hi = 1.55;
  return PAD.left + ((v - lo) / (hi - lo)) * (W - PAD.left - PAD.right);
}

function drawScene() {
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, W, H);

  const lineY = 160;

  // Title
  ctx.font = 'bold 13px system-ui';
  ctx.fillStyle = COL.text;
  ctx.textAlign = 'center';
  ctx.fillText('Closing the Gap: Lower Bounds on C_1a', W / 2, 28);

  // Number line
  ctx.strokeStyle = COL.grid;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(PAD.left, lineY);
  ctx.lineTo(W - PAD.right, lineY);
  ctx.stroke();

  // Ticks
  ctx.font = '11px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'center';
  for (let v = 1.0; v <= 1.55; v += 0.1) {
    const x = xForVal(v);
    ctx.strokeStyle = COL.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, lineY - 5);
    ctx.lineTo(x, lineY + 5);
    ctx.stroke();
    ctx.fillText(v.toFixed(1), x, lineY + 22);
  }

  // Reference lines
  // World record lower bound
  const wrX = xForVal(1.2802);
  ctx.strokeStyle = COL.gold;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 3]);
  ctx.beginPath();
  ctx.moveTo(wrX, lineY - 70);
  ctx.lineTo(wrX, lineY + 55);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = '10px system-ui';
  ctx.fillStyle = COL.gold;
  ctx.textAlign = 'center';
  ctx.fillText('World record: 1.2802', wrX, lineY + 70);

  // Upper bound
  const ubX = xForVal(1.5029);
  ctx.strokeStyle = COL.red;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 3]);
  ctx.beginPath();
  ctx.moveTo(ubX, lineY - 70);
  ctx.lineTo(ubX, lineY + 55);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = COL.red;
  ctx.fillText('Upper bound: 1.5029', ubX, lineY + 70);

  // Unknown region
  ctx.fillStyle = 'rgba(255,255,255,0.03)';
  ctx.fillRect(wrX, lineY - 50, ubX - wrX, 100);
  ctx.font = '11px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'center';
  ctx.fillText('Unknown region', (wrX + ubX) / 2, lineY - 58);

  // Proven region (fills from left to best proven bound)
  let bestLB = 1.0;
  const allPoints = showProjected ? [...POINTS, ...PROJECTED] : POINTS;
  const visiblePts = allPoints.slice(0, visiblePoints);
  for (const p of visiblePts) {
    if (p.lb > bestLB) bestLB = p.lb;
  }

  if (bestLB > 1.0) {
    const provenX = xForVal(bestLB);
    const grad = ctx.createLinearGradient(PAD.left, 0, provenX, 0);
    grad.addColorStop(0, 'rgba(78,168,222,0.15)');
    grad.addColorStop(1, 'rgba(129,199,132,0.25)');
    ctx.fillStyle = grad;
    ctx.fillRect(PAD.left, lineY - 50, provenX - PAD.left, 100);

    ctx.font = 'bold 12px system-ui';
    ctx.fillStyle = COL.green;
    ctx.textAlign = 'right';
    ctx.fillText(`Proven: >= ${bestLB.toFixed(3)}`, provenX - 4, lineY - 56);
  }

  // Data points
  const colors = {
    1: COL.blue,
    2: COL.green,
    3: COL.gold,
  };

  for (let i = 0; i < visiblePts.length; i++) {
    const p = visiblePts[i];
    const x = xForVal(p.lb);
    const yOff = -35 + (i % 3) * -18;
    const color = colors[p.order] || COL.purple;

    if (p.projected) {
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(x, lineY, 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.5;
    } else {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, lineY, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Label
    ctx.strokeStyle = `${color}44`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, lineY - 8);
    ctx.lineTo(x, lineY + yOff + 4);
    ctx.stroke();

    ctx.font = p.projected ? 'italic 10px system-ui' : 'bold 10px system-ui';
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.fillText(p.label, x, lineY + yOff);

    // Tooltip-style time
    if (p.time && !p.projected) {
      ctx.font = '9px system-ui';
      ctx.fillStyle = COL.textLight;
      const timeStr = p.time < 60 ? `${p.time.toFixed(1)}s` : `${(p.time / 60).toFixed(1)}min`;
      ctx.fillText(timeStr, x, lineY + 36 + (i % 2) * 14);
    }
  }

  // Legend
  const legY = H - 28;
  ctx.font = '11px system-ui';
  const legendItems = [
    { color: COL.blue, label: 'Order 1' },
    { color: COL.green, label: 'Order 2' },
    { color: COL.gold, label: 'Order 3' },
  ];
  let legX = PAD.left;
  for (const item of legendItems) {
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.arc(legX + 6, legY, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = COL.textLight;
    ctx.textAlign = 'left';
    ctx.fillText(item.label, legX + 16, legY + 4);
    legX += 90;
  }

  if (showProjected) {
    ctx.fillStyle = COL.purple;
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = COL.purple;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(legX + 6, legY, 5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COL.textLight;
    ctx.fillText('Projected', legX + 16, legY + 4);
  }

  // Step counter
  ctx.font = 'bold 12px system-ui';
  ctx.fillStyle = COL.blue;
  ctx.textAlign = 'right';
  const total = allPoints.length;
  ctx.fillText(
    visiblePoints >= total ? `All ${total} results shown` : `${visiblePoints} / ${total}`,
    W - PAD.right, 28
  );
}

async function playAnimation() {
  if (playing) return;
  playing = true;
  visiblePoints = 0;
  drawScene();

  const allPoints = showProjected ? [...POINTS, ...PROJECTED] : POINTS;
  for (let i = 0; i < allPoints.length && playing; i++) {
    visiblePoints = i + 1;
    drawScene();
    await delay(500);
  }
  playing = false;
}

function resetAnimation() {
  playing = false;
  const allPoints = showProjected ? [...POINTS, ...PROJECTED] : POINTS;
  visiblePoints = allPoints.length;
  drawScene();
}

export function initConvergence() {
  canvas = document.getElementById('conv-canvas');
  ctx = canvas.getContext('2d');

  visiblePoints = POINTS.length;
  drawScene();

  document.getElementById('conv-play').addEventListener('click', playAnimation);
  document.getElementById('conv-reset').addEventListener('click', resetAnimation);

  document.getElementById('conv-projected').addEventListener('click', (e) => {
    showProjected = !showProjected;
    e.target.textContent = showProjected ? 'Hide projected' : 'Show projected';
    e.target.classList.toggle('active', showProjected);
    const allPoints = showProjected ? [...POINTS, ...PROJECTED] : POINTS;
    visiblePoints = allPoints.length;
    drawScene();
  });
}
