// Section 6: The Farkas Certificate
// Binary search animation on a number line

import { BISECTION_D4_O3, BISECTION_D6_O3 } from './lasserre-data.js';
import { tween, delay, lerp, clamp } from './animate.js';

const W = 900, H = 340;
const PAD = { left: 60, right: 60, top: 60, bottom: 80 };

const COL = {
  bg: '#1a1d27', grid: '#2a2d3a', text: '#e0e0e0', textLight: '#999',
  blue: '#4ea8de', green: '#81c784', red: '#ef5350',
  gold: '#ffb74d', greenPale: 'rgba(129,199,132,0.15)', redPale: 'rgba(239,83,80,0.15)',
};

let canvas, ctx;
let currentData = BISECTION_D4_O3;
let currentLabel = 'd=4, order 3';
let steps = [];
let animStep = -1;
let playing = false;

function xForT(t) {
  const lo = 0.95, hi = 1.20;
  return PAD.left + ((t - lo) / (hi - lo)) * (W - PAD.left - PAD.right);
}

function drawNumberLine() {
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, W, H);

  const lineY = 140;
  const lo = 0.95, hi = 1.20;

  // Title
  ctx.font = 'bold 13px system-ui';
  ctx.fillStyle = COL.text;
  ctx.textAlign = 'center';
  ctx.fillText(`Binary Search for Lower Bound  (${currentLabel})`, W / 2, 28);

  // Main line
  ctx.strokeStyle = COL.grid;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(PAD.left, lineY);
  ctx.lineTo(W - PAD.right, lineY);
  ctx.stroke();

  // Tick marks
  ctx.font = '11px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'center';
  for (let t = 0.95; t <= 1.20; t += 0.05) {
    const x = xForT(t);
    ctx.strokeStyle = COL.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, lineY - 6);
    ctx.lineTo(x, lineY + 6);
    ctx.stroke();
    ctx.fillText(t.toFixed(2), x, lineY + 22);
  }

  // Reference line: val(d)
  const valD = currentData === BISECTION_D4_O3 ? 1.102 : 1.171;
  const valX = xForT(valD);
  ctx.strokeStyle = COL.gold;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(valX, lineY - 40);
  ctx.lineTo(valX, lineY + 40);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = '10px system-ui';
  ctx.fillStyle = COL.gold;
  ctx.fillText(`val(d) = ${valD}`, valX, lineY - 46);

  // Current search bracket
  if (steps.length > 0) {
    let bracketLo = 0.95, bracketHi = 1.20;
    for (let i = 0; i <= Math.min(animStep, steps.length - 1); i++) {
      const s = steps[i];
      if (s.feasible) bracketLo = Math.max(bracketLo, s.t);
      else bracketHi = Math.min(bracketHi, s.t);
    }

    // Proven region (green)
    const provenX = xForT(bracketLo);
    ctx.fillStyle = COL.greenPale;
    ctx.fillRect(PAD.left, lineY - 30, provenX - PAD.left, 60);

    // Excluded region (red)
    const excludedX = xForT(bracketHi);
    ctx.fillStyle = COL.redPale;
    ctx.fillRect(excludedX, lineY - 30, W - PAD.right - excludedX, 60);

    // Bracket labels
    ctx.font = 'bold 12px system-ui';
    ctx.fillStyle = COL.green;
    ctx.textAlign = 'center';
    ctx.fillText(`lb >= ${bracketLo.toFixed(4)}`, (PAD.left + provenX) / 2, lineY - 36);

    if (bracketHi < 1.20) {
      ctx.fillStyle = COL.red;
      ctx.fillText(`infeasible`, (excludedX + W - PAD.right) / 2, lineY - 36);
    }
  }

  // Draw each step marker
  for (let i = 0; i <= Math.min(animStep, steps.length - 1); i++) {
    const s = steps[i];
    const x = xForT(s.t);
    const markerY = lineY;

    // Marker
    ctx.fillStyle = s.feasible ? COL.green : COL.red;
    ctx.beginPath();
    ctx.arc(x, markerY, 8, 0, Math.PI * 2);
    ctx.fill();

    // Checkmark or X
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 12px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(s.feasible ? '✓' : '✗', x, markerY + 4);

    // Step number
    ctx.font = '9px system-ui';
    ctx.fillStyle = COL.textLight;
    ctx.fillText(`${i + 1}`, x, markerY + 24);
  }

  // Legend
  const legY = H - 40;
  ctx.font = '12px system-ui';

  ctx.fillStyle = COL.green;
  ctx.beginPath();
  ctx.arc(PAD.left + 10, legY, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'left';
  ctx.fillText('Feasible (solution exists at t)', PAD.left + 24, legY + 4);

  ctx.fillStyle = COL.red;
  ctx.beginPath();
  ctx.arc(PAD.left + 310, legY, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = COL.textLight;
  ctx.fillText('Infeasible (no solution) = bound is proven!', PAD.left + 324, legY + 4);

  // Step counter
  if (animStep >= 0) {
    ctx.font = 'bold 13px system-ui';
    ctx.fillStyle = COL.blue;
    ctx.textAlign = 'right';
    const stepLabel = animStep < steps.length
      ? `Step ${animStep + 1} / ${steps.length}`
      : `Done — ${steps.length} steps`;
    ctx.fillText(stepLabel, W - PAD.right, 28);
  }
}

async function playAnimation() {
  if (playing) return;
  playing = true;
  steps = [...currentData];
  animStep = -1;
  drawNumberLine();

  const speedSlider = document.getElementById('fk-speed');
  const baseDelay = 200;

  for (let i = 0; i < steps.length && playing; i++) {
    animStep = i;
    drawNumberLine();

    const speed = parseInt(speedSlider.value);
    const d = baseDelay + (100 - speed) * 10;
    await delay(d);
  }

  playing = false;
}

function resetAnimation() {
  playing = false;
  steps = [...currentData];
  animStep = steps.length - 1;
  drawNumberLine();
}

export function initFarkas() {
  canvas = document.getElementById('fk-canvas');
  ctx = canvas.getContext('2d');

  steps = [...currentData];
  animStep = -1;
  drawNumberLine();

  document.getElementById('fk-play').addEventListener('click', playAnimation);
  document.getElementById('fk-reset').addEventListener('click', resetAnimation);

  document.getElementById('fk-d4').addEventListener('click', () => {
    currentData = BISECTION_D4_O3;
    currentLabel = 'd=4, order 3';
    document.getElementById('fk-d4').classList.add('active');
    document.getElementById('fk-d6').classList.remove('active');
    steps = [...currentData];
    animStep = -1;
    drawNumberLine();
  });

  document.getElementById('fk-d6').addEventListener('click', () => {
    currentData = BISECTION_D6_O3;
    currentLabel = 'd=6, order 3';
    document.getElementById('fk-d6').classList.add('active');
    document.getElementById('fk-d4').classList.remove('active');
    steps = [...currentData];
    animStep = -1;
    drawNumberLine();
  });
}
