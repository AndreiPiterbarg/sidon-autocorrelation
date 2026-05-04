// Section 1: The Original Problem
// Ternary simplex heatmap showing val(d) = min max_W mu^T M_W mu

import { buildWindowMatrices, computeMaxWindow, baryToXY } from './lasserre-data.js';
import { tween, delay, lerp, clamp } from './animate.js';

const W = 700, H = 460;
const TRI_CX = 300, TRI_CY = 260, TRI_R = 200;
const BAR_X = 560, BAR_W = 110, BAR_TOP = 60, BAR_H = 340;

const COL = {
  bg: '#1a1d27', grid: '#2a2d3a', text: '#e0e0e0', textLight: '#999',
  blue: '#4ea8de', blueLight: '#64b5f6', red: '#ef5350', green: '#81c784',
  gold: '#ffb74d',
};

let canvas, ctx;
let windows3;
let heatmapData;
let animDot = { b0: 1/3, b1: 1/3, b2: 1/3 };
let bestDot = null;
let playing = false, animId = null, progress = 0;
let path = [];

const GRID_RES = 60;

function triVertices() {
  return [
    baryToXY(1, 0, 0, TRI_CX, TRI_CY, TRI_R),
    baryToXY(0, 1, 0, TRI_CX, TRI_CY, TRI_R),
    baryToXY(0, 0, 1, TRI_CX, TRI_CY, TRI_R),
  ];
}

function precomputeHeatmap() {
  const data = [];
  let minVal = Infinity, maxVal = -Infinity;
  let bestB = null;
  for (let i = 0; i <= GRID_RES; i++) {
    for (let j = 0; j <= GRID_RES - i; j++) {
      const k = GRID_RES - i - j;
      const b0 = i / GRID_RES, b1 = j / GRID_RES, b2 = k / GRID_RES;
      if (b0 < 0.001 && b1 < 0.001) continue;
      if (b0 < 0.001 && b2 < 0.001) continue;
      if (b1 < 0.001 && b2 < 0.001) continue;
      const mu = [b0, b1, b2];
      const { value } = computeMaxWindow(mu, windows3);
      data.push({ b0, b1, b2, value });
      if (value < minVal) { minVal = value; bestB = { b0, b1, b2 }; }
      if (value > maxVal) maxVal = value;
    }
  }
  return { data, minVal, maxVal, bestB };
}

function valToColor(v, lo, hi) {
  const t = clamp((v - lo) / (hi - lo), 0, 1);
  const r = Math.round(30 + t * 200);
  const g = Math.round(80 - t * 50);
  const b = Math.round(180 - t * 130);
  return `rgb(${r},${g},${b})`;
}

function drawTriangle() {
  const verts = triVertices();
  ctx.strokeStyle = COL.grid;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(verts[0][0], verts[0][1]);
  ctx.lineTo(verts[1][0], verts[1][1]);
  ctx.lineTo(verts[2][0], verts[2][1]);
  ctx.closePath();
  ctx.stroke();

  ctx.font = '12px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'center';
  ctx.fillText('bin 1 = 100%', verts[0][0], verts[0][1] - 14);
  ctx.fillText('bin 2 = 100%', verts[1][0] - 10, verts[1][1] + 22);
  ctx.fillText('bin 3 = 100%', verts[2][0] + 10, verts[2][1] + 22);
}

function drawHeatmap() {
  const { data, minVal, maxVal } = heatmapData;
  const cellR = TRI_R / GRID_RES * 1.2;
  for (const pt of data) {
    const [x, y] = baryToXY(pt.b0, pt.b1, pt.b2, TRI_CX, TRI_CY, TRI_R);
    ctx.fillStyle = valToColor(pt.value, minVal, maxVal);
    ctx.beginPath();
    ctx.arc(x, y, cellR, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawDot(b0, b1, b2, color, r = 8) {
  const [x, y] = baryToXY(b0, b1, b2, TRI_CX, TRI_CY, TRI_R);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.stroke();
  return [x, y];
}

function drawPath() {
  if (path.length < 2) return;
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  const [x0, y0] = baryToXY(path[0].b0, path[0].b1, path[0].b2, TRI_CX, TRI_CY, TRI_R);
  ctx.moveTo(x0, y0);
  for (let i = 1; i < path.length; i++) {
    const [x, y] = baryToXY(path[i].b0, path[i].b1, path[i].b2, TRI_CX, TRI_CY, TRI_R);
    ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawWindowBars(mu) {
  const vals = windows3.map(w => {
    let v = 0;
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        v += mu[i] * w.M[i][j] * mu[j];
    return v;
  });
  const maxV = Math.max(...vals);
  const n = vals.length;
  const barW = BAR_W / n - 2;

  ctx.fillStyle = COL.bg;
  ctx.fillRect(BAR_X - 10, BAR_TOP - 20, BAR_W + 30, BAR_H + 50);

  ctx.font = '11px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'center';
  ctx.fillText('Window values', BAR_X + BAR_W / 2, BAR_TOP - 6);

  for (let i = 0; i < n; i++) {
    const barH = (vals[i] / 6) * BAR_H;
    const x = BAR_X + i * (barW + 2);
    const y = BAR_TOP + BAR_H - barH;
    const isMax = Math.abs(vals[i] - maxV) < 0.001;
    ctx.fillStyle = isMax ? COL.red : COL.blue;
    ctx.globalAlpha = isMax ? 1.0 : 0.6;
    ctx.fillRect(x, y, barW, barH);
    ctx.globalAlpha = 1;
  }

  ctx.fillStyle = COL.gold;
  ctx.font = 'bold 14px system-ui';
  ctx.textAlign = 'left';
  ctx.fillText(`max = ${maxV.toFixed(3)}`, BAR_X, BAR_TOP + BAR_H + 20);
}

function drawTitle() {
  ctx.font = 'bold 13px system-ui';
  ctx.fillStyle = COL.text;
  ctx.textAlign = 'left';
  ctx.fillText('Simplex: every point is a distribution over 3 bins', 20, 28);

  if (bestDot) {
    ctx.font = '12px system-ui';
    ctx.fillStyle = COL.green;
    ctx.fillText(`Best found: val(3) = ${bestDot.value.toFixed(4)}`, 20, 48);
  }
}

function render() {
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, W, H);

  drawHeatmap();
  drawTriangle();
  drawPath();

  if (bestDot) {
    drawDot(bestDot.b0, bestDot.b1, bestDot.b2, COL.green, 6);
  }

  const mu = [animDot.b0, animDot.b1, animDot.b2];
  drawDot(animDot.b0, animDot.b1, animDot.b2, '#fff', 8);
  drawWindowBars(mu);
  drawTitle();
}

function generateSearchPath(steps) {
  const pts = [];
  let b0 = 0.9, b1 = 0.05, b2 = 0.05;
  const best = heatmapData.bestB;

  for (let i = 0; i < steps; i++) {
    const t = i / steps;
    const noise = 0.04 * Math.sin(i * 0.7) * (1 - t);
    b0 = lerp(0.9, best.b0, t) + noise;
    b1 = lerp(0.05, best.b1, t) + noise * 0.7;
    b2 = 1 - b0 - b1;
    b0 = clamp(b0, 0.01, 0.98);
    b1 = clamp(b1, 0.01, 0.98 - b0);
    b2 = 1 - b0 - b1;
    pts.push({ b0, b1, b2 });
  }
  return pts;
}

let searchPath = [];

async function playAnimation() {
  if (playing) return;
  playing = true;
  searchPath = generateSearchPath(120);
  path = [];
  bestDot = null;
  let bestVal = Infinity;

  const label = document.getElementById('prob-label');
  const scrubber = document.getElementById('prob-scrubber');

  for (let i = 0; i < searchPath.length && playing; i++) {
    animDot = searchPath[i];
    path.push({ ...animDot });
    progress = (i / searchPath.length) * 100;
    scrubber.value = progress;

    const mu = [animDot.b0, animDot.b1, animDot.b2];
    const { value } = computeMaxWindow(mu, windows3);
    if (value < bestVal) {
      bestVal = value;
      bestDot = { ...animDot, value };
    }

    label.textContent = `Step ${i + 1} / ${searchPath.length}`;
    render();
    await delay(50);
  }

  if (playing) {
    label.textContent = 'Done — minimum found';
  }
  playing = false;
}

function resetAnimation() {
  playing = false;
  animDot = { b0: 1/3, b1: 1/3, b2: 1/3 };
  bestDot = null;
  path = [];
  progress = 0;
  document.getElementById('prob-scrubber').value = 0;
  document.getElementById('prob-label').textContent = 'Ready';
  render();
}

export function initProblem() {
  canvas = document.getElementById('prob-canvas');
  ctx = canvas.getContext('2d');

  windows3 = buildWindowMatrices(3);
  heatmapData = precomputeHeatmap();

  document.getElementById('prob-play').addEventListener('click', playAnimation);
  document.getElementById('prob-reset').addEventListener('click', resetAnimation);

  document.getElementById('prob-scrubber').addEventListener('input', (e) => {
    if (playing) return;
    if (!searchPath.length) searchPath = generateSearchPath(120);
    const idx = Math.round((e.target.value / 100) * (searchPath.length - 1));
    path = searchPath.slice(0, idx + 1);
    animDot = searchPath[idx] || { b0: 1/3, b1: 1/3, b2: 1/3 };

    let bestVal = Infinity;
    bestDot = null;
    for (const pt of path) {
      const mu = [pt.b0, pt.b1, pt.b2];
      const { value } = computeMaxWindow(mu, windows3);
      if (value < bestVal) { bestVal = value; bestDot = { ...pt, value }; }
    }
    render();
  });

  render();
}
