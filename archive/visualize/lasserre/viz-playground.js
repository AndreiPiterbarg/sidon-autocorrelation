// Section 5: Build Your Own Relaxation (interactive playground)
// Draggable bar chart + constraint toggle switches

import { buildWindowMatrices, computeMaxWindow, PLAYGROUND_BOUNDS, VAL_D } from './lasserre-data.js';
import { tween, clamp, setupHiDPI } from './animate.js';

const W = 440, H = 220;
const PAD = { top: 30, right: 16, bottom: 36, left: 46 };

const COL = {
  bg: '#1a1d27', grid: '#2a2d3a', text: '#e0e0e0', textLight: '#999',
  blue: '#4ea8de', blueLight: '#64b5f6', green: '#81c784', gold: '#ffb74d',
  red: '#ef5350',
};

let canvas, ctx;
let windows4;
const D = 4, M = 20;
let masses = [5, 5, 5, 5];
let dragging = false, dragIdx = -1;

const CONSTRAINTS = [
  { key: 'nonneg',  name: 'y >= 0',              desc: 'Nonnegativity',            on: false },
  { key: 'norm',    name: 'y_0 = 1',             desc: 'Normalization',            on: false },
  { key: 'psd1',    name: 'M_1(y) PSD',          desc: 'Moment matrix (order 1)',  on: false },
  { key: 'psd2',    name: 'M_2(y) PSD',          desc: 'Moment matrix (order 2)',  on: false },
  { key: 'loc',     name: 'Localizing PSD',      desc: 'Localizing matrices',      on: false },
  { key: 'win',     name: 'Window PSD',          desc: 'Window constraints',       on: false },
  { key: 'psd3',    name: 'M_3(y) PSD',          desc: 'Order 3 (full)',           on: false },
];

function getActiveKey() {
  const active = CONSTRAINTS.filter(c => c.on).map(c => c.key);
  if (active.length === 0) return 'none';
  if (active.includes('psd3')) return 'nonneg+norm+psd3+loc+win';
  const parts = [];
  if (active.includes('nonneg')) parts.push('nonneg');
  if (active.includes('norm')) parts.push('norm');
  if (active.includes('psd2')) parts.push('psd2');
  else if (active.includes('psd1')) parts.push('psd1');
  if (active.includes('loc')) parts.push('loc');
  if (active.includes('win')) parts.push('win');
  const key = parts.join('+');
  return PLAYGROUND_BOUNDS[key] !== undefined ? key : 'nonneg+norm';
}

function getBound() {
  return PLAYGROUND_BOUNDS[getActiveKey()] || 0;
}

function drawHist() {
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, W, H);

  const d = D;
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const barW = plotW / d - 4;

  ctx.strokeStyle = COL.grid;
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + plotH * (1 - i / 4);
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(W - PAD.right, y);
    ctx.stroke();
    ctx.fillStyle = COL.textLight;
    ctx.font = '10px system-ui';
    ctx.textAlign = 'right';
    ctx.fillText((i * M / 4).toFixed(0), PAD.left - 6, y + 3);
  }

  for (let i = 0; i < d; i++) {
    const x = PAD.left + i * (plotW / d) + 2;
    const h = (masses[i] / M) * plotH;
    const y = PAD.top + plotH - h;
    ctx.fillStyle = COL.blue;
    ctx.globalAlpha = 0.8;
    ctx.fillRect(x, y, barW, h);
    ctx.globalAlpha = 1;

    ctx.fillStyle = COL.text;
    ctx.font = 'bold 12px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(masses[i], x + barW / 2, y - 6);

    ctx.fillStyle = COL.textLight;
    ctx.font = '10px system-ui';
    ctx.fillText(`bin ${i + 1}`, x + barW / 2, H - PAD.bottom + 14);
  }

  ctx.font = 'bold 11px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'center';
  ctx.fillText(`Mass distribution (total = ${M})`, W / 2, 16);
}

function updateDisplays() {
  const mu = masses.map(m => m / M);
  const { value } = computeMaxWindow(mu, windows4);
  const bound = getBound();
  const valD = VAL_D[4];

  document.getElementById('pg-trueval-num').textContent = value.toFixed(4);

  const pct = clamp((bound / valD) * 100, 0, 100);
  document.getElementById('pg-bound-bar').style.width = `${pct}%`;
  document.getElementById('pg-bound-val').textContent = bound.toFixed(3);
  document.getElementById('pg-gap-pct').textContent = `${pct.toFixed(1)}%`;

  const compEl = document.getElementById('pg-comparison');
  if (compEl) {
    const holds = value >= bound - 0.001;
    compEl.innerHTML = `
      <div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;">
        <div><span style="color:${COL.gold};font-weight:700;">Your peak: ${value.toFixed(4)}</span></div>
        <div style="color:${COL.textLight};">≥</div>
        <div><span style="color:${COL.blue};font-weight:700;">SDP bound: ${bound.toFixed(3)}</span></div>
        <div style="color:${COL.textLight};">≥</div>
        <div><span style="color:${COL.textLight};">true val(4) = ${valD.toFixed(3)}</span></div>
      </div>
      <div style="margin-top:6px;font-size:12px;color:${COL.textLight};">
        ${holds
          ? 'The SDP bound holds for <em>every</em> distribution — try to get your peak close to it!'
          : ''}
      </div>
    `;
  }
}

function buildConstraintStack() {
  const container = document.getElementById('pg-constraints');
  container.innerHTML = '';

  for (const c of CONSTRAINTS) {
    const row = document.createElement('div');
    row.className = 'constraint-row' + (c.on ? ' active' : '');

    const toggle = document.createElement('label');
    toggle.className = 'constraint-toggle';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = c.on;
    const slider = document.createElement('span');
    slider.className = 'slider';
    toggle.appendChild(input);
    toggle.appendChild(slider);

    const nameSpan = document.createElement('span');
    nameSpan.className = 'constraint-name';
    nameSpan.textContent = c.name;

    const descSpan = document.createElement('span');
    descSpan.className = 'constraint-desc';
    descSpan.textContent = c.desc;

    row.appendChild(toggle);
    row.appendChild(nameSpan);
    row.appendChild(descSpan);
    container.appendChild(row);

    input.addEventListener('change', () => {
      c.on = input.checked;

      if (input.checked) {
        if (c.key === 'psd3') {
          for (const cc of CONSTRAINTS) cc.on = true;
        }
        if (c.key === 'win' || c.key === 'loc') {
          CONSTRAINTS.find(x => x.key === 'psd2').on = true;
          CONSTRAINTS.find(x => x.key === 'nonneg').on = true;
          CONSTRAINTS.find(x => x.key === 'norm').on = true;
        }
        if (c.key === 'psd2' || c.key === 'psd1') {
          CONSTRAINTS.find(x => x.key === 'nonneg').on = true;
          CONSTRAINTS.find(x => x.key === 'norm').on = true;
        }
      }

      buildConstraintStack();
      updateDisplays();
    });
  }
}

function handleDrag(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = W / rect.width;
  const scaleY = H / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const barW = plotW / D;

  const idx = Math.floor((mx - PAD.left) / barW);
  if (idx < 0 || idx >= D) return -1;

  const newMass = Math.round(clamp((1 - (my - PAD.top) / plotH) * M, 0, M));
  return { idx, newMass };
}

function redistributeMass(idx, newMass) {
  const oldMass = masses[idx];
  const diff = newMass - oldMass;
  if (diff === 0) return;

  masses[idx] = newMass;
  let remaining = M - masses.reduce((a, b) => a + b, 0);

  const others = Array.from({ length: D }, (_, i) => i).filter(i => i !== idx);
  while (remaining !== 0 && others.length > 0) {
    const share = Math.sign(remaining);
    for (const i of others) {
      if (remaining === 0) break;
      const newVal = masses[i] + share;
      if (newVal >= 0 && newVal <= M) {
        masses[i] = newVal;
        remaining -= share;
      }
    }
    if (remaining !== 0) {
      const canAdjust = others.some(i =>
        (remaining > 0 && masses[i] < M) || (remaining < 0 && masses[i] > 0)
      );
      if (!canAdjust) break;
    }
  }
}

function setPreset(name) {
  switch (name) {
    case 'uniform': masses = [5, 5, 5, 5]; break;
    case 'concentrated': masses = [14, 2, 2, 2]; break;
    case 'symmetric': masses = [3, 7, 7, 3]; break;
    case 'random':
      masses = [0, 0, 0, 0];
      for (let i = 0; i < M; i++) masses[Math.floor(Math.random() * D)]++;
      break;
  }
  drawHist();
  updateDisplays();
}

export function initPlayground() {
  canvas = document.getElementById('pg-hist');
  ctx = setupHiDPI(canvas, W, H);
  windows4 = buildWindowMatrices(4);

  buildConstraintStack();

  document.querySelectorAll('#playground .preset-row .btn-sm').forEach(btn => {
    btn.addEventListener('click', () => setPreset(btn.dataset.preset));
  });

  canvas.addEventListener('mousedown', (e) => {
    const result = handleDrag(e);
    if (result && result.idx >= 0) {
      dragging = true;
      dragIdx = result.idx;
      redistributeMass(result.idx, result.newMass);
      drawHist();
      updateDisplays();
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const result = handleDrag(e);
    if (result && result.idx >= 0) {
      redistributeMass(result.idx, result.newMass);
      drawHist();
      updateDisplays();
    }
  });

  window.addEventListener('mouseup', () => { dragging = false; dragIdx = -1; });

  drawHist();
  updateDisplays();
}
