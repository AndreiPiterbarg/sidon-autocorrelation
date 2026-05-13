// Section 3: Moment Matrices & PSD
// Left: distribution + moment matrix grid. Right: eigenvalue visualization.
// Both panels use the same d=3 distribution for consistency.

import { eigenvalues3x3 } from './lasserre-data.js';
import { tween, delay, lerp, clamp, setupHiDPI } from './animate.js';

const LW = 440, LH = 380;
const RW = 440, RH = 380;

const COL = {
  bg: '#1a1d27', grid: '#2a2d3a', text: '#e0e0e0', textLight: '#999',
  blue: '#4ea8de', blueLight: '#64b5f6', red: '#ef5350',
  green: '#81c784', greenPale: '#1a3d1a', redPale: '#3d1a1a',
  gold: '#ffb74d',
};

let distCanvas, distCtx, psdCanvas, psdCtx;
const D = 3;
let dist = [0.40, 0.25, 0.35];
let hoveredCell = null;
let tamperAmount = 0;

function momentMatrix() {
  const M = Array.from({ length: D }, () => new Float64Array(D));
  for (let i = 0; i < D; i++)
    for (let j = 0; j < D; j++)
      M[i][j] = dist[i] * dist[j];
  return M;
}

function tamperedMatrix() {
  const M = momentMatrix();
  if (tamperAmount > 0) {
    M[0][1] += tamperAmount * 0.15;
    M[1][0] += tamperAmount * 0.15;
    M[0][2] -= tamperAmount * 0.10;
    M[2][0] -= tamperAmount * 0.10;
  }
  return M;
}

function drawDistPanel() {
  const ctx = distCtx;
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, LW, LH);

  const barW = 80, gap = 20;
  const totalW = D * barW + (D - 1) * gap;
  const startX = (LW - totalW) / 2;
  const barTop = 40, barMaxH = 90;

  ctx.font = 'bold 12px system-ui';
  ctx.fillStyle = COL.text;
  ctx.textAlign = 'center';
  ctx.fillText('Distribution (d = 3 bins)', LW / 2, 22);

  for (let i = 0; i < D; i++) {
    const x = startX + i * (barW + gap);
    const h = dist[i] * barMaxH / 0.5;
    const y = barTop + barMaxH - h;
    const highlight = hoveredCell && (hoveredCell[0] === i || hoveredCell[1] === i);
    ctx.fillStyle = highlight ? COL.gold : COL.blue;
    ctx.globalAlpha = highlight ? 1 : 0.7;
    ctx.fillRect(x, y, barW, h);
    ctx.globalAlpha = 1;

    ctx.font = '11px system-ui';
    ctx.fillStyle = COL.textLight;
    ctx.textAlign = 'center';
    ctx.fillText(`bin ${i + 1}`, x + barW / 2, barTop + barMaxH + 16);
    ctx.fillStyle = COL.text;
    ctx.font = 'bold 12px system-ui';
    ctx.fillText(dist[i].toFixed(2), x + barW / 2, y - 8);
  }

  const M = momentMatrix();
  const gridSize = 52, gridGap = 3;
  const gridTotalW = D * gridSize + (D - 1) * gridGap;
  const gridX = (LW - gridTotalW) / 2;
  const gridY = 170;

  ctx.font = 'bold 11px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'center';
  ctx.fillText('Moment Matrix  M[i, j] = bin_i × bin_j', LW / 2, gridY - 12);

  const maxVal = Math.max(...Array.from({ length: D * D }, (_, k) => M[Math.floor(k / D)][k % D]));
  for (let i = 0; i < D; i++) {
    for (let j = 0; j < D; j++) {
      const x = gridX + j * (gridSize + gridGap);
      const y = gridY + i * (gridSize + gridGap);
      const v = M[i][j];
      const t = maxVal > 0 ? v / maxVal : 0;

      const isHovered = hoveredCell && hoveredCell[0] === i && hoveredCell[1] === j;

      ctx.fillStyle = isHovered
        ? COL.gold
        : `rgb(${Math.round(26 + t * 52)}, ${Math.round(29 + t * 100)}, ${Math.round(39 + t * 170)})`;
      ctx.fillRect(x, y, gridSize, gridSize);

      if (isHovered) {
        ctx.strokeStyle = COL.gold;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, gridSize, gridSize);
      }

      ctx.fillStyle = t > 0.5 ? '#fff' : COL.textLight;
      ctx.font = '11px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(3), x + gridSize / 2, y + gridSize / 2 + 4);
    }
  }

  ctx.font = '10px system-ui';
  ctx.fillStyle = COL.textLight;
  for (let i = 0; i < D; i++) {
    ctx.textAlign = 'right';
    ctx.fillText(`${i + 1}`, gridX - 8, gridY + i * (gridSize + gridGap) + gridSize / 2 + 4);
    ctx.textAlign = 'center';
    ctx.fillText(`${i + 1}`, gridX + i * (gridSize + gridGap) + gridSize / 2, gridY + D * (gridSize + gridGap) + 14);
  }
}

function drawPSDPanel() {
  const ctx = psdCtx;
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, RW, RH);

  const M = tamperedMatrix();
  const eigs = eigenvalues3x3(M);
  const isPSD = eigs.every(e => e >= -1e-10);

  ctx.font = 'bold 12px system-ui';
  ctx.fillStyle = COL.text;
  ctx.textAlign = 'center';
  ctx.fillText('Same Matrix — Eigenvalue Check', RW / 2, 22);

  const cellSize = 48, gapM = 2;
  const mTotalW = D * cellSize + (D - 1) * gapM;
  const mGridX = (RW - mTotalW) / 2;
  const mGridY = 42;

  for (let i = 0; i < D; i++) {
    for (let j = 0; j < D; j++) {
      const x = mGridX + j * (cellSize + gapM);
      const y = mGridY + i * (cellSize + gapM);
      const v = M[i][j];
      const t = clamp(Math.abs(v) / 0.2, 0, 1);
      ctx.fillStyle = v < -0.001
        ? `rgba(239,83,80,${0.2 + t * 0.5})`
        : `rgb(${Math.round(26 + t * 40)}, ${Math.round(29 + t * 80)}, ${Math.round(39 + t * 150)})`;
      ctx.fillRect(x, y, cellSize, cellSize);

      ctx.fillStyle = '#fff';
      ctx.font = '10px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(3), x + cellSize / 2, y + cellSize / 2 + 4);
    }
  }

  const eigSectionY = mGridY + D * (cellSize + gapM) + 24;
  ctx.font = 'bold 11px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'left';
  ctx.fillText('Eigenvalues:', 30, eigSectionY);

  const barX = 30, barW = 240, barH = 20, barGap = 10;
  const maxAbs = Math.max(0.01, ...eigs.map(Math.abs));

  for (let i = 0; i < 3; i++) {
    const y = eigSectionY + 10 + i * (barH + barGap);
    const e = eigs[i];
    const w = (Math.abs(e) / maxAbs) * barW;

    ctx.fillStyle = e >= -1e-10 ? COL.green : COL.red;
    ctx.font = '11px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText(`λ${i + 1} = ${e.toFixed(4)}`, barX, y + 4);

    const bx = barX + 120;
    ctx.fillStyle = COL.grid;
    ctx.fillRect(bx, y - 6, barW - 100, barH);

    ctx.fillStyle = e >= -1e-10 ? COL.green : COL.red;
    const bw = (Math.abs(e) / maxAbs) * (barW - 100);
    ctx.fillRect(bx, y - 6, bw, barH);
  }

  const verdictY = eigSectionY + 10 + 3 * (barH + barGap) + 16;
  ctx.font = 'bold 16px system-ui';
  ctx.textAlign = 'center';
  if (isPSD) {
    ctx.fillStyle = COL.green;
    ctx.fillText('✓ PSD — Consistent', RW / 2, verdictY);
    ctx.font = '12px system-ui';
    ctx.fillStyle = COL.textLight;
    ctx.fillText('These moments could come from a real distribution', RW / 2, verdictY + 22);
  } else {
    ctx.fillStyle = COL.red;
    ctx.fillText('✗ NOT PSD — Impossible!', RW / 2, verdictY);
    ctx.font = '12px system-ui';
    ctx.fillStyle = COL.textLight;
    ctx.fillText('No real distribution produces these moments', RW / 2, verdictY + 22);
  }

  ctx.font = '11px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'center';
  const tampLabel = tamperAmount > 0 ? `Tampered: ${(tamperAmount * 100).toFixed(0)}%` : 'Valid moments (from distribution on left)';
  ctx.fillText(tampLabel, RW / 2, RH - 12);
}

async function autoPlayPSD() {
  tamperAmount = 0;
  drawPSDPanel();
  await delay(2000);

  for (let i = 0; i <= 20; i++) {
    tamperAmount = i / 20;
    drawPSDPanel();
    await delay(80);
  }

  await delay(2000);

  for (let i = 20; i >= 0; i--) {
    tamperAmount = i / 20;
    drawPSDPanel();
    await delay(60);
  }
}

export function initMoments() {
  distCanvas = document.getElementById('mom-dist');
  distCtx = setupHiDPI(distCanvas, LW, LH);
  psdCanvas = document.getElementById('mom-psd');
  psdCtx = setupHiDPI(psdCanvas, RW, RH);

  distCanvas.addEventListener('mousemove', (e) => {
    const rect = distCanvas.getBoundingClientRect();
    const scaleX = LW / rect.width;
    const scaleY = LH / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;

    const gridSize = 52, gridGap = 3;
    const gridTotalW = D * gridSize + (D - 1) * gridGap;
    const gridX = (LW - gridTotalW) / 2;
    const gridY = 170;

    hoveredCell = null;
    for (let i = 0; i < D; i++) {
      for (let j = 0; j < D; j++) {
        const x = gridX + j * (gridSize + gridGap);
        const y = gridY + i * (gridSize + gridGap);
        if (mx >= x && mx <= x + gridSize && my >= y && my <= y + gridSize) {
          hoveredCell = [i, j];
          break;
        }
      }
      if (hoveredCell) break;
    }

    drawDistPanel();

    const status = document.getElementById('mom-status');
    if (hoveredCell) {
      const [i, j] = hoveredCell;
      const v = dist[i] * dist[j];
      status.textContent = `M[${i + 1},${j + 1}] = bin${i + 1} × bin${j + 1} = ${dist[i].toFixed(2)} × ${dist[j].toFixed(2)} = ${v.toFixed(4)}`;
      status.style.borderColor = COL.gold;
    } else {
      status.textContent = 'Hover over matrix cells to see the connection to the distribution.';
      status.style.borderColor = '';
    }
  });

  distCanvas.addEventListener('mouseleave', () => {
    hoveredCell = null;
    drawDistPanel();
    document.getElementById('mom-status').textContent = 'Hover over matrix cells to see the connection to the distribution.';
    document.getElementById('mom-status').style.borderColor = '';
  });

  drawDistPanel();
  drawPSDPanel();
  autoPlayPSD();
}
