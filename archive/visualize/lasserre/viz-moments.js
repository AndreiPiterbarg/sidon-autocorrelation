// Section 3: Moment Matrices & PSD
// Left: distribution + moment matrix grid. Right: eigenvalue visualization.

import { eigenvalues3x3 } from './lasserre-data.js';
import { tween, delay, lerp, clamp } from './animate.js';

const COL = {
  bg: '#1a1d27', grid: '#2a2d3a', text: '#e0e0e0', textLight: '#999',
  blue: '#4ea8de', blueLight: '#64b5f6', red: '#ef5350',
  green: '#81c784', greenPale: '#1a3d1a', redPale: '#3d1a1a',
  gold: '#ffb74d',
};

let distCanvas, distCtx, psdCanvas, psdCtx;
let dist = [0.3, 0.2, 0.15, 0.35];
let hoveredCell = null;
let animPhase = 0;
let tamperAmount = 0;

function momentMatrix(d) {
  const M = Array.from({ length: d }, () => new Float64Array(d));
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      M[i][j] = dist[i] * dist[j];
  return M;
}

function drawDistPanel() {
  const W = 440, H = 300;
  const ctx = distCtx;
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, W, H);

  const d = dist.length;
  const barW = 50, gap = 12;
  const totalW = d * barW + (d - 1) * gap;
  const startX = (W - totalW) / 2;
  const barTop = 30, barMaxH = 90;

  // Title
  ctx.font = 'bold 12px system-ui';
  ctx.fillStyle = COL.text;
  ctx.textAlign = 'center';
  ctx.fillText('Distribution (d = 4 bins)', W / 2, 20);

  // Bars
  for (let i = 0; i < d; i++) {
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
    ctx.fillText(dist[i].toFixed(2), x + barW / 2, y - 6);
  }

  // Moment matrix grid
  const M = momentMatrix(d);
  const gridSize = 48, gridGap = 3;
  const gridTotalW = d * gridSize + (d - 1) * gridGap;
  const gridX = (W - gridTotalW) / 2;
  const gridY = 155;

  ctx.font = 'bold 11px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'center';
  ctx.fillText('Moment Matrix  M(y)', W / 2, gridY - 10);

  const maxVal = Math.max(...M.flat());
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
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
      ctx.font = '10px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(3), x + gridSize / 2, y + gridSize / 2 + 4);
    }
  }

  // Row/col labels
  ctx.font = '10px system-ui';
  ctx.fillStyle = COL.textLight;
  for (let i = 0; i < d; i++) {
    ctx.textAlign = 'right';
    ctx.fillText(`${i + 1}`, gridX - 6, gridY + i * (gridSize + gridGap) + gridSize / 2 + 4);
    ctx.textAlign = 'center';
    ctx.fillText(`${i + 1}`, gridX + i * (gridSize + gridGap) + gridSize / 2, gridY + d * (gridSize + gridGap) + 12);
  }
}

function drawPSDPanel() {
  const W = 440, H = 300;
  const ctx = psdCtx;
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, W, H);

  // Build a 3x3 moment matrix from first 3 bins
  const M = [
    [dist[0] * dist[0], dist[0] * dist[1], dist[0] * dist[2]],
    [dist[1] * dist[0], dist[1] * dist[1], dist[1] * dist[2]],
    [dist[2] * dist[0], dist[2] * dist[1], dist[2] * dist[2]],
  ];

  // Apply tampering
  if (tamperAmount > 0) {
    M[0][1] += tamperAmount * 0.15;
    M[1][0] += tamperAmount * 0.15;
    M[0][2] -= tamperAmount * 0.10;
    M[2][0] -= tamperAmount * 0.10;
  }

  const eigs = eigenvalues3x3(M);
  const isPSD = eigs.every(e => e >= -1e-10);

  // Title
  ctx.font = 'bold 12px system-ui';
  ctx.fillStyle = COL.text;
  ctx.textAlign = 'center';
  ctx.fillText('Eigenvalues of 3x3 Moment Matrix', W / 2, 20);

  // Matrix display (small)
  const cellSize = 52, gapM = 3;
  const mGridX = 40, mGridY = 45;
  ctx.font = 'bold 11px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'left';
  ctx.fillText('Matrix:', mGridX, 40);

  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const x = mGridX + j * (cellSize + gapM);
      const y = mGridY + 10 + i * (cellSize + gapM);
      const v = M[i][j];
      const t = clamp(Math.abs(v) / 0.15, 0, 1);
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

  // Eigenvalue bars
  const barX = 240, barY = 60, barW = 160, barH = 24, barGap = 12;
  ctx.font = 'bold 11px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'left';
  ctx.fillText('Eigenvalues:', barX, 55);

  const maxAbs = Math.max(0.01, ...eigs.map(Math.abs));

  for (let i = 0; i < 3; i++) {
    const y = barY + i * (barH + barGap);
    const e = eigs[i];
    const w = (Math.abs(e) / maxAbs) * barW;

    ctx.fillStyle = COL.grid;
    ctx.fillRect(barX, y, barW, barH);

    ctx.fillStyle = e >= -1e-10 ? COL.green : COL.red;
    ctx.fillRect(barX, y, w, barH);

    ctx.font = '11px system-ui';
    ctx.fillStyle = e >= -1e-10 ? COL.green : COL.red;
    ctx.textAlign = 'left';
    ctx.fillText(`λ${i + 1} = ${e.toFixed(5)}`, barX + barW + 8, y + barH / 2 + 4);
  }

  // Verdict
  const verdictY = barY + 3 * (barH + barGap) + 20;
  ctx.font = 'bold 16px system-ui';
  ctx.textAlign = 'center';
  if (isPSD) {
    ctx.fillStyle = COL.green;
    ctx.fillText('PSD — Consistent', W / 2, verdictY);
    ctx.font = '12px system-ui';
    ctx.fillStyle = COL.textLight;
    ctx.fillText('These moments could come from a real distribution', W / 2, verdictY + 20);
  } else {
    ctx.fillStyle = COL.red;
    ctx.fillText('NOT PSD — Impossible!', W / 2, verdictY);
    ctx.font = '12px system-ui';
    ctx.fillStyle = COL.textLight;
    ctx.fillText('No real distribution produces these moments', W / 2, verdictY + 20);
  }

  // Tamper slider label
  ctx.font = '11px system-ui';
  ctx.fillStyle = COL.textLight;
  ctx.textAlign = 'center';
  const tampLabel = tamperAmount > 0 ? `Tampered: ${(tamperAmount * 100).toFixed(0)}%` : 'Valid moments';
  ctx.fillText(tampLabel, W / 2, H - 10);
}

async function autoPlayPSD() {
  // Phase 1: valid moments
  tamperAmount = 0;
  drawPSDPanel();
  await delay(1500);

  // Phase 2: gradually tamper
  for (let i = 0; i <= 20; i++) {
    tamperAmount = i / 20;
    drawPSDPanel();
    await delay(80);
  }

  await delay(2000);

  // Phase 3: revert
  for (let i = 20; i >= 0; i--) {
    tamperAmount = i / 20;
    drawPSDPanel();
    await delay(60);
  }
}

export function initMoments() {
  distCanvas = document.getElementById('mom-dist');
  distCtx = distCanvas.getContext('2d');
  psdCanvas = document.getElementById('mom-psd');
  psdCtx = psdCanvas.getContext('2d');

  // Mouse hover on moment matrix
  distCanvas.addEventListener('mousemove', (e) => {
    const rect = distCanvas.getBoundingClientRect();
    const scaleX = 440 / rect.width;
    const scaleY = 300 / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;

    const d = dist.length;
    const gridSize = 48, gridGap = 3;
    const gridTotalW = d * gridSize + (d - 1) * gridGap;
    const gridX = (440 - gridTotalW) / 2;
    const gridY = 155;

    hoveredCell = null;
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
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
