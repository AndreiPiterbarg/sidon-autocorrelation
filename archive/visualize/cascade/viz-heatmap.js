// Section 3: Window Scanner Heatmap
// Animated heatmap showing the window scanning process.

import { scanWindows, onHeatmapRequest, getState } from './viz-challenge.js';
import { delay } from './animate.js';

const COLORS = {
  empty: '#1e2030',
  low: '#1a3a5c',
  mid: '#2a6090',
  high: '#c07050',
  hot: '#ef5350',
  bg: '#1a1d27',
  axis: '#999',
  text: '#e0e0e0',
  textLight: '#777',
  pruneFlash: '#ff4444',
};

let canvas, ctx;
let currentWindows = null;
let currentThreshold = 1.30;
let currentD = 4;
let scanning = false;
let scanAbort = false;

function getSpeedMs() {
  const slider = document.getElementById('hm-speed');
  const v = parseInt(slider.value);
  if (v < 20) return 150;
  if (v < 40) return 80;
  if (v < 60) return 30;
  if (v < 80) return 10;
  return 0;
}

function updateSpeedLabel() {
  const v = parseInt(document.getElementById('hm-speed').value);
  const label = document.getElementById('hm-speed-label');
  if (v < 20) label.textContent = 'Slow';
  else if (v < 40) label.textContent = 'Medium-Slow';
  else if (v < 60) label.textContent = 'Medium';
  else if (v < 80) label.textContent = 'Fast';
  else label.textContent = 'Instant';
}

function cellColor(tv, threshold) {
  const ratio = tv / threshold;
  if (ratio < 0.5) return COLORS.low;
  if (ratio < 0.8) return COLORS.mid;
  if (ratio < 1.0) return COLORS.high;
  return COLORS.hot;
}

function drawEmpty() {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = COLORS.textLight;
  ctx.font = '14px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('Test a distribution in Section 2, then click "Scan" to watch the window search.', W / 2, H / 2);
}

function drawHeatmapFrame(windows, revealedCells) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (!windows || windows.length === 0) { drawEmpty(); return; }

  const padL = 60, padR = 20, padT = 30, padB = 50;
  const nRows = windows.length;
  const nCols = windows[0].length;
  const cellW = Math.floor((W - padL - padR) / nCols);
  const cellH = Math.floor((H - padT - padB) / nRows);

  // Draw cells
  for (let r = 0; r < nRows; r++) {
    for (let c = 0; c < nCols; c++) {
      if (c >= windows[r].length) continue;
      const x = padL + c * cellW;
      const y = padT + r * cellH;

      const cellIdx = r * nCols + c;
      if (revealedCells != null && !revealedCells.has(cellIdx)) {
        ctx.fillStyle = COLORS.empty;
      } else {
        const tv = windows[r][c].tv;
        ctx.fillStyle = cellColor(tv, currentThreshold);
      }

      ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);

      // Show value in cell if large enough
      if (cellW > 35 && cellH > 18 && (revealedCells == null || revealedCells.has(cellIdx))) {
        ctx.fillStyle = COLORS.text;
        ctx.font = '9px system-ui';
        ctx.textAlign = 'center';
        ctx.fillText(windows[r][c].tv.toFixed(2), x + cellW / 2, y + cellH / 2 + 3);
      }
    }
  }

  // X axis: position k
  ctx.fillStyle = COLORS.textLight;
  ctx.font = '10px system-ui';
  ctx.textAlign = 'center';
  for (let c = 0; c < nCols; c++) {
    ctx.fillText(String(c), padL + c * cellW + cellW / 2, H - padB + 16);
  }
  ctx.fillText('Starting position k', padL + (nCols * cellW) / 2, H - 8);

  // Y axis: window length ell
  ctx.textAlign = 'right';
  for (let r = 0; r < nRows; r++) {
    const ell = windows[r][0].ell;
    ctx.fillText(`ell=${ell}`, padL - 6, padT + r * cellH + cellH / 2 + 3);
  }

  ctx.save();
  ctx.translate(12, padT + (nRows * cellH) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('Window length', 0, 0);
  ctx.restore();

  // Threshold line indicator
  ctx.fillStyle = COLORS.hot;
  ctx.font = 'bold 10px system-ui';
  ctx.textAlign = 'right';
  ctx.fillText(`threshold = ${currentThreshold.toFixed(2)}`, W - padR, padT - 10);

  // Legend
  const legX = padL, legY = padT - 14;
  const legW = 12;
  const legColors = [COLORS.low, COLORS.mid, COLORS.high, COLORS.hot];
  const legLabels = ['Low', '', '', 'Exceeds'];
  for (let i = 0; i < legColors.length; i++) {
    ctx.fillStyle = legColors[i];
    ctx.fillRect(legX + i * (legW + 3), legY, legW, legW);
  }
  ctx.fillStyle = COLORS.textLight;
  ctx.font = '9px system-ui';
  ctx.textAlign = 'left';
  ctx.fillText('Low', legX + 4 * (legW + 3) - legW, legY + 10);
}

async function runScan() {
  if (!currentWindows) return;
  scanning = true;
  scanAbort = false;

  const verdict = document.getElementById('hm-verdict');
  verdict.textContent = 'Scanning...';
  verdict.style.color = COLORS.text;

  const revealed = new Set();
  const nRows = currentWindows.length;
  const nCols = currentWindows[0].length;
  const speedMs = getSpeedMs();
  let pruneCell = null;

  for (let r = 0; r < nRows && !scanAbort; r++) {
    for (let c = 0; c < currentWindows[r].length && !scanAbort; c++) {
      const cellIdx = r * nCols + c;
      revealed.add(cellIdx);
      drawHeatmapFrame(currentWindows, revealed);

      if (currentWindows[r][c].tv >= currentThreshold) {
        pruneCell = currentWindows[r][c];
        // Flash the prune cell
        const padL = 60, padT = 30;
        const cellW = Math.floor((canvas.width - 80) / nCols);
        const cellH = Math.floor((canvas.height - 80) / nRows);
        const x = padL + c * cellW;
        const y = padT + r * cellH;
        ctx.strokeStyle = COLORS.pruneFlash;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, cellW, cellH);

        verdict.innerHTML = `<strong style="color:${COLORS.hot}">PRUNED!</strong> ` +
          `Window (ell=${pruneCell.ell}, k=${pruneCell.k}): ` +
          `test value = <b>${pruneCell.tv.toFixed(4)}</b> &ge; ${currentThreshold.toFixed(2)}`;
        canvas.className = 'pruned';
        scanning = false;
        return;
      }

      if (speedMs > 0) await delay(speedMs);
    }
  }

  if (!scanAbort) {
    drawHeatmapFrame(currentWindows, null);
    verdict.innerHTML = `<strong style="color:#4dac26">SURVIVES!</strong> ` +
      `No window exceeds the threshold at this resolution. The cascade would refine further.`;
    canvas.className = 'survives';
  }
  scanning = false;
}

function resetHeatmap() {
  scanAbort = true;
  scanning = false;
  canvas.className = '';
  const verdict = document.getElementById('hm-verdict');
  verdict.textContent = '';
  if (currentWindows) {
    drawHeatmapFrame(currentWindows, null);
  } else {
    drawEmpty();
  }
}

export function initHeatmap() {
  canvas = document.getElementById('hm-canvas');
  ctx = canvas.getContext('2d');

  drawEmpty();

  // Listen for data from the challenge section
  onHeatmapRequest((windows, threshold, d) => {
    currentWindows = windows;
    currentThreshold = threshold;
    currentD = d;
    drawHeatmapFrame(windows, null);
    document.getElementById('hm-verdict').textContent = 'Ready — click "Scan" to animate the window checking process.';
  });

  document.getElementById('hm-speed').addEventListener('input', updateSpeedLabel);
  updateSpeedLabel();

  document.getElementById('hm-scan').addEventListener('click', () => {
    if (scanning) { scanAbort = true; return; }
    if (!currentWindows) {
      const st = getState();
      currentWindows = scanWindows(st.a, 4, 20);
      currentThreshold = st.c_target;
    }
    runScan();
  });

  document.getElementById('hm-reset').addEventListener('click', resetHeatmap);
}
