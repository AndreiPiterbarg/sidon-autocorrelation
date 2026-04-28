// Section 2: "Can You Beat the Algorithm?" — interactive game

let D = 4, M = 20;
const PAD = { top: 24, right: 16, bottom: 36, left: 46 };

const COL = {
  leftBin:    '#64b5f6',
  rightBin:   '#4ea8de',
  asymDom:    '#e57373',
  asymSub:    '#555',
  convNormal: '#4ea8de',
  convPruned: '#e57373',
  threshold:  '#ef5350',
  grid:       '#2a2d3a',
  axis:       '#777',
  center:     '#444',
  text:       '#e0e0e0',
  textLight:  '#999',
  white:      '#e0e0e0',
  green:      '#81c784',
};

// ============================================================
// Math (ported from demo/app.js)
// ============================================================

function convolve(a) {
  const d = a.length;
  const c = new Float64Array(2 * d - 1);
  for (let i = 0; i < d; i++) {
    if (a[i] === 0) continue;
    for (let j = 0; j < d; j++) c[i + j] += a[i] * a[j];
  }
  return c;
}

function normalizeConv(c, d, m) {
  const factor = (2 * d) / (m * m);
  return Array.from(c, v => v * factor);
}

function checkAsymmetry(a, d, m, c_target) {
  if (m === 0) return { n1: 0, n2: 0, bound: 0, pruned: false };
  const half = Math.floor(d / 2);
  let n1 = 0;
  for (let i = 0; i < half; i++) n1 += a[i];
  const n2 = m - n1;
  const bound = 4 * Math.max(n1, n2) ** 2 / (m * m);
  return { n1, n2, bound, pruned: bound >= c_target };
}

function computeTestValue(a, d, m) {
  const n_half = d / 2;
  const scale = (4 * n_half) / m;
  const aScaled = a.map(v => v * scale);
  const convLen = 2 * d - 1;
  const conv = new Float64Array(convLen);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      conv[i + j] += aScaled[i] * aScaled[j];

  const cum = new Float64Array(convLen);
  cum[0] = conv[0];
  for (let k = 1; k < convLen; k++) cum[k] = cum[k - 1] + conv[k];

  let best = 0, bestEll = 0, bestK = 0;
  for (let ell = 2; ell <= 2 * d; ell++) {
    const nConv = ell - 1;
    for (let sLo = 0; sLo <= convLen - nConv; sLo++) {
      const sHi = sLo + nConv - 1;
      let ws = cum[sHi];
      if (sLo > 0) ws -= cum[sLo - 1];
      const tv = ws / (4 * n_half * ell);
      if (tv > best) { best = tv; bestEll = ell; bestK = sLo; }
    }
  }
  return { tv: best, ell: bestEll, k: bestK };
}

function correction(m, n_half) {
  return (4 * n_half / 2) * (2 / m + 1 / (m * m));
}

export function scanWindows(a, d, m) {
  const n_half = d / 2;
  const scale = (4 * n_half) / m;
  const aScaled = a.map(v => v * scale);
  const convLen = 2 * d - 1;
  const conv = new Float64Array(convLen);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      conv[i + j] += aScaled[i] * aScaled[j];

  const cum = new Float64Array(convLen);
  cum[0] = conv[0];
  for (let k = 1; k < convLen; k++) cum[k] = cum[k - 1] + conv[k];

  const results = [];
  for (let ell = 2; ell <= 2 * d; ell++) {
    const nConvVals = ell - 1;
    const row = [];
    for (let sLo = 0; sLo <= convLen - nConvVals; sLo++) {
      const sHi = sLo + nConvVals - 1;
      let ws = cum[sHi];
      if (sLo > 0) ws -= cum[sLo - 1];
      const tv = ws / (4 * n_half * ell);
      row.push({ ell, k: sLo, tv });
    }
    results.push(row);
  }
  return results;
}

// ============================================================
// Child generation for refinement display
// ============================================================

function generateChildrenFromParent(parent, m) {
  const d = parent.length;
  const splits = parent.map(v => {
    const opts = [];
    for (let a = 0; a <= v; a++) opts.push([a, v - a]);
    return opts;
  });

  const children = [];
  const indices = new Array(d).fill(0);
  const maxes = splits.map(s => s.length);

  while (true) {
    const child = new Array(d * 2);
    for (let i = 0; i < d; i++) {
      child[2 * i] = splits[i][indices[i]][0];
      child[2 * i + 1] = splits[i][indices[i]][1];
    }
    children.push(child);

    let carry = d - 1;
    while (carry >= 0) {
      indices[carry]++;
      if (indices[carry] < maxes[carry]) break;
      indices[carry] = 0;
      carry--;
    }
    if (carry < 0) break;
  }
  return children;
}

function evaluateChildSimple(child, childD, m, c_target) {
  const asym = checkAsymmetry(child, childD, m, c_target);
  if (asym.pruned) return { pruned: true };
  const n_half = childD / 2;
  const corr = correction(m, n_half);
  const tvResult = computeTestValue(child, childD, m);
  if (tvResult.tv >= c_target + corr) return { pruned: true };
  return { pruned: false };
}

function canonicalFilter(children) {
  return children.filter(a => {
    const rev = [...a].reverse();
    for (let i = 0; i < a.length; i++) {
      if (a[i] < rev[i]) return true;
      if (a[i] > rev[i]) return false;
    }
    return true;
  });
}

// ============================================================
// State
// ============================================================

const state = {
  a: [5, 5, 5, 5],
  c_target: 1.30,
  tested: false,
  result: null,
  levelReached: -1,
};

let drag = { active: false, bar: -1, startY: 0, startVal: 0 };
let heatmapCallback = null;

const HINTS = [
  'Symmetric distributions (left half = right half) avoid the asymmetry prune.',
  'Spread mass evenly — concentrated peaks create high autoconvolution peaks.',
  'Try [8, 1, 1, 10] — it survives at d=4! But can its children survive at d=8?',
  'Lower the target c to make survival easier — try c = 1.15.',
  'The algorithm always wins: at fine enough resolution, every distribution is pruned.',
];
let hintIdx = 0;

export function onHeatmapRequest(cb) { heatmapCallback = cb; }
export function getState() { return state; }

// ============================================================
// Array helpers
// ============================================================

function setBar(idx, newVal) {
  const a = state.a;
  newVal = Math.max(0, Math.min(M, Math.round(newVal)));
  let delta = newVal - a[idx];
  if (delta === 0) return;
  a[idx] = newVal;
  let remaining = -delta;
  let pass = 0;
  while (remaining !== 0 && pass < M + D) {
    if (remaining > 0) {
      let bi = -1, bv = Infinity;
      for (let i = 0; i < D; i++) if (i !== idx && a[i] < bv) { bv = a[i]; bi = i; }
      if (bi < 0) break;
      a[bi]++; remaining--;
    } else {
      let bi = -1, bv = -Infinity;
      for (let i = 0; i < D; i++) if (i !== idx && a[i] > 0 && a[i] > bv) { bv = a[i]; bi = i; }
      if (bi < 0) break;
      a[bi]--; remaining++;
    }
    pass++;
  }
}

// ============================================================
// Canvas drawing
// ============================================================

function plotArea(canvas) {
  const W = canvas.width, H = canvas.height;
  return { W, H, x0: PAD.left, y0: PAD.top, pw: W - PAD.left - PAD.right, ph: H - PAD.top - PAD.bottom };
}

function drawGrid(ctx, area, yMax) {
  ctx.strokeStyle = COL.grid; ctx.lineWidth = 0.5;
  ctx.fillStyle = COL.textLight; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = (yMax * i) / 4;
    const y = area.y0 + area.ph - (val / yMax) * area.ph;
    ctx.beginPath(); ctx.moveTo(area.x0, y); ctx.lineTo(area.x0 + area.pw, y); ctx.stroke();
    ctx.fillText(Number.isInteger(val) ? val : val.toFixed(2), area.x0 - 4, y + 3.5);
  }
}

function drawHistogram(canvas) {
  const ctx = canvas.getContext('2d');
  const area = plotArea(canvas);
  const { x0, y0, pw, ph, W, H } = area;
  ctx.clearRect(0, 0, W, H);

  const a = state.a;
  const asym = checkAsymmetry(a, D, M, state.c_target);
  const half = Math.floor(D / 2);
  const yMax = M;
  const barW = pw / D;

  drawGrid(ctx, area, yMax);

  for (let i = 0; i < D; i++) {
    const bh = (a[i] / yMax) * ph;
    const bx = x0 + i * barW;
    const by = y0 + ph - bh;
    const isLeft = i < half;

    let fill;
    if (asym.pruned) {
      const isDom = (asym.n1 >= asym.n2) ? isLeft : !isLeft;
      fill = isDom ? COL.asymDom : COL.asymSub;
    } else {
      fill = isLeft ? COL.leftBin : COL.rightBin;
    }

    ctx.fillStyle = fill;
    ctx.fillRect(bx + 2, by, barW - 4, bh);

    if (a[i] > 0 && bh > 15) {
      ctx.fillStyle = COL.white; ctx.font = 'bold 11px system-ui'; ctx.textAlign = 'center';
      ctx.fillText(a[i], bx + barW / 2, by + 14);
    }
  }

  // Divider
  const divX = x0 + (half / D) * pw;
  ctx.strokeStyle = '#555'; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(divX, y0); ctx.lineTo(divX, y0 + ph); ctx.stroke();
  ctx.setLineDash([]);

  // Axes
  ctx.strokeStyle = COL.axis; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y0 + ph); ctx.lineTo(x0 + pw, y0 + ph); ctx.stroke();

  // X labels
  ctx.fillStyle = COL.textLight; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  ctx.fillText('-1/4', x0, H - 6);
  ctx.fillText('0', x0 + pw / 2, H - 6);
  ctx.fillText('1/4', x0 + pw, H - 6);

  // Half labels
  ctx.font = '11px system-ui'; ctx.textAlign = 'center';
  ctx.fillStyle = COL.leftBin;
  ctx.fillText(`n1=${asym.n1}`, x0 + pw / 4, y0 - 6);
  ctx.fillStyle = COL.rightBin;
  ctx.fillText(`n2=${asym.n2}`, x0 + 3 * pw / 4, y0 - 6);
}

function drawConvolution(canvas, tvResult) {
  const ctx = canvas.getContext('2d');
  const area = plotArea(canvas);
  const { x0, y0, pw, ph, W, H } = area;
  ctx.clearRect(0, 0, W, H);

  const a = state.a;
  const c = convolve(a);
  const norm = normalizeConv(c, D, M);
  const maxVal = Math.max(...norm);
  const yMax = Math.max(maxVal * 1.1, 0.5);
  const n = norm.length;

  const xc = i => x0 + (i / (n - 1)) * pw;
  const yc = v => y0 + ph - (v / yMax) * ph;

  drawGrid(ctx, area, yMax);

  // Curve (always neutral color — pruning is decided by windowed test value, not raw max)
  ctx.globalAlpha = 0.12; ctx.fillStyle = COL.convNormal;
  ctx.beginPath(); ctx.moveTo(xc(0), yc(0));
  for (let i = 0; i < n; i++) ctx.lineTo(xc(i), yc(norm[i]));
  ctx.lineTo(xc(n - 1), yc(0)); ctx.closePath(); ctx.fill();
  ctx.globalAlpha = 1;

  ctx.strokeStyle = COL.convNormal; ctx.lineWidth = 2; ctx.lineJoin = 'round';
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xc(i), y = yc(norm[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Highlight the best window if provided
  if (tvResult) {
    const n_half = D / 2;
    const convLen = 2 * D - 1;
    // The window spans sLo..sLo+ell-2 in convolution space (nConv = ell-1 points)
    const nConv = tvResult.ell - 1;
    const sLo = tvResult.k;
    const sHi = sLo + nConv - 1;
    if (nConv >= 1 && sHi < convLen) {
      const wx0 = xc(sLo), wx1 = xc(Math.min(sHi, n - 1));
      ctx.fillStyle = 'rgba(100,181,246,0.18)';
      ctx.fillRect(wx0, y0, wx1 - wx0 + (pw / (n - 1)), ph);
      ctx.strokeStyle = '#4ea8de'; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
      ctx.strokeRect(wx0, y0, wx1 - wx0 + (pw / (n - 1)), ph);
      ctx.setLineDash([]);

      // Windowed average line
      const n_half_loc = D / 2;
      // tv is in computeTestValue units; convert back to normalizeConv y-axis:
      // normalizeConv factor = 2D/M², computeTestValue factor = (4n_half/M)²/(4n_half*ell) per point
      // The displayed average of norm values over the window:
      let wsum = 0;
      for (let i = sLo; i <= sHi; i++) wsum += norm[i];
      const wavg = wsum / nConv;
      const yWavg = yc(wavg);
      ctx.strokeStyle = COL.convPruned; ctx.lineWidth = 1.5; ctx.setLineDash([6, 3]);
      ctx.beginPath(); ctx.moveTo(wx0, yWavg); ctx.lineTo(wx1 + pw / (n - 1), yWavg); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // L∞ label (informational — not the pruning criterion)
  const maxIdx = norm.indexOf(maxVal);
  const mx = xc(maxIdx), my = yc(maxVal);
  ctx.beginPath(); ctx.arc(mx, my, 4, 0, 2 * Math.PI);
  ctx.fillStyle = COL.convNormal; ctx.fill();
  ctx.fillStyle = COL.textLight; ctx.font = '10px system-ui';
  ctx.textAlign = maxIdx < n / 2 ? 'left' : 'right';
  ctx.fillText(`‖f★f‖∞ = ${maxVal.toFixed(3)}`, mx + (maxIdx < n / 2 ? 7 : -7), my - 6);

  // Axes
  ctx.strokeStyle = COL.axis; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y0 + ph); ctx.lineTo(x0 + pw, y0 + ph); ctx.stroke();

  ctx.fillStyle = COL.textLight; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  ctx.fillText('-1/2', x0, H - 6);
  ctx.fillText('0', x0 + pw / 2, H - 6);
  ctx.fillText('1/2', x0 + pw, H - 6);

  ctx.fillStyle = COL.textLight; ctx.font = '9px system-ui'; ctx.textAlign = 'left';
  ctx.fillText('f ★ f', x0 + 2, y0 + 11);
}

// ============================================================
// Result panel
// ============================================================

function showResult(asym, tvResult) {
  const panel = document.getElementById('ch-result');
  const n_half = D / 2;
  const corr = correction(M, n_half);
  const threshold = state.c_target + corr;

  if (asym.pruned) {
    panel.className = 'result-panel result-asym';
    panel.innerHTML = `
      <strong>Pruned at d=${D}: asymmetry.</strong>
      The ${asym.n1 >= asym.n2 ? 'left' : 'right'} half holds
      ${Math.max(asym.n1, asym.n2)} of ${M} mass units.
      Asymmetry bound = <b>${asym.bound.toFixed(3)}</b>
      &ge; ${state.c_target.toFixed(2)}.<br>
      <em>The heavier half alone forces ‖f★f‖<sub>∞</sub> &ge; c.</em>
    `;
    return;
  }

  if (tvResult.tv >= threshold) {
    panel.className = 'result-panel result-pruned';
    panel.innerHTML = `
      <strong>Pruned at d=${D}: windowed test value.</strong>
      Best window (ell=${tvResult.ell}, k=${tvResult.k}):
      test value = <b>${tvResult.tv.toFixed(4)}</b>
      &ge; threshold <b>${threshold.toFixed(4)}</b>
      (= c + correction = ${state.c_target.toFixed(2)} + ${corr.toFixed(4)}).<br>
      <em>The windowed average proves ‖f★f‖<sub>∞</sub> &ge; c for this distribution.</em>
    `;
    return;
  }

  // Survives — show refinement
  const childD = D * 2;
  const rawChildren = generateChildrenFromParent(state.a, M);
  const canonical = canonicalFilter(rawChildren);
  const tested = canonical.map(c => evaluateChildSimple(c, childD, M, state.c_target));
  const childSurvivors = tested.filter(c => !c.pruned).length;
  const childTotal = canonical.length;
  const childPrunePct = childTotal > 0
    ? ((1 - childSurvivors / childTotal) * 100).toFixed(1)
    : '100.0';

  panel.className = 'result-panel result-survives';
  panel.innerHTML = `
    <strong>Survives at d=${D}.</strong>
    Windowed test value = <b>${tvResult.tv.toFixed(4)}</b>
    &lt; threshold <b>${threshold.toFixed(4)}</b>.<br>
    The algorithm refines to d=${childD}: tested <b>${childTotal}</b> children,
    <b>${childPrunePct}%</b> pruned, <b>${childSurvivors}</b> survive.
    ${childSurvivors === 0
      ? `<br><strong style="color:var(--green-light);">All children pruned at d=${childD} — no escape at finer resolution.</strong>`
      : `<br><em>${childSurvivors} survivors would be refined further at d=${childD * 2}…</em>`}
  `;
}

function clearResult() {
  const panel = document.getElementById('ch-result');
  panel.className = 'result-panel';
  panel.innerHTML = '<em>Click "Test My Distribution" to see the result.</em>';
}

// ============================================================
// Interaction
// ============================================================

function render() {
  drawHistogram(document.getElementById('ch-hist'));
  if (state.tested) {
    const tvResult = computeTestValue(state.a, D, M);
    drawConvolution(document.getElementById('ch-conv'), tvResult);
  } else {
    const canvas = document.getElementById('ch-conv');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#1a1d27';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#777';
    ctx.font = '13px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Click "Test My Distribution" to compute the autoconvolution.', canvas.width / 2, canvas.height / 2);
  }
}

function initUniform() {
  state.a = new Array(D).fill(0);
  const perBin = Math.floor(M / D);
  const rem = M - perBin * D;
  for (let i = 0; i < D; i++) state.a[i] = perBin + (i < rem ? 1 : 0);
}

function setPreset(name) {
  state.tested = false;
  clearResult();
  switch (name) {
    case 'uniform':
      initUniform();
      break;
    case 'random':
      state.a = new Array(D).fill(0);
      for (let i = 0; i < M; i++) state.a[Math.floor(Math.random() * D)]++;
      break;
    case 'concentrated':
      state.a = new Array(D).fill(0);
      state.a[Math.floor(D / 2)] = M;
      break;
    case 'split':
      state.a = new Array(D).fill(0);
      state.a[0] = Math.floor(M / 2);
      state.a[D - 1] = M - state.a[0];
      break;
    case 'lopsided':
      state.a = new Array(D).fill(0);
      let rem2 = M;
      for (let i = 0; i < D - 1 && rem2 > 0; i++) {
        const v = Math.max(1, Math.floor((D - i) * 0.6));
        state.a[i] = Math.min(v, rem2);
        rem2 -= state.a[i];
      }
      state.a[D - 1] += rem2;
      break;
  }
  render();
}

function testDistribution() {
  state.tested = true;
  const asym = checkAsymmetry(state.a, D, M, state.c_target);
  const tvResult = computeTestValue(state.a, D, M);
  drawConvolution(document.getElementById('ch-conv'), tvResult);
  showResult(asym, tvResult);

  if (heatmapCallback) {
    const windows = scanWindows(state.a, D, M);
    heatmapCallback(windows, state.c_target, D);
  }
}

export function initChallenge() {
  const histCanvas = document.getElementById('ch-hist');

  initUniform();
  render();
  clearResult();

  // d select
  document.getElementById('ch-d-slider').addEventListener('change', e => {
    D = parseInt(e.target.value);
    initUniform();
    state.tested = false;
    clearResult();
    render();
  });

  // c_target slider
  document.getElementById('ch-c-slider').addEventListener('input', e => {
    state.c_target = parseFloat(e.target.value);
    document.getElementById('ch-c-val').textContent = state.c_target.toFixed(2);
    state.tested = false;
    clearResult();
    render();
  });

  // Presets
  document.querySelectorAll('[data-preset]').forEach(btn => {
    btn.addEventListener('click', () => setPreset(btn.dataset.preset));
  });

  // Test button
  document.getElementById('ch-test').addEventListener('click', testDistribution);

  // Hint button
  document.getElementById('ch-hint').addEventListener('click', () => {
    const panel = document.getElementById('ch-result');
    panel.className = 'result-panel';
    panel.innerHTML = `<em style="color:var(--gold);">&#128161; ${HINTS[hintIdx % HINTS.length]}</em>`;
    hintIdx++;
  });

  // Drag interaction
  histCanvas.addEventListener('mousedown', e => {
    const bar = canvasBarIdx(histCanvas, e.clientX);
    if (bar < 0) return;
    drag = { active: true, bar, startY: e.clientY, startVal: state.a[bar] };
    histCanvas.style.cursor = 'grabbing';
    e.preventDefault();
  });

  document.addEventListener('mousemove', e => {
    if (!drag.active) return;
    const area = plotArea(histCanvas);
    const rect = histCanvas.getBoundingClientRect();
    const scaleY = histCanvas.height / rect.height;
    const dy = (drag.startY - e.clientY) * scaleY;
    setBar(drag.bar, drag.startVal + Math.round(dy * M / area.ph));
    state.tested = false;
    clearResult();
    render();
  });

  document.addEventListener('mouseup', () => {
    if (!drag.active) return;
    drag.active = false;
    histCanvas.style.cursor = 'ns-resize';
  });

  // Touch
  histCanvas.addEventListener('touchstart', e => {
    const t = e.touches[0];
    const bar = canvasBarIdx(histCanvas, t.clientX);
    if (bar < 0) return;
    drag = { active: true, bar, startY: t.clientY, startVal: state.a[bar] };
    e.preventDefault();
  }, { passive: false });

  document.addEventListener('touchmove', e => {
    if (!drag.active) return;
    const t = e.touches[0];
    const area = plotArea(histCanvas);
    const rect = histCanvas.getBoundingClientRect();
    const scaleY = histCanvas.height / rect.height;
    const dy = (drag.startY - t.clientY) * scaleY;
    setBar(drag.bar, drag.startVal + Math.round(dy * M / area.ph));
    state.tested = false;
    clearResult();
    render();
    e.preventDefault();
  }, { passive: false });

  document.addEventListener('touchend', () => { drag.active = false; });

  histCanvas.addEventListener('mousemove', e => {
    if (drag.active) return;
    histCanvas.style.cursor = canvasBarIdx(histCanvas, e.clientX) >= 0 ? 'ns-resize' : 'default';
  });
}

function canvasBarIdx(canvas, clientX) {
  const rect = canvas.getBoundingClientRect();
  const { pw } = plotArea(canvas);
  const scaleX = canvas.width / rect.width;
  const cx = (clientX - rect.left) * scaleX;
  if (cx < PAD.left || cx > PAD.left + pw) return -1;
  return Math.max(0, Math.min(D - 1, Math.floor((cx - PAD.left) / pw * D)));
}
