// Section 4: Refinement Tree
// Shows how a d=4 parent splits into d=8 children, with pruning status.
// Green (surviving) nodes can be clicked to drill down to their children.

import { delay } from './animate.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const W = 900, H = 500;
const COLORS = {
  pruned: '#ef5350',
  survivor: '#81c784',
  pending: '#64b5f6',
  link: '#2a2d3a',
  text: '#e0e0e0',
  textLight: '#999',
  bg: '#1a1d27',
};

const ROOT_CONFIG = [8, 1, 1, 10];
const M = 20;
const C_TARGET = 1.30;

let svg, tooltip;
let treeStack = [];
let currentTreeData = null;

// ============================================================
// Math (same as challenge)
// ============================================================

function convolve(a) {
  const d = a.length;
  const c = new Float64Array(2 * d - 1);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++) c[i + j] += a[i] * a[j];
  return c;
}

function checkAsymmetry(a, d, m, c_target) {
  const half = d / 2;
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

// ============================================================
// Generate children: split each parent bin into 2
// ============================================================

function generateChildren(parent, m) {
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

function evaluateChild(child, m, c_target) {
  const d = child.length;
  const n_half = d / 2;
  const asym = checkAsymmetry(child, d, m, c_target);
  if (asym.pruned) {
    return { a: child, pruned: true, reason: 'asymmetry', bound: asym.bound };
  }
  const tv = computeTestValue(child, d, m);
  const corr = correction(m, n_half);
  const threshold = c_target + corr;
  if (tv.tv >= threshold) {
    return { a: child, pruned: true, reason: `window (ell=${tv.ell}, k=${tv.k})`, testValue: tv.tv, threshold };
  }
  return { a: child, pruned: false, reason: 'survives', testValue: tv.tv, threshold };
}

// ============================================================
// Build tree data
// ============================================================

function buildTreeData(rootConfig) {
  const children = generateChildren(rootConfig, M);
  const evaluated = children.map(c => evaluateChild(c, M, C_TARGET));

  const canonical = evaluated.filter(e => {
    const a = e.a, rev = [...a].reverse();
    for (let i = 0; i < a.length; i++) {
      if (a[i] < rev[i]) return true;
      if (a[i] > rev[i]) return false;
    }
    return true;
  });

  return {
    root: { a: rootConfig, d: rootConfig.length, pruned: false, reason: `survives at d=${rootConfig.length}` },
    children: canonical,
  };
}

// ============================================================
// SVG helpers
// ============================================================

function el(tag, attrs, text) {
  const e = document.createElementNS(SVG_NS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

// ============================================================
// Expand node (drill-down)
// ============================================================

function expandNode(node) {
  if (node.a.length >= 16) {
    // At d>=32 the tree is too large to enumerate interactively
    const infoDiv = document.createElement('div');
    infoDiv.style.cssText = 'padding:12px;color:var(--gold);font-size:13px;';
    infoDiv.textContent = `At d=${node.a.length * 2}, the number of children is too large to display interactively. The real algorithm uses GPU acceleration for this scale.`;
    const container = document.getElementById('tree-container');
    container.insertBefore(infoDiv, container.querySelector('#tree-svg'));
    setTimeout(() => infoDiv.remove(), 4000);
    return;
  }

  treeStack.push(currentTreeData);
  currentTreeData = buildTreeData(node.a);
  renderTree(document.getElementById('tree-container'), currentTreeData);
}

// ============================================================
// SVG rendering
// ============================================================

function renderTree(container, treeData) {
  svg = container.querySelector('#tree-svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = '';
  tooltip = document.getElementById('tree-tooltip');

  const { root, children } = treeData;
  const survivors = children.filter(c => !c.pruned);
  const pruned = children.filter(c => c.pruned);
  const total = children.length;

  const rootX = W / 2, rootY = 60;
  const childY = 280;
  const maxVisible = Math.min(total, 60);
  const spacing = Math.min(14, (W - 80) / maxVisible);
  const startX = W / 2 - (maxVisible * spacing) / 2;

  // Sample proportionally from survivors and pruned so both groups are visible
  function sampleEvenly(arr, n) {
    if (arr.length <= n) return arr.slice();
    const step = arr.length / n;
    return Array.from({ length: n }, (_, i) => arr[Math.floor(i * step)]);
  }
  const survShow = Math.round((survivors.length / Math.max(total, 1)) * maxVisible);
  const prunShow = maxVisible - survShow;
  const sorted = [...sampleEvenly(survivors, survShow), ...sampleEvenly(pruned, prunShow)];

  // Back button (when navigated into a subtree)
  if (treeStack.length > 0) {
    const backG = el('g', { class: 'tree-node' });
    backG.appendChild(el('rect', { x: 10, y: 10, width: 64, height: 24, rx: 4, fill: '#2a2d3a', stroke: '#4ea8de', 'stroke-width': 1 }));
    backG.appendChild(el('text', { x: 42, y: 26, 'text-anchor': 'middle', 'font-size': '11px', fill: '#4ea8de' }, '← Back'));
    backG.style.cursor = 'pointer';
    backG.addEventListener('click', () => {
      currentTreeData = treeStack.pop();
      renderTree(container, currentTreeData);
    });
    svg.appendChild(backG);
  }

  // Draw links
  for (let i = 0; i < sorted.length; i++) {
    const cx = startX + i * spacing + spacing / 2;
    svg.appendChild(el('line', {
      x1: rootX, y1: rootY + 12,
      x2: cx, y2: childY - 8,
      stroke: COLORS.link, 'stroke-width': 0.8,
    }));
  }

  // Root node
  const rootG = el('g', { class: 'tree-node', 'data-idx': 'root' });
  rootG.appendChild(el('circle', { cx: rootX, cy: rootY, r: 14, fill: COLORS.survivor }));
  rootG.appendChild(el('text', {
    x: rootX, y: rootY + 4,
    'text-anchor': 'middle', 'font-size': '9px', fill: '#fff', 'font-weight': '700',
  }, `[${root.a.join(',')}]`));
  svg.appendChild(rootG);

  // Root label
  svg.appendChild(el('text', {
    x: rootX, y: rootY - 24,
    'text-anchor': 'middle', 'font-size': '12px', fill: COLORS.text, 'font-weight': '600',
  }, `Parent: d=${root.d}, m=${M}`));

  // Children
  for (let i = 0; i < sorted.length; i++) {
    const child = sorted[i];
    const cx = startX + i * spacing + spacing / 2;
    const color = child.pruned ? COLORS.pruned : COLORS.survivor;
    const nodeG = el('g', { class: 'tree-node', 'data-idx': String(i) });
    nodeG.appendChild(el('circle', { cx, cy: childY, r: 5, fill: color }));
    svg.appendChild(nodeG);

    nodeG.addEventListener('mouseenter', (ev) => showTooltip(ev, child));
    nodeG.addEventListener('mouseleave', hideTooltip);

    // Click to expand surviving nodes
    if (!child.pruned) {
      nodeG.style.cursor = 'pointer';
      nodeG.addEventListener('click', (ev) => {
        ev.stopPropagation();
        hideTooltip();
        expandNode(child);
      });
    }
  }

  // Ellipsis if truncated
  if (total > maxVisible) {
    svg.appendChild(el('text', {
      x: startX + maxVisible * spacing + 10, y: childY + 4,
      'font-size': '12px', fill: COLORS.textLight,
    }, `...+${total - maxVisible} more`));
  }

  // Summary stats
  svg.appendChild(el('text', {
    x: W / 2, y: childY + 50,
    'text-anchor': 'middle', 'font-size': '12px', fill: COLORS.text,
  }, `d=${root.d * 2} children: ${total} total`));

  const prunedCount = pruned.length;
  const survCount = survivors.length;
  const prunePct = ((prunedCount / total) * 100).toFixed(1);

  svg.appendChild(el('text', {
    x: W / 2, y: childY + 68,
    'text-anchor': 'middle', 'font-size': '12px', fill: COLORS.textLight,
  }, `${prunedCount} pruned (${prunePct}%) | ${survCount} survive`));

  // Click-to-expand hint (only show when survivors exist)
  if (survCount > 0) {
    svg.appendChild(el('text', {
      x: W / 2, y: childY + 86,
      'text-anchor': 'middle', 'font-size': '11px', fill: '#4ea8de',
    }, 'Click a green node to drill down to its children'));
  }

  // Color legend
  const legY = childY + 110;
  svg.appendChild(el('circle', { cx: W / 2 - 80, cy: legY, r: 5, fill: COLORS.pruned }));
  svg.appendChild(el('text', {
    x: W / 2 - 70, y: legY + 4, 'font-size': '11px', fill: COLORS.textLight,
  }, 'Pruned'));
  svg.appendChild(el('circle', { cx: W / 2 + 20, cy: legY, r: 5, fill: COLORS.survivor }));
  svg.appendChild(el('text', {
    x: W / 2 + 30, y: legY + 4, 'font-size': '11px', fill: COLORS.textLight,
  }, 'Survives (click to expand)'));

  // Root tooltip
  rootG.addEventListener('mouseenter', (ev) => showTooltip(ev, {
    a: root.a, pruned: false, reason: root.reason,
  }));
  rootG.addEventListener('mouseleave', hideTooltip);
}

function showTooltip(ev, node) {
  tooltip.classList.remove('hidden');

  let html = `<canvas width="180" height="50" style="display:block;width:100%;border-radius:4px;margin-bottom:6px;" id="tt-canvas"></canvas>`;
  html += `<div style="font-weight:600;margin-bottom:4px;">[${node.a.join(', ')}]</div>`;
  if (node.pruned) {
    html += `<div style="color:${COLORS.pruned};">Pruned: ${node.reason}</div>`;
    if (node.testValue != null) {
      html += `<div style="font-size:11px;color:#888;">Test value: ${node.testValue.toFixed(4)} (threshold: ${node.threshold.toFixed(4)})</div>`;
    }
    if (node.bound != null) {
      html += `<div style="font-size:11px;color:#888;">Asymmetry bound: ${node.bound.toFixed(4)}</div>`;
    }
  } else {
    html += `<div style="color:${COLORS.survivor};">${node.reason}</div>`;
    if (node.testValue != null) {
      html += `<div style="font-size:11px;color:#888;">Test value: ${node.testValue.toFixed(4)} (threshold: ${node.threshold.toFixed(4)})</div>`;
    }
    if (!node.pruned && node.a.length < 16) {
      html += `<div style="font-size:11px;color:#4ea8de;margin-top:4px;">Click to expand children</div>`;
    }
  }

  tooltip.innerHTML = html;

  const ttCanvas = tooltip.querySelector('#tt-canvas');
  if (ttCanvas) {
    const ttCtx = ttCanvas.getContext('2d');
    ttCtx.fillStyle = '#1a1d27';
    ttCtx.fillRect(0, 0, 180, 50);
    const max = Math.max(...node.a);
    const barW = 180 / node.a.length;
    for (let i = 0; i < node.a.length; i++) {
      const bh = max > 0 ? (node.a[i] / max) * 44 : 0;
      ttCtx.fillStyle = node.pruned ? '#e57373' : '#64b5f6';
      ttCtx.fillRect(i * barW + 1, 50 - bh, barW - 2, bh);
    }
  }

  tooltip.style.left = (ev.clientX + 12) + 'px';
  tooltip.style.top = (ev.clientY - 20) + 'px';
}

function hideTooltip() {
  tooltip.classList.add('hidden');
}

export function initTree() {
  currentTreeData = buildTreeData(ROOT_CONFIG);
  treeStack = [];
  const container = document.getElementById('tree-container');
  renderTree(container, currentTreeData);
}
