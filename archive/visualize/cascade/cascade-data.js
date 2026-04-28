// Embedded data for cascade visualizations.
// Sources: data/cpu_cascade_20260325_162306.json (c=1.30)
//          data/cpu_cascade_20260319_201644.json (c=1.40)
//          proof/lower_bound_proof.tex lines 202-209

export const CASCADE_130 = {
  c_target: 1.30,
  m: 20,
  n_half: 2,
  d0: 4,
  total_time: 247.6,
  proven_at: 'L4',
  levels: [
    { label: 'L0', d: 4,  parents: 891,       children: 891,          survivors: 376,    pruned_asym: 190, pruned_test: 325, elapsed: 0.008 },
    { label: 'L1', d: 8,  parents: 376,        children: 238792,      survivors: 13540,  elapsed: 22.9 },
    { label: 'L2', d: 16, parents: 13540,      children: 96448738,    survivors: 116646, elapsed: 4.9 },
    { label: 'L3', d: 32, parents: 116646,     children: 6859053079,  survivors: 256,    elapsed: 208.6 },
    { label: 'L4', d: 64, parents: 256,        children: 62159872,    survivors: 0,      elapsed: 9.8 },
  ],
};

export const CASCADE_140 = {
  c_target: 1.40,
  m: 20,
  n_half: 2,
  d0: 4,
  total_time: 252272,
  proven_at: 'L5',
  levels: [
    { label: 'L0', d: 4,   parents: 891,        children: 891,              survivors: 345,      elapsed: 0.36 },
    { label: 'L1', d: 8,   parents: 345,         children: 228405,           survivors: 48443,    elapsed: 12.0 },
    { label: 'L2', d: 16,  parents: 48443,       children: 341204198,        survivors: 7499382,  elapsed: 28.7 },
    { label: 'L3', d: 32,  parents: 7499382,     children: 426897334375,     survivors: 147279894, elapsed: 56769 },
    { label: 'L4', d: 64,  parents: 147279894,   children: 38717374906892,   survivors: 76829,    elapsed: 251301 },
    { label: 'L5', d: 128, parents: 76829,       children: 69763661824,      survivors: 0,        elapsed: 822.5 },
  ],
};

export const BOUND_HISTORY = [
  { year: 2002, value: 1.5708, label: 'Schinzel-Schmidt', detail: 'π/2', type: 'upper' },
  { year: 2004, value: 1.1828, label: 'Martin-O\'Bryant', type: 'lower' },
  { year: 2017, value: 1.28,   label: 'Cloninger-Steinerberger', type: 'lower' },
  { year: 2026, value: 1.5029, label: 'TTT Discover', type: 'upper' },
  { year: 2026, value: 1.40,   label: 'This work', type: 'lower', highlight: true },
];

// Sample d=4 configurations for the challenge game (m=20)
// Picked to illustrate different pruning outcomes at c_target=1.30
export const SAMPLE_CONFIGS = [
  { a: [5, 5, 5, 5], note: 'Uniform' },
  { a: [10, 0, 0, 10], note: 'Split peaks' },
  { a: [0, 0, 0, 20], note: 'Concentrated' },
  { a: [15, 2, 2, 1], note: 'Lopsided left' },
  { a: [3, 7, 7, 3], note: 'Center-heavy' },
  { a: [4, 6, 6, 4], note: 'Gentle center' },
  { a: [6, 4, 4, 6], note: 'Edge-heavy' },
  { a: [2, 8, 8, 2], note: 'Strong center' },
];

// Precomputed refinement tree: a d=4 survivor and its d=8 children
// Parent: [3, 7, 7, 3] (symmetric, survives L0 at c=1.30)
// Each child splits parent bins: parent[i] -> child[2i] + child[2i+1]
// with child[2i] + child[2i+1] = 2 * parent[i] (scaled to child m)
export const TREE_DATA = {
  root: {
    d: 4, m: 20, a: [3, 7, 7, 3],
    pruned: false, reason: 'survives',
  },
  children: [],  // will be populated by generate_tree_data.py
};

export function fmtLarge(n) {
  if (n >= 1e12) return (n / 1e12).toFixed(2) + 'T';
  if (n >= 1e9)  return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6)  return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3)  return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

export function fmtTime(secs) {
  if (secs < 1) return (secs * 1000).toFixed(0) + 'ms';
  if (secs < 60) return secs.toFixed(1) + 's';
  if (secs < 3600) return (secs / 60).toFixed(1) + 'min';
  return (secs / 3600).toFixed(1) + 'hr';
}
