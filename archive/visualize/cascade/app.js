// Main orchestrator: scroll observer, section lifecycle

import { initFunnel } from './viz-funnel.js';
import { initChallenge } from './viz-challenge.js';
import { initHeatmap } from './viz-heatmap.js';
import { initTree } from './viz-tree.js';
import { initTimeline } from './viz-timeline.js';

const sections = [
  { id: 'funnel',    init: initFunnel },
  { id: 'challenge', init: initChallenge },
  { id: 'heatmap',   init: initHeatmap },
  { id: 'tree',      init: initTree },
  { id: 'timeline',  init: initTimeline },
];

const initialized = new Set();

const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (entry.isIntersecting && !initialized.has(entry.target.id)) {
      const sec = sections.find(s => s.id === entry.target.id);
      if (sec) {
        sec.init();
        initialized.add(sec.id);
      }
    }
  }
}, { threshold: 0.15 });

for (const sec of sections) {
  const el = document.getElementById(sec.id);
  if (el) observer.observe(el);
}
