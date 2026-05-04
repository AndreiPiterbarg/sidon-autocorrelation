// Main orchestrator: scroll observer, section lifecycle

import { initProblem } from './viz-problem.js';
import { initRelax } from './viz-relax.js';
import { initMoments } from './viz-moments.js';
import { initHierarchy } from './viz-hierarchy.js';
import { initPlayground } from './viz-playground.js';
import { initFarkas } from './viz-farkas.js';
import { initConvergence } from './viz-convergence.js';

const sections = [
  { id: 'problem',     init: initProblem },
  { id: 'relax',       init: initRelax },
  { id: 'moments',     init: initMoments },
  { id: 'hierarchy',   init: initHierarchy },
  { id: 'playground',  init: initPlayground },
  { id: 'farkas',      init: initFarkas },
  { id: 'convergence', init: initConvergence },
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
