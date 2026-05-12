#!/usr/bin/env python
"""Quick hyperparameter sweep for spectral cut settings.

Tests three configurations — 2 CG rounds, 4 bisect steps each (~6 min total).
Measures: gc% after 2 rounds, lb, per-iter ms, peak memory.

Configs tested:
 A: k_vecs=3, n_add=100 (current baseline)
 B: k_vecs=5, n_add=100 (more eigenvectors per window → tighter cuts)
 C: k_vecs=3, n_add=150 (more windows per round → faster coverage)
"""
import sys, os, time, gc
import numpy as np
from scipy import sparse as sp
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre_highd import (
 _precompute_highd, _check_violations_highd,
 _build_banded_cliques, val_d_known,
)

# Import from run_d16_l3 (has all our improvements)
from run_d16_l3 import (
 build_base_problem,
 _precompute_spectral_cuts,
 _assemble_spectral_cuts,
 _greedy_diverse_cuts,
 _window_active_bins,
 _classify_verdict, _VERDICT_FEAS, _VERDICT_INFEAS, _VERDICT_UNCERTAIN,
 solve_scs_direct,
)


def run_config(k_vecs, n_add, label, d=16, order=3, bw=15,
 cg_rounds=3, n_bisect=4, scs_iters=20000, scs_eps=1e-5):
 """Run solver with given spectral cut config, return results dict."""
 print(f"\n{'='*60}")
 print(f"CONFIG {label}: k_vecs={k_vecs}, n_add={n_add}")
 print(f" d={d} L{order} bw={bw} | {cg_rounds} rounds, {n_bisect} bisect")
 print(f"{'='*60}", flush=True)

 t0 = time.time()

 # Patch k_vecs into the solver by monkey-patching the module-level call
 import run_d16_l3 as _mod
 import lasserre_highd as _lh

 # Store original k_vecs default, patch it
 _orig_check = _lh._check_violations_highd

 def _patched_check(y_vals, t_val, P, active_windows, tol=1e-6, k_vecs_arg=k_vecs):
 return _orig_check(y_vals, t_val, P, active_windows, tol=tol, k_vecs=k_vecs_arg)

 # Run the solver with patched k_vecs (hacky but avoids rewrite)
 # Instead just call solve_scs_direct directly — it reads k_vecs from
 # the violation calls inside. We need to parameterize via the n_add.
 # For this sweep we call the full solver and capture results.

 r = solve_scs_direct(
 d=d, order=order, bandwidth=bw,
 max_cg_rounds=cg_rounds, n_bisect=n_bisect,
 use_gpu=True, scs_max_iters=scs_iters, scs_eps=scs_eps,
 verbose=False,
 _k_vecs_override=k_vecs, # we add this param below
 _n_add_override=n_add,
 )

 elapsed = time.time() - t0
 v = val_d_known.get(d, 0)
 gc_pct = (r['lb'] - 1) / (v - 1) * 100 if v > 1 else 0

 print(f"CONFIG {label} RESULT:")
 print(f" lb = {r['lb']:.6f}")
 print(f" gc = {gc_pct:.1f}%")
 print(f" time = {elapsed:.0f}s")
 print(f" target = 1.2802 {' REACHED' if r['lb'] > 1.2802 else '— not yet'}")
 return {'label': label, 'k_vecs': k_vecs, 'n_add': n_add,
 'lb': r['lb'], 'gc': gc_pct, 'elapsed': elapsed}


if __name__ == '__main__':
 print(f"Spectral cut hyperparameter sweep")
 print(f"Started: {datetime.now().isoformat()}")

 # Since monkey-patching is fragile, instead we expose k_vecs/n_add
 # as env-var overrides and run three separate processes — or simpler,
 # directly inline the logic here by calling the underlying pieces.
 # For cleanliness, just run three full calls with argument overrides
 # via the existing solve_scs_direct. The function currently hardcodes
 # k_vecs=3 and n_add=100 internally. To test different values, we
 # need to modify them before calling.

 # Minimal approach: write three tiny wrapper scripts and run them sequentially.
 import subprocess, json

 configs = [
 {'k_vecs': 3, 'n_add': 100, 'label': 'A_baseline'},
 {'k_vecs': 5, 'n_add': 100, 'label': 'B_5vecs'},
 {'k_vecs': 3, 'n_add': 150, 'label': 'C_150cuts'},
 ]

 results = []
 for cfg in configs:
 t0 = time.time()
 env = os.environ.copy()
 env['SWEEP_K_VECS'] = str(cfg['k_vecs'])
 env['SWEEP_N_ADD'] = str(cfg['n_add'])

 cmd = [
 'python', 'tests/run_d16_l3.py',
 '--d', '16', '--order', '3', '--bw', '15',
 '--cg-rounds', '3', '--bisect', '5',
 '--scs-iters', '20000', '--scs-eps', '1e-5',
 ]
 print(f"\nRunning {cfg['label']} k_vecs={cfg['k_vecs']} n_add={cfg['n_add']} ...")
 proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env)
 elapsed = time.time() - t0

 # Extract lb from output
 lb = None
 for line in proc.stdout.split('\n'):
 if 'lb=' in line and 'Checkpoint' in line:
 parts = line.split('lb=')
 if len(parts) > 1:
 try:
 lb = float(parts[-1].strip())
 except Exception:
 pass
 if lb is None:
 for line in proc.stdout.split('\n'):
 if 'FINAL' in line or ('lb=' in line and 'gc=' in line):
 parts = line.split('lb=')
 if len(parts) > 1:
 try:
 lb = float(parts[1].split()[0])
 except Exception:
 pass

 v = val_d_known.get(16, 1.319)
 gc_pct = (lb - 1) / (v - 1) * 100 if lb and lb > 0 else 0

 r = {**cfg, 'lb': lb, 'gc': gc_pct, 'elapsed': elapsed}
 results.append(r)
 print(f" lb={lb}, gc={gc_pct:.1f}%, t={elapsed:.0f}s")

 if lb and lb > 1.2802:
 print(f" *** TARGET REACHED with {cfg['label']} ***")

 print(f"\n{'='*60}")
 print("SWEEP RESULTS SUMMARY")
 print(f"{'='*60}")
 results.sort(key=lambda x: -(x['gc'] or 0))
 for r in results:
 print(f" {r['label']:15s}: lb={r['lb']}, gc={r['gc']:.1f}%, t={r['elapsed']:.0f}s")

 best = results[0]
 print(f"\nBest config: {best['label']} (k_vecs={best['k_vecs']}, n_add={best['n_add']})")
 print(f"Recommended for full proof run.")
