"""Plot the val(d) trajectory and a power-law / logistic extrapolation.

Reads ``lasserre/trajectory/results/d{d}_trajectory.json`` for d in
DIMS, plots certified val(d) on a log-d axis with horizontal reference
lines at 1.2802 (CS17) and 1.281 (target), and overlays two
extrapolation fits.

Outputs ``lasserre/trajectory/trajectory.pdf`` (matplotlib only).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_completed(results_dir: Path, dims: List[int]) -> List[Tuple[int, float]]:
    pts = []
    for d in dims:
        p = results_dir / f'd{d}_trajectory.json'
        if not p.exists():
            continue
        with open(p) as f:
            r = json.load(f)
        if r['status'] != 'COMPLETED':
            continue
        if r['largest_certified_t'] is None:
            continue
        pts.append((d, float(r['largest_certified_t'])))
    return pts


def _fit_power_law(pts: List[Tuple[int, float]],
                   asymptote: float = 1.50992
                   ) -> Tuple[callable, str]:
    """Fit val(d) = A - B * d^{-alpha}, with A clamped at the
    Matolcsi-Vinuesa upper bound 1.50992 by default — this is the
    natural asymptote since val(d) <= C_{1a} <= 1.50992 for every d.

    Returns (predict_fn, summary_str).
    """
    if len(pts) < 2:
        return (lambda d: float('nan'), 'insufficient points')
    A = asymptote
    ds = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    # log(A - y) = log(B) - alpha * log(d)
    diffs = A - ys
    if np.any(diffs <= 0):
        # data point already at/above asymptote — pick a tighter A
        A = max(ys.max() + 0.01, A)
        diffs = A - ys
    L = np.log(diffs)
    X = np.log(ds)
    # Least squares: L = log(B) - alpha * X
    M = np.vstack([np.ones_like(X), -X]).T
    coef, *_ = np.linalg.lstsq(M, L, rcond=None)
    logB, alpha = coef
    B = math.exp(logB)
    fn = lambda d, A=A, B=B, alpha=alpha: A - B * (float(d) ** (-alpha))
    summary = f'val(d) ≈ {A:.4f} − {B:.4f}·d^{{−{alpha:.3f}}}'
    return fn, summary


def _fit_logistic(pts: List[Tuple[int, float]]) -> Tuple[callable, str]:
    """Fit a 3-parameter logistic: val(d) = L / (1 + exp(-k*(log d - log d0)))
    asymptoting at L. Approximation by least-squares in log d.

    With only 3-4 points this is dramatically under-determined; we report
    it explicitly as a "soft cap" alternative model to the power law.
    """
    if len(pts) < 3:
        return (lambda d: float('nan'), 'need >=3 points')
    ds = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    # crude grid search over (L, k, d0)
    L_grid = np.linspace(ys.max() + 0.01, 1.55, 25)
    k_grid = np.linspace(0.4, 4.0, 12)
    d0_grid = np.linspace(2.0, 200.0, 30)
    best = (1e9, None)
    for L in L_grid:
        for k_ in k_grid:
            for d0 in d0_grid:
                pred = L / (1.0 + np.exp(-k_ * (np.log(ds) - np.log(d0))))
                err = float(((pred - ys) ** 2).sum())
                if err < best[0]:
                    best = (err, (L, k_, d0))
    if best[1] is None:
        return (lambda d: float('nan'), 'fit failed')
    L, k_, d0 = best[1]
    fn = lambda d, L=L, k=k_, d0=d0: L / (1.0 + math.exp(-k * (math.log(d) - math.log(d0))))
    summary = f'val(d) ≈ {L:.4f} / (1 + exp(−{k_:.2f}·(ln d − ln {d0:.1f})))'
    return fn, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', type=str,
                    default=str(Path(_HERE) / 'results'))
    ap.add_argument('--out', type=str,
                    default=str(Path(_HERE) / 'trajectory.pdf'))
    ap.add_argument('--dims', type=int, nargs='+', default=[4, 8, 16, 32])
    ap.add_argument('--predict_at', type=int, nargs='+',
                    default=[64, 128])
    args = ap.parse_args()

    pts = _load_completed(Path(args.results_dir), args.dims)
    if not pts:
        raise SystemExit('no completed trajectory points found')

    # Fits
    pl_fn, pl_str = _fit_power_law(pts)
    log_fn, log_str = _fit_logistic(pts)

    # Predictions
    print('\n=== predictions ===')
    print(f'  power law:  {pl_str}')
    print(f'  logistic:   {log_str}')
    for d in args.predict_at:
        pl = pl_fn(d)
        lg = log_fn(d)
        print(f'  d={d:4d}: power-law={pl:.4f}  logistic={lg:.4f}')

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.set_xscale('log', base=2)
    d_dense = np.geomspace(min(p[0] for p in pts) * 0.95, max(args.predict_at) * 1.05, 80)
    pl_curve = [pl_fn(d) for d in d_dense]
    log_curve = [log_fn(d) for d in d_dense]

    ax.plot(d_dense, pl_curve, '--', color='steelblue', alpha=0.85,
            label=f'power-law fit')
    ax.plot(d_dense, log_curve, ':', color='darkorange', alpha=0.85,
            label=f'logistic fit')

    # Reference lines
    ax.axhline(1.2802, color='red', linewidth=1.0, alpha=0.7,
                label='CS 2017: 1.2802')
    ax.axhline(1.281, color='green', linewidth=1.0, alpha=0.7,
                label='target: 1.281')

    # Data
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            'o', color='black', markersize=8, label='measured (Farkas-cert LB)')
    for d, v in pts:
        ax.annotate(f'{v:.4f}', (d, v),
                    textcoords='offset points', xytext=(8, -3), fontsize=8)

    # Predictions as crosses
    for d in args.predict_at:
        pl = pl_fn(d)
        lg = log_fn(d)
        ax.plot(d, pl, 'x', color='steelblue', markersize=10)
        ax.plot(d, lg, '+', color='darkorange', markersize=12)

    ax.set_xlabel('d')
    ax.set_ylabel('certified lower bound on val(d)')
    ax.set_title('val(d) trajectory: certified Lasserre lower bounds (order=2, b=7)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, format='pdf', bbox_inches='tight')
    print(f'\n[saved] {args.out}')


if __name__ == '__main__':
    main()
