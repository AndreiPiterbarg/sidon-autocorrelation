#!/usr/bin/env python
"""Grid-point cascade for c=1.28-1.35 at small S.

Grid-point proofs DON'T need box cert — they prove TV >= c at every
discrete composition. The cascade refines survivors until all are pruned.

At small S (10-30), the number of compositions is manageable and the
cascade converges quickly. The question is: does it converge at all?

If it does, we have a grid-point proof. To upgrade to a rigorous continuous
proof, we'd need a separate argument (e.g., Lipschitz bound on TV as a
function of continuous masses, with grid spacing 1/S).
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))

from pruning import count_compositions


def run_cascade_test(d0, S, c_target, max_levels=8, n_workers=None):
    from run_cascade import run_cascade
    n = count_compositions(d0, S)
    print(f"\n{'='*70}")
    print(f"CASCADE: d0={d0}, S={S}, c_target={c_target}")
    print(f"  L0 compositions: {n:,}")
    print(f"{'='*70}")

    t0 = time.time()
    info = run_cascade(
        n_half=d0 / 2.0, m=20, c_target=c_target,
        max_levels=max_levels, n_workers=n_workers,
        verbose=True, output_dir='data_sweep',
        coarse_S=S, d0=d0,
    )
    elapsed = time.time() - t0

    proven = 'proven_at' in info
    if proven:
        print(f"\n  GRID-POINT PROOF: converged at {info['proven_at']} "
              f"in {elapsed:.1f}s")
        box_ok = info.get('box_certified', False)
        if box_ok:
            print(f"  *** RIGOROUS PROOF: C_{{1a}} >= {c_target} ***")
    else:
        last = info.get('levels', [{}])[-1] if info.get('levels') else {}
        print(f"\n  NOT PROVEN: {last.get('survivors_out', '?')} survivors "
              f"at d={last.get('d_child', '?')} after {elapsed:.1f}s")
    return info


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()

    os.makedirs('data_sweep', exist_ok=True)

    # Strategy: small S, moderate d0, let cascade run deep.
    # d0=2 is best for cascade (fewest L0 survivors, more levels to prune).
    # S=10-30 keeps composition counts manageable.

    configs = [
        # c=1.28: should be easier (below current best)
        (2, 15, 1.28), (2, 20, 1.28), (2, 25, 1.28), (2, 30, 1.28),
        (3, 10, 1.28), (3, 15, 1.28), (3, 20, 1.28),
        (4, 10, 1.28), (4, 15, 1.28), (4, 20, 1.28),
        # c=1.30: the target
        (2, 15, 1.30), (2, 20, 1.30), (2, 25, 1.30), (2, 30, 1.30),
        (3, 10, 1.30), (3, 15, 1.30), (3, 20, 1.30),
        (4, 10, 1.30), (4, 15, 1.30), (4, 20, 1.30),
        # c=1.32, 1.35: stretch goals
        (2, 15, 1.32), (2, 20, 1.32),
        (3, 10, 1.32), (3, 15, 1.32),
        (4, 10, 1.32), (4, 15, 1.32),
        (2, 15, 1.35), (2, 20, 1.35),
        (3, 10, 1.35), (3, 15, 1.35),
        (4, 10, 1.35), (4, 15, 1.35),
        # c=1.40, 1.45: can the coarse grid even converge?
        (2, 10, 1.40), (2, 15, 1.40), (2, 20, 1.40),
        (3, 10, 1.40), (4, 10, 1.40),
        (2, 10, 1.45), (2, 15, 1.45),
        (3, 10, 1.45), (4, 10, 1.45),
        (2, 10, 1.50), (3, 10, 1.50), (4, 10, 1.50),
    ]

    proven_configs = []
    for d0, S, c in configs:
        n = count_compositions(d0, S)
        if n > 50_000_000:
            print(f"\nSKIP d0={d0}, S={S}: {n:,} compositions")
            continue

        # Clean checkpoints
        d = 'data_sweep'
        for f in os.listdir(d):
            if f.startswith('checkpoint_'):
                try:
                    os.remove(os.path.join(d, f))
                except OSError:
                    pass

        info = run_cascade_test(d0, S, c, n_workers=args.workers)
        if 'proven_at' in info:
            proven_configs.append((d0, S, c, info))

    print(f"\n{'='*70}")
    print("SUMMARY OF GRID-POINT PROOFS")
    print(f"{'='*70}")
    if proven_configs:
        for d0, S, c, info in proven_configs:
            box = "RIGOROUS" if info.get('box_certified') else "grid-point"
            print(f"  C_{{1a}} >= {c}: d0={d0}, S={S}, "
                  f"proved at {info['proven_at']} ({box})")
        best_c = max(c for _, _, c, _ in proven_configs)
        print(f"\n  BEST: C_{{1a}} >= {best_c}")
    else:
        print("  No proofs found.")


if __name__ == '__main__':
    main()
