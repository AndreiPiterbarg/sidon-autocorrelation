"""Baseline results from Lasserre Level-2 v2 notebook.

These are the known bounds on V(P) from the notebook sweep (P=2..15).
"""

# Baseline results: {P: {'lasserre2_lb': ..., 'shor_lb': ..., 'primal_ub': ..., 'time': ..., 'flat_ext': ...}}
BASELINE = {
    2:  {'lasserre2_lb': 1.777778, 'shor_lb': 1.333333, 'primal_ub': 1.777778, 'time': 0.5,  'flat_ext': True,  'ratio': 1.000000},
    3:  {'lasserre2_lb': 1.706667, 'shor_lb': 1.200000, 'primal_ub': 1.706667, 'time': 1.0,  'flat_ext': True,  'ratio': 1.000000},
    4:  {'lasserre2_lb': 1.644465, 'shor_lb': 1.142857, 'primal_ub': 1.644465, 'time': 2.2,  'flat_ext': True,  'ratio': 1.000000},
    5:  {'lasserre2_lb': 1.632651, 'shor_lb': 1.111111, 'primal_ub': 1.633817, 'time': 5.4,  'flat_ext': True,  'ratio': 0.999287},
    6:  {'lasserre2_lb': 1.585589, 'shor_lb': 1.090909, 'primal_ub': 1.600883, 'time': 9.2,  'flat_ext': False, 'ratio': 0.990447},
    7:  {'lasserre2_lb': 1.581729, 'shor_lb': 1.076923, 'primal_ub': 1.591746, 'time': 17.1, 'flat_ext': False, 'ratio': 0.993707},
    8:  {'lasserre2_lb': 1.548249, 'shor_lb': 1.066667, 'primal_ub': 1.580150, 'time': 35.1, 'flat_ext': False, 'ratio': 0.979811},
    9:  {'lasserre2_lb': 1.545321, 'shor_lb': 1.058824, 'primal_ub': 1.578073, 'time': 88.1, 'flat_ext': False, 'ratio': 0.979246},
    10: {'lasserre2_lb': 1.524610, 'shor_lb': 1.052632, 'primal_ub': 1.566436, 'time': 146.7,'flat_ext': False, 'ratio': 0.973299},
    11: {'lasserre2_lb': 1.519106, 'shor_lb': 1.047619, 'primal_ub': 1.562873, 'time': 240.9,'flat_ext': False, 'ratio': 0.971996},
    12: {'lasserre2_lb': 1.506925, 'shor_lb': 1.043478, 'primal_ub': 1.559773, 'time': 442.9,'flat_ext': False, 'ratio': 0.966118},
    13: {'lasserre2_lb': 1.503012, 'shor_lb': 1.040000, 'primal_ub': 1.559581, 'time': 851.0,'flat_ext': False, 'ratio': 0.963728},
    14: {'lasserre2_lb': 1.492672, 'shor_lb': 1.037037, 'primal_ub': 1.552608, 'time': 1580.8,'flat_ext': False,'ratio': 0.961397},
    15: {'lasserre2_lb': 1.485952, 'shor_lb': 1.034483, 'primal_ub': 1.550623, 'time': 2464.0,'flat_ext': False,'ratio': 0.958294},
}


def compare_with_baseline(P, new_lb, new_time, method_name="new"):
    """Compare a new lower bound with the baseline at given P."""
    if P in BASELINE:
        bl = BASELINE[P]
        diff = new_lb - bl['lasserre2_lb']
        speedup = bl['time'] / new_time if new_time > 0 else float('inf')
        ratio = new_lb / bl['primal_ub'] if bl['primal_ub'] > 0 else 0
        print(f"  P={P:2d}: {method_name:>20s} LB={new_lb:.6f} | baseline LB={bl['lasserre2_lb']:.6f} | "
              f"diff={diff:+.6f} | primal UB={bl['primal_ub']:.6f} | ratio={ratio:.6f} | "
              f"time={new_time:.1f}s (baseline {bl['time']:.1f}s, {speedup:.1f}x)")
        return {'P': P, 'new_lb': new_lb, 'baseline_lb': bl['lasserre2_lb'],
                'diff': diff, 'primal_ub': bl['primal_ub'], 'ratio': ratio,
                'new_time': new_time, 'baseline_time': bl['time'], 'speedup': speedup}
    else:
        print(f"  P={P:2d}: {method_name:>20s} LB={new_lb:.6f} | no baseline | time={new_time:.1f}s")
        return {'P': P, 'new_lb': new_lb, 'baseline_lb': None, 'diff': None,
                'new_time': new_time}


if __name__ == '__main__':
    print("Baseline Lasserre Level-2 results:")
    print(f"{'P':>3} | {'Lass-2 LB':>10} | {'Shor LB':>10} | {'Primal UB':>10} | {'Ratio':>7} | {'Flat':>5} | {'Time':>7}")
    print("-" * 75)
    for P in sorted(BASELINE.keys()):
        bl = BASELINE[P]
        flat = 'YES' if bl['flat_ext'] else 'no'
        print(f"{P:>3} | {bl['lasserre2_lb']:>10.6f} | {bl['shor_lb']:>10.6f} | "
              f"{bl['primal_ub']:>10.6f} | {bl['ratio']:>7.4f} | {flat:>5} | {bl['time']:>6.1f}s")
