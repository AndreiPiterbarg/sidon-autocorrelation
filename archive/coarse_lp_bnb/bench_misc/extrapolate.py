"""Extrapolation analysis for cascade convergence."""

print('='*70)
print('SURVIVOR ESTIMATE SUMMARY: d0=2, m=20, c_target=1.35')
print('='*70)

# === MEASURED DATA ===
print('\nMEASURED DATA:')
print(f'  L0 (d=2):  26 survivors (exact)')
print(f'  L1 (d=4):  66,992 survivors (exact, all 26 parents)')
print(f'  L2 (d=8):  ~460 billion survivors (10-parent random sample)')

print(f'\n  L0->L1 expansion: {66992/26:,.0f}x')
print(f'  L1->L2 expansion: {460e9/66992:,.0f}x')

# === REFERENCE DATA from c_target=1.40 d0=4 (flat correction) ===
print('\nREFERENCE (c_target=1.40, d0=4, flat correction=0.41):')
print('  L0 (d=4):  467 survivors')
print('  L1 (d=8):  4,690 unique survivors (10x expansion)')
print('  L2 (d=16): 868,456 unique survivors (185x expansion)')
print('  -> Expansion GREW from L1 to L2')

# === THRESHOLD ANALYSIS ===
print('\nEFFECTIVE THRESHOLD (determines pruning power):')
print('  Flat correction (C&S Lemma 3): 2/m + 1/m^2 = 0.1025')
print('  This is dimension-INDEPENDENT: same at d=4 and d=64')
print('  So effective threshold = 1.35 + 0.1025 = 1.4525 at ALL levels')
print('  W-refined is per-window and <= flat, so slightly tighter')

# === L3 CARTESIAN PRODUCT SIZES ===
print('\nL3 PARENT CARTESIAN PRODUCTS (d=8->d=16, measured):')
print('  Parent 1: 625 billion children (after tightening)')
print('  Parent 2: 336 TRILLION children')
print('  Parent 3: 20 TRILLION children')
print('  -> Single L3 parent takes HOURS to DAYS on CPU')

# === EXTRAPOLATION ===
print('\n' + '='*70)
print('EXTRAPOLATION TO L3+')
print('='*70)

# L2 stats: avg 20.9M children/parent, avg 6.9M survivors/parent
# survival rate at L2: ~33%
# L3 children per parent: ~10^12 to 10^14 (much bigger)
# Even if survival rate drops to 0.1% at L3:
surv_rate_l3_optimistic = 0.001  # very generous
avg_cp_l3 = 1e13  # geometric mean of measured samples
surv_per_parent_l3 = avg_cp_l3 * surv_rate_l3_optimistic  # 10 billion
n_parents_l3 = 460e9
l3_raw = n_parents_l3 * surv_per_parent_l3

print(f'\n  460B parents x ~10^13 children/parent x 0.1% survival')
print(f'  = {l3_raw:.1e} raw L3 survivors')
print(f'  Even with 100x dedup: {l3_raw/100:.1e}')

print(f'\n  Compute required for L3 alone:')
print(f'    Total children: {n_parents_l3 * avg_cp_l3:.1e}')
throughput = 7e6  # children/sec/core
cores = 64
gpu_speedup = 64  # H100 vs CPU core
total_sec_cpu = n_parents_l3 * avg_cp_l3 / (throughput * cores)
total_sec_gpu = total_sec_cpu / gpu_speedup
print(f'    At 7M ch/sec/core x 64 cores: {total_sec_cpu/3600/24/365:.1e} years (CPU)')
print(f'    With 64x GPU speedup: {total_sec_gpu/3600/24/365:.1e} years (64 H100s)')

# === WHAT ABOUT CONVERGENCE AT L4+? ===
print('\n' + '='*70)
print('DOES THE CASCADE EVER CONVERGE?')
print('='*70)
print('''
The expansion ratio measures survivors_out / survivors_in per level.
For the cascade to converge, this ratio must drop BELOW 1.

Measured expansion ratios:
  L0->L1:  2,577x
  L1->L2:  6,866,050x (GROWING)

For convergence, you need the pruning to kill MORE children than
the Cartesian product creates. At each level:
  - Branching factor: each parent bin splits into ~R choices (R = range width)
  - d bins means Cartesian product ~ R^d
  - But d doubles each level, so branching is EXPONENTIAL in level

  Level  d_child  Cursor positions  Typical range  Approx CP
  L1     4        2                 ~31            ~961
  L2     8        4                 ~50            ~6.25M
  L3     16       8                 ~70            ~5.7e14
  L4     32       16               ~80            ~1.2e30
  L5     64       32               ~90            ~10^62

The Cartesian product grows as R^(d/2) ~ R^(2^level).
Pruning power grows polynomially (more windows, but each window
check is O(d)).

CONCLUSION: The branching grows doubly-exponentially while pruning
grows polynomially. The cascade DIVERGES for c_target=1.35.

The only way to make it work is to start at a higher d0 (d0=4 or
d0=8) so you skip the explosive early levels, but this requires
MANY more L0 compositions to enumerate.
''')

# Final table
print('='*70)
print('FINAL TABLE')
print('='*70)
print(f'{"Level":<8} {"Dim":<6} {"Survivors":<25} {"Expansion":<15} {"Method"}')
print('-'*70)
data = [
    (0, 2, 26, '-', 'exact'),
    (1, 4, 66992, '2,577x', 'exact'),
    (2, 8, 4.6e11, '6,866,050x', '10-parent sample'),
    (3, 16, None, None, 'INTRACTABLE (single parent takes days)'),
]
for lvl, d, surv, exp, method in data:
    if surv is None:
        s = '>>10^18'
    elif surv > 1e12:
        s = f'~{surv/1e12:.1f}T'
    elif surv > 1e9:
        s = f'~{surv/1e9:.0f}B'
    elif surv > 1e6:
        s = f'~{surv/1e6:.0f}M'
    else:
        s = f'{surv:,}'
    e = exp if exp else 'N/A'
    print(f'L{lvl:<7} d={d:<4} {s:<25} {e:<15} {method}')
