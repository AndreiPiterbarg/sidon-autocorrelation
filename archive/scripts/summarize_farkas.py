"""Print a summary table of all certified Farkas bounds.

Usage:
    python3.11 scripts/summarize_farkas.py
"""
import json
import sys
from pathlib import Path
from fractions import Fraction

RESULTS_DIR = Path('data/farkas_results')

# Literature baselines
VAL_D_KNOWN = {4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241, 12: 1.271,
                14: 1.284, 16: 1.319, 18: 1.30, 20: 1.315}
BOYER_LI = 1.2802  # current record

def main():
    print('=' * 75)
    print('CERTIFIED VAL(d) LOWER BOUNDS (Farkas-CG, rational-rounded)')
    print('=' * 75)
    print(f'{"d":>4} | {"order":>5} | {"certified lb":>18} | {"val(d) est":>10} | '
          f'{"gap":>8} | {"vs 1.2802":>10}')
    print('-' * 75)

    files = sorted(RESULTS_DIR.glob('d*_o*.json'))
    if not files:
        print('No results yet.')
        return

    total_certs = 0
    best_d16_or_above = None
    for fn in files:
        d_part = fn.stem.split('_')[0]   # 'd16'
        o_part = fn.stem.split('_')[1]   # 'o3'
        try:
            data = json.loads(fn.read_text())
        except Exception as e:
            print(f'{fn.name}: CORRUPTED ({e})')
            continue
        d = data['d']
        order = data['order']
        lb_str = data['lb_rig_decimal']
        lb_flt = float(lb_str)
        vd = VAL_D_KNOWN.get(d, None)
        gap_str = f'{vd - lb_flt:.5f}' if vd is not None else 'n/a'
        bl_cmp = f'{lb_flt - BOYER_LI:+.5f}'
        bl_mark = ' *BEATS*' if lb_flt > BOYER_LI else ''
        print(f'{d:>4} | {order:>5} | {lb_str:>18} | '
              f'{vd if vd else "?":>10} | {gap_str:>8} | {bl_cmp}{bl_mark}')
        total_certs += 1
        if d >= 16 and lb_flt > BOYER_LI:
            best_d16_or_above = (d, lb_str)

    print('-' * 75)
    print(f'Total certified: {total_certs}')
    if best_d16_or_above:
        d, lb = best_d16_or_above
        print(f'\n*** NEW BOUND: val({d}) > {lb}  (beats Boyer-Li 1.2802) ***')
    print()


if __name__ == '__main__':
    main()
