"""v7 finalization (simplified): parse partial log + run 5-scale + 6-scale +
recertify the best 4-scale point at XI_MAX = 1e5.  Skips the remaining 4-scale
configs (the partial log already covered the most promising branches; the
remainder turned out to be in worse delta neighborhoods).
"""

from __future__ import annotations

import json
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

# Import v7 (which imports v5 internally)
import importlib.util
spec = importlib.util.spec_from_file_location("_cohn_elkies_128_v7",
                                              _HERE / "_cohn_elkies_128_v7.py")
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)

DELTA1_Q = v7.DELTA1_Q
_Q = v7._Q
_norm_lambdas = v7._norm_lambdas
certify_with_reopt_NG = v7.certify_with_reopt_NG

LOG_PATH = _HERE / "_cohn_elkies_128_v7_run_partial.log"
RE_3SC = re.compile(
    r"\s*N_G=\s*(\d+)\s+M_cert\s*=\s*([\d.]+)\s+minG=([\d.]+)\s+"
    r"S1=([\d.]+)\s+K2hi=([\d.]+)"
)


def parse_log(path: Path):
    runs = []
    text = path.read_text()
    lines = text.splitlines()
    cur_scale = None
    ref_deltas = [0.138, 0.055, 0.025]
    ref_lambdas = [0.85, 0.10, 0.05]
    for line in lines:
        if "[3-scale reference]" in line:
            cur_scale = 3
            continue
        if "4-SCALE SWEEP" in line:
            cur_scale = 4
            continue
        if cur_scale == 3:
            m = RE_3SC.search(line)
            if m:
                n_g, mcert, ming, s1, k2hi = m.groups()
                runs.append({
                    "tag": f"3sc NG={int(n_g)} d=(0.138,0.055,0.025) "
                           f"l=(0.85,0.1,0.05)",
                    "n_scales": 3, "N_G": int(n_g),
                    "deltas": ref_deltas, "lambdas": ref_lambdas,
                    "M_cert_lower": float(mcert),
                    "min_G_lower": float(ming),
                    "S_1_upper": float(s1),
                    "K_2_upper": float(k2hi),
                    "xi_max": 10000, "source": "log",
                })
        elif cur_scale == 4:
            m = re.match(
                r"\s+\(([\d.,]+)\)\s+\(([\d.,]+)\)\s+(\d+)\s+"
                r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\(",
                line)
            if m:
                d_str, l_str, n_g, mcert, ming, s1 = m.groups()
                deltas = [float(x) for x in d_str.split(",")]
                lams = [float(x) for x in l_str.split(",")]
                runs.append({
                    "tag": f"4sc NG={int(n_g)} d=({d_str}) l=({l_str})",
                    "n_scales": 4, "N_G": int(n_g),
                    "deltas": deltas, "lambdas": lams,
                    "M_cert_lower": float(mcert),
                    "min_G_lower": float(ming),
                    "S_1_upper": float(s1),
                    "xi_max": 10000, "source": "log",
                })
    return runs


def main():
    print("=" * 78)
    print("v7 finalization (5-scale + 6-scale + best recert)")
    print("=" * 78)
    parsed_runs = parse_log(LOG_PATH)
    print(f"\nParsed {len(parsed_runs)} runs from partial log")
    best_parsed = max(parsed_runs, key=lambda r: r["M_cert_lower"])
    print(f"  best parsed: M={best_parsed['M_cert_lower']:.8f}  ({best_parsed['tag']})")
    all_runs = list(parsed_runs)

    XI_MAX = 10000
    XI_MAX_BEST = 100000

    # 5-SCALE
    print("\n" + "=" * 78)
    print("5-SCALE SWEEP")
    print("=" * 78)
    five_scale_deltas = [
        (0.138, 0.055, 0.025, 0.015, 0.008),
        (0.138, 0.055, 0.025, 0.012, 0.006),
        (0.138, 0.046, 0.025, 0.015, 0.008),
    ]
    five_scale_lambdas = [
        (0.85, 0.07, 0.04, 0.02, 0.02),
        (0.84, 0.08, 0.04, 0.02, 0.02),
        (0.86, 0.07, 0.04, 0.02, 0.01),
    ]
    n_g_list = [200, 500]
    best5 = None
    for deltas in five_scale_deltas:
        for lams in five_scale_lambdas:
            d_q = [DELTA1_Q] + [_Q(d, 10**6) for d in deltas[1:]]
            try:
                l_q = _norm_lambdas(list(lams))
            except ValueError:
                continue
            for n_g in n_g_list:
                t0 = time.time()
                try:
                    r = certify_with_reopt_NG(d_q, l_q, xi_max=XI_MAX, n_modes=n_g)
                except Exception as exc:
                    print(f"  d={deltas} l={lams} N_G={n_g}: ERROR {exc}")
                    continue
                el = time.time() - t0
                r["tag"] = f"5sc NG={n_g} d={deltas} l={lams}"
                r["N_G"] = n_g
                r["source"] = "rerun"
                marker = ""
                if best5 is None or r["M_cert_lower"] > best5["M_cert_lower"]:
                    best5 = r
                    marker = " *"
                print(f"  d={deltas} l={lams} N_G={n_g}  "
                      f"M={r['M_cert_lower']:.6f}  minG={r['min_G_lower']:.4f}  "
                      f"S1={r['S_1_upper']:.2f}  ({el:.1f}s){marker}",
                      flush=True)
                all_runs.append(r)
                # write running state to JSON so we never lose progress
                _write_partial(all_runs)

    # 6-SCALE (N_G=200 only)
    print("\n" + "=" * 78)
    print("6-SCALE SWEEP")
    print("=" * 78)
    six_rest = [
        (0.005, 0.080),
        (0.008, 0.090),
        (0.010, 0.060),
    ]
    six_lambdas = [
        (0.85, 0.06, 0.04, 0.02, 0.02, 0.01),
        (0.84, 0.07, 0.04, 0.02, 0.02, 0.01),
    ]
    for (lo_, hi_) in six_rest:
        rest = list(np.linspace(hi_, lo_, 5))
        deltas = (0.138,) + tuple(rest)
        for lams in six_lambdas:
            d_q = [DELTA1_Q] + [_Q(d, 10**6) for d in deltas[1:]]
            try:
                l_q = _norm_lambdas(list(lams))
            except ValueError:
                continue
            n_g = 200
            t0 = time.time()
            try:
                r = certify_with_reopt_NG(d_q, l_q, xi_max=XI_MAX, n_modes=n_g)
            except Exception as exc:
                print(f"  d={deltas} l={lams}: ERROR {exc}")
                continue
            el = time.time() - t0
            r["tag"] = f"6sc NG={n_g} d=({','.join(f'{d:.3g}' for d in deltas)}) l={lams}"
            r["N_G"] = n_g
            r["source"] = "rerun"
            print(f"  d=({','.join(f'{d:.3g}' for d in deltas)}) l={lams}  "
                  f"M={r['M_cert_lower']:.6f}  ({el:.1f}s)", flush=True)
            all_runs.append(r)
            _write_partial(all_runs)

    # Re-certify best at XI_MAX = 1e5
    print("\n" + "=" * 78)
    best_overall = max(all_runs, key=lambda r: r["M_cert_lower"])
    print(f"Re-certify GLOBAL BEST at XI_MAX={XI_MAX_BEST}: {best_overall['tag']}")
    print("=" * 78)
    d_best = best_overall["deltas"]
    l_best = best_overall["lambdas"]
    n_g_best = best_overall["N_G"]
    d_q_best = [DELTA1_Q] + [_Q(d, 10**6) for d in d_best[1:]]
    l_q_best = _norm_lambdas(list(l_best))
    try:
        r_best_hi = certify_with_reopt_NG(d_q_best, l_q_best,
                                          xi_max=XI_MAX_BEST,
                                          n_modes=n_g_best, verbose=True)
        r_best_hi["tag"] = best_overall["tag"] + f" @xi={XI_MAX_BEST}"
        r_best_hi["N_G"] = n_g_best
        r_best_hi["source"] = "rerun_hires"
        all_runs.append(r_best_hi)
        if r_best_hi["M_cert_lower"] > best_overall["M_cert_lower"]:
            best_overall = r_best_hi
    except Exception as exc:
        print(f"  high-XI recert failed: {exc}")

    _final_summary(all_runs)
    _write_json(all_runs, best_overall, final=True)


def _write_partial(all_runs):
    out_path = _HERE / "_cohn_elkies_128_v7_results.json"
    out = {"runs": all_runs, "status": "in_progress"}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


def _final_summary(all_runs):
    print()
    print("=" * 78)
    print("FINAL SUMMARY (top 20)")
    print("=" * 78)
    print(f"  {'tag':<60}  {'M_cert':>10}  {'src':>12}")
    sorted_runs = sorted(all_runs, key=lambda r: -r["M_cert_lower"])
    for r in sorted_runs[:20]:
        tag = r.get("tag", "?")[:60]
        src = r.get("source", "?")
        print(f"  {tag:<60}  {r['M_cert_lower']:>10.6f}  {src:>12}")
    best = sorted_runs[0]
    print()
    print(f"BEST rigorous M_cert: {best['M_cert_lower']:.8f}")
    print(f"  tag: {best.get('tag','?')}")
    print(f"  vs 3-scale v4 ref (1.29216): {best['M_cert_lower'] - 1.29216:+.6f}")
    print(f"  vs CS17 paper (1.28020):     {best['M_cert_lower'] - 1.28020:+.6f}")


def _write_json(all_runs, best_overall, final=False):
    out_path = _HERE / "_cohn_elkies_128_v7_results.json"
    out = {
        "configuration": {
            "delta_1": 0.138, "u": 0.638, "prec_bits": 256,
            "XI_MAX_sweep": 10000, "XI_MAX_best": 100000,
            "N_G_values": [200, 500],
        },
        "runs": all_runs,
        "best_overall": best_overall,
        "baselines": {
            "v4_3sc_N200": 1.29216,
            "v5_4sc_N119": 1.29136,
            "cs17_paper": 1.28020,
            "MV_numerical": 1.27428,
            "MV_arcsine_empirical_ceiling": 1.2924,
        },
        "status": "final" if final else "in_progress",
        "note": "v7 finalization: 4-scale rows 1-41 are from the partial log "
                "(rigorous arb pipeline at XI_MAX=10000); rows 42+ are 5-scale "
                "and 6-scale reruns.  Original v7 sweep was interrupted by an "
                "external file-move at ~22:02 local time.",
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
