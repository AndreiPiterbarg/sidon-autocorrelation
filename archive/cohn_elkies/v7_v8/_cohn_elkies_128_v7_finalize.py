"""Parse the partial v7 log, run the remaining sweeps (5-scale, 6-scale),
recertify the best 4-scale point at XI_MAX = 1e5, and emit the final
_cohn_elkies_128_v7_results.json.

This is a finalization step after the original v7 run was interrupted by an
external file-move (OneDrive auto-archiver) at ~22:02 local time.
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
sys.path.insert(0, str(_HERE / "delsarte_dual"))

# Import v7 utilities (relies on v5 internally)
import importlib.util
spec = importlib.util.spec_from_file_location("_cohn_elkies_128_v7",
                                              _HERE / "_cohn_elkies_128_v7.py")
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)

DELTA1_Q = v7.DELTA1_Q
_Q = v7._Q
_norm_lambdas = v7._norm_lambdas
certify_with_reopt_NG = v7.certify_with_reopt_NG
configure_precision = v7.configure_precision


# ---------------------------------------------------------------------------
# Parse the partial log
# ---------------------------------------------------------------------------
LOG_PATH = _HERE / "_cohn_elkies_128_v7_run_partial.log"

RE_3SC = re.compile(
    r"\s*N_G=\s*(\d+)\s+M_cert\s*=\s*([\d.]+)\s+minG=([\d.]+)\s+"
    r"S1=([\d.]+)\s+K2hi=([\d.]+)"
)
RE_SWEEP = re.compile(
    r"\s*\(([\d.,]+)\)\s+\(([\d.,]+)\)\s+(\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\("
)


def parse_log(path: Path):
    runs = []
    text = path.read_text()
    lines = text.splitlines()
    cur_scale = None
    # 3-scale reference: deltas/lambdas appear in a header line above the rows
    ref_deltas = [0.138, 0.055, 0.025]
    ref_lambdas = [0.85, 0.10, 0.05]
    in_4scale = False
    for line in lines:
        if "[3-scale reference]" in line:
            cur_scale = 3
            continue
        if "4-SCALE SWEEP" in line:
            cur_scale = 4
            in_4scale = True
            continue
        if "5-SCALE SWEEP" in line:
            cur_scale = 5
            in_4scale = False
            continue
        if cur_scale == 3:
            m = RE_3SC.search(line)
            if m:
                n_g, mcert, ming, s1, k2hi = m.groups()
                runs.append({
                    "tag": f"3sc NG={int(n_g)} d={tuple(ref_deltas)} l={tuple(ref_lambdas)}",
                    "n_scales": 3,
                    "N_G": int(n_g),
                    "deltas": ref_deltas,
                    "lambdas": ref_lambdas,
                    "M_cert_lower": float(mcert),
                    "min_G_lower": float(ming),
                    "S_1_upper": float(s1),
                    "K_2_upper": float(k2hi),
                    "xi_max": 10000,
                    "source": "log",
                })
        elif cur_scale == 4:
            m = RE_SWEEP.search(line)
            if m:
                d_str, l_str, n_g, mcert, ming, s1 = m.groups()
                deltas = [float(x) for x in d_str.split(",")]
                lams = [float(x) for x in l_str.split(",")]
                # K2hi not captured in the truncated 4-scale rows; try a wider regex
                wider = re.search(
                    r"\(([\d.,]+)\)\s+\(([\d.,]+)\)\s+(\d+)\s+"
                    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                    line)
                runs.append({
                    "tag": f"4sc NG={int(n_g)} d={tuple(deltas)} l={tuple(lams)}",
                    "n_scales": 4,
                    "N_G": int(n_g),
                    "deltas": deltas,
                    "lambdas": lams,
                    "M_cert_lower": float(mcert),
                    "min_G_lower": float(ming),
                    "S_1_upper": float(s1),
                    "xi_max": 10000,
                    "source": "log",
                })
    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("v7 finalization: parse partial log + 5-scale + 6-scale + best recert")
    print("=" * 78)

    # 1) Parse what we have
    parsed_runs = parse_log(LOG_PATH)
    print(f"\nParsed {len(parsed_runs)} runs from partial log")

    best_parsed = max(parsed_runs, key=lambda r: r["M_cert_lower"])
    print(f"  best parsed: {best_parsed['tag']}")
    print(f"  M_cert_lower = {best_parsed['M_cert_lower']:.8f}")

    all_runs = list(parsed_runs)

    XI_MAX = 10000
    XI_MAX_BEST = 100000

    # 2) Finish the remaining 4-scale configs (we have 41 of 60; finish 42-60)
    # Original lists:
    four_scale_deltas = [
        (0.138, 0.055, 0.025, 0.020),
        (0.138, 0.055, 0.025, 0.012),
        (0.138, 0.046, 0.025, 0.015),
        (0.138, 0.055, 0.030, 0.015),
        (0.138, 0.046, 0.025, 0.012),
        (0.138, 0.055, 0.020, 0.010),
    ]
    four_scale_lambdas = [
        (0.85, 0.08, 0.04, 0.03),
        (0.85, 0.10, 0.03, 0.02),
        (0.86, 0.08, 0.04, 0.02),
        (0.84, 0.10, 0.04, 0.02),
        (0.85, 0.09, 0.04, 0.02),
    ]
    n_g_list = [200, 500]
    # done set: build (deltas_tuple, lams_tuple, n_g) tuples
    done = set()
    for r in parsed_runs:
        if r["n_scales"] == 4:
            done.add((tuple(round(d, 5) for d in r["deltas"]),
                      tuple(round(l, 4) for l in r["lambdas"]),
                      r["N_G"]))

    print("\n" + "=" * 78)
    print("FINISHING 4-SCALE (configs not in partial log)")
    print("=" * 78)

    for deltas in four_scale_deltas:
        for lams in four_scale_lambdas:
            d_q = [DELTA1_Q] + [_Q(d, 10**6) for d in deltas[1:]]
            try:
                l_q = _norm_lambdas(list(lams))
            except ValueError:
                continue
            # the lambdas after normalization may have small numerical drift;
            # check using the requested lams tuple
            lams_norm = [float(q) for q in l_q]
            for n_g in n_g_list:
                key = (tuple(round(d, 5) for d in deltas),
                       tuple(round(l, 4) for l in lams_norm),
                       n_g)
                # match in done either by raw lams or by normalized lams
                # actually the parsed log stored the FLOAT lams from the
                # certifier output, which equal the normalized ones; build
                # alt keys
                alt_key = (tuple(round(d, 5) for d in deltas),
                           tuple(round(l, 4) for l in lams),
                           n_g)
                if key in done or alt_key in done:
                    continue
                t0 = time.time()
                try:
                    r = certify_with_reopt_NG(d_q, l_q,
                                              xi_max=XI_MAX, n_modes=n_g)
                except Exception as exc:
                    print(f"  d={deltas} l={lams} N_G={n_g}: ERROR {exc}")
                    continue
                el = time.time() - t0
                r["tag"] = f"4sc NG={n_g} d={tuple(deltas)} l={tuple(lams)}"
                r["N_G"] = n_g
                r["source"] = "rerun"
                d_str = "(" + ",".join(f"{d:.4g}" for d in deltas) + ")"
                l_str = "(" + ",".join(f"{l:.3g}" for l in lams) + ")"
                print(f"  {d_str:<28} {l_str:<28} {n_g:>5} "
                      f"{r['M_cert_lower']:>10.6f} ({el:.1f}s)")
                all_runs.append(r)

    # 3) 5-SCALE
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
                    r = certify_with_reopt_NG(d_q, l_q,
                                              xi_max=XI_MAX, n_modes=n_g)
                except Exception as exc:
                    print(f"  d={deltas} l={lams} N_G={n_g}: ERROR {exc}")
                    continue
                el = time.time() - t0
                r["tag"] = f"5sc NG={n_g} d={tuple(deltas)} l={tuple(lams)}"
                r["N_G"] = n_g
                r["source"] = "rerun"
                d_str = "(" + ",".join(f"{d:.4g}" for d in deltas) + ")"
                l_str = "(" + ",".join(f"{l:.3g}" for l in lams) + ")"
                marker = ""
                if best5 is None or r["M_cert_lower"] > best5["M_cert_lower"]:
                    best5 = r
                    marker = " *"
                print(f"  {d_str:<32} {l_str:<32} {n_g:>5} "
                      f"{r['M_cert_lower']:>10.6f} ({el:.1f}s){marker}")
                all_runs.append(r)

    # 4) 6-SCALE (only N_G=200, fewer configs for thoroughness)
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
                r = certify_with_reopt_NG(d_q, l_q,
                                          xi_max=XI_MAX, n_modes=n_g)
            except Exception as exc:
                print(f"  d={deltas} l={lams}: ERROR {exc}")
                continue
            el = time.time() - t0
            r["tag"] = f"6sc NG={n_g} d={tuple(round(d,4) for d in deltas)} l={tuple(lams)}"
            r["N_G"] = n_g
            r["source"] = "rerun"
            d_str = "(" + ",".join(f"{d:.3g}" for d in deltas) + ")"
            l_str = "(" + ",".join(f"{l:.2g}" for l in lams) + ")"
            print(f"  {d_str:<48} {l_str:<32} {n_g:>5} "
                  f"{r['M_cert_lower']:>10.6f} ({el:.1f}s)")
            all_runs.append(r)

    # 5) Recertify global best at XI_MAX = 1e5
    print("\n" + "=" * 78)
    best_overall = max(all_runs, key=lambda r: r["M_cert_lower"])
    print(f"Re-certify GLOBAL BEST at XI_MAX={XI_MAX_BEST}: {best_overall['tag']}")
    print("=" * 78)
    d_best = best_overall["deltas"]
    l_best = best_overall["lambdas"]
    n_g_best = best_overall["N_G"]
    d_q_best = [DELTA1_Q] + [_Q(d, 10**6) for d in d_best[1:]]
    l_q_best = _norm_lambdas(list(l_best))
    r_best_hi = certify_with_reopt_NG(d_q_best, l_q_best,
                                      xi_max=XI_MAX_BEST,
                                      n_modes=n_g_best, verbose=True)
    r_best_hi["tag"] = best_overall["tag"] + f" @xi={XI_MAX_BEST}"
    r_best_hi["N_G"] = n_g_best
    r_best_hi["source"] = "rerun_hires"
    all_runs.append(r_best_hi)
    if r_best_hi["M_cert_lower"] > best_overall["M_cert_lower"]:
        best_overall = r_best_hi

    # 6) Also recertify the best 4-scale parsed-from-log point at high XI_MAX
    # so that the final result has a fully self-contained certificate.
    best_4sc_log = max(
        (r for r in parsed_runs if r["n_scales"] == 4),
        key=lambda r: r["M_cert_lower"])
    print("\n" + "=" * 78)
    print(f"Re-certify BEST 4-SCALE (log) at XI_MAX={XI_MAX_BEST}: {best_4sc_log['tag']}")
    print("=" * 78)
    d4 = best_4sc_log["deltas"]
    l4 = best_4sc_log["lambdas"]
    n_g4 = best_4sc_log["N_G"]
    d_q4 = [DELTA1_Q] + [_Q(d, 10**6) for d in d4[1:]]
    l_q4 = _norm_lambdas(list(l4))
    r4_hi = certify_with_reopt_NG(d_q4, l_q4, xi_max=XI_MAX_BEST,
                                  n_modes=n_g4, verbose=True)
    r4_hi["tag"] = best_4sc_log["tag"] + f" @xi={XI_MAX_BEST}"
    r4_hi["N_G"] = n_g4
    r4_hi["source"] = "rerun_hires"
    all_runs.append(r4_hi)
    if r4_hi["M_cert_lower"] > best_overall["M_cert_lower"]:
        best_overall = r4_hi

    _final_summary(all_runs)
    _write_json(all_runs, best_overall)


def _final_summary(all_runs):
    print()
    print("=" * 78)
    print("FINAL SUMMARY (top 20)")
    print("=" * 78)
    print(f"  {'tag':<60}  {'M_cert':>10}  {'src':>10}")
    print(f"  {'-'*60}  {'-'*10}  {'-'*10}")
    sorted_runs = sorted(all_runs, key=lambda r: -r["M_cert_lower"])
    for r in sorted_runs[:20]:
        tag = r.get("tag", "?")[:60]
        src = r.get("source", "?")
        print(f"  {tag:<60}  {r['M_cert_lower']:>10.6f}  {src:>10}")
    best = sorted_runs[0]
    print()
    print(f"BEST rigorous M_cert: {best['M_cert_lower']:.8f}")
    print(f"  tag: {best.get('tag','?')}")
    print(f"  vs 3-scale ref (1.29216): {best['M_cert_lower'] - 1.29216:+.6f}")
    print(f"  vs v5 best (1.29136):     {best['M_cert_lower'] - 1.29136:+.6f}")
    print(f"  vs CS17 (1.28020):        {best['M_cert_lower'] - 1.28020:+.6f}")
    print("=" * 78)


def _write_json(all_runs, best_overall):
    out_path = _HERE / "_cohn_elkies_128_v7_results.json"
    out = {
        "configuration": {
            "delta_1": float(DELTA1_Q),
            "u": 0.638,
            "prec_bits": 256,
            "XI_MAX_sweep": 10000,
            "XI_MAX_best": 100000,
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
        "note": "v7 finalization: 4-scale rows 1-41 are from the partial log; "
                "rows 42+ and 5-/6-scale are reruns after the original v7 was "
                "interrupted by external file-move at ~22:02.",
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
