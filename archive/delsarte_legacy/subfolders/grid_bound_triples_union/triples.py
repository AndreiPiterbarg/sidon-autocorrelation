"""Build a family of (delta, K, G) triples -> PhiMMParams for the union search.

Each triple supplies its own (delta, u, coeffs, min_G, gain_a, K2, k_n, S1).
K is always the arcsine kernel at scale delta; G is the OPTIMAL cosine
polynomial for that delta, produced off-line by the semi-infinite QP in
``mv_qp_optimize.solve_mv_qp`` (run via ``_run_qp_sweep.py`` and cached in
``G_by_delta/*.json``).  We load those cached coefficients here and compile
into fully rigorous PhiMMParams via the Phase-2 path.

Design
------
* delta sweep fixed at {0.10, 0.115, 0.13, 0.138, 0.15, 0.165, 0.18}.
* u = 1/2 + delta (period-u convention).
* n_coeffs = 119 (same as MV's reference fit).
* K2 surrogate = 0.5747 / delta (inherited from Martin-O'Bryant Lemma 3.2;
  the arcsine density rescales as (1/delta)*beta(x/delta), which makes
  ||K||_2^2 scale as 1/delta; MV state the period-delta integral bound
  ||K||_2^2 < 0.5747/delta).  The same constant 0.5747 is re-used for every
  delta in the sweep; this is conservative but consistent with MV's own
  practice across delta scans.
* Coefficients parsed losslessly from cached JSON (stored as Python repr
  floats, i.e. exact IEEE-754 doubles), converted to fmpq via Fraction.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.phi_mm import PhiMMParams
from delsarte_dual.grid_bound.coeffs import MV_K2_NUMERATOR


# Canonical sweep (as exact rationals) required by the union-of-triples spec.
# Each entry is (delta_numer, delta_denom).  u_q is derived as 1/2 + delta.
DELTA_SWEEP: tuple[fmpq, ...] = (
    fmpq(10,  100),    # 0.10
    fmpq(115, 1000),   # 0.115
    fmpq(13,  100),    # 0.13
    fmpq(138, 1000),   # 0.138  (canonical MV)
    fmpq(15,  100),    # 0.15
    fmpq(165, 1000),   # 0.165
    fmpq(18,  100),    # 0.18
)

DELTA_STR_MAP = {
    fmpq(10,  100):  "0.10",
    fmpq(115, 1000): "0.115",
    fmpq(13,  100):  "0.13",
    fmpq(138, 1000): "0.138",
    fmpq(15,  100):  "0.15",
    fmpq(165, 1000): "0.165",
    fmpq(18,  100):  "0.18",
}


def _delta_to_json_name(delta_q: fmpq) -> str:
    """Stable filename for cached QP coefficients at a given delta."""
    s = DELTA_STR_MAP.get(delta_q)
    if s is None:
        # Fall back to a deterministic p_q form
        s = f"{delta_q.p}_{delta_q.q}"
    return f"delta_{s}.json"


def u_of_delta(delta_q: fmpq) -> fmpq:
    """u = 1/2 + delta  (MV period-u convention)."""
    return fmpq(1, 2) + delta_q


def _float_to_fmpq(x: float) -> fmpq:
    """Lossless IEEE-754 double -> fmpq (same value exactly)."""
    frac = Fraction(x)
    return fmpq(frac.numerator, frac.denominator)


def load_qp_coeffs_json(
    delta_q: fmpq,
    g_by_delta_dir: str | None = None,
) -> tuple[tuple[fmpq, ...], dict]:
    """Load cached QP coefficients for a given delta.  Returns (coeffs_fmpq, raw_json).

    Raises FileNotFoundError if no cache exists.  Raises RuntimeError if the
    cached record reports status != 'OK'.
    """
    if g_by_delta_dir is None:
        g_by_delta_dir = os.path.join(os.path.dirname(__file__), "G_by_delta")
    path = os.path.join(g_by_delta_dir, _delta_to_json_name(delta_q))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached QP coefficients for delta={delta_q} at {path}.  "
            f"Run _run_qp_sweep.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("status", "OK") != "OK":
        raise RuntimeError(
            f"Cached QP at {path} has status={raw['status']}; QP did not converge."
        )
    if raw["n_coeffs"] != len(raw["a_opt"]):
        raise RuntimeError(f"Corrupt cache: n_coeffs/a_opt mismatch in {path}")
    coeffs = tuple(_float_to_fmpq(float(v)) for v in raw["a_opt"])
    return coeffs, raw


@dataclass(frozen=True)
class Triple:
    """One (delta, K, G) triple pre-compiled as rigorous PhiMMParams.

    ``params.delta``, ``params.u``, and ``params.n_coeffs`` identify the triple.
    ``idx`` is the triple's position in the family (0-based).
    """
    idx: int
    params: PhiMMParams

    @property
    def delta(self) -> fmpq:
        return self.params.delta

    @property
    def u(self) -> fmpq:
        return self.params.u


def build_triple(
    idx: int,
    delta_q: fmpq,
    N_max: int = 1,
    *,
    K2_times_delta: fmpq = MV_K2_NUMERATOR,
    n_cells_min_G: int = 8192,
    prec_bits: int = 256,
    g_by_delta_dir: str | None = None,
    coeffs_override: Sequence[fmpq] | None = None,
) -> Triple:
    """Compile a single triple into fully rigorous PhiMMParams.

    Arguments
    ---------
    idx            family position (0-based), for logging and certificate ids.
    delta_q        delta as exact fmpq.
    N_max          multi-moment level; ALL triples in a family must share this.
    K2_times_delta MV's regularised ||K||_2^2 * delta (default 0.5747).
    coeffs_override if supplied, use these instead of loading from cache.

    Returns a Triple carrying PhiMMParams ready for phi_mm.
    """
    u_q = u_of_delta(delta_q)
    if coeffs_override is None:
        coeffs, _raw = load_qp_coeffs_json(delta_q, g_by_delta_dir=g_by_delta_dir)
    else:
        coeffs = tuple(coeffs_override)
    params = PhiMMParams.from_mv(
        N_max=N_max,
        delta=delta_q,
        u=u_q,
        coeffs=coeffs,
        K2_times_delta=K2_times_delta,
        n_cells_min_G=n_cells_min_G,
        prec_bits=prec_bits,
    )
    return Triple(idx=idx, params=params)


def build_family(
    delta_list: Sequence[fmpq],
    N_max: int = 1,
    *,
    K2_times_delta: fmpq = MV_K2_NUMERATOR,
    n_cells_min_G: int = 8192,
    prec_bits: int = 256,
    g_by_delta_dir: str | None = None,
    coeffs_overrides: dict | None = None,
) -> list[Triple]:
    """Compile a whole family of triples.  Skips deltas with unavailable cache."""
    coeffs_overrides = coeffs_overrides or {}
    out: list[Triple] = []
    for i, d in enumerate(delta_list):
        override = coeffs_overrides.get(d)
        try:
            t = build_triple(
                idx=i, delta_q=d, N_max=N_max,
                K2_times_delta=K2_times_delta,
                n_cells_min_G=n_cells_min_G, prec_bits=prec_bits,
                g_by_delta_dir=g_by_delta_dir,
                coeffs_override=override,
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[triples] SKIP delta={d} ({e.__class__.__name__}: {e})")
            continue
        out.append(t)
    return out


__all__ = [
    "DELTA_SWEEP",
    "DELTA_STR_MAP",
    "u_of_delta",
    "load_qp_coeffs_json",
    "Triple",
    "build_triple",
    "build_family",
]
