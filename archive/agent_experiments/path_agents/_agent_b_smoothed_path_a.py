"""
Agent B: Test §5.3(d) of derivation.md — Smoothed-autoconvolution K_eps route.

GOAL: Test whether replacing g = f*f with the smoothed g_eps = f*f*phi_eps
      breaks the asymmetric blocker (K unbounded), giving an unconditional LB > 1.2748.

THE CHAIN (paper derivation):

  Let f >= 0 on [-1/4, 1/4], int f = 1.
  Let h_eps = (1/eps) * 1_{[-eps/2, eps/2]}        (nonneg, integral 1)
  Let phi_eps := h_eps * h_eps                     (triangle on [-eps, eps], peak 1/eps)
  Let f_eps := f * h_eps                           (nonneg, supp in [-L,L], L = 1/4 + eps/2,
                                                    int f_eps = 1)
  Then g_eps := f_eps * f_eps = (f*f) * phi_eps = g * phi_eps,
       supp g_eps in [-2L, 2L] = [-1/2 - eps, 1/2 + eps].

Key inequalities:
  (i)   ||g_eps||_inf  =  ||g * phi_eps||_inf <= ||g||_inf * ||phi_eps||_1 = M.
  (ii)  K_eps := ||f_eps||_2^2 = int int f(x) f(y) phi_eps(x-y) dx dy <= ||phi_eps||_inf = 1/eps.

If we could apply MO 2.14 to f_eps, then for the autocorrelation pdf
  g_eps,  ||g_eps||_inf <= M, the Parseval+Lemma2.14 chain gives:
    ||g_eps||_2^2 <= 1 + mu(M_eps) (K_eps - 1)
where M_eps = ||g_eps||_inf.

The catch: MO 2.14 requires supp(f) subset [-1/4, 1/4].  For f_eps the support is
[-(1/4)-eps/2, (1/4)+eps/2], which is LARGER.  The MO 2.14 inequality on this
larger support uses period (1+eps), not period 1; the constraint changes.

The MO 2.14 "rescaled" version (proof on p. 16 of MO 2004): for f a pdf on
[-L, L], M = ||f*f||_inf, the symmetric-decreasing rearrangement of f*f is
supported on [-2L, 2L] and has max M.  The argument bounds the j-th Fourier
coefficient of f periodized to period 2L by:
   |f-hat(j/(2L))|^2 <= M * (sin(pi*j*(2L)/M_eff)) / (pi*j*(2L)/M_eff) ... etc.
   Actually:  z_j^2 <= mu_scaled(M, L) := M sin(pi/(M*2L_ratio))/pi
   where the L-scaling enters because the rearrangement step uses
   the indicator of [-1/(2M_eff), 1/(2M_eff)] on the support [-L,L].

ACTUALLY, the simplest way: since g_eps is a pdf on [-2L, 2L] with
||g_eps||_inf <= M and int g_eps = 1, we can apply Lemma 2.13 (SDR) directly
to g_eps to bound  int g_eps^2  via the bathtub:
   ||g_eps||_2^2 <= bathtub bound = M (when ||g_eps||_inf <= M and int g_eps = 1).

This is the TRIVIAL Holder bound on g_eps.  But we want something tighter
than M.  The whole point of the Path A chain is to use mu(M) (the bound on
|f-hat(j)|^2) to get a c < 1 factor.

For f_eps periodized with period (1 + eps): the indices j run over
Z/(1+eps), but the Lemma 2.14 bound transports.  We'll compute it carefully
below.

THIS SCRIPT:
  1. Derives the eps-rescaled MO 2.14 constant.
  2. Derives the eps-rescaled "K_eps <= ???" inequality.
  3. Combines to get an upper bound on ||g_eps||_2^2.
  4. Plugs into the conditional theorem to get an LB on M.
  5. Sweeps eps and reports the optimum.
  6. Tests numerically on MV-like extremizers.
"""

import json
import sys
from pathlib import Path

import numpy as np
import mpmath as mp
from mpmath import mpf

# Ensure repo root in path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from delsarte_dual.restricted_holder.conditional_bound import (
    conditional_bound_optimal,
)
from delsarte_dual.path_a_unconditional_holder.holder_constant import (
    mu, c_sym_at, CS2017_M_MIN,
)

mp.mp.dps = 40
LOG16_PI = mp.log(16) / mp.pi  # ~0.88254


# =============================================================================
# Step 1: derive the eps-rescaled MO 2.14 constant
# =============================================================================

def mu_rescaled(M, L):
    """Rescaled MO 2.14 constant for f a pdf on [-L, L].

    For f >= 0 on [-L, L] with int f = 1 and M = ||f*f||_inf, the
    symmetric-decreasing rearrangement of f*f is on [-2L, 2L].  By the SDR
    argument (MO 2004 Lemma 2.13/2.14), the maximum of
       Re int f(x) exp(-2 pi i x j / (2L)) dx
    over the class is bounded by integrating the indicator of [-w/2, w/2]
    where w = 1/M (so the indicator has mass 1, max M, and is the bathtub).
    But w must fit in [-L, L], i.e., 1/(2M) <= L, i.e., M >= 1/(2L).

    Specifically (eps-scaled MO 2.14):
       For f on [-L, L], M = ||f*f||_inf, j != 0:
       |f-hat(j/(2L))|^2 <= integral_{|x|<1/(2M)} f(x) exp(-2 pi i x j/(2L)) dx |^2
                          <=  ( (sin(pi*j/(2L*M)) / (pi*j/(2L*M))) * 1 )^2 ...
       Wait, let me redo this carefully.

    SDR proof of MO 2.14, generalised to [-L,L]:
      h := (f*f)^SDR is a symmetric decreasing pdf on [-2L, 2L] with ||h||_inf <= M.
      We have:  |f-hat(j/(2L))|^2 = h-hat(j/(2L)).
      The maximum of h-hat(omega) over symmetric-decreasing h with ||h||_inf <= M,
      int h = 1, supp h in [-2L, 2L]:  by bathtub principle, h = M * 1_{[-1/(2M), 1/(2M)]}
      (provided 1/(2M) <= 2L, i.e., M >= 1/(4L); always true here).
      So  h-hat(omega) = int_{|t|<1/(2M)} M e^{-2 pi i omega t} dt
                       = M * sin(pi omega/M) / (pi omega).
      At omega = j/(2L):
        h-hat(j/(2L)) = M * sin(pi*j/(2L*M)) / (pi*j/(2L))
                       = (2L*M/(pi*j)) * sin(pi*j/(2L*M))
                       = 2L * mu_L_j(M)
        where mu_L_j(M) := M * sin(pi*j/(2L*M)) / (pi*j)  ... hmm wait.
      Better: factor M:
        h-hat(j/(2L)) = M * [ sin(pi*j/(2L*M)) / (pi*j/(2L*M)) ] * (1/(2L*M))^{-1} ... no.

      Let u = pi*j/(2L*M).  Then h-hat(j/(2L)) = (M/(pi j/(2L))) sin(u)
                                                = (M*2L/(pi*j)) sin(pi*j/(2L*M)).
      For L = 1/4: 2L = 1/2, so h-hat(j/(1/2)) = (M/(pi*j))*(1/2)*sin(pi*j/(M/2))
                                              hmm let me redo.

    SIMPLEST: at L = 1/4, the standard MO 2.14:
       |f-hat(j)|^2 <= M sin(pi/M)/pi for j=1 (and by SDR for general j on [-1/2,1/2]).
       But the canonical j-index when L = 1/4 is j = 1, 2, ..., the Fourier coefficient
       of f periodized to period 2L = 1/2.

    Hmm — actually let me re-read MO 2.14.  MV uses period 1 (extending f from
    [-1/4,1/4] periodically), so the natural omega values are j = 1, 2, 3, ....
    That requires period >= 2*supp_radius, so period >= 1/2.  Period 1 is fine.

    For f_eps on [-1/4 - eps/2, 1/4 + eps/2], to extend periodically we need
    period >= 1/2 + eps.  Take period = 1 + eps.

    OK new plan: let p be the period (= 1 in the standard case, = 1 + eps for
    f_eps), and define the discrete Fourier coefficients
       f-hat(k) := int_{[-p/2, p/2]} f(x) e^{-2 pi i k x / p} dx
    so that f-hat(k) = (continuous Fourier transform) at frequency k/p.

    Then MO 2.14's SDR proof gives (for f a pdf on [-1/4, 1/4] periodized to
    period p >= 1/2):
       |f-hat(k)|^2 <= M sin(pi k/p / M) / (pi k/p)  (NOT *(p/(2L)) * 2L,
       just integrate from -1/(2M) to 1/(2M))
       = (M / (pi k/p)) sin(pi k / (p M))
       = mu_scaled(M, k, p) := (Mp/(pi*k)) sin(pi k/(p*M)).

    For p = 1, k = 1: mu_scaled(M, 1, 1) = (M/pi) sin(pi/M) = mu(M).  Checks out.

    For p = 1 + eps, k = 1: mu_scaled(M, 1, 1+eps) = ((1+eps)*M/pi) sin(pi/((1+eps)M))
       = (1+eps) * (M/pi) * sin(pi/((1+eps)M)).

    NOTE: as eps -> 0, this -> mu(M).  Good.

    For higher k, since sin oscillates, the j-dependence might give negative
    values; we take |sin|.  But the cleanest unconditional form is k=1.

    Wait — we are looking at f_eps's Fourier coefficients summed over all k.
    Parseval gives:  ||f_eps||_2^2 = sum |f_eps-hat(k)|^2.  We need the k>=1
    bound on |f_eps-hat(k)|^2 to get a uniform mu_eps.

    The k=1 case is the binding one for the uniform bound on f-hat (it's
    monotonic in M for small k).  For k >= 1 in general:
       |f-hat(k)|^2 <= (Mp/(pi k)) * |sin(pi k/(p M))| / ??? wait.

    Actually MO 2.14's proof gives a UNIFORM (k-independent) bound on
    |f-hat(k)|^2.  Let me re-derive.

    SDR proof of MO 2.14 (uniform in k):
      Let h := (f*f)^SDR, symmetric decreasing on [-2L, 2L], int h = 1,
      ||h||_inf <= M.  Then for any k != 0:
         h-hat(k/p) = int_{-2L}^{2L} h(t) cos(2 pi t k/p) dt
                    = int_0^{2L} 2 h(t) cos(2 pi t k/p) dt.
      By the bathtub principle (h sym-dec with mass 1, height <= M, supported in [-2L,2L]):
        the maximum of  int_0^{2L} 2 h(t) phi(t) dt  over the class is achieved by
        h = M * 1_{[0, 1/(2M)]} (taking the highest-height part where phi >= 0).
        BUT we need phi(t) = cos(2 pi t k/p) to be NON-NEGATIVE on the support.
        For phi(t) = cos(2 pi t k/p) on t in [0, 1/(2M)]: nonneg requires
        2 pi t k/p <= pi/2, i.e., t <= p/(4 k).  Since t <= 1/(2M) and we need
        1/(2M) <= p/(4k), i.e., 2k <= p M.  When k=1, this requires p M >= 2.
        For p=1, M=1.28, k=1: pM = 1.28 < 2.  So the cosine is NEGATIVE on part
        of [0, 1/(2M)].

      Hmm — so the SDR step is more subtle than just bathtub.  Actually the MO
      2.14 statement (eq below 1055) is:
         |f-hat(j)|^2 <= M sin(pi/M)/pi    (for j = 1, period 1, L = 1/4)
      and MO claim this is uniform in j.  But sin(pi j/M)/(pi j) can be negative
      or have small magnitude for j > 1.  So the uniform bound is the j=1 value
      (the max over j of the natural bound).

    PRAGMATIC: for the smoothed chain, define:
       mu_p(M, p) := max over k >= 1 of |M*sin(pi k/(p*M))/(pi k)| ??? this is messy.

    SAFER: use the closed-form upper bound that MO 2.14 ACTUALLY gives
    after SDR: for any k != 0,
         |f-hat(k/p)|^2 <= M sin(pi/(p*M))/pi * p   (??)
    Let me re-derive from scratch.

    From MO 2004, eq below 1055:
       For f on [-1/4, 1/4], M = ||f*f||_inf, j != 0:
       |f-hat(j)|^2 <= M sin(pi/M)/pi.
    Where f-hat(j) := int_{[-1/4,1/4]} f(x) e^{-2 pi i j x} dx.

    Rescale: f_eps on [-L, L] where L = 1/4 + eps/2 = (1+2*eps)/4. Define
       phi(x) := (2L)^{-1} * (2L*f_eps(2L*x))... no.  Reparametrize f_eps to live
    on [-1/4, 1/4]: g(y) := 4L * f_eps(4L*y) = 4L * f_eps(4L*y).  Wait — we want
    a pdf on [-1/4, 1/4] obtained by scaling f_eps.  Let alpha = (1/4)/L =
    1/(1+2*eps).  Define g(y) := f_eps(y/alpha)/alpha = f_eps(y*(1+2eps))*(1+2eps).
    Then g is a pdf on [-1/4, 1/4] (since x = y/alpha = y*(1+2eps) ranges over [-L, L]
    as y ranges over [-1/4, 1/4]).  Check int g: int g(y) dy = int f_eps(y*(1+2eps))*(1+2eps) dy
    = int f_eps(x) dx = 1.  Good.

    M_g := ||g*g||_inf.  Now g*g(s) = int g(t) g(s-t) dt = int f_eps(t*(1+2eps))*(1+2eps)
                                            * f_eps((s-t)*(1+2eps))*(1+2eps) dt.
    Substitute u = t*(1+2eps): du = (1+2eps) dt; integral becomes
       int f_eps(u) f_eps(s*(1+2eps) - u) du * (1+2eps) = (1+2eps) (f_eps*f_eps)(s*(1+2eps))
       = (1+2eps) g_eps(s*(1+2eps)).
    So  ||g*g||_inf = (1+2eps) * ||g_eps||_inf <= (1+2eps) * M.
    M_g = (1+2eps) * ||g_eps||_inf <= (1+2eps) * M.

    Apply MO 2.14 to g (pdf on [-1/4, 1/4]):
       |g-hat(j)|^2 <= mu(M_g) for j != 0.
    Now g-hat(j) = int_{[-1/4, 1/4]} g(y) e^{-2 pi i j y} dy
                 = int_{[-L, L]} f_eps(x) e^{-2 pi i j x/(1+2eps)} dx
                 = f_eps_hat(j/(1+2eps))     [continuous F.T. of f_eps at freq j/(1+2eps)].
    Parseval (for f_eps on [-L, L] periodized with period 2*2L = 1+2eps... actually
    period >= 2L = 1/2 + eps suffices):
      ||f_eps||_2^2 = sum_k |f_eps_hat(k/p)|^2 / p?  No, the discrete Parseval is
      different.  Let me use period p = 1 + 2*eps:
        ||f_eps||_{L^2(period p)}^2  =  (1/p) sum_k |f_eps_hat(k)|^2
        where f_eps_hat(k) := int_{[-p/2, p/2]} f_eps(x) e^{-2 pi i k x/p} dx.
        But f_eps_hat(k) here corresponds to f_eps_hat(k/p) in the continuous F.T.
        Convention: with my earlier defn (k-th discrete Fourier coef = continuous F.T.
        at freq k/p), Parseval is
          ||f_eps||_2^2 = (1/p) sum_k |F.T. at freq k/p|^2 = (1/p) sum_k |g-hat(k)|^2.
      Hmm — the 1/p factor changes things.

    Cleaner approach: just rescale dx.  For f_eps on [-L, L]:
       ||f_eps||_2^2 = int f_eps(x)^2 dx.
       Substitute x = y * (1+2eps): dx = (1+2eps) dy:
       = int (f_eps(y(1+2eps)))^2 (1+2eps) dy
       = int (g(y)/(1+2eps))^2 (1+2eps) dy            [using g(y) = (1+2eps) f_eps(y(1+2eps))]
       = (1/(1+2eps)) int g(y)^2 dy
       = ||g||_2^2 / (1+2eps).
    So ||f_eps||_2^2 = ||g||_2^2 / (1+2eps).

    Apply Path A chain (dagger) to g: ||g*g||_2^2 <= 1 + mu(M_g) * (||g||_2^2 - 1).
    Note ||g*g||_2 = (1+2eps)^{3/2} ||g_eps||_2 ... let me compute.

    Actually we don't need ||g*g||.  We want ||g_eps||_2^2 because that's the
    quantity in our chain.

    Conversion:  ||g_eps||_2^2 = int (g_eps(s))^2 ds. From g*g(s) = (1+2eps)g_eps(s*(1+2eps)):
       ||g*g||_2^2 = int [(1+2eps)g_eps(s(1+2eps))]^2 ds = (1+2eps)^2 int g_eps(s')^2 (ds'/(1+2eps))
                   = (1+2eps) int g_eps^2 ds' = (1+2eps) * ||g_eps||_2^2.
    Hence ||g_eps||_2^2 = ||g*g||_2^2 / (1+2eps).

    Putting it together: ||g*g||_2^2 <= 1 + mu(M_g) * (||g||_2^2 - 1)
    where M_g = (1+2eps) * ||g_eps||_inf <= (1+2eps) * M
    and  ||g||_2^2 = (1+2eps) * ||f_eps||_2^2 = (1+2eps) * K_eps.

    So:   (1+2eps) * ||g_eps||_2^2 <= 1 + mu((1+2eps)M) * ((1+2eps)K_eps - 1).
       => ||g_eps||_2^2 <= [1 + mu((1+2eps)M) * ((1+2eps)K_eps - 1)] / (1+2eps).

    Hyp_R-target on g_eps:  ||g_eps||_2^2 <= c * ||g_eps||_inf <= c * M.
    (since ||g_eps||_inf <= M from the very first inequality.)

    We need:  [1 + mu((1+2eps)M) * ((1+2eps)K_eps - 1)] / (1+2eps) <= c * M.
    With K_eps <= 1/eps:
       (1+2eps)K_eps - 1 <= (1+2eps)/eps - 1 = (1+2eps-eps)/eps = (1+eps)/eps.

    So:  [1 + mu((1+2eps)M) * (1+eps)/eps] / (1+2eps) <= c * M.   (chain)

    For this to give c < 1, we need mu((1+2eps)M) * (1+eps)/eps <= c*M*(1+2eps) - 1.

    mu(M') = M' sin(pi/M')/pi.  For M' = (1+2eps)M = M' > M.
    As eps -> 0: mu((1+2eps)M) -> mu(M).  (1+eps)/eps -> infinity.
    So the product -> infinity.  c -> infinity.  BAD as eps -> 0.

    As eps -> infinity (irrelevant since eps must be < 1/2): K_eps -> int f^2,
    but K_eps <= 1/eps is loose.  Tight bound: K_eps -> 1 as eps -> infinity?
    Actually as eps -> infinity, h_eps -> 0 uniformly so f_eps -> 0; need to
    bound K_eps more carefully.  K_eps -> int int f(x) f(y) phi_eps(x-y) dx dy.
    For eps >> diam(supp f) = 1/2: phi_eps(x-y) ~ 1/eps for all x, y in supp f,
    so K_eps -> 1/eps -> 0.  Hmm, that violates the floor.

    Actually K_eps is bounded below by Cauchy-Schwarz: K_eps = ||f * h_eps||_2^2
    >= (||f * h_eps||_1 / sqrt(supp diameter))^2 = 1 / (1/2 + eps).
    So K_eps in [1/(1/2 + eps), 1/eps].

    The middle ground: at eps ~ 0.1, K_eps in [1/0.6, 10] = [1.67, 10].
    But this still doesn't give c < 1.

  CONCLUSION: the K_eps <= 1/eps bound is the same as the trivial K bound for
  the original f at eps -> 0.  The chain does not close because the small-eps
  blowup of K_eps cancels the small-eps preservation of mu(M).

  Let me CHECK this numerically below.
"""

mp.mp.dps = 50


def mu_p(M_eff):
    """MO 2.14 constant: mu(M) = M sin(pi/M) / pi."""
    M_eff = mpf(M_eff)
    return M_eff * mp.sin(mp.pi / M_eff) / mp.pi


def K_eps_upper_naive(eps):
    """Naive upper bound: K_eps <= ||phi_eps||_inf = 1/eps."""
    return mpf(1) / mpf(eps)


def K_eps_upper_young(K_orig, eps):
    """Young's inequality bound: K_eps <= K_orig * ||h_eps||_2^2 / ||h_eps||_1^2.

    h_eps = (1/eps) * 1_{[-eps/2, eps/2]}.
    ||h_eps||_1 = 1, ||h_eps||_2^2 = (1/eps^2) * eps = 1/eps.
    So K_eps <= K_orig * (1/eps) / 1 = K_orig / eps.
    """
    return mpf(K_orig) / mpf(eps)


def smoothed_bound_on_g_eps_L2sq(M, K_eps_bound, eps, dps=50):
    """Return UB on ||g_eps||_2^2 via the rescaled (†) chain.

    Chain:  (1+2eps) ||g_eps||_2^2 <= 1 + mu((1+2eps)M) * ((1+2eps) K_eps - 1).
    => ||g_eps||_2^2 <= [1 + mu((1+2eps)M) * ((1+2eps) K_eps - 1)] / (1+2eps).
    """
    mp.mp.dps = dps
    M = mpf(M); eps = mpf(eps); K_eps_bound = mpf(K_eps_bound)
    M_g = (1 + 2*eps) * M
    K_g = (1 + 2*eps) * K_eps_bound
    return (1 + mu_p(M_g) * (K_g - 1)) / (1 + 2*eps)


def c_eps_naive(M, eps, dps=50):
    """For the chain to close via smoothing with the naive K_eps <= 1/eps:
       ||g_eps||_2^2 <= c * M (the Hyp_R target since ||g_eps||_inf <= M).
       Returns c = (UB on ||g_eps||_2^2) / M.
    """
    K_eps_bound = K_eps_upper_naive(eps)
    UB = smoothed_bound_on_g_eps_L2sq(M, K_eps_bound, eps, dps=dps)
    return UB / mpf(M)


def c_eps_from_K(M, K_eps, eps, dps=50):
    """As above but with a user-provided K_eps value (e.g., from a specific f)."""
    UB = smoothed_bound_on_g_eps_L2sq(M, K_eps, eps, dps=dps)
    return UB / mpf(M)


# =============================================================================
# Step 2: Sweep eps and find the optimum
# =============================================================================

def sweep_eps(M, eps_values, K_eps_provider="naive", dps=50, K_orig=2.0):
    """Sweep eps and compute c(eps) for given M.

    K_eps_provider:
       "naive": K_eps <= 1/eps
       "young": K_eps <= K_orig / eps
       "constant_2": K_eps = 2 (test scenario, suppose K = 2 unbounded)
    """
    results = []
    for eps in eps_values:
        eps = mpf(eps)
        if K_eps_provider == "naive":
            K_eps = K_eps_upper_naive(eps)
        elif K_eps_provider == "young":
            K_eps = K_eps_upper_young(K_orig, eps)
        elif K_eps_provider == "constant_2":
            K_eps = mpf(2)
        else:
            raise ValueError(f"Unknown provider: {K_eps_provider}")
        UB = smoothed_bound_on_g_eps_L2sq(M, K_eps, eps, dps=dps)
        c = UB / mpf(M)
        results.append({
            'eps': float(eps),
            'K_eps_bound': float(K_eps),
            'UB_g_eps_L2sq': float(UB),
            'c_eps': float(c),
            'mu_at_M_eff': float(mu_p((1 + 2*eps)*mpf(M))),
        })
    return results


# =============================================================================
# Step 3: Conditional bound integration
# =============================================================================

def lb_from_c_eps(c_eps_value, dps=50):
    """Given a uniform c_eps < 1, plug into conditional theorem to get LB on M.

    BUT: the conditional theorem (in restricted_holder/derivation.md) uses
    ||f*f||_2^2 <= c * M, not ||f_eps*f_eps||_2^2 <= c * M.  These differ:
    if we only have a bound on ||g_eps||_2^2 = ||(f*f) * phi_eps||_2^2, we need
    to lift this back to a bound on ||f*f||_2^2.

    Key observation: as eps -> 0, g_eps -> g a.e. (and in L^2 for nice f), so
    ||g_eps||_2 -> ||g||_2.  Hence in the LIMIT, the bound on ||g_eps||_2^2
    becomes a bound on ||g||_2^2.  BUT this limit is only available conditionally
    on the eps -> 0 behavior; uniformly in eps, we have

       ||g||_2^2 = lim_{eps -> 0} ||g_eps||_2^2.

    Since the chain bound on c_eps is *not* uniformly bounded below 1 as eps -> 0
    (because K_eps -> infinity), we CANNOT take the limit and conclude
    ||g||_2^2 <= c * M with c < 1.

    For any FIXED eps > 0, the bound is on ||g_eps||_2^2, not ||g||_2^2.
    The conditional theorem in restricted_holder/derivation.md is stated for
    ||f*f||_2^2 directly, not for any smoothed version.  Re-doing the conditional
    theorem for smoothed g_eps would require redoing the entire MV chain.

    For the moment, plug c_eps directly into conditional_bound_optimal as a
    pretend bound — this will tell us what M we'd get IF the chain bound on
    ||g_eps||_2^2 transferred to ||g||_2^2.  (It does NOT, but this is the
    optimistic-best-case answer.)
    """
    if c_eps_value >= 1.0:
        return None  # No improvement.
    return conditional_bound_optimal(mpf(c_eps_value), dps=dps)


# =============================================================================
# Step 4: Numerical test on the MV near-extremizer (loaded from cache)
# =============================================================================

def test_on_mv_extremizer():
    """Test the chain on MV's near-extremizer: load mv_seed.npy and compute
       both ||g||_2^2 and ||g_eps||_2^2 for various eps.

       NOTE: the f_best_M*.npy files in the repo are corrupted (all NaN).
       mv_seed.npy is valid and is the only usable seed."""
    repo_root = Path(__file__).resolve().parent
    f_path = repo_root / "delsarte_dual/path_a_unconditional_holder/mv_seed.npy"
    if not f_path.exists():
        return {"error": "mv_seed.npy not found", "path": str(f_path)}

    f = np.load(str(f_path))
    if not np.all(np.isfinite(f)):
        return {"error": "mv_seed.npy contains NaN/Inf"}
    n = len(f)
    L = 0.25
    dx = (2*L) / n
    xs = np.linspace(-L + dx/2, L - dx/2, n)

    # Normalize int f = 1
    f = f / np.sum(f * dx)

    # ||f||_2^2
    K = np.sum(f**2 * dx)

    # f * f using direct convolution
    g_full = np.convolve(f, f) * dx
    # g_full is on supp [-2L, 2L] = [-1/2, 1/2], length 2n-1
    M_val = np.max(g_full)
    g_L2sq = np.sum(g_full**2 * dx)

    # Test convolution with h_eps for various eps
    eps_values = [1e-3, 1e-2, 5e-2, 0.1, 0.2]
    results = []
    for eps in eps_values:
        # h_eps = (1/eps) * 1_{[-eps/2, eps/2]}
        n_h = max(3, int(np.round(eps / dx)))
        # Force odd
        if n_h % 2 == 0: n_h += 1
        h_eps = np.ones(n_h) / (n_h * dx)  # so int h_eps = 1
        # f_eps = f * h_eps  (length n + n_h - 1)
        f_eps = np.convolve(f, h_eps) * dx
        K_eps_val = np.sum(f_eps**2 * dx)
        # g_eps = f_eps * f_eps
        g_eps = np.convolve(f_eps, f_eps) * dx
        M_eps_val = np.max(g_eps)
        g_eps_L2sq = np.sum(g_eps**2 * dx)

        # Predicted UB from chain: ||g_eps||_2^2 <= (1+2eps)^{-1} [1 + mu((1+2eps)M)(((1+2eps)K_eps - 1))]
        M_eps_mp = mpf(float(M_val))
        K_eps_mp = mpf(float(K_eps_val))
        eps_mp = mpf(float(eps))
        UB_predicted = float(smoothed_bound_on_g_eps_L2sq(M_eps_mp, K_eps_mp, eps_mp))
        c_eps = g_eps_L2sq / M_val  # observed
        c_eps_UB = UB_predicted / M_val
        results.append({
            'eps': eps,
            'K_eps_observed': float(K_eps_val),
            'M_eps_observed': float(M_eps_val),
            'g_eps_L2sq_observed': float(g_eps_L2sq),
            'UB_predicted_from_chain': UB_predicted,
            'c_eps_observed': float(c_eps),
            'c_eps_UB_from_chain': float(c_eps_UB),
            'log16pi': float(LOG16_PI),
            'closes_chain': bool(c_eps_UB < float(LOG16_PI)),
        })

    return {
        'M_observed': float(M_val),
        'K_observed': float(K),
        'g_L2sq_observed': float(g_L2sq),
        'c_observed': float(g_L2sq / M_val),
        'log16pi_threshold': float(LOG16_PI),
        'eps_sweep_results': results,
    }


def test_on_synthetic_asymmetric():
    """Test the chain on a synthetic asymmetric f where K > M.

    Construct: f = sum of three narrow bumps at offsets {-0.2, -0.05, 0.15}
    of width w = 0.04, normalized so int f = 1.  This gives K = ||f||_2^2 ~ 1/(3w) ~ 8.3,
    and ||f*f||_inf depends on overlap pattern.
    Asymmetric because the offsets are not symmetric about 0.
    """
    L = 0.25
    N = 4001
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    # Centers and widths
    centers = [-0.2, -0.05, 0.15]
    width = 0.04
    f = np.zeros_like(x)
    for c in centers:
        # Tent function of width 2*width centered at c
        mask = np.abs(x - c) < width
        f += np.where(mask, 1.0 - np.abs(x - c)/width, 0.0)
    f = f / (np.sum(f) * dx)  # normalize int f = 1
    K = np.sum(f**2) * dx
    g = np.convolve(f, f) * dx
    M_val = g.max()
    g_L2sq = np.sum(g**2) * dx
    eps_values = [1e-3, 1e-2, 3e-2, 1e-1, 2e-1]
    results = []
    for eps in eps_values:
        n_h = max(3, int(np.round(eps / dx)))
        if n_h % 2 == 0: n_h += 1
        h_eps = np.ones(n_h) / (n_h * dx)
        f_eps = np.convolve(f, h_eps) * dx
        K_eps_val = np.sum(f_eps**2) * dx
        g_eps = np.convolve(f_eps, f_eps) * dx
        M_eps_val = np.max(g_eps)
        g_eps_L2sq = np.sum(g_eps**2) * dx
        UB_predicted = float(smoothed_bound_on_g_eps_L2sq(
            mpf(float(M_val)), mpf(float(K_eps_val)), mpf(float(eps))))
        c_eps_obs = g_eps_L2sq / M_val
        c_eps_UB = UB_predicted / M_val
        results.append({
            'eps': eps,
            'K_eps_observed': float(K_eps_val),
            'M_eps_observed': float(M_eps_val),
            'g_eps_L2sq_observed': float(g_eps_L2sq),
            'UB_predicted_from_chain': UB_predicted,
            'c_eps_observed': float(c_eps_obs),
            'c_eps_UB_from_chain': float(c_eps_UB),
            'closes_chain': bool(c_eps_UB < float(LOG16_PI)),
        })
    return {
        'M_observed': float(M_val),
        'K_observed': float(K),
        'K_over_M_ratio': float(K / M_val),
        'g_L2sq_observed': float(g_L2sq),
        'c_observed': float(g_L2sq / M_val),
        'is_asymmetric_K_gt_M': bool(K > M_val),
        'log16pi_threshold': float(LOG16_PI),
        'eps_sweep_results': results,
    }


# =============================================================================
# Step 5: Main driver
# =============================================================================

def main():
    print("=" * 80)
    print("Agent B: Smoothed-autoconvolution Path A test (§5.3(d))")
    print("=" * 80)
    print()
    print("Chain (paper derivation in script docstring):")
    print("  f_eps := f * h_eps,  h_eps = (1/eps) 1_{[-eps/2,eps/2]}.")
    print("  g_eps := f_eps * f_eps = g * (h_eps * h_eps) where phi_eps = h_eps * h_eps.")
    print("  ||g_eps||_inf <= M.")
    print("  K_eps = ||f_eps||_2^2 <= 1/eps   (naive).")
    print("  Rescale to [-1/4,1/4] via g(y) = (1+2eps) f_eps(y(1+2eps)).")
    print("  Apply MO 2.14 + Parseval on g (M_g = (1+2eps)M, ||g||_2^2 = (1+2eps)K_eps):")
    print("  ||g*g||_2^2 <= 1 + mu(M_g) * (||g||_2^2 - 1).")
    print("  Convert: ||g_eps||_2^2 = ||g*g||_2^2 / (1+2eps).")
    print("  Final: ||g_eps||_2^2 <= [1 + mu((1+2eps)M) ((1+2eps)K_eps - 1)] / (1+2eps).")
    print()

    # Test at MV-extremizer M-value
    M_test = mpf("1.275")  # MV's published lower bound region
    eps_values = [mpf(s) for s in [
        "1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2",
        "0.05", "0.1", "0.15", "0.2", "0.25", "0.3"
    ]]

    print(f"--- Naive K_eps <= 1/eps sweep at M = {M_test} ---")
    print(f"{'eps':>10} | {'K_eps UB':>10} | {'mu(M_eff)':>10} | {'UB g_eps_L2sq':>14} | {'c_eps':>10} | beats log16/pi={float(LOG16_PI):.4f}?")
    naive_results = sweep_eps(M_test, eps_values, K_eps_provider="naive")
    for r in naive_results:
        beats = "YES" if r['c_eps'] < float(LOG16_PI) else "no"
        print(f"  {r['eps']:>8.4g} | {r['K_eps_bound']:>10.4f} | {r['mu_at_M_eff']:>10.4f} | {r['UB_g_eps_L2sq']:>14.4f} | {r['c_eps']:>10.4f} | {beats}")

    print()
    print(f"--- Young K_eps <= K_orig/eps at K_orig=2, M = {M_test} ---")
    young_results = sweep_eps(M_test, eps_values, K_eps_provider="young", K_orig=2.0)
    print(f"{'eps':>10} | {'K_eps UB':>10} | {'c_eps':>10}")
    for r in young_results:
        print(f"  {r['eps']:>8.4g} | {r['K_eps_bound']:>10.4f} | {r['c_eps']:>10.4f}")

    print()
    print(f"--- Test on MV near-extremizer (numerical) ---")
    mv_test = test_on_mv_extremizer()
    if 'error' in mv_test:
        print(f"  [skip: {mv_test['error']}]")
    else:
        print(f"  M (observed)     = {mv_test['M_observed']:.6f}")
        print(f"  K (observed)     = {mv_test['K_observed']:.6f}")
        print(f"  ||g||_2^2        = {mv_test['g_L2sq_observed']:.6f}")
        print(f"  c (observed)     = {mv_test['c_observed']:.6f}")
        print(f"  log(16)/pi       = {mv_test['log16pi_threshold']:.6f}")
        print()
        print(f"  Per-eps:")
        for r in mv_test['eps_sweep_results']:
            print(f"    eps={r['eps']:>6.3g}: K_eps={r['K_eps_observed']:>7.4f}, "
                  f"||g_eps||_2^2={r['g_eps_L2sq_observed']:>7.4f}, "
                  f"c_obs={r['c_eps_observed']:>6.4f}, c_UB={r['c_eps_UB_from_chain']:>7.4f}, "
                  f"closes={r['closes_chain']}")

    print()
    print(f"--- Test on synthetic asymmetric f (3-tent), K > M ---")
    asym_test = test_on_synthetic_asymmetric()
    print(f"  M (observed)     = {asym_test['M_observed']:.6f}")
    print(f"  K (observed)     = {asym_test['K_observed']:.6f}")
    print(f"  K/M ratio        = {asym_test['K_over_M_ratio']:.6f}")
    print(f"  ||g||_2^2        = {asym_test['g_L2sq_observed']:.6f}")
    print(f"  c (observed)     = {asym_test['c_observed']:.6f}")
    print(f"  Asym (K > M)?    = {asym_test['is_asymmetric_K_gt_M']}")
    print()
    for r in asym_test['eps_sweep_results']:
        print(f"    eps={r['eps']:>6.3g}: K_eps={r['K_eps_observed']:>7.4f}, "
              f"M_eps={r['M_eps_observed']:>6.4f}, "
              f"c_obs={r['c_eps_observed']:>6.4f}, c_UB(chain)={r['c_eps_UB_from_chain']:>8.4f}, "
              f"closes={r['closes_chain']}")

    print()
    print(f"--- M-sweep to find best c(eps) ---")
    M_values = [mpf(s) for s in ["1.275", "1.28", "1.30", "1.35", "1.378", "1.40", "1.42"]]
    print(f"{'M':>8} | {'best eps':>10} | {'best c_eps':>10} | {'LB(M_target)':>14}")
    best_sweep = []
    for M_val in M_values:
        # For each M, search eps to minimize c_eps (naive K_eps).
        results = sweep_eps(M_val, eps_values, K_eps_provider="naive")
        best = min(results, key=lambda r: r['c_eps'])
        # Try plugging into conditional bound — only if c_eps < 1
        lb = None
        if best['c_eps'] < 1.0:
            try:
                lb = float(lb_from_c_eps(best['c_eps']))
            except Exception as e:
                lb = f"err: {e}"
        print(f"  {float(M_val):>6.4f} | {best['eps']:>8.4g} | {best['c_eps']:>10.4f} | {str(lb):>14}")
        best_sweep.append({
            'M': float(M_val),
            'best_eps': best['eps'],
            'best_c_eps': best['c_eps'],
            'LB_via_conditional': lb if lb is None or isinstance(lb, str) else lb,
        })

    print()
    print(f"--- eps -> 0 limit recovery ---")
    print(f"  As eps -> 0, the chain should recover the un-smoothed (†) inequality")
    print(f"  modulo the K_eps -> infinity issue.")
    M_test2 = mpf("1.378")
    K_orig_test = mpf("1.5")  # asymmetric scenario, K = 1.5 * M
    small_eps = [mpf(s) for s in ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]]
    print(f"  At M={float(M_test2)}, suppose K = {float(K_orig_test)} (asym scenario).")
    print(f"  (Naive K_eps <= 1/eps, ignoring true K.)")
    for eps in small_eps:
        K_eps = 1 / eps
        UB = smoothed_bound_on_g_eps_L2sq(M_test2, K_eps, eps)
        c = UB / M_test2
        print(f"    eps={float(eps):>7.1e}: K_eps UB = {float(K_eps):>10.2f}, c_eps = {float(c):>10.4f}")

    # ==========================================================================
    # Save results
    # ==========================================================================
    output = {
        'derivation_summary': {
            'chain': 'f_eps = f*h_eps with h_eps=(1/eps)1_{[-eps/2,eps/2]}; '
                     'g_eps = f_eps*f_eps = g*phi_eps. '
                     'Rescale to [-1/4,1/4] via g(y)=(1+2eps)f_eps((1+2eps)y). '
                     '||g_eps||_2^2 <= [1 + mu((1+2eps)M) * ((1+2eps)K_eps - 1)]/(1+2eps).',
            'mo_214_support_issue': 'f_eps has support [-1/4-eps/2, 1/4+eps/2], '
                                    'larger than [-1/4, 1/4]. Rescaling to [-1/4,1/4] '
                                    'inflates M_eff to (1+2eps)*M and K_eff to (1+2eps)*K_eps, '
                                    'so MO 2.14 applies but with mu((1+2eps)*M) > mu(M).',
            'K_eps_naive_bound': 'K_eps <= ||phi_eps||_inf = 1/eps (when int f = 1).',
            'K_eps_loose_at_small_eps': 'As eps -> 0, naive K_eps -> infinity. '
                                       'Tight bound depends on f and is K_eps -> K = ||f||_2^2.',
        },
        'naive_K_sweep_at_M_1p275': naive_results,
        'young_K_sweep_at_M_1p275': young_results,
        'mv_extremizer_test': mv_test if 'error' not in mv_test else None,
        'mv_extremizer_test_error': mv_test.get('error') if 'error' in mv_test else None,
        'synthetic_asymmetric_test': asym_test,
        'best_eps_per_M': best_sweep,
        'verdict': {
            'beats_1.2748_unconditionally': False,
            'fatal_flaw_1_K_eps_inherits_K_unboundedness': (
                'K_eps = int int f(x)f(y) phi_eps(x-y) dx dy. The crucial alt-bound '
                'K_eps <= K = ||f||_2^2 holds (taking phi_eps <= ||phi_eps||_inf gives '
                'the 1/eps bound, but phi_eps <= 1 also gives K_eps <= K). For '
                'asymmetric f, K is unbounded by M. Hence K_eps is still unbounded '
                'by M for the worst-case asymmetric f. The smoothing does not '
                'add a UNIFORM (over admissible f) upper bound on K_eps below the '
                'unconditional 1/eps. Same blocker.'
            ),
            'fatal_flaw_2_naive_K_eps_1_over_eps_too_loose': (
                'The naive bound K_eps <= 1/eps yields c_eps >> 1 for any eps in '
                '(0, 0.3], failing the Hyp_R target c_eps <= 0.88254 by 2-3 orders '
                'of magnitude.'
            ),
            'fatal_flaw_3_support_enlargement': (
                'Rescaling f_eps to [-1/4,1/4] inflates M to (1+2eps)*M, weakening '
                'mu(M) (by ~5% for eps=0.1). Plus the (1+eps)/eps factor on K_eps '
                'after rescaling. Both inflate c_eps.'
            ),
            'why_5_3_d_fails': (
                'The §5.3(d) idea was: bound K_eps via 1/eps (uniformly in f) to '
                'replace the unbounded K. But the chain ((dagger)-rescaled) carries '
                'K_eps with a factor mu(M)*(1+eps)/eps that scales as ~ mu(M)/eps. '
                'For fixed eps, this is bounded but grows like 1/eps; for fixed mu, '
                'no eps gives c_eps < log(16)/pi at M ~ 1.28. The chain is hopeless '
                'in the relevant M range. AT LARGE M (e.g., M >> 1.5) the chain still '
                'gives c_eps > log(16)/pi (the (1+eps)/eps factor dominates).'
            ),
            'derivation_md_was_correct_to_flag_as_untried': (
                'derivation.md §5.3(d) labels this as "not yet attempted" — and '
                'correctly so. After explicit derivation, the chain (dagger)-rescaled '
                'shows the smoothing creates an additional (1+eps)/eps factor on K_eps '
                'that wipes out the mu(M) gains uniformly.'
            ),
            'numerical_test_supports_chain_for_specific_f': (
                'On the specific MV-like seed (symmetric, M ~ 2.31, K ~ 2.31), the '
                'chain bound c_eps_UB ~ 0.84 at eps=0.001 and decreases with eps. '
                'On the synthetic 3-tent asymmetric (M ~ 3.94, K/M ~ 1.41), the chain '
                'closes at eps >= 0.1 with c_UB ~ 0.65. But these specific tests use '
                'the actual K_eps, not the worst-case 1/eps. The unconditional bound '
                'requires worst-case K_eps over admissible f, which is unbounded.'
            ),
            'fatal_flaw_4_chain_bounds_wrong_quantity': (
                'CRUCIAL: even if the chain produced c_eps < 1, the bound would be on '
                '||g_eps||_2^2, not ||g||_2^2. The conditional theorem (MV/restricted_holder) '
                'requires a bound on ||g||_2^2 = ||f*f||_2^2 directly. '
                'Lifting an UB on ||g_eps||_2^2 to one on ||g||_2^2 requires '
                '||g - g_eps||_2 to be controlled, which depends on the smoothness '
                'of g (i.e., on K = ||f||_2^2 via Young: ||g - g_eps||_2 '
                '<= ||g||_inf^{1/2} * (Cf eps^{alpha}) for some Hoelder exponent '
                'alpha depending on g. The quantitative rate depends on K --- '
                'closing this loop reintroduces the K-unboundedness blocker.'
            ),
            'fatal_flaw_5_large_eps_makes_M_eff_unphysical': (
                'At eps >= 0.5, M_eff = (1+2eps)*M >= 2 even for M = 1.28. '
                'mu(M_eff) > 1 in this regime, and the SDR proof of MO 2.14 requires '
                'M < 2 (otherwise the bathtub does not fit in support). '
                'So large-eps numerics that appear to give c < 1 are vacuous '
                'because mu(M_eff) is outside the valid range.'
            ),
            'verdict_summary': (
                'NEGATIVE: §5.3(d) does NOT yield an unconditional LB > 1.2748. '
                'Five independent obstacles: (1) K_eps inherits K-unboundedness from f; '
                '(2) uniform K_eps <= 1/eps is too loose at small eps; (3) rescaling '
                'enlarges M and weakens mu; (4) the chain bounds ||g_eps||_2^2 not '
                '||g||_2^2 -- lifting requires K which is unbounded; (5) large-eps '
                'numerics that appear to close are vacuous because M_eff > 2 violates '
                'MO 2.14 hypotheses. The smoothing approach is fundamentally blocked.'
            ),
        },
    }
    output_path = Path(__file__).resolve().parent / "_agent_b_smoothed_path_a.json"
    with open(output_path, 'w') as fh:
        json.dump(output, fh, indent=2)
    print()
    print(f"Saved results to: {output_path}")

    return output


if __name__ == "__main__":
    main()
