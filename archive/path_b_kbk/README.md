# archive/path_b_kbk/

Path B — Krein–Boas–Kac (KBK) SDP attempt to close Hyp_R unconditionally.

## Goal

Prove the restricted-Hölder hypothesis Hyp_R$(c_*, M_{\max})$ with
$c_* = \log(16)/\pi$ at $M_{\max} = 1.378$ via a phase-aware Bochner +
KBK-localising SDP. A successful closure would yield $C_{1a} \ge 1.378$
unconditionally.

## Files

- `__init__.py`  — empty package marker.
- `kbk_sdp.py`   — KBK SDP builder (phase-aware Bochner lift, $y$-Toeplitz
                   localising blocks, KBK constraint family).
- `run_kbk.py`   — driver: baseline (no KBK), phase-aware-only baseline,
                   and full KBK sweep over $(N, K_{\text{trunc}}, K_{\text{ub}})$.

## Status

DEAD. The KBK relaxation hit Shor / Lasserre looseness at $M = 1.378$ —
SDP value never crossed the $c_*$ threshold required to certify Hyp_R.
Writeup archived prior to publication; code retained here in case a future
tighter localiser revives the attack.
