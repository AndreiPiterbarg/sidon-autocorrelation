"""Sharper-Markov route to break MV's 1.276 ceiling.

The MV master inequality has a tight Cauchy-Schwarz step but a loose
Markov step

    ||h||_2^2  =  ||(f*f)||_2^2  <=  ||f*f||_infty  =  M.

Replacing this with the sharper

    ||h||_2^2  <=  1 + mu(M) * (||f||_2^2 - 1)        (Lemma 1 + Plancherel)

gives a strict improvement IF we can bound ||f||_2^2 <= b_bar(M) with
b_bar small enough.  The breakeven at M=1.276 is b_bar < 2.08; achieving
b_bar ~ 1.5-2 would lift the bound to ~1.30.

This module:
  * `b_aux_shor.py`     Shor SDP for the auxiliary max ||f||_2^2 problem.
  * `b_aux_lasserre.py` Lasserre level-2 tighter bound.
  * `master_sharper.py` New master inequality with b_bar plugged in.
  * `verify.py`         Rigorous wrapper; mpmath / interval certification.

Author chain: derived 2026-05 from Agent 1 / Agent 2 / Agent 3 synthesis;
all three agents independently identified ||f||_2^2 control as the only
viable path past the MV ceiling.
"""
