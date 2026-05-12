"""Multi-frequency MO 2004 framework: Prop 2.11 (m=3) + Lemma 2.17 + Lemma 2.14.

Goal:  Combine the m=3 case of Martin-O'Bryant 2004 Proposition 2.11 with the
joint linear constraint of MO 2004 Lemma 2.17 in the Matolcsi-Vinuesa dual
framework, and report whether this lifts the rigorous lower bound on
C_{1a} = inf ||f*f||_oo above the Cloninger-Steinerberger 2017 value 1.2802.

Modules:
    qp_solver     The master inequality + bisection.
    certificate   Dual certificate (only emitted if the QP optimum > 1.2802).
"""
