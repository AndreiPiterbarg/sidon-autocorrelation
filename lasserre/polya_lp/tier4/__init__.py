"""Tier-4 PDLP -> active-set -> MOSEK polish -> Jansson rigorize pipeline.

Layers (each is a separate module):
  pdlp_robust.py   -- GPU-resident restarted Halpern-PDHG (NEW; replaces
                      the broken lasserre/polya_lp/pdlp.py for this pipeline)
  active_set.py    -- extract optimal active set from a coarse PDLP solution
  polish.py        -- MOSEK polish on the reduced LP at 1e-9 tolerance
  rigorize.py      -- Jansson 2004 directed-down-rounding rigorous LB
  driver.py        -- composes all of the above into tier4_solve(d, R)

Design constraints (per user):
  * NO modifications to existing lasserre/polya_lp/*.py — only new files.
  * Local validation on RTX 3080 Laptop (8 GB) at d <= 16, small R first.
"""
