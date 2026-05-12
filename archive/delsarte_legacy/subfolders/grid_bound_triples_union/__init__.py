"""Grid-bound pipeline extended to a UNION of (delta, K, G) triples.

MV Remark #2 (mv_construction_detailed.md lines 493-499) points out that each
(delta, K, G) triple produces a forbidden set F_i subset [0, sqrt(||f*f||_inf)]
for z_1.  A single triple leaves a non-forbidden slice; a FAMILY of triples
whose forbidden slices COVER the full z_1 range certifies the bound directly.

In the full 2N-D (a, b) cell-search this becomes: a cell is forbidden if ANY
triple's Phi_MM (with its own delta, u, k_n, gain_a) has upper < 0.  This
effectively enlarges the dual ansatz by sweeping delta -- each delta re-tunes
G_i optimally -- and can push M_cert above the single-triple ceiling.

Files
-----
 - ``triples.py``           build per-delta PhiMMParams
 - ``phi_union.py``         evaluate every triple's Phi on one cell
 - ``cell_search_union.py`` cell-search that rejects when ANY triple rejects
 - ``bisect_union.py``      driver + certificate emitter
 - ``certify_union.py``     independent verifier
 - ``G_by_delta/``          QP coefficients per delta (JSON)
 - ``certificates/``        output certificates
"""
