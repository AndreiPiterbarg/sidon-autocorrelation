# Sanity check report — `compact_sidon`

Date: 2026-05-12

## Goal

Per `README.md` → *Running the full audit*: instantiate a Python venv in
`C:\Python312\envs`, install `requirements.txt`, and run
`python audit_consistency.py`.

## Steps taken

1. **Inspected `README.md`** to confirm the documented entry point
   (`python audit_consistency.py`) and the precision settings it advertises
   (256-bit `flint.arb`, full pipeline rerun, 8 check sections).

2. **Inspected `requirements.txt`**. Listed runtime deps:
   `numpy>=2.0`, `numba>=0.60`, `joblib>=1.3`, `matplotlib>=3.8`,
   `runpod>=1.7.0`, `python-dotenv>=1.0`, `mpmath>=1.3`, `sympy>=1.12`,
   `scipy>=1.10`. `python-flint>=0.6` listed but **commented out** as
   "Optional, used by delsarte_dual if available".

3. **Created the venv**:

   ```bash
   C:/Python312/python.exe -m venv C:/Python312/envs/compact_sidon
   ```

4. **Upgraded pip** in the new venv (`pip 24.3.1` → `pip 26.1.1`).

5. **Installed `requirements.txt`** — succeeded, large transitive closure
   driven mostly by `runpod` (FastAPI, boto3, cryptography, …).

6. **First audit attempt failed** with:

   ```
   ModuleNotFoundError: No module named 'flint'
   ```

   Audit script imports `from flint import arb, fmpq, ctx` unconditionally
   (`audit_consistency.py:54`), so `python-flint` is **not actually optional**
   despite the comment in `requirements.txt`.

7. **Installed `python-flint`** → `python-flint-0.8.0`.

8. **Second audit attempt failed** with:

   ```
   ModuleNotFoundError: No module named 'cvxpy'
   ```

   imported from `delsarte_dual/grid_bound_alt_kernel/optimize_G.py:104`
   (`solve_qp_for_kernel`), which the audit invokes from
   `compile_ground_truth` (`audit_consistency.py:194`). `cvxpy` is not in
   `requirements.txt` at all.

9. **Installed `cvxpy`** → `cvxpy-1.8.2` (plus solver backends
   `clarabel`, `osqp`, `scs`, `highspy`).

10. **Third audit attempt completed end-to-end** — pipeline re-ran at
    `n=200, prec=256, n_cells_min_G=32768`. 50 checks executed, 1 failed.

## Certifier output (interval arithmetic)

```
k_1   in [0.9212465899364083, 0.9212465899364083]
K_2   in [4.7888234212591554, 4.7889051816332424]
S_1   in [29.8409043109686642, 29.8409043109686642]
min G in [0.9999798648629120, 0.9999798648629120]
gain  in [0.2100921585851016, 0.2100921585851016]
```

## Audit result

`TOTAL: 50 checks, 1 failed` — `VERDICT: AUDIT FAILED`.

### Single failing check

Section **D. Tight-decimal claims across all surfaces**:

> `min_G >= 0.99997987`

Certifier produced `min G = 0.99997986486…`, which is strictly less than the
claimed `0.99997987` by ~5e-9. The published surfaces (LaTeX / README / Lean /
JSON anchors / docs) appear to round the computed value up at the 8th decimal,
whereas the certifier value rounds down, so the inequality fails.

### Possible fixes

1. Relax the claim on all five surfaces to `min_G >= 0.99997986`
   (matches certifier; conservative).
2. Re-run with more cells / higher precision to push `min G` above
   `0.99997987` before patching anchors.

## Suggested follow-up to `requirements.txt`

The audit cannot run without `python-flint` and `cvxpy`. Recommend:

- Uncomment `python-flint>=0.6` (or move it into the required section).
- Add `cvxpy` to the required section.
