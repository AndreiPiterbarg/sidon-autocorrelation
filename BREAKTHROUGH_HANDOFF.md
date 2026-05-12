# Breakthrough Run — Handoff to Pod Agent

## Goal
Prove `C_{1a} > 1.2802` (current LB) via Polya/Handelman LP at `d=16`, with breathing room. Per the M1 model (`gap(R) = C_d/R`, `val_16 ≈ 1.319`, `C_16 ≈ 1.027`), the run that should clear it is **d=16 R=35**, predicted α ≈ 1.290 (+0.010 above current LB).

The launch script tries a trajectory `R = 28 → 30 → 33 → 35` so we (a) confirm the model along the way and (b) have a fallback bound if the largest case times out.

## Pod requirements
- **CPU pod** (no GPU needed — MOSEK IPM is CPU only)
- **System RAM ≥ 256 GB** (768 GB is plenty; the largest case is predicted at ~50-150 GB factor)
- **≥ 16 cores** (32+ preferred — MOSEK IPM scales with threads on Cholesky)
- Ubuntu 20.04+ with Python 3.10+
- 50 GB free disk

## What the agent must do

### 1. SSH in
```sh
ssh -i "<path to private_key (23).pem>" ubuntu@<NEW_POD_IP>
```
(Use the same private key as the prior pod, in `~/Downloads/private_key (23).pem` on the user's Windows machine.)

### 2. Install deps
```sh
pip3 install --user numpy scipy mosek highspy psutil
```

### 3. Add MOSEK license
The license file is at `C:/Users/andre/mosek/mosek.lic` on the user's Windows machine. Copy it to the pod:
```sh
# from local Windows shell:
ssh -i "<key>" ubuntu@<POD_IP> "mkdir -p ~/mosek"
scp -i "<key>" "C:/Users/andre/mosek/mosek.lic" ubuntu@<POD_IP>:~/mosek/mosek.lic
```
Verify with:
```sh
ssh -i "<key>" ubuntu@<POD_IP> "python3 -c 'import mosek
with mosek.Env() as e, e.Task() as t:
    t.appendvars(1); t.putvarbound(0, mosek.boundkey.lo, 0.0, 0.0); t.putcj(0, -1.0)
    t.putobjsense(mosek.objsense.minimize); t.optimize()
    print(\"MOSEK_OK\")'"
```
Expected output: `MOSEK_OK`.

### 4. Upload code
From the user's Windows machine, in the `compact_sidon` repo directory:
```sh
tar -czf /tmp/sidon_polya.tar.gz lasserre/polya_lp _breakthrough_run.py
scp -i "<key>" /tmp/sidon_polya.tar.gz ubuntu@<POD_IP>:~/sidon.tar.gz
ssh -i "<key>" ubuntu@<POD_IP> "mkdir -p ~/sidon && cd ~/sidon && tar xzf ~/sidon.tar.gz"
```

### 5. Launch
On the pod:
```sh
cd ~/sidon
nohup python3 -u _breakthrough_run.py > breakthrough.log 2>&1 &
echo "PID=$!"
```
Save the PID — you'll need it to check liveness.

### 6. Monitor
```sh
# tail the human-readable log:
tail -f ~/sidon/breakthrough.log

# inspect cumulative results so far:
cat ~/sidon/breakthrough_results.json
```

Each case prints `BUILD_DONE rows=… vars=… nnz=… t_build=…s` immediately, then MOSEK IPM iterations every ~minute, then a `RESULT:` JSON line on success.

### 7. Expected runtime per case
| Case | Predicted α | rows | nnz | RAM peak | wall time |
|---|---|---|---|---|---|
| d=16 R=28 | 1.282 | ~25M | ~250M | ~25 GB | ~30-60 min |
| d=16 R=30 | 1.285 | ~30M | ~300M | ~30 GB | ~45-90 min |
| d=16 R=33 | 1.288 | ~80M | ~700M | ~70 GB | ~2-4 h |
| d=16 R=35 | 1.290 | ~145M | ~1.2B | ~120 GB | ~3-6 h |

Total worst case: ~12 h. Per-task wall budget is 6 h (configured in script), so individual cases that hang get cut off and the next one starts.

### 8. Success criteria
A case `succeeds` if its `RESULT:` JSON has `alpha > 1.2802`. The script prints `** CLEARS 1.2802 by +Δ **` when it does. The summary at the bottom tells you the best α achieved overall.

If **any case** clears 1.2802, the run is a success — report back the (d, R, α) tuple. The R=35 case is the keeper since it has the largest margin.

### 9. Reporting back
Send to the user:
- Final `breakthrough_results.json` (zip or paste)
- Summary table from the bottom of `breakthrough.log`
- The best α and its (d, R)
- Total wall time consumed

If everything fails (all cases time out or return non-OPTIMAL): paste the MOSEK iteration log from one failing case and the relevant section of `breakthrough.log` so we can diagnose.

## What NOT to change
- **Do not** flip `eliminate_c_slacks` to `True` in `_breakthrough_run.py`. The eliminated form makes MOSEK solve_form=DUAL build a 776 M-nonzero KKT factor and stall (we measured this at d=16 R=12 — it sat 9+ minutes per case).
- **Do not** disable the Z/2 reduction (`use_z2=True`). Doubles the LP size for nothing.
- **Do not** swap to HiGHS unless MOSEK is unrecoverable — HiGHS doesn't have the IPM tolerances we want for rigorous bounds.

## What to do if MOSEK runs out of memory
Two options, in order:
1. **Ride it**: lower R. If R=35 OOMs, the run schedule already includes R=33, R=30, R=28 — the script will continue to the next case after the OOM-ed case errors out. R=30 still gives α ≈ 1.285 (clears 1.2802 with +0.005 margin).
2. **Last resort**: enable `eliminate_c_slacks=True` AND change `solve.py` to use `intpnt_solve_form = mosek.solveform.primal` instead of `dual`. This is untested for these LPs — only do this if (1) failed and you must squeeze memory.

## Status of the codebase
- Vectorized build (3-17× speedup over the prior naive Python loops, matters for d=16 R≥30 build phases).
- Z/2 symmetry reduction: ON.
- MOSEK IPM with tuned options (presolve eliminator cascading, basis identification OFF, all cores).
- `c_slack` elimination: implemented but **disabled** for this run (see above).

## Question to bounce back to the user if blocked
- Pod IP / host?
- Pod cores / RAM (to confirm we have ≥ 256 GB and ≥ 16 cores)?
- Time budget (default: ride to completion or ~12 h, whichever first)?
