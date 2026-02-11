# Optimization Experiment Results — Session 2

## Setup
- Compute budget: 60 seconds per method at P=200 (unless noted)
- P values tested: 200 (focused on beating P=200 baseline)
- Baseline: Hybrid LSE+Polyak = **1.510357** (best of 3 trials @ 90s), **1.512053** (60s single trial)
- Hardware: Local machine, single-threaded per restart
- All solutions verified via `autoconv_coeffs(x, P)` and stored as JSON

## Baseline Results
| P | Best Value (60s) | Best Value (90s, 3 trials) | Mean (90s) |
|---|------------------|---------------------------|------------|
| 200 | 1.512053 | 1.510357 | 1.511353 |

## Experiments

(experiments will be added below as they complete)
