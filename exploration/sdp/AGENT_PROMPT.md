# SDP Lower Bound Agent — Exploration Sprint

## Setup

You are working in `exploration/sdp/` within the sidon-autocorrelation repository. Activate the virtual environment with `.venv` before running anything.

**First**: Read `RESULTS.md` in this folder thoroughly. It contains the current best results, what works, what's been tried, what failed, and the promising directions. Do not skip this step. Also read `core_utils.py` and `exp_fast_lasserre.py` to understand the existing infrastructure and best approach. Skim the files in `promising/` to understand what's been partially explored.

## Objective

Your goal is to **maximize the SDP lower bound on V(P)** — the discretized version of C_{1a}. The current best lower bounds come from Lasserre Level-2 (see RESULTS.md). You want to either:

1. **Tighten the bound at existing P values** (beat the numbers in RESULTS.md)
2. **Scale to larger P** (P=25, 30, 50+) where current methods are too slow
3. **Both**

Higher lower bounds are better. The theoretical target is C_{1a} ~ 1.5029.

## Constraints

- You are running on a **laptop** with limited compute. No GPU, no cloud. Plan accordingly — prefer methods that run in minutes, not hours. If something takes more than ~10 minutes for a single run, it's probably too expensive.
- You have **3 hours** total. If you hit a Claude usage limit, just wait for it to reset and continue where you left off.
- Use MOSEK as the SDP solver (it's already installed and used in the existing code). Fall back to SCS if needed for larger problems.

## Approach

Spend your time trying as many **different promising techniques** as possible. Do not get stuck polishing one approach — breadth over depth. If something isn't working after a reasonable implementation effort, move on.

Some directions to consider (but don't limit yourself to these):
- Improve the promising experiments in `promising/` (they each have a "How to fix" column in RESULTS.md)
- Novel SDP formulations or relaxation hierarchies
- Exploit the specific structure of the A_k matrices (antidiagonal, 1-2 nonzeros per row)
- Cutting planes or constraint generation to tighten existing relaxations
- Continuous (non-discretized) SDP formulations
- Smarter dimensionality reduction to push Lasserre-2 to higher P
- Combine techniques that individually get close to Lasserre-2

**Do NOT** retry the dead ends listed in RESULTS.md. They are dead for mathematically fundamental reasons.

## Implementation Standards

- Keep implementations **simple and correct**. No overengineering. A clean 100-line script that works beats a 500-line framework that's buggy.
- **Test on small P first** (P=5 or P=8) where you can verify against known results before scaling up.
- Each experiment should be a standalone `.py` file in this folder.
- Name files descriptively: `exp_<technique_name>.py`
- Print results clearly: P value, lower bound obtained, time taken.
- Reuse utilities from `core_utils.py` where applicable — don't reinvent the wheel.

## Results Tracking

As you work, keep a running log. When you are done (or time is up), create a file called `AGENT_RESULTS.md` in this folder with:

1. **Summary table**: Every technique you tried, the P values tested, the bounds obtained, and wall-clock time
2. **What worked**: Rank your approaches by effectiveness
3. **What didn't work**: Brief notes on why, so future work doesn't repeat it
4. **Best result**: Your single best lower bound at each P, and which method produced it
5. **Recommendations**: What you'd try next with more time/compute
