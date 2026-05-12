#!/usr/bin/env julia
#=
Lasserre hierarchy via TSSOS — exploiting term + correlative sparsity.

The full SDP at d=16 order=3 has a 969×969 moment matrix (74K variables,
496 × 153×153 PSD constraints). TSSOS decomposes this into many small
block-diagonal SDPs by exploiting:

1. TERM SPARSITY: each window constraint t - TV_W(mu) only involves
   monomials mu_i*mu_j where i+j falls in the window. The degree-4
   moments mu_i*mu_j*mu_a*mu_b are sparse — most don't appear in any
   single constraint. TSSOS detects this and decomposes the 969×969
   moment matrix into smaller blocks.

2. CORRELATIVE SPARSITY: variables mu_i and mu_j only interact through
   windows containing i+j. For small windows (ell=2), only a few pairs
   interact. CS-TSSOS detects this and decomposes the problem into
   overlapping cliques.

Expected speedup: 100-1000× for large d, because O(969³) factorization
becomes O(sum of block_k³) where each block_k << 969.

Usage:
  julia tests/lasserre_tssos.jl                          # d=4, order=2
  julia tests/lasserre_tssos.jl --d 8 --order 2
  julia tests/lasserre_tssos.jl --d 16 --order 2
  julia tests/lasserre_tssos.jl --d 16 --order 3         # the big one

Install deps first:
  julia -e 'using Pkg; Pkg.add(["DynamicPolynomials", "MosekTools", "JuMP"])'
  julia -e 'using Pkg; Pkg.add(url="https://github.com/wangjie212/TSSOS")'
=#

using DynamicPolynomials
using TSSOS
using Printf
using LinearAlgebra

# =====================================================================
# Build the polynomial optimization problem
# =====================================================================

function build_pop(d::Int)
    """Build the POP for val(d) = min_mu max_W TV_W(mu).

    Variables: mu[1:d] (mass distribution), t (upper bound)
    Minimize: t
    Subject to:
      mu[i] >= 0              (d inequality constraints)
      t - TV_W(mu) >= 0       (n_win inequality constraints)
      sum(mu) - 1 = 0         (1 equality constraint)
    """

    # Variables
    @polyvar mu[1:d] t

    x = [mu..., t]   # all variables

    # Objective: minimize t
    obj = t

    # Build window constraints
    conv_len = 2d - 1
    ineq_constraints = Polynomial{true, Float64}[]

    # mu_i >= 0
    for i in 1:d
        push!(ineq_constraints, mu[i] + 0.0)  # ensure Float64
    end

    # t - TV_W(mu) >= 0 for each window
    n_win = 0
    for ell in 2:(2d)
        for s_lo in 0:(conv_len - ell + 1)
            # TV_W = (2d/ell) * sum_{i+j in [s_lo, s_lo+ell-2]} mu[i+1]*mu[j+1]
            # (0-indexed i,j converted to 1-indexed mu)
            tv = zero(Polynomial{true, Float64})
            coeff = 2.0 * d / ell
            for i in 0:(d-1)
                for j in 0:(d-1)
                    s = i + j
                    if s >= s_lo && s <= s_lo + ell - 2
                        tv += coeff * mu[i+1] * mu[j+1]
                    end
                end
            end
            push!(ineq_constraints, t - tv)
            n_win += 1
        end
    end

    # Equality: sum(mu) = 1
    eq_constraint = sum(mu) - 1.0

    # Assemble pop = [obj, ineq_1, ..., ineq_m, eq_1]
    pop = [obj; ineq_constraints; eq_constraint]

    return pop, x, n_win
end


# =====================================================================
# Solve with different TSSOS strategies
# =====================================================================

function solve_dense_lasserre(pop, x, relax_order; verbose=true)
    """Standard Lasserre (no sparsity) — baseline."""
    verbose && println("  Solving dense Lasserre (order=$relax_order)...")
    t0 = time()
    opt, sol, data = tssos(pop, x, relax_order,
        numeq=1, TS=false, QUIET=!verbose, solution=false)
    elapsed = time() - t0
    verbose && @printf("  Dense: opt=%.10f, time=%.1fs\n", opt, elapsed)
    return opt, elapsed, data
end


function solve_tssos_block(pop, x, relax_order; verbose=true)
    """TSSOS with block term sparsity (no correlative sparsity)."""
    verbose && println("  Solving TSSOS-block (order=$relax_order)...")
    t0 = time()
    opt, sol, data = tssos(pop, x, relax_order,
        numeq=1, TS="block", QUIET=!verbose, solution=false)
    elapsed = time() - t0
    verbose && @printf("  TSSOS-block: opt=%.10f, time=%.1fs\n", opt, elapsed)
    if verbose && hasfield(typeof(data), :blocksize)
        try
            bs = data.blocksize
            println("  Block sizes: max=$(maximum(bs[1])), ",
                    "n_blocks=$(length(bs[1])), ",
                    "total=$(sum(bs[1]))")
        catch; end
    end
    return opt, elapsed, data
end


function solve_tssos_md(pop, x, relax_order; verbose=true)
    """TSSOS with MD term sparsity."""
    verbose && println("  Solving TSSOS-MD (order=$relax_order)...")
    t0 = time()
    opt, sol, data = tssos(pop, x, relax_order,
        numeq=1, TS="MD", QUIET=!verbose, solution=false)
    elapsed = time() - t0
    verbose && @printf("  TSSOS-MD: opt=%.10f, time=%.1fs\n", opt, elapsed)
    return opt, elapsed, data
end


function solve_cs_tssos(pop, x, relax_order; verbose=true)
    """CS-TSSOS: correlative + term sparsity."""
    verbose && println("  Solving CS-TSSOS (order=$relax_order)...")
    t0 = time()
    opt, sol, data = cs_tssos(pop, x, relax_order,
        numeq=1, CS="MF", TS="block", QUIET=!verbose, solution=false)
    elapsed = time() - t0
    verbose && @printf("  CS-TSSOS: opt=%.10f, time=%.1fs\n", opt, elapsed)
    return opt, elapsed, data
end


function solve_cs_tssos_md(pop, x, relax_order; verbose=true)
    """CS-TSSOS with MD sparsity."""
    verbose && println("  Solving CS-TSSOS-MD (order=$relax_order)...")
    t0 = time()
    opt, sol, data = cs_tssos(pop, x, relax_order,
        numeq=1, CS="MF", TS="MD", QUIET=!verbose, solution=false)
    elapsed = time() - t0
    verbose && @printf("  CS-TSSOS-MD: opt=%.10f, time=%.1fs\n", opt, elapsed)
    return opt, elapsed, data
end


# =====================================================================
# Main benchmark
# =====================================================================

function main()
    # Parse args
    d = 4
    order = 2
    for i in 1:length(ARGS)
        if ARGS[i] == "--d" && i < length(ARGS)
            d = parse(Int, ARGS[i+1])
        elseif ARGS[i] == "--order" && i < length(ARGS)
            order = parse(Int, ARGS[i+1])
        end
    end

    println("=" ^ 60)
    println("LASSERRE via TSSOS: d=$d, order=$order (degree $(2*order))")
    println("=" ^ 60)
    println()

    # Build the POP
    println("Building POP...")
    t0 = time()
    pop, x, n_win = build_pop(d)
    build_time = time() - t0
    n_ineq = d + n_win
    @printf("  %d variables, %d inequality constraints, 1 equality\n",
            length(x), n_ineq)
    @printf("  %d windows, build time: %.2fs\n", n_win, build_time)
    println()

    results = []

    # 1. Dense Lasserre (baseline) — skip for large problems
    if d <= 8 || (d <= 16 && order <= 2)
        try
            opt, elapsed, _ = solve_dense_lasserre(pop, x, order)
            push!(results, ("Dense Lasserre", opt, elapsed))
        catch e
            println("  Dense Lasserre FAILED: $e")
            push!(results, ("Dense Lasserre", NaN, NaN))
        end
        println()
    else
        println("  Dense Lasserre: SKIPPED (too large)")
        push!(results, ("Dense Lasserre", NaN, NaN))
        println()
    end

    # 2. TSSOS-block
    try
        opt, elapsed, data = solve_tssos_block(pop, x, order)
        push!(results, ("TSSOS-block", opt, elapsed))
    catch e
        println("  TSSOS-block FAILED: $e")
        push!(results, ("TSSOS-block", NaN, NaN))
    end
    println()

    # 3. TSSOS-MD
    try
        opt, elapsed, _ = solve_tssos_md(pop, x, order)
        push!(results, ("TSSOS-MD", opt, elapsed))
    catch e
        println("  TSSOS-MD FAILED: $e")
        push!(results, ("TSSOS-MD", NaN, NaN))
    end
    println()

    # 4. CS-TSSOS
    try
        opt, elapsed, _ = solve_cs_tssos(pop, x, order)
        push!(results, ("CS-TSSOS", opt, elapsed))
    catch e
        println("  CS-TSSOS FAILED: $e")
        push!(results, ("CS-TSSOS", NaN, NaN))
    end
    println()

    # 5. CS-TSSOS-MD
    try
        opt, elapsed, _ = solve_cs_tssos_md(pop, x, order)
        push!(results, ("CS-TSSOS-MD", opt, elapsed))
    catch e
        println("  CS-TSSOS-MD FAILED: $e")
        push!(results, ("CS-TSSOS-MD", NaN, NaN))
    end
    println()

    # Summary
    println("=" ^ 60)
    println("SUMMARY: d=$d, order=$order")
    println("=" ^ 60)
    @printf("%-20s %14s %10s %10s\n", "Method", "val(d) bound", "Time", "Speedup")
    @printf("%-20s %14s %10s %10s\n", "-"^20, "-"^14, "-"^10, "-"^10)

    baseline_time = NaN
    for (name, opt, elapsed) in results
        if !isnan(elapsed) && isnan(baseline_time)
            baseline_time = elapsed
        end
    end

    for (name, opt, elapsed) in results
        if isnan(elapsed)
            @printf("%-20s %14s %10s %10s\n", name, "FAILED", "--", "--")
        else
            speedup = isnan(baseline_time) ? 1.0 : baseline_time / elapsed
            @printf("%-20s %14.10f %9.1fs %9.1fx\n", name, opt, elapsed, speedup)
        end
    end

    println()
    # Report best
    valid = [(name, opt, elapsed) for (name, opt, elapsed) in results if !isnan(elapsed)]
    if !isempty(valid)
        best_name, best_opt, best_time = valid[argmin([t for (_, _, t) in valid])]
        println("Best: $best_name ($(round(best_time, digits=1))s)")
        @printf("val(%d) >= %.10f\n", d, best_opt)
    end
end

main()
