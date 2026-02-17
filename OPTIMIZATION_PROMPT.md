# GPU Optimization Task: Sidon Autocorrelation Branch-and-Prune

You are an expert CUDA/GPU optimization engineer. Your task is to maximize the throughput of a mathematically-rigorous branch-and-prune algorithm running on an NVIDIA A100-SXM4-80GB GPU (80 GB HBM2e, 108 SMs, compute capability 8.0, 2039 GB/s bandwidth, 19.5 TFLOPS FP64).

## The Goal

We are trying to prove that the autoconvolution constant c >= 1.20 using the Cloninger-Steinerberger exhaustive verification algorithm (arXiv:1403.7988). The production entry point is `run_proof.py`. The algorithm works hierarchically:

1. **Level 0** (n=3, d=6 bins, m=50): Enumerate all ~664 billion lattice points in B_{3,50}. Prune via asymmetry + canonical filtering + windowed autoconvolution test. Extract survivors.
2. **Level 1** (n=6, d=12 bins): For each Level 0 survivor parent, generate all child refinements via Cartesian product splits. Each parent component b_i splits into (c_{2i}, c_{2i+1}) with c_{2i}+c_{2i+1}=2*b_i, giving prod(2*b_i+1) children per parent. Prune children. Extract survivors.
3. **Level 2** (n=12, d=24 bins): Same refinement on Level 1 survivors.
4. **Level 3** (n=24, d=48 bins): Same refinement on Level 2 survivors.

If all configurations are eliminated at any level, the proof is complete: c >= 1.20 is formally proven.
