function sdpnal_bench()
% Quick SDPNAL+ scaling benchmark: d=4 and d=6 L3 full.
% Times each solve, measures memory, saves to bench_results.mat.
% Takes ~1-2 minutes total.
%
% Usage (in MATLAB, with SDPNAL+ on path):
%   >> addpath('C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\tests')
%   >> sdpnal_bench
%
% After completion, bench_results.mat is dropped in data/sdpnal_bench/
% for Python to read and extrapolate.

    repo_root  = 'C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon';
    bench_dir  = fullfile(repo_root, 'data', 'sdpnal_bench');
    test_dir   = fullfile(repo_root, 'tests');
    addpath(test_dir);

    cases = {'bench_d4_l3_bw3', 'bench_d6_l3_bw5'};
    results = struct();

    fprintf('\n========== SDPNAL+ benchmark ==========\n');
    for k = 1:length(cases)
        tag = cases{k};
        fprintf('\n--- %s ---\n', tag);

        % Clear any variables from the previous iteration so stale
        % state doesn't leak into this solve.
        clearvars -except cases results bench_dir test_dir k tag repo_root;

        input_mat  = fullfile(bench_dir, [tag, '_input.mat']);
        load(input_mat);  %#ok<LOAD>

        % Memory before — defensive: `memory` struct fields differ
        % across MATLAB versions and platforms.
        try
            m0 = memory;
        catch
            m0 = struct();
        end
        if ~isfield(m0, 'MemUsedMATLAB'), m0.MemUsedMATLAB = 0; end

        % Solve.
        t0 = tic;
        try
            sdpnalplus_solve;
            solve_ok = true;
        catch err
            fprintf('  ERROR: %s\n', err.message);
            fprintf('  ---- full stack ----\n');
            disp(getReport(err, 'extended'));
            fprintf('  --------------------\n');
            solve_ok = false;
            obj       = NaN;
            termcode  = -999;
            kkt       = NaN;
            pri_res   = NaN;
            dual_res  = NaN;
            iter_main = 0;
            iter_sub  = 0;
        end
        wall = toc(t0);

        % Memory after — defensive fallback.
        try
            m1 = memory;
        catch
            m1 = struct();
        end
        if ~isfield(m1, 'MemUsedMATLAB'), m1.MemUsedMATLAB = 0; end

        results.(tag).wall_s           = wall;
        results.(tag).termcode         = termcode;
        results.(tag).kkt              = kkt;
        results.(tag).pri_res          = pri_res;
        results.(tag).dual_res         = dual_res;
        results.(tag).iter_main        = iter_main;
        results.(tag).iter_sub         = iter_sub;
        results.(tag).mem_matlab_MB    = m1.MemUsedMATLAB / 1e6;
        results.(tag).mem_delta_MB     = (m1.MemUsedMATLAB - m0.MemUsedMATLAB) / 1e6;
        % Physical memory: try newer layout (.PhysicalMemory.Total),
        % then older flat fields, then leave as NaN.
        if isfield(m1, 'PhysicalMemory') && isfield(m1.PhysicalMemory, 'Total')
            results.(tag).phys_total_GB = m1.PhysicalMemory.Total / 1e9;
            results.(tag).phys_avail_GB = m1.PhysicalMemory.Available / 1e9;
        elseif isfield(m1, 'SystemMemory') && isfield(m1.SystemMemory, 'Available')
            results.(tag).phys_total_GB = NaN;   % not exposed in this layout
            results.(tag).phys_avail_GB = m1.SystemMemory.Available / 1e9;
        else
            results.(tag).phys_total_GB = NaN;
            results.(tag).phys_avail_GB = NaN;
        end
        results.(tag).solve_ok         = solve_ok;

        fprintf('  wall       = %.2fs\n', wall);
        fprintf('  termcode   = %d  (0=solved, 1=pri infeas)\n', termcode);
        fprintf('  KKT        = %.2e\n', kkt);
        fprintf('  iter_main  = %d,  iter_sub = %d\n', iter_main, iter_sub);
        fprintf('  mem_matlab = %.1f MB (delta %.1f MB)\n', ...
                results.(tag).mem_matlab_MB, results.(tag).mem_delta_MB);
        fprintf('  phys memory= %.1f GB available / %.1f GB total\n', ...
                results.(tag).phys_avail_GB, results.(tag).phys_total_GB);
    end

    out_path = fullfile(bench_dir, 'bench_results.mat');
    save(out_path, 'results', '-v7.3');
    fprintf('\n========== DONE ==========\n');
    fprintf('Saved: %s\n', out_path);
    fprintf('Now back in Python, run: python tests/interpret_bench.py\n');
end
