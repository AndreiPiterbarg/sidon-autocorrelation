% SDPNAL+ wrapper for the Sidon-autocorrelation Lasserre SDP driver.
%
% NOTE: This is a SCRIPT, not a function. It runs in the caller's
% workspace so A, b, c, K_f, K_l, K_s, etc. must already be defined
% there (either by a `load(input.mat)` or by direct assignment).
%
%
% This script is invoked from the Python driver
% (tests/lasserre_sdpnalplus.py) via either:
%   - the persistent MATLAB engine (workspace variables pushed directly)
%   - a -batch subprocess call (workspace populated by a preceding
%     `load(input_mat)` call).
%
% Both code paths leave the SeDuMi-form problem data in the workspace,
% then invoke this function.  It converts to SDPNAL+ block format via
% `read_sedumi`, configures tuned OPTIONS, runs `sdpnalplus`, and writes
% the results back into the workspace (and, if `output_path` is set, to
% disk for the subprocess path).
%
% Required workspace inputs:
%   A              [m × n_u]  sparse constraint matrix, SeDuMi primal form
%   b              [m × 1]    RHS
%   c              [n_u × 1]  objective (zeros in feasibility mode)
%   K_f            scalar      # free variables (1 — the t variable)
%   K_l            scalar      # nonneg variables (n_y + |s_l_slack|)
%   K_s            [q × 1]     PSD block sizes
%   sdpnal_tol     scalar      KKT tolerance (e.g. 1e-7)
%   sdpnal_maxiter scalar      max outer iterations
%   t_val_py       scalar      t fixed in A's window-PSD rows (-1 if N/A)
%   n_y            scalar      # moment variables (informational)
%   have_warm      scalar      1 to use X0/y0/Z0 warm-start
%   X0, y0, Z0     cells/vec   warm-start payload (only if have_warm=1)
%   output_path    char        (optional) save all outputs to this .mat
%
% Outputs (written to workspace):
%   obj            scalar      primal optimum (= t for 'optimize' mode)
%   termcode       scalar      0=solved, 1=pri infeas, 2=dual infeas, -1=maxiter
%   kkt            scalar      max(rel gap, pinfeas, dinfeas)
%   pri_res        scalar      info.pinfeas
%   dual_res       scalar      info.dinfeas
%   iter_main      scalar      outer iterations
%   iter_sub       scalar      inner CG / Newton iterations
%   t_opt          scalar      the optimized t value (NaN if unavailable)
%   has_t_opt      scalar      1 if t_opt is meaningful, else 0
%   runtime        scalar      wall-clock seconds inside sdpnalplus()
%   X_out, y_out, Z_out         warm-start payload for the next call

    % ============================================================
    %                     Build K struct
    % ============================================================
    K = struct();
    if exist('K_f', 'var') && K_f > 0
        K.f = double(K_f);
    end
    if exist('K_l', 'var') && K_l > 0
        K.l = double(K_l);
    end
    if exist('K_s', 'var') && ~isempty(K_s)
        K.s = double(K_s(:)).';  % row vector per SeDuMi convention
    end

    % If Python wrote A in chunks (MAT5 per-variable 4 GB limit), assemble
    % now.  Expected workspace vars: n_A_chunks, A_n_rows, A_n_cols,
    % A_chunk_1 ... A_chunk_{n}.  Backward-compat: if `A` exists, use it.
    if ~exist('A', 'var') && exist('n_A_chunks', 'var') && n_A_chunks >= 1
        fprintf('[sdpnalplus_solve] reassembling A from %d chunks...\n', ...
                n_A_chunks);
        t_assemble = tic;
        chunks = cell(1, double(n_A_chunks));
        for kk = 1:double(n_A_chunks)
            chunks{kk} = eval(sprintf('A_chunk_%d', kk));
        end
        A = vertcat(chunks{:});
        clear chunks;
        for kk = 1:double(n_A_chunks)
            evalin('caller', sprintf('clear A_chunk_%d', kk));
            clear(sprintf('A_chunk_%d', kk));
        end
        fprintf('[sdpnalplus_solve]   assembled in %.1fs, size %dx%d nnz=%d\n', ...
                toc(t_assemble), size(A, 1), size(A, 2), nnz(A));
    end

    if ~issparse(A)
        A = sparse(A);
    end
    b = full(b(:));
    c = full(c(:));

    fprintf('[sdpnalplus_solve] A: %d × %d, nnz=%d\n', ...
            size(A, 1), size(A, 2), nnz(A));
    if isfield(K, 'f'), K_f_print = K.f; else, K_f_print = 0; end
    if isfield(K, 'l'), K_l_print = K.l; else, K_l_print = 0; end
    if isfield(K, 's'), K_s_print = K.s; else, K_s_print = []; end
    fprintf('[sdpnalplus_solve] K: f=%d l=%d |s|=%d (blocks)\n', ...
            K_f_print, K_l_print, length(K_s_print));

    % ============================================================
    %               SeDuMi → SDPNAL+ block format
    % ============================================================
    t_convert = tic;
    [blk, At, C_blk, b_sdp] = read_sedumi(A, b, c, K);
    fprintf('[sdpnalplus_solve] read_sedumi: %.1fs; %d blocks\n', ...
            toc(t_convert), size(blk, 1));
    for bi = 1:size(blk, 1)
        fprintf('  blk{%d}: type=%s size=%s\n', bi, blk{bi, 1}, ...
                mat2str(blk{bi, 2}));
    end

    % ============================================================
    %                   OPTIONS tuning
    % ============================================================
    OPTIONS = struct();
    OPTIONS.tol           = double(sdpnal_tol);
    OPTIONS.maxiter       = double(sdpnal_maxiter);
    OPTIONS.printlevel    = 2;         % full progress in log
    OPTIONS.scale_data    = 2;         % aggressive column + row scaling
    OPTIONS.stopoption    = 0;         % strict KKT stopping
    OPTIONS.ADMplus       = 1;         % ADMM pre-phase before SSN
    OPTIONS.BBTtol        = min(double(sdpnal_tol), 1e-8);
    OPTIONS.AATsolve      = struct('method', 'iterative');
    % For large problems, this matters: the default direct solve on
    % AA^T blows up at m > 1e5.  Iterative (PCG) keeps memory bounded.

    % Warm-start (only if workspace vars are consistent in shape with
    % the current blk structure; if sizes differ we silently skip so
    % a bisection step with a changed problem doesn't blow up).
    if exist('have_warm', 'var') && double(have_warm) > 0 && ...
            exist('X0', 'var') && exist('y0', 'var') && exist('Z0', 'var')
        try
            if iscell(X0) && size(X0, 1) == size(blk, 1)
                OPTIONS.X0 = X0;
                OPTIONS.y0 = y0;
                OPTIONS.Z0 = Z0;
                fprintf('[sdpnalplus_solve] warm-start: enabled\n');
            end
        catch
            fprintf('[sdpnalplus_solve] warm-start: skipped (shape mismatch)\n');
        end
    end

    % ============================================================
    %                      Solve
    % ============================================================
    t_solve = tic;
    try
        [obj_all, X, s_out, y_out, Z1_out, Z2_out, ybar_out, v_out, ...
            info, runhist] = sdpnalplus(blk, At, C_blk, b_sdp, ...
                                         [], [], [], [], [], OPTIONS);
    catch err
        fprintf('[sdpnalplus_solve] sdpnalplus threw: %s\n', err.message);
        obj = NaN;
        termcode = -999;
        kkt = NaN; pri_res = NaN; dual_res = NaN;
        iter_main = 0; iter_sub = 0;
        t_opt = NaN; has_t_opt = 0;
        runtime = toc(t_solve);
        X_out = {}; y_out = []; Z_out = {};
        if exist('output_path', 'var')
            save(output_path, 'obj', 'termcode', 'kkt', 'pri_res', ...
                 'dual_res', 'iter_main', 'iter_sub', 't_opt', ...
                 'has_t_opt', 'runtime');
        end
        return;   % exit the script early
    end
    runtime = toc(t_solve);
    fprintf('[sdpnalplus_solve] sdpnalplus: %.1fs  termcode=%d\n', ...
            runtime, info.termcode);

    % ============================================================
    %                   Extract outputs
    % ============================================================

    % Primal objective (first element of obj_all is primal; second is dual).
    if isnumeric(obj_all) && ~isempty(obj_all)
        obj = double(obj_all(1));
    else
        obj = NaN;
    end

    termcode = double(info.termcode);
    if isfield(info, 'pinfeas'), pri_res  = double(info.pinfeas); else, pri_res  = NaN; end
    if isfield(info, 'dinfeas'), dual_res = double(info.dinfeas); else, dual_res = NaN; end
    if isfield(info, 'relgap'),  relgap   = double(info.relgap);  else, relgap   = NaN; end
    kkt       = max([abs(relgap), abs(pri_res), abs(dual_res)]);
    if isfield(info, 'iter'),    iter_main = double(info.iter);   else, iter_main = 0;  end
    if isfield(info, 'itersub')
        iter_sub = double(info.itersub);
    elseif isfield(info, 'iter_sub')
        iter_sub = double(info.iter_sub);
    else
        iter_sub = 0;
    end

    % t_opt — the free block is the first one when K.f > 0.
    t_opt = NaN;
    has_t_opt = 0;
    if iscell(X) && ~isempty(X) && isfield(K, 'f') && K.f > 0
        try
            x1 = X{1};
            if ~isempty(x1)
                t_opt = double(x1(1));
                has_t_opt = 1;
            end
        catch
            % leave NaN
        end
    end

    % Warm-start payloads are the primal X, dual y, dual Z returned by
    % SDPNAL+.  Left in caller's workspace as X_out / y_out / Z_out for
    % the next bisection call to pick up.
    X_out = X;
    Z_out = Z1_out;
    % (y_out is already set by the sdpnalplus() call.)

    % If invoked via subprocess, persist to disk so Python can read it.
    if exist('output_path', 'var')
        save(output_path, 'obj', 'termcode', 'kkt', 'pri_res', ...
             'dual_res', 'iter_main', 'iter_sub', 't_opt', ...
             'has_t_opt', 'runtime');
        fprintf('[sdpnalplus_solve] saved: %s\n', output_path);
    end

    % --------- MATLAB memory cleanup ---------
    % Drop the big intermediate objects (svec-form At, OPTIONS struct,
    % runhist) once the outputs are safely in obj/termcode/etc.
    clear blk At C_blk b_sdp OPTIONS info runhist s_out ybar_out ...
          v_out Z2_out obj_all Z1_out;
