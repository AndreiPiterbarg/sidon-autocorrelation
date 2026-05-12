function verify_sdpnal_install()
% Verify SDPNAL+ is installed and callable.
%
% Usage (from MATLAB):
%   >> cd 'C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\tests'
%   >> verify_sdpnal_install
%
% Expected final output:
%   === SDPNAL+ INSTALLATION VERIFIED ===
%
% If this fails, the Python driver will also fail. Fix this first.

    fprintf('=== SDPNAL+ installation verification ===\n\n');

    % ----- 1. MATLAB version -----
    fprintf('[1/5] MATLAB release: %s\n', version('-release'));

    % ----- 2. SDPNAL+ on path -----
    if exist('sdpnalplus', 'file') ~= 2
        error(['sdpnalplus is NOT on the MATLAB path. ', ...
               'Run addpath(genpath(''C:\\path\\to\\SDPNALplus'')); ', ...
               'savepath; and retry.']);
    end
    fprintf('[2/5] sdpnalplus located at:\n        %s\n', which('sdpnalplus'));

    % ----- 3. read_sedumi available (MEX-compiled helpers) -----
    if exist('read_sedumi', 'file') == 0
        error(['read_sedumi not found. SDPNAL+ install is incomplete. ', ...
               'Run Installmex(1) from the SDPNAL+ root, then savepath.']);
    end
    fprintf('[3/5] read_sedumi available.\n');

    % ----- 4. MEX compilation check -----
    % Presence of the mexw64/mexa64 binaries indicates Installmex ran.
    try
        mex_test = which('mexFvec');  % one of SDPNAL+''s core MEX files
        if isempty(mex_test)
            warning(['MEX helpers (e.g. mexFvec) not found. SDPNAL+ may ', ...
                     'run slowly without them. Re-run Installmex(1) with ', ...
                     'a compiler configured (mex -setup C++).']);
        else
            fprintf('[4/5] MEX helpers compiled: %s\n', mex_test);
        end
    catch
        fprintf('[4/5] MEX check skipped (non-fatal).\n');
    end

    % ----- 5. Solve a 2x2 test SDP -----
    % min trace(C*X)  s.t.  trace(X) = 1, X >= 0 (2x2 PSD)
    % with C = diag(1, 2).  Optimal: X = [[1,0],[0,0]], obj = 1.
    fprintf('[5/5] Running 2x2 test SDP ...\n');

    blk      = cell(1, 2);
    blk{1,1} = 's';
    blk{1,2} = 2;

    % Constraint coefficient for trace(X) = X11 + X22.
    % SDPT3/SDPNAL+ svec order (upper triangle column-major):
    %   [X11; sqrt(2)*X12; X22]  for a 2x2 matrix.
    % So trace = [1; 0; 1] in svec.
    At       = cell(1, 1);
    At{1}    = sparse([1; 0; 1]);

    C_blk    = cell(1, 1);
    C_blk{1} = [1 0; 0 2];

    b        = 1;

    OPTIONS  = struct('tol',         1e-8, ...
                      'maxiter',     100, ...
                      'printlevel',  0);

    t0 = tic;
    [obj_all, X, ~, ~, ~, ~, ~, ~, info, ~] = ...
        sdpnalplus(blk, At, C_blk, b, [], [], [], [], [], OPTIONS);
    elapsed = toc(t0);

    if ~isfield(info, 'termcode')
        error('sdpnalplus returned no termcode. SDPNAL+ install is broken.');
    end

    if info.termcode ~= 0
        error(['Test SDP did not solve cleanly (termcode=%d). ', ...
               'Inspect SDPNAL+ installation.'], info.termcode);
    end

    obj_val = obj_all(1);
    if abs(obj_val - 1.0) > 1e-4
        error(['Test SDP objective differs from expected 1.0: got %.6f. ', ...
               'SDPNAL+ is producing incorrect results. ', ...
               'Do not proceed until this is fixed.'], obj_val);
    end

    fprintf('      termcode = %d (0 = solved)\n', info.termcode);
    fprintf('      objective = %.6f (expected 1.0)\n', obj_val);
    fprintf('      runtime = %.2fs\n', elapsed);

    fprintf('\n=== SDPNAL+ INSTALLATION VERIFIED ===\n');
    fprintf('Next step: from Python,\n');
    fprintf('  python tests/verify_matlabengine.py\n');
end
