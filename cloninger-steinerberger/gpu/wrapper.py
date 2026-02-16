"""Python ctypes wrapper for GPU CUDA kernels.

Loads the compiled shared library and provides typed Python functions
that call into the CUDA host code.
"""
import os
import ctypes
import numpy as np
import platform


# Library handle (lazy-loaded)
_lib = None
_lib_path = None


def _get_lib_path():
    """Get the path to the compiled shared library."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if platform.system() == 'Windows':
        return os.path.join(script_dir, 'kernels.dll')
    else:
        return os.path.join(script_dir, 'kernels.so')


def _load_lib():
    """Load the shared library, building if necessary."""
    global _lib, _lib_path

    if _lib is not None:
        return _lib

    lib_path = _get_lib_path()

    if not os.path.exists(lib_path):
        # Try to build
        from . import build
        result = build.build(verbose=True)
        if result is None:
            raise RuntimeError("Failed to build CUDA kernels")
        lib_path = result

    try:
        _lib = ctypes.CDLL(lib_path)
        _lib_path = lib_path
    except OSError as e:
        raise RuntimeError(f"Failed to load CUDA library: {e}")

    # Set up function signatures
    _setup_signatures(_lib)

    return _lib


def _setup_signatures(lib):
    """Set up ctypes function signatures."""

    # gpu_check_cuda() -> int
    lib.gpu_check_cuda.restype = ctypes.c_int
    lib.gpu_check_cuda.argtypes = []

    # gpu_get_device_name(char* buf, int buf_len) -> int
    lib.gpu_get_device_name.restype = ctypes.c_int
    lib.gpu_get_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]

    # gpu_get_free_memory() -> long long
    lib.gpu_get_free_memory.restype = ctypes.c_longlong
    lib.gpu_get_free_memory.argtypes = []

    # gpu_find_best_bound_direct(d, S, n_half, m, init_min_eff,
    #     result_min_eff*, result_min_config*) -> int
    lib.gpu_find_best_bound_direct.restype = ctypes.c_int
    lib.gpu_find_best_bound_direct.argtypes = [
        ctypes.c_int,       # d
        ctypes.c_int,       # S
        ctypes.c_int,       # n_half
        ctypes.c_int,       # m
        ctypes.c_double,    # init_min_eff
        ctypes.POINTER(ctypes.c_double),  # result_min_eff
        ctypes.POINTER(ctypes.c_int),     # result_min_config
    ]

    # gpu_run_single_level(d, S, n_half, m, c_target,
    #     n_fp32_skipped*, n_pruned_asym*, n_pruned_test*, n_survivors*,
    #     min_test_val*, min_test_config*) -> int
    lib.gpu_run_single_level.restype = ctypes.c_int
    lib.gpu_run_single_level.argtypes = [
        ctypes.c_int,       # d
        ctypes.c_int,       # S
        ctypes.c_int,       # n_half
        ctypes.c_int,       # m
        ctypes.c_double,    # c_target
        ctypes.POINTER(ctypes.c_longlong),  # n_fp32_skipped
        ctypes.POINTER(ctypes.c_longlong),  # n_pruned_asym
        ctypes.POINTER(ctypes.c_longlong),  # n_pruned_test
        ctypes.POINTER(ctypes.c_longlong),  # n_survivors
        ctypes.POINTER(ctypes.c_double),    # min_test_val
        ctypes.POINTER(ctypes.c_int),       # min_test_config
    ]

    # gpu_run_single_level_extract(d, S, n_half, m, c_target,
    #     n_fp32_skipped*, n_pruned_asym*, n_pruned_test*, n_survivors*,
    #     min_test_val*, min_test_config*,
    #     survivor_configs*, n_extracted*, max_survivors) -> int
    lib.gpu_run_single_level_extract.restype = ctypes.c_int
    lib.gpu_run_single_level_extract.argtypes = [
        ctypes.c_int,       # d
        ctypes.c_int,       # S
        ctypes.c_int,       # n_half
        ctypes.c_int,       # m
        ctypes.c_double,    # c_target
        ctypes.POINTER(ctypes.c_longlong),  # n_fp32_skipped
        ctypes.POINTER(ctypes.c_longlong),  # n_pruned_asym
        ctypes.POINTER(ctypes.c_longlong),  # n_pruned_test
        ctypes.POINTER(ctypes.c_longlong),  # n_survivors
        ctypes.POINTER(ctypes.c_double),    # min_test_val
        ctypes.POINTER(ctypes.c_int),       # min_test_config
        ctypes.POINTER(ctypes.c_int),       # survivor_configs
        ctypes.POINTER(ctypes.c_int),       # n_extracted
        ctypes.c_int,                       # max_survivors
    ]

    # gpu_run_single_level_extract_streamed(d, S, n_half, m, c_target,
    #     n_fp32_skipped*, n_pruned_asym*, n_pruned_test*, n_survivors*,
    #     min_test_val*, min_test_config*,
    #     survivor_file_path, n_extracted*) -> int
    lib.gpu_run_single_level_extract_streamed.restype = ctypes.c_int
    lib.gpu_run_single_level_extract_streamed.argtypes = [
        ctypes.c_int,       # d
        ctypes.c_int,       # S
        ctypes.c_int,       # n_half
        ctypes.c_int,       # m
        ctypes.c_double,    # c_target
        ctypes.POINTER(ctypes.c_longlong),  # n_fp32_skipped
        ctypes.POINTER(ctypes.c_longlong),  # n_pruned_asym
        ctypes.POINTER(ctypes.c_longlong),  # n_pruned_test
        ctypes.POINTER(ctypes.c_longlong),  # n_survivors
        ctypes.POINTER(ctypes.c_double),    # min_test_val
        ctypes.POINTER(ctypes.c_int),       # min_test_config
        ctypes.c_char_p,                    # survivor_file_path
        ctypes.POINTER(ctypes.c_longlong),  # n_extracted
    ]

    # gpu_refine_parents(d_parent, parent_configs*, num_parents, m, c_target,
    #     total_asym*, total_test*, total_survivors*,
    #     min_test_val*, min_test_config*,
    #     survivor_configs*, n_extracted*, max_survivors,
    #     time_budget_sec) -> int
    lib.gpu_refine_parents.restype = ctypes.c_int
    lib.gpu_refine_parents.argtypes = [
        ctypes.c_int,                       # d_parent
        ctypes.POINTER(ctypes.c_int),       # parent_configs
        ctypes.c_int,                       # num_parents
        ctypes.c_int,                       # m
        ctypes.c_double,                    # c_target
        ctypes.POINTER(ctypes.c_longlong),  # total_asym
        ctypes.POINTER(ctypes.c_longlong),  # total_test
        ctypes.POINTER(ctypes.c_longlong),  # total_survivors
        ctypes.POINTER(ctypes.c_double),    # min_test_val
        ctypes.POINTER(ctypes.c_int),       # min_test_config
        ctypes.POINTER(ctypes.c_int),       # survivor_configs
        ctypes.POINTER(ctypes.c_int),       # n_extracted
        ctypes.c_int,                       # max_survivors
        ctypes.c_double,                    # time_budget_sec
    ]


def is_available():
    """Check if CUDA GPU is available."""
    try:
        lib = _load_lib()
        return lib.gpu_check_cuda() == 1
    except (RuntimeError, OSError):
        return False


def get_device_name():
    """Get the name of the CUDA device."""
    lib = _load_lib()
    buf = ctypes.create_string_buffer(256)
    ret = lib.gpu_get_device_name(buf, 256)
    if ret != 0:
        return "Unknown"
    return buf.value.decode('utf-8')


def get_free_memory():
    """Get free GPU memory in bytes."""
    lib = _load_lib()
    return lib.gpu_get_free_memory()


def max_survivors_for_dim(d, reserve_gb=0.5):
    """Compute max survivors that fit in both GPU and host memory.

    GPU reserve is small (~0.5 GB for working buffers); the C code
    further clamps via cudaMemGetInfo after allocating actual working
    buffers.  Capped at 2B (int32 atomicAdd limit).
    """
    per_survivor = d * 4  # d ints, 4 bytes each

    # GPU capacity
    free_gpu = get_free_memory()
    if free_gpu > 0:
        usable_gpu = max(free_gpu - int(reserve_gb * 1024**3), 0)
        gpu_cap = usable_gpu // per_survivor if usable_gpu > 0 else 10_000_000
    else:
        gpu_cap = 10_000_000  # fallback

    # Host capacity (use at most 50% of available RAM for survivor buffer)
    host_avail = 0
    try:
        import psutil
        host_avail = psutil.virtual_memory().available
    except ImportError:
        try:
            if hasattr(os, 'sysconf'):
                pages = os.sysconf('SC_AVPHYS_PAGES')
                page_size = os.sysconf('SC_PAGE_SIZE')
                if pages > 0 and page_size > 0:
                    host_avail = pages * page_size
        except (ValueError, OSError):
            pass
    if host_avail > 0:
        host_cap = int(host_avail * 0.5) // per_survivor
    else:
        host_cap = gpu_cap  # can't measure host RAM; assume >= GPU

    n = min(gpu_cap, host_cap)
    return max(min(n, 2_000_000_000), 1)  # int32 atomic limit


def find_best_bound_direct(d, S, n_half, m, init_min_eff):
    """GPU version of find_best_bound_direct.

    Parameters
    ----------
    d : int
        Number of bins (must be 4 or 6).
    S : int
        Total mass (S=m convention).
    n_half : int
        Paper's n.
    m : int
        Grid resolution.
    init_min_eff : float
        Initial minimum effective value (from uniform config).

    Returns
    -------
    (min_eff, min_config) : (float, np.ndarray of int32)
    """
    lib = _load_lib()

    result_min_eff = ctypes.c_double(0.0)
    result_min_config = (ctypes.c_int * d)()

    ret = lib.gpu_find_best_bound_direct(
        d, S, n_half, m, init_min_eff,
        ctypes.byref(result_min_eff),
        result_min_config)

    if ret != 0:
        raise RuntimeError(f"GPU find_best_bound_direct failed (error {ret})")

    config = np.array([result_min_config[i] for i in range(d)], dtype=np.int32)
    return result_min_eff.value, config


def run_single_level(d, S, n_half, m, c_target):
    """GPU version of run_single_level.

    Parameters
    ----------
    d : int
        Number of bins (must be 4 or 6).
    S : int
        Total mass.
    n_half : int
        Paper's n.
    m : int
        Grid resolution.
    c_target : float
        Target lower bound to prove.

    Returns
    -------
    dict with keys: n_pruned_asym, n_pruned_test, n_survivors,
                    min_test_val, min_test_config
    """
    lib = _load_lib()

    n_fp32_skipped = ctypes.c_longlong(0)
    n_pruned_asym = ctypes.c_longlong(0)
    n_pruned_test = ctypes.c_longlong(0)
    n_survivors = ctypes.c_longlong(0)
    min_test_val = ctypes.c_double(0.0)
    min_test_config = (ctypes.c_int * d)()

    ret = lib.gpu_run_single_level(
        d, S, n_half, m, c_target,
        ctypes.byref(n_fp32_skipped),
        ctypes.byref(n_pruned_asym),
        ctypes.byref(n_pruned_test),
        ctypes.byref(n_survivors),
        ctypes.byref(min_test_val),
        min_test_config)

    if ret != 0:
        raise RuntimeError(f"GPU run_single_level failed (error {ret})")

    config = np.array([min_test_config[i] for i in range(d)], dtype=np.int32)

    return {
        'n_fp32_skipped': n_fp32_skipped.value,
        'n_pruned_asym': n_pruned_asym.value,
        'n_pruned_test': n_pruned_test.value,
        'n_survivors': n_survivors.value,
        'min_test_val': min_test_val.value,
        'min_test_config': config,
    }


def run_single_level_extract(d, S, n_half, m, c_target, max_survivors=2000000000):
    """GPU version of run_single_level with survivor config extraction.

    Parameters
    ----------
    d : int
        Number of bins (must be 4 or 6).
    S : int
        Total mass.
    n_half : int
        Paper's n.
    m : int
        Grid resolution.
    c_target : float
        Target lower bound to prove.
    max_survivors : int
        Maximum number of survivor configs to extract.

    Returns
    -------
    dict with keys: n_pruned_asym, n_pruned_test, n_survivors,
                    min_test_val, min_test_config, survivor_configs, n_extracted
    """
    lib = _load_lib()

    n_fp32_skipped = ctypes.c_longlong(0)
    n_pruned_asym = ctypes.c_longlong(0)
    n_pruned_test = ctypes.c_longlong(0)
    n_survivors = ctypes.c_longlong(0)
    min_test_val = ctypes.c_double(0.0)
    min_test_config = (ctypes.c_int * d)()

    survivor_buf = np.empty(max_survivors * d, dtype=np.int32)
    survivor_ptr = survivor_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    n_extracted = ctypes.c_int(0)

    ret = lib.gpu_run_single_level_extract(
        d, S, n_half, m, c_target,
        ctypes.byref(n_fp32_skipped),
        ctypes.byref(n_pruned_asym),
        ctypes.byref(n_pruned_test),
        ctypes.byref(n_survivors),
        ctypes.byref(min_test_val),
        min_test_config,
        survivor_ptr,
        ctypes.byref(n_extracted),
        max_survivors)

    if ret != 0:
        raise RuntimeError(f"GPU run_single_level_extract failed (error {ret})")

    config = np.array([min_test_config[i] for i in range(d)], dtype=np.int32)
    n_ext = n_extracted.value

    if n_ext > 0:
        survivor_configs = survivor_buf[:n_ext * d].reshape(n_ext, d).copy()
    else:
        survivor_configs = np.empty((0, d), dtype=np.int32)

    return {
        'n_fp32_skipped': n_fp32_skipped.value,
        'n_pruned_asym': n_pruned_asym.value,
        'n_pruned_test': n_pruned_test.value,
        'n_survivors': n_survivors.value,
        'min_test_val': min_test_val.value,
        'min_test_config': config,
        'survivor_configs': survivor_configs,
        'n_extracted': n_ext,
    }


def refine_parents(d_parent, parent_configs_array, m, c_target,
                   max_survivors=10000000, time_budget_sec=0.0):
    """GPU refinement: process parent survivors through child-level pruning.

    Parameters
    ----------
    d_parent : int
        Parent dimension (6, 12, or 24).
    parent_configs_array : np.ndarray of shape (num_parents, d_parent), int32
        Parent bin configurations.
    m : int
        Grid resolution.
    c_target : float
        Target lower bound.
    max_survivors : int
        Max child survivors to extract.
    time_budget_sec : float
        Time limit in seconds (0 = no limit).

    Returns
    -------
    dict with keys: total_asym, total_test, total_survivors,
                    min_test_val, min_test_config, survivor_configs,
                    n_extracted, timed_out
    """
    lib = _load_lib()

    parent_configs_array = np.ascontiguousarray(parent_configs_array, dtype=np.int32)
    num_parents = parent_configs_array.shape[0]
    d_child = 2 * d_parent

    total_asym = ctypes.c_longlong(0)
    total_test = ctypes.c_longlong(0)
    total_survivors = ctypes.c_longlong(0)
    min_test_val = ctypes.c_double(0.0)
    min_test_config = (ctypes.c_int * d_child)()

    survivor_buf = np.empty(max_survivors * d_child, dtype=np.int32)
    survivor_ptr = survivor_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    n_extracted = ctypes.c_int(0)

    parent_ptr = parent_configs_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    ret = lib.gpu_refine_parents(
        d_parent,
        parent_ptr,
        num_parents,
        m,
        c_target,
        ctypes.byref(total_asym),
        ctypes.byref(total_test),
        ctypes.byref(total_survivors),
        ctypes.byref(min_test_val),
        min_test_config,
        survivor_ptr,
        ctypes.byref(n_extracted),
        max_survivors,
        time_budget_sec)

    if ret < 0:
        raise RuntimeError(f"GPU refine_parents failed (error {ret})")

    config = np.array([min_test_config[i] for i in range(d_child)], dtype=np.int32)
    n_ext = n_extracted.value

    if n_ext > 0:
        survivor_configs = survivor_buf[:n_ext * d_child].reshape(n_ext, d_child).copy()
    else:
        survivor_configs = np.empty((0, d_child), dtype=np.int32)

    return {
        'total_asym': total_asym.value,
        'total_test': total_test.value,
        'total_survivors': total_survivors.value,
        'min_test_val': min_test_val.value,
        'min_test_config': config,
        'survivor_configs': survivor_configs,
        'n_extracted': n_ext,
        'timed_out': (ret == 1),
    }


def run_single_level_extract_streamed(d, S, n_half, m, c_target, output_path):
    """GPU survivor extraction with chunked streaming to disk.

    Survivors are written to a binary file instead of being held in memory.
    The file contains packed int32[d] per survivor, no header.

    Parameters
    ----------
    d : int
        Number of bins (must be 4 or 6).
    S : int
        Total mass.
    n_half : int
        Paper's n.
    m : int
        Grid resolution.
    c_target : float
        Target lower bound to prove.
    output_path : str
        Path for the binary survivor file.

    Returns
    -------
    dict with keys: n_pruned_asym, n_pruned_test, n_survivors,
                    min_test_val, min_test_config,
                    survivor_file, n_extracted
    """
    lib = _load_lib()

    print(f"STREAMED wrapper: d={d}, S={S}, n_half={n_half}, m={m}, "
          f"c_target={c_target}", flush=True)
    print(f"STREAMED wrapper: output file: {output_path}", flush=True)

    n_fp32_skipped = ctypes.c_longlong(0)
    n_pruned_asym = ctypes.c_longlong(0)
    n_pruned_test = ctypes.c_longlong(0)
    n_survivors = ctypes.c_longlong(0)
    min_test_val = ctypes.c_double(0.0)
    min_test_config = (ctypes.c_int * d)()
    n_extracted = ctypes.c_longlong(0)

    path_bytes = output_path.encode('utf-8')

    ret = lib.gpu_run_single_level_extract_streamed(
        d, S, n_half, m, c_target,
        ctypes.byref(n_fp32_skipped),
        ctypes.byref(n_pruned_asym),
        ctypes.byref(n_pruned_test),
        ctypes.byref(n_survivors),
        ctypes.byref(min_test_val),
        min_test_config,
        path_bytes,
        ctypes.byref(n_extracted))

    if ret == -4:
        # Disk full — partial extraction. Counting data is still valid.
        n_ext = n_extracted.value
        print(f"STREAMED wrapper: WARNING: disk write failed after "
              f"{n_ext:,} survivors (partial extraction)", flush=True)
    elif ret != 0:
        raise RuntimeError(
            f"GPU run_single_level_extract_streamed failed (error {ret})")

    config = np.array([min_test_config[i] for i in range(d)], dtype=np.int32)
    if ret != -4:
        n_ext = n_extracted.value

    print(f"STREAMED wrapper: extracted {n_ext:,} survivors", flush=True)
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"STREAMED wrapper: file size: {file_size / (1024**2):.1f} MB",
              flush=True)

    return {
        'n_fp32_skipped': n_fp32_skipped.value,
        'n_pruned_asym': n_pruned_asym.value,
        'n_pruned_test': n_pruned_test.value,
        'n_survivors': n_survivors.value,
        'min_test_val': min_test_val.value,
        'min_test_config': config,
        'survivor_file': output_path,
        'n_extracted': n_ext,
    }


def load_survivors_chunk(filepath, d, offset=0, count=None):
    """Load a chunk of survivors from a binary file.

    Parameters
    ----------
    filepath : str
        Path to binary survivor file (packed int32[d] per survivor).
    d : int
        Dimension (number of ints per survivor).
    offset : int
        Number of survivors to skip from the start.
    count : int or None
        Number of survivors to read. None = read all remaining.

    Returns
    -------
    np.ndarray of shape (n, d), dtype=int32
    """
    per_survivor = d * 4  # bytes
    file_size = os.path.getsize(filepath)
    total_survivors = file_size // per_survivor

    if offset >= total_survivors:
        return np.empty((0, d), dtype=np.int32)

    if count is None:
        count = total_survivors - offset
    else:
        count = min(count, total_survivors - offset)

    byte_offset = offset * per_survivor
    data = np.fromfile(filepath, dtype=np.int32,
                       count=count * d, offset=byte_offset)
    return data.reshape(-1, d)
