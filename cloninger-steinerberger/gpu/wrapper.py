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
    #     n_pruned_asym*, n_pruned_test*, n_survivors*,
    #     min_test_val*, min_test_config*) -> int
    lib.gpu_run_single_level.restype = ctypes.c_int
    lib.gpu_run_single_level.argtypes = [
        ctypes.c_int,       # d
        ctypes.c_int,       # S
        ctypes.c_int,       # n_half
        ctypes.c_int,       # m
        ctypes.c_double,    # c_target
        ctypes.POINTER(ctypes.c_longlong),  # n_pruned_asym
        ctypes.POINTER(ctypes.c_longlong),  # n_pruned_test
        ctypes.POINTER(ctypes.c_longlong),  # n_survivors
        ctypes.POINTER(ctypes.c_double),    # min_test_val
        ctypes.POINTER(ctypes.c_int),       # min_test_config
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


def find_best_bound_direct(d, S, n_half, m, init_min_eff):
    """GPU version of find_best_bound_direct.

    Parameters
    ----------
    d : int
        Number of bins (must be 4 or 6).
    S : int
        Total mass (4 * n_half * m).
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

    n_pruned_asym = ctypes.c_longlong(0)
    n_pruned_test = ctypes.c_longlong(0)
    n_survivors = ctypes.c_longlong(0)
    min_test_val = ctypes.c_double(0.0)
    min_test_config = (ctypes.c_int * d)()

    ret = lib.gpu_run_single_level(
        d, S, n_half, m, c_target,
        ctypes.byref(n_pruned_asym),
        ctypes.byref(n_pruned_test),
        ctypes.byref(n_survivors),
        ctypes.byref(min_test_val),
        min_test_config)

    if ret != 0:
        raise RuntimeError(f"GPU run_single_level failed (error {ret})")

    config = np.array([min_test_config[i] for i in range(d)], dtype=np.int32)

    return {
        'n_pruned_asym': n_pruned_asym.value,
        'n_pruned_test': n_pruned_test.value,
        'n_survivors': n_survivors.value,
        'min_test_val': min_test_val.value,
        'min_test_config': config,
    }
