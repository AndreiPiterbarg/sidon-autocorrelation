#pragma once
/*
 * C dispatch functions (extern "C") for Python ctypes wrapper.
 *
 * Contents:
 *   - gpu_check_cuda()
 *   - gpu_get_device_name()
 *   - gpu_find_best_bound_direct()  -> dispatches to D=4 or D=6
 *   - gpu_run_single_level()        -> dispatches to D=4 or D=6
 *
 * Depends on: host_find_min.cuh (find_best_bound_direct_d4/d6)
 *             host_prove.cuh (run_single_level_d4/d6)
 */

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {

EXPORT int gpu_check_cuda() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) return 0;
    return 1;
}

EXPORT int gpu_get_device_name(char* buf, int buf_len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return -1;
    snprintf(buf, buf_len, "%s (CC %d.%d, %d MB)",
             prop.name, prop.major, prop.minor,
             (int)(prop.totalGlobalMem / (1024 * 1024)));
    return 0;
}

EXPORT int gpu_find_best_bound_direct(
    int d, int S, int n_half, int m,
    double init_min_eff,
    double* result_min_eff,
    int* result_min_config
) {
    switch (d) {
        case 4: return find_best_bound_direct_d4(S, n_half, m,
                    init_min_eff, result_min_eff, result_min_config);
        case 6: return find_best_bound_direct_d6(S, n_half, m,
                    init_min_eff, result_min_eff, result_min_config);
        default:
            fprintf(stderr, "GPU: d=%d not supported (only d=4,6)\n", d);
            return -2;
    }
}

EXPORT int gpu_run_single_level(
    int d, int S, int n_half, int m,
    double c_target,
    long long* n_pruned_asym,
    long long* n_pruned_test,
    long long* n_survivors,
    double* min_test_val,
    int* min_test_config
) {
    switch (d) {
        case 4: return run_single_level_d4(S, n_half, m, c_target,
                    n_pruned_asym, n_pruned_test, n_survivors,
                    min_test_val, min_test_config);
        case 6: return run_single_level_d6(S, n_half, m, c_target,
                    n_pruned_asym, n_pruned_test, n_survivors,
                    min_test_val, min_test_config);
        default:
            fprintf(stderr, "GPU: d=%d not supported (only d=4,6)\n", d);
            return -2;
    }
}

}  /* extern "C" */
