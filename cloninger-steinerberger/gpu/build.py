"""Build script for CUDA GPU kernels.

Compiles kernels.cu into a shared library (DLL on Windows, .so on Linux).
Auto-detects CUDA toolkit and MSVC paths.
"""
import os
import sys
import glob
import subprocess
import platform


def find_nvcc():
    """Find nvcc compiler."""
    # Check PATH first
    for ext in ['', '.exe']:
        try:
            result = subprocess.run(['nvcc' + ext, '--version'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                return 'nvcc' + ext
        except FileNotFoundError:
            pass

    # Check common CUDA toolkit locations
    if platform.system() == 'Windows':
        cuda_dirs = glob.glob(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*')
        cuda_dirs.sort(reverse=True)  # newest first
        for d in cuda_dirs:
            nvcc = os.path.join(d, 'bin', 'nvcc.exe')
            if os.path.exists(nvcc):
                return nvcc
    else:
        for d in ['/usr/local/cuda', '/usr/local/cuda-12', '/usr/local/cuda-13']:
            nvcc = os.path.join(d, 'bin', 'nvcc')
            if os.path.exists(nvcc):
                return nvcc

    return None


def find_msvc():
    """Find MSVC cl.exe for Windows host compilation.

    Prefers VS 2022 (supported by CUDA 13.1) over newer versions.
    """
    if platform.system() != 'Windows':
        return None

    # Search in VS Build Tools and VS installations
    # Prefer 2022 (supported by CUDA 13.x), then 2019, then others
    search_roots = [
        r'C:\Program Files (x86)\Microsoft Visual Studio',
        r'C:\Program Files\Microsoft Visual Studio',
    ]
    candidates = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for cl_path in glob.glob(os.path.join(root, '**', 'Hostx64', 'x64', 'cl.exe'),
                                 recursive=True):
            candidates.append(os.path.dirname(cl_path))

    # Sort: prefer paths containing "2022" or "2019"
    def sort_key(path):
        if '2022' in path: return 0
        if '2019' in path: return 1
        return 2
    candidates.sort(key=sort_key)
    if candidates:
        return candidates[0]

    return None


def build(verbose=True):
    """Build the CUDA kernels into a shared library."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(script_dir, 'kernels.cu')

    if not os.path.exists(src):
        print(f"Error: {src} not found")
        return None

    nvcc = find_nvcc()
    if nvcc is None:
        print("Error: nvcc not found. Install CUDA Toolkit.")
        return None

    if verbose:
        print(f"Using nvcc: {nvcc}")

    # Output library
    if platform.system() == 'Windows':
        lib_name = 'kernels.dll'
    else:
        lib_name = 'kernels.so'
    output = os.path.join(script_dir, lib_name)

    # Build command
    cmd = [
        nvcc,
        '-O3',
        '--shared',
        '-o', output,
        src,
    ]

    # Target architectures: on Linux (remote pod) only build for A100
    # to keep compile times short. On Windows build for multiple archs.
    if platform.system() == 'Windows':
        cmd += [
            '-gencode', 'arch=compute_80,code=sm_80',   # A100
            '-gencode', 'arch=compute_86,code=sm_86',   # RTX 3080/3090
            '-gencode', 'arch=compute_89,code=sm_89',   # RTX 4090
            '-gencode', 'arch=compute_90,code=sm_90',   # H100
        ]
    else:
        cmd += [
            '-gencode', 'arch=compute_80,code=sm_80',   # A100
        ]

    # Windows-specific: set MSVC host compiler
    if platform.system() == 'Windows':
        msvc_dir = find_msvc()
        if msvc_dir:
            cmd.extend(['-ccbin', msvc_dir])
            if verbose:
                print(f"Using MSVC: {msvc_dir}")
        # Use dynamic CRT
        cmd.extend(['-Xcompiler', '/MD'])
    else:
        cmd.extend(['-Xcompiler', '-fPIC,-fopenmp'])
        cmd.extend(['-lgomp'])

    # Print register info for profiling
    cmd.extend(['--ptxas-options=-v'])

    if verbose:
        print(f"Compiling: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                # ptxas register info goes to stderr
                for line in result.stderr.split('\n'):
                    if 'registers' in line.lower() or 'warning' in line.lower():
                        print(f"  {line.strip()}")

        if result.returncode != 0:
            print(f"Build failed (exit code {result.returncode})")
            if result.stderr:
                print(result.stderr)
            return None

        if verbose:
            size_mb = os.path.getsize(output) / (1024 * 1024)
            print(f"Built: {output} ({size_mb:.1f} MB)")

        return output

    except subprocess.TimeoutExpired:
        print("Build timed out (600s)")
        return None
    except Exception as e:
        print(f"Build error: {e}")
        return None


if __name__ == '__main__':
    result = build(verbose=True)
    if result:
        print(f"\nSuccess: {result}")
    else:
        print("\nBuild failed")
        sys.exit(1)
