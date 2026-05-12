#!/bin/bash
set -e

echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y -qq libopenblas-dev pkg-config > /dev/null 2>&1

echo "=== Installing Python deps ==="
pip install -q scipy Mosek meson meson-python ninja

echo "=== Setting up cuDSS ==="
CUDSS_DIR=/tmp/libcudss-linux-x86_64-0.7.1.4_cuda12-archive
if [ ! -d "$CUDSS_DIR" ]; then
    cd /tmp
    wget -q https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.7.1.4_cuda12-archive.tar.xz
    tar xf libcudss-linux-x86_64-0.7.1.4_cuda12-archive.tar.xz
fi

# Write pkg-config file using python (avoids shell escaping issues)
python3 -c "
p = '/tmp/libcudss-linux-x86_64-0.7.1.4_cuda12-archive'
import os
os.makedirs(f'{p}/lib/pkgconfig', exist_ok=True)
with open(f'{p}/lib/pkgconfig/cudss.pc', 'w') as f:
    f.write(f'prefix={p}\n')
    f.write('libdir=\${prefix}/lib\n')
    f.write('includedir=\${prefix}/include\n\n')
    f.write('Name: cudss\nDescription: cuDSS\nVersion: 0.7.1\n')
    f.write('Libs: -L\${libdir} -lcudss\n')
    f.write('Cflags: -I\${includedir}\n')
"

export PKG_CONFIG_PATH=$CUDSS_DIR/lib/pkgconfig:$PKG_CONFIG_PATH
echo "pkg-config cudss: $(pkg-config --libs cudss)"

echo "=== Cloning scs-python ==="
rm -rf /tmp/scs-python
git clone --recursive https://github.com/bodono/scs-python.git /tmp/scs-python 2>&1 | tail -3

echo "=== Building SCS with cuDSS GPU ==="
pip uninstall -y scs 2>/dev/null || true
cd /tmp/scs-python
PKG_CONFIG_PATH=$CUDSS_DIR/lib/pkgconfig \
pip install --no-build-isolation \
    -Csetup-args=-Dlink_cudss=true \
    -Csetup-args=-Dint32=true \
    . 2>&1 | tail -10

echo "=== Verifying ==="
export LD_LIBRARY_PATH=$CUDSS_DIR/lib:$LD_LIBRARY_PATH
python3 -c "
import scs
print('SCS', scs.__version__)
print('int_size:', scs.__sizeof_int__)
mods = [x for x in dir(scs) if x.startswith('_scs')]
print('modules:', mods)

import numpy as np
from scipy import sparse as sp

A = sp.csc_matrix(np.array([[-1.0, 0], [0, -1.0], [1, 1]], dtype=np.float64))
data = {'A': A, 'b': np.array([0.0, 0.0, 1.0]), 'c': np.array([1.0, 0.0])}
cone = {'l': 3}

# Direct
s1 = scs.SCS(data, cone, max_iters=100, verbose=False)
sol1 = s1.solve()
print(f'Direct: {sol1[\"info\"][\"status\"]}, x={sol1[\"x\"]}')

# Indirect
try:
    s2 = scs.SCS(data, cone, max_iters=100, verbose=False, use_indirect=True)
    sol2 = s2.solve()
    print(f'Indirect: {sol2[\"info\"][\"status\"]}, x={sol2[\"x\"]}')
except Exception as e:
    print(f'Indirect FAIL: {e}')
    # Try with linear_solver kwarg
    try:
        s2b = scs.SCS(data, cone, max_iters=100, verbose=False, linear_solver='indirect')
        sol2b = s2b.solve()
        print(f'linear_solver=indirect: {sol2b[\"info\"][\"status\"]}')
    except Exception as e2:
        print(f'linear_solver=indirect FAIL: {e2}')

# GPU cuDSS
try:
    s3 = scs.SCS(data, cone, max_iters=100, verbose=False, gpu=True)
    sol3 = s3.solve()
    print(f'GPU: {sol3[\"info\"][\"status\"]}, x={sol3[\"x\"]}')
except Exception as e:
    print(f'GPU FAIL: {e}')
    # Try with linear_solver kwarg
    try:
        s3b = scs.SCS(data, cone, max_iters=100, verbose=False, linear_solver='cudss')
        sol3b = s3b.solve()
        print(f'linear_solver=cudss: {sol3b[\"info\"][\"status\"]}')
    except Exception as e2:
        print(f'linear_solver=cudss FAIL: {e2}')
    try:
        s3c = scs.SCS(data, cone, max_iters=100, verbose=False, linear_solver='gpu')
        sol3c = s3c.solve()
        print(f'linear_solver=gpu: {sol3c[\"info\"][\"status\"]}')
    except Exception as e3:
        print(f'linear_solver=gpu FAIL: {e3}')
"

echo "=== DONE ==="
