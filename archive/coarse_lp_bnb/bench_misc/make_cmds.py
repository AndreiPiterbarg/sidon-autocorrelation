import base64, os
tf = os.path.join(os.environ.get('LOCALAPPDATA',''), 'Temp', 'sidon_gpu.tar.gz')
b64s = base64.b64encode(open(tf,'rb').read()).decode()
lines = [
    'cd /workspace && mkdir -p sidon && cd sidon',
    'echo %s | base64 -d > /tmp/sidon.tar.gz' % b64s,
    'tar xzf /tmp/sidon.tar.gz 2>/dev/null',
    'pip install numpy numba -q 2>&1 | tail -1',
    'echo FILES_READY',
    'cp gpu/cascade_kernel.cu gpu/cascade_host.cu gpu/cascade_kernel.h /tmp/',
    'cd /tmp && nvcc -arch=sm_90 -O3 -ftz=false -prec-div=true -prec-sqrt=true -fmad=false -lineinfo cascade_kernel.cu cascade_host.cu -o cascade_prover 2>&1 && echo BUILD_OK || echo BUILD_FAILED',
    'cp /tmp/cascade_prover /workspace/sidon/ 2>/dev/null',
    'cd /workspace/sidon',
    'python3 run_proof.py --m 35 --c_targets 1.28,1.30,1.33,1.35,1.37,1.40 --workers 32',
    'echo ALL_DONE',
    'exit',
]
of = os.path.join(os.environ.get('LOCALAPPDATA',''), 'Temp', 'gpu_final.txt')
with open(of, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Written %d bytes to %s' % (os.path.getsize(of), of))
