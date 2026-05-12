"""Check what code _spawn_bnb generates."""
import ast
import sys
sys.path.insert(0, '.')
from cert_pipeline.bnb_phase import _spawn_bnb, BnBPhaseConfig
from pathlib import Path

# Monkey-patch subprocess.Popen to capture the code arg.
import subprocess
captured = {}
orig = subprocess.Popen
class FakePopen:
    def __init__(self, args, **kw):
        captured['args'] = args
        captured['kw'] = kw
    def __getattr__(self, k):
        return None
subprocess.Popen = FakePopen

cfg = BnBPhaseConfig(d=22, target_str='1.2805')
_spawn_bnb(cfg, Path('/home/ubuntu/sidon'))

code = captured['args'][3]  # -c <code>
print('--- generated code ---')
print(code)
print('--- end ---')
try:
    ast.parse(code)
    print('PARSES OK')
except SyntaxError as e:
    print(f'SYNTAX ERROR: {e}')
