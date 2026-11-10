"""Run the frozen study without any historical PYTHONPATH entries."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'assets/basis'),
               str(ROOT / 'assets/history'), str(ROOT / 'assets/support')]
for key, value in dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', FREQDUET_DIAGNOSTIC_COMMIT='2b0115af7b',
        FREQDUET_HANDOFF_COMMIT='17043d5c03', FREQDUET_RESOURCE_COMMIT='eec7a20c71').items():
    os.environ[key] = value

p = argparse.ArgumentParser()
p.add_argument('command', choices=('import_check', 'native_engineering', 'case'))
p.add_argument('--root', type=Path, default=ROOT / 'runs')
p.add_argument('--regime')
p.add_argument('--day', type=int)
p.add_argument('--arm')
a = p.parse_args()
a.assets = ROOT / 'assets'
for name in ('benchmark_summary.json', 'training_equivalence/summary.json'):
    target = a.root / 'runtime/runtime_optimization_20261014' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(a.assets / 'runtime/traffic/runtime_optimization_20261014' / name, target)
import study
study.NATIVE_CODE = ROOT / 'python'
if a.command == 'import_check':
    f = study.bind(a.assets)
    f.base.fixed.pilot.joint.optimized_runtime(a.root / 'runtime/import_check')
elif a.command == 'native_engineering':
    day = study.read(ROOT / 'python/protocol.json')['engineering_day']
    study.collect(a, 'engineering', day, 'fixed_bilateral', True)
else:
    study.collect(a, a.regime, a.day, a.arm)
sources = {name: str(Path(m.__file__).resolve()) for name, m in list(sys.modules.items())
    if getattr(m, '__file__', None) and str(m.__file__).startswith('/home/zhengliang01/scheduleurm_work/')
    and not Path(m.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())}
external = {name: path for name, path in sources.items() if not Path(path).is_relative_to(ROOT)}
assert not external, external
print('FREQDUET_PORTABLE_COMPLETE', a.command, len(sources), flush=True)
