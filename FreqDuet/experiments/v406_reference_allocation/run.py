"""Run the fixed-comparator allocation study against the frozen physical runtime."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent.parent / 'FreqDuet-v404-publication-closure/portable_runtime'
ASSETS = RUNTIME / 'assets'
sys.path[:0] = [str(HERE), str(RUNTIME / 'python'), str(ASSETS / 'basis'),
               str(ASSETS / 'history'), str(ASSETS / 'support')]
for key, value in dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', FREQDUET_DIAGNOSTIC_COMMIT='2b0115af7b',
        FREQDUET_HANDOFF_COMMIT='17043d5c03', FREQDUET_RESOURCE_COMMIT='eec7a20c71').items():
    os.environ[key] = value

import study
from coordination import Controller, Manager, SimpleCap
from worker import helper

study.NATIVE_CODE = RUNTIME / 'python'
study.HERE = HERE
PLAN = study.read(HERE / 'protocol.json')
NATIVE_AUDIT = study.make_audit
ARMS = PLAN['arms']


def audit(frozen, credit, arm, regime):
    if arm == 'cap_10':
        return NATIVE_AUDIT(frozen, credit, arm, regime)
    native_type = type(NATIVE_AUDIT(frozen, credit, 'fixed_bilateral', regime))

    class CoordinationAudit(native_type):
        def attach(self, runner):
            super().attach(runner)
            self.bridge.detach()
            self.manager = Manager(self.env, self.credit, arm == 'coordinated', arm)
            self.variant = self.manager.arm
            self.bridge = (SimpleCap if arm == 'simple_cap' else Controller)(
                self.env, self.launches, self.credit, self.manager)
            from env.bus import Bus
            self.bridge.attach(Bus)

        def finish(self, runner, records, holds):
            super().finish(runner, records, holds)
            assert not self.bridge.pending
            assert all(d['actual_s'] >= d['public_s'] for d in self.dispatches)
            counts = self.bridge.activity
            assert counts['transfers_proposed'] == sum(counts[k] for k in
                ('transfers_followed', 'transfers_abstained', 'transfers_expired',
                 'transfers_not_reached'))

    return CoordinationAudit('joint', credit)


def portable_imports():
    prefix = RUNTIME.parent.parent
    packages = Path(sys.prefix).resolve()
    external = []
    for module in tuple(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and str(file).startswith(str(prefix) + '/'):
            path = Path(file).resolve()
            if not any(path.is_relative_to(root) for root in (RUNTIME, HERE, packages)):
                external.append(str(path))
    assert not external, external


def prepare(root):
    for name in ('benchmark_summary.json', 'training_equivalence/summary.json'):
        target = root / 'runtime/runtime_optimization_20261014' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(ASSETS / 'runtime/traffic/runtime_optimization_20261014' / name, target)


def checks(args):
    from tests import checks as run_checks
    passed = run_checks()
    prepare(args.root)
    frozen = study.bind(ASSETS)
    frozen.base.fixed.pilot.joint.optimized_runtime(args.root / 'runtime/import_check')
    portable_imports()
    workspace = args.root.parent.parent
    days = {str(d) for values in PLAN['days'].values() for d in values}
    for project in workspace.glob('FreqDuet-*'):
        if project == args.root.parent:
            continue
        for d in days:
            assert not list(project.glob(f'run*/cases/*/{d}')), (project, d)
        for p in project.glob('source*/protocol.json'):
            doc = json.loads(p.read_text())
            declared = doc.get('days', {})
            values = (v for group in declared.values() for v in group) if isinstance(declared, dict) else declared
            assert not days.intersection(map(str, values)), p
    study.write(args.root/'checks/summary/results.json', dict(passed=True, checks=passed,
        physical_starts=0, inspected_days=sorted(days)))


def case(args):
    assert study.read(args.root/'checks/summary/results.json')['passed']
    assert args.arm in ARMS and args.day in PLAN['days'][args.regime]
    study.make_audit = audit
    args.assets = ASSETS
    study.collect(args, args.regime, args.day, args.arm)
    portable_imports()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('checks', 'case'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--regime')
    p.add_argument('--day', type=int)
    p.add_argument('--arm')
    a = p.parse_args()
    globals()[a.command](a)
