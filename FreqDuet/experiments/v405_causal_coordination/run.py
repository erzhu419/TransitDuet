"""Run only the frozen V405 study against the standalone V404 environment."""

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
from coordination import Controller, Manager
from worker import helper

study.NATIVE_CODE = RUNTIME / 'python'
study.HERE = HERE
PLAN = study.read(HERE / 'protocol.json')
NATIVE_AUDIT = study.make_audit
ARMS = PLAN['arms']


def audit(frozen, credit, arm, regime):
    if arm not in ('hold_transfer', 'coordinated'):
        return NATIVE_AUDIT(frozen, credit, arm, regime)
    native_type = type(NATIVE_AUDIT(frozen, credit, 'fixed_bilateral', regime))

    class CoordinationAudit(native_type):
        def attach(self, runner):
            super().attach(runner)
            self.bridge.detach()
            self.manager = Manager(self.env, self.credit, arm == 'coordinated')
            self.variant = self.manager.arm
            self.bridge = Controller(self.env, self.launches, self.credit, self.manager)
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
    result, _, out = study.collect(args, args.regime, args.day, args.arm)
    original = args.root.parent / 'run02/runtime' / f'{args.regime}_{args.day}_{args.arm}'
    original = original / 'ready' / f'v404_{args.regime}_{args.day}_{args.arm}.json'
    if original.exists() and args.root.name != 'run02':
        previous = study.read(original)
        same = all(result['results'][k] == value for k, value in previous.items())
        assert same, 'record-schema repair changed the completed original path'
        study.write(out/'summary/original_endpoint_equivalence.json',
                    dict(passed=True, fields=len(previous), control_code_changed=False))
    portable_imports()


def aggregate(args):
    from statistics import mean
    physical, archive = helper(ASSETS, 'physical'), helper(ASSETS, 'gate_default')
    out, n = {}, 0
    for regime, days in PLAN['days'].items():
        rows = {}
        for day in days:
            arms, reference = {}, None
            for arm in ARMS:
                path = args.root / 'cases' / regime / str(day) / arm
                if not (path / 'summary/results.json').exists():
                    path = args.root.parent / 'run02/cases' / regime / str(day) / arm
                row = study.read(path / 'summary/results.json')
                trace = archive.raw(path / 'raw/path.pkl.gz')
                assert row['physical_account']['passed']
                if reference is None:
                    reference = trace['uniforms']
                receipt = physical.crn_pair(reference, trace['uniforms'])
                assert receipt['exact'] and receipt['common_events'] == 5200
                published = [p for e in trace['slow_events'] if e['kind'] == 'publish'
                             for p in e['plans']]
                dispatches = {d['trip']: d for d in trace['dispatches']}
                realized = sum(dispatches[e['trip']]['actual_s'] == e['release_s'] for e in published)
                arms[arm] = dict(metrics=row['endpoints'], wall_s=row['wall_s'],
                    graph_statuses=row['graph']['statuses'],
                    overrides=row['graph']['actual_overrides'],
                    coordination=row['graph'].get('coordination', {}),
                    scheduled_edits=len(published), departed_at_edit=realized,
                    earliness=row['operating_detail']['early_departures'],
                    unserved=row['passenger_detail']['unserved'])
                n += 1
            rows[day] = arms
        comparisons = {}
        for baseline in ARMS[:-1]:
            by_day = {d:{m:v - a[baseline]['metrics'][m]
                         for m, v in a['coordinated']['metrics'].items()} for d, a in rows.items()}
            comparisons[baseline] = dict(by_day=by_day, mean={
                m:mean(d[m] for d in by_day.values()) for m in next(iter(by_day.values()))})
        cap = comparisons['cap_10']
        gate = all(d[m] <= 0 for d in cap['by_day'].values()
                   for m in ('cv_squared', 'public_delay_s_per_day', 'readiness_s_per_trip'))
        out[regime] = dict(days=rows, comparisons=comparisons, cap_gate=gate
                           and cap['mean']['journey_s_per_passenger'] < 0)
    assert n == PLAN['physical_start_cap']
    charged = sum(len(list((args.root.parent / name).glob('cases/*/*/*/started.json')))
                  for name in ('run02', 'run03'))
    study.write(args.root/'aggregate/summary/results.json', dict(completed=True,
        physical_starts=charged, completed_unique_paths=n, all_paired=True,
        regimes=out, training_updates=0,
        promotion=False, passed_development_screen=all(v['cap_gate'] for v in out.values())))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('checks', 'case', 'aggregate'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--regime')
    p.add_argument('--day', type=int)
    p.add_argument('--arm')
    a = p.parse_args()
    globals()[a.command](a)
