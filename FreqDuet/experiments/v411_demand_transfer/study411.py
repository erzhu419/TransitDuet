"""Run an unchanged V410 pair under two new demand settings."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
OLD = WORK / 'FreqDuet-v410-within-day-control'
sys.path.insert(0, str(OLD / 'source01'))
import study410 as old
import numpy as np

read, write = old.read, old.write
PLAN = read(HERE / 'protocol.json')
DAYS, ARMS = PLAN['days'], PLAN['arms']
CAP, FIXED, RULE = old.CAP, old.FIXED, old.RULE


def preflight(root):
    days = [d for ds in DAYS.values() for d in ds]
    assert len(days) == len(set(days)) and len(days) * len(ARMS) == PLAN['physical_start_cap']
    assert ARMS == old.PLAN['arms']
    for project in WORK.glob('FreqDuet-*'):
        if not project.is_dir() or project == HERE.parent:
            continue
        for day in days:
            assert not list(project.glob(f'run*/cases/*/{day}')), (project, day)
        for manifest in project.glob('source*/protocol.json'):
            assert not set(map(str, days)) & old.base.declared_days(read(manifest)), manifest
    gate = read(OLD / 'run01/calibration/summary/ready.json')
    ref = read(OLD / 'run01/aggregate/summary/results.json')
    assert ref['complete'] and ref['passed'] and ref['calibration'] == gate
    assert gate['passed'] and gate['tilt'] == 0 and gate['variant'] == 'original'
    assert all(sum(row.values()) > 0 for row in gate['counts'].values())
    old.frozen.prepare(root)
    bound = old.frozen.study.bind(old.frozen.ASSETS)
    bound.base.fixed.pilot.joint.optimized_runtime(root / 'runtime/import_check')
    write(root / 'preflight/summary/results.json', dict(
        passed=True, physical_starts=0, days=DAYS, frozen_gate=gate,
        regime_settings=PLAN['regimes']))


def case(root, regime, day):
    gate = read(root / 'preflight/summary/results.json')
    assert gate['passed'] and day in DAYS[regime]
    if day != DAYS[regime][0]:
        assert read(root / 'engineering/summary/ready.json')['passed']
    # These two bindings only select the manifest and environment setting.
    # V410's policy, RNG seed, kernel, and endpoint validation are untouched.
    old.HERE, old.REGIME = HERE, regime
    original = old.make_audit

    def checked_audit(runtime, credit, arm, settings, *args):
        audit = original(runtime, credit, arm, settings, *args)

        class ScaleAudit(type(audit)):
            def finish(self, runner, records, holds):
                super().finish(runner, records, holds)
                self.contract['applied_demand_scale'] = runner.env._demand_scale
                assert self.contract['applied_demand_scale'] == settings['demand_scale']
                assert self.contract['physical_vehicle_count'] == 12

        return ScaleAudit('joint', credit)

    old.make_audit = checked_audit
    row = old.collect(root, day, ARMS, gate['frozen_gate'])
    row['regime'] = regime
    for arm in ARMS:
        receipt = read(root / 'cases' / regime / str(day) / arm / 'summary/results.json')
        row['arms'][arm]['applied_demand_scale'] = receipt['contract']['applied_demand_scale']
        row['arms'][arm]['graph_statuses'] = receipt['graph']['statuses']
    write(root / 'pairs' / regime / str(day) / 'summary/results.json', row)


def engineering(root):
    first = {}
    for r, ds in DAYS.items():
        row = read(root / 'pairs' / r / str(ds[0]) / 'summary/results.json')
        assert row['regime'] == r and row['day'] == ds[0]
        assert row['paired_events'] == 5200 and set(row['arms']) == set(ARMS)
        assert all(a['applied_demand_scale'] == PLAN['regimes'][r]['demand_scale']
                   for a in row['arms'].values())
        first[r] = dict(day=ds[0], generated=row['generated'])
    write(root / 'engineering/summary/ready.json',
          dict(passed=True, physical_starts=6, first=first, effect_gate=False))


def aggregate(root):
    assert read(root / 'engineering/summary/ready.json')['passed']
    starts = len(list((root / 'cases').glob('*/*/*/started.json')))
    assert starts == PLAN['physical_start_cap']
    regimes = {}
    for i, (regime, days) in enumerate(DAYS.items()):
        rows = {d: read(root / 'pairs' / regime / str(d) / 'summary/results.json') for d in days}
        assert all(row['day'] == d and row['regime'] == regime
                   and set(row['arms']) == set(ARMS) for d, row in rows.items())
        draws = np.random.default_rng(PLAN['bootstrap_seed'] + i).integers(
            0, len(days), size=(old.PLAN['bootstrap_draws'], len(days)))
        effects = {}
        for a, b in ((CAP, RULE), (CAP, FIXED), (RULE, FIXED)):
            effect = {}
            for k in rows[days[0]]['arms'][a]['metrics']:
                v = np.asarray([r['arms'][a]['metrics'][k]-r['arms'][b]['metrics'][k]
                                for r in rows.values()])
                assert np.isfinite(v).all()
                effect[k] = dict(mean=float(v.mean()), by_day=dict(zip(days, v.tolist())),
                                 ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist())
            effects[f'{a}_minus_{b}'] = effect
        primary = effects[f'{CAP}_minus_{RULE}']
        matching = {k: -lim <= primary[k]['ci95'][0] <= primary[k]['ci95'][1] <= lim
                    for k, lim in old.margins().items()}
        outcomes = {old.base.JOURNEY: primary[old.base.JOURNEY]['ci95'][1] < 0,
                    **{k: primary[k]['ci95'][1] <= 0 for k in old.base.SAFETY}}
        regimes[regime] = dict(matching=matching, outcomes=outcomes, effects=effects,
            passed=all(matching.values()) and all(outcomes.values()), by_day=rows)
    write(root / 'aggregate/summary/results.json', dict(complete=True, physical_starts=starts,
          training_updates=0, main_policy_changed=False, regimes=regimes))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('preflight', 'case', 'engineering', 'aggregate'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--regime')
    p.add_argument('--day', type=int)
    a = p.parse_args()
    if a.phase == 'case':
        case(a.root, a.regime, a.day)
    else:
        globals()[a.phase](a.root)
