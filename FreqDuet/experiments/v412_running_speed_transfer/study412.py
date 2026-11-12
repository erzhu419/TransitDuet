"""Frozen V410 controllers under a prospective road-speed input shift."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
sys.path.insert(0, str(WORK / 'FreqDuet-v411-demand-transfer/source01'))
import study411 as prior
import numpy as np
import pandas as pd

old, read, write = prior.old, prior.read, prior.write
PLAN = read(HERE / 'protocol.json')
DAYS, ARMS, SHIFT = PLAN['days'], PLAN['arms'], PLAN['road_profile_shift_mps']
CAP, RULE = prior.CAP, prior.RULE


def shifted_profile(frame, delta):
    # Route.set_routes consumes the five geometry fields and then hourly speeds.
    assert set(frame.columns[:5]) == {'route_id', 'start_stop', 'end_stop', 'distance', 'V_max'}
    profile = frame.iloc[:, 5:].astype(float) + delta
    assert profile.shape[1] and np.isfinite(profile.to_numpy()).all()
    shifted = frame.copy()
    for column in profile:
        shifted[column] = profile[column]
    return shifted


def preflight(root):
    days = DAYS['normal']
    assert days == DAYS['slow'] and len(days) == len(set(days)) == 8
    assert sum(map(len, DAYS.values())) * len(ARMS) == PLAN['physical_start_cap']
    assert ARMS == old.PLAN['arms'] and all(r == old.PLAN['regimes']['heavy']
                                          for r in PLAN['regimes'].values())
    for project in WORK.glob('FreqDuet-*'):
        if not project.is_dir() or project == HERE.parent:
            continue
        for day in days:
            assert not list(project.glob(f'run*/cases/*/{day}')), (project, day)
        for manifest in project.glob('source*/protocol.json'):
            assert not set(map(str, days)) & old.base.declared_days(read(manifest)), manifest
    gate = read(prior.OLD / 'run01/calibration/summary/ready.json')
    ref = read(prior.OLD / 'run01/aggregate/summary/results.json')
    assert ref['complete'] and ref['passed'] and ref['calibration'] == gate
    old.frozen.prepare(root)
    bound = old.frozen.study.bind(old.frozen.ASSETS)
    joint = bound.base.fixed.pilot.joint
    _, runner, load_config, _ = joint.optimized_runtime(root / 'runtime/import_check')
    cfg = load_config(str(joint.MODEL_ROOT / 'configs_freqduet' / (joint.CURRENT + '.yaml')))
    data = Path(sys.modules[runner.__module__].SCRIPT_DIR) / cfg['env']['path'] / 'data'
    frame = pd.read_excel(data / 'route_news.xlsx')
    shifted = shifted_profile(frame, SHIFT['slow'])
    assert shifted.iloc[:, :5].equals(frame.iloc[:, :5])
    np.testing.assert_array_equal(shifted.iloc[:, 5:].to_numpy(),
                                  frame.iloc[:, 5:].to_numpy(dtype=float) + SHIFT['slow'])
    write(root / 'preflight/summary/results.json', dict(
        passed=True, physical_starts=0, days=DAYS, frozen_gate=gate, shift=SHIFT,
        speed_input_shape=list(frame.iloc[:, 5:].shape)))


def case(root, regime, day):
    gate = read(root / 'preflight/summary/results.json')
    assert gate['passed'] and day in DAYS[regime]
    if day != DAYS[regime][0]:
        assert read(root / 'engineering/summary/ready.json')['passed']
    old.HERE, old.REGIME = HERE, regime
    original, delta = old.make_audit, SHIFT[regime]

    def checked_audit(runtime, credit, arm, settings, *args):
        audit = original(runtime, credit, arm, settings, *args)

        class SpeedAudit(type(audit)):
            def attach(self, runner):
                super().attach(runner)
                frame = self.env.routes_set
                self.expected = frame.iloc[:, 5:].to_numpy(dtype=float) + delta
                if delta:
                    self.env.routes_set = shifted_profile(frame, delta)

            def finish(self, runner, records, holds):
                super().finish(runner, records, holds)
                np.testing.assert_array_equal(
                    [r.speed_history.to_numpy(dtype=float) for r in self.env.routes], self.expected)
                assert self.env._demand_scale == 1.2 and self.contract['physical_vehicle_count'] == 12
                self.contract['applied_road_profile_shift_mps'] = delta

        return SpeedAudit('joint', credit)

    old.make_audit = checked_audit
    row = old.collect(root, day, ARMS, gate['frozen_gate'])
    row['regime'] = regime
    for arm in ARMS:
        receipt = read(root / 'cases' / regime / str(day) / arm / 'summary/results.json')
        row['arms'][arm]['road_profile_shift_mps'] = receipt['contract']['applied_road_profile_shift_mps']
        row['arms'][arm]['graph_statuses'] = receipt['graph']['statuses']
    write(root / 'pairs' / regime / str(day) / 'summary/results.json', row)


def engineering(root):
    first = {}
    for r, ds in DAYS.items():
        row = read(root / 'pairs' / r / str(ds[0]) / 'summary/results.json')
        assert row['regime'] == r and row['day'] == ds[0]
        assert row['paired_events'] == 5200 and set(row['arms']) == set(ARMS)
        assert all(a['road_profile_shift_mps'] == SHIFT[r] for a in row['arms'].values())
        first[r] = dict(day=ds[0], generated=row['generated'])
    assert first['normal']['generated'] == first['slow']['generated']
    write(root / 'engineering/summary/ready.json',
          dict(passed=True, physical_starts=6, first=first, effect_gate=False))


def aggregate(root):
    days = DAYS['normal']
    for day in days:
        normal, slow = (read(root / 'pairs' / r / str(day) / 'summary/results.json') for r in DAYS)
        assert normal['generated'] == slow['generated'], day
    prior.PLAN, prior.DAYS, prior.ARMS = PLAN, DAYS, ARMS
    prior.aggregate(root)
    out = root / 'aggregate/summary'
    results = read(out / 'results.json')
    contrast = f'{CAP}_minus_{RULE}'
    normal, slow = (results['regimes'][r]['effects'][contrast] for r in ('normal', 'slow'))
    draws = np.random.default_rng(PLAN['bootstrap_seed'] + 2).integers(
        0, len(days), size=(old.PLAN['bootstrap_draws'], len(days)))
    effects = {}
    for metric in (old.DOSE, old.base.JOURNEY, *old.base.SAFETY):
        v = np.asarray([slow[metric]['by_day'][str(d)] - normal[metric]['by_day'][str(d)] for d in days])
        effects[metric] = dict(mean=float(v.mean()), by_day=dict(zip(days, v.tolist())),
                              ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist())
    write(out / 'interaction.json', dict(contrast='slow minus normal of capped minus transition',
          descriptive=True, effects=effects))


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
