"""Calibrate and evaluate a within-day, native-bin transition control."""

import argparse
from collections import Counter
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
OLD = WORK / 'FreqDuet-v409-holding-dose'
sys.path.insert(0, str(OLD / 'source02'))
import study409 as base
import numpy as np

frozen, read, write = base.frozen, base.read, base.write
PLAN = read(HERE / 'protocol.json')
REGIME, CAP, DOSE = base.REGIME, base.CAP, base.DOSE
FIXED, RULE = 'fixed_cap_10', 'transition'
BINS = (0, 5, 10)
DAYS, DEV = PLAN['days'][REGIME], PLAN['development_days']
SHARES = tuple(f'bin_{h}_share' for h in BINS)


def draw(table, ref, legal, tilt, u):
    legal = tuple(h for h in BINS if h in legal)
    weights = [table[str(ref)][str(h)] * math.exp(tilt * (h - ref) / 5)
               for h in legal]
    total = sum(weights)
    if not total:
        return ref
    threshold = u * total
    for h, weight in zip(legal, weights):
        threshold -= weight
        if threshold < 0:
            return h
    return next(h for h, w in reversed(list(zip(legal, weights))) if w > 0)


def make_audit(runtime, credit, arm, regime, day, table, tilt):
    if arm in (CAP, FIXED):
        return base.audit(runtime, credit, arm, regime)
    original = base.audit(runtime, credit, FIXED, regime)

    class Transition(frozen.FixedCap):
        def observe(self, bus, legal, native_action):
            profile = super().observe(bus, legal, native_action)
            action = draw(table, profile['shadow_action_s'], legal, tilt, self.rng.random())
            profile['shadow_action_s'] = action
            self.execution_records[-1]['action_s'] = action
            return profile

    class RuleAudit(type(original)):
        def attach(self, runner):
            from env.bus import Bus
            super().attach(runner)
            self.bridge.detach()
            self.bridge = Transition(self.env, self.launches, self.credit, self.manager, 10)
            self.bridge.rng = random.Random(PLAN['policy_seed'] + day)
            self.bridge.attach(Bus)

    return RuleAudit('joint', credit)


def metrics(row):
    return dict(row['metrics'], **{f'bin_{h}_share': row['bins'].get(str(h), 0)/5200
                                  for h in BINS})


def preflight(root):
    assert PLAN['regimes'][REGIME] == base.PLAN['regimes'][REGIME]
    assert set(DEV).isdisjoint(DAYS) and len(DAYS) == len(set(DAYS)) == 16
    assert len(DEV) * len(PLAN['tilts']) + len(DAYS) * len(PLAN['arms']) == PLAN['physical_start_cap']
    reserved = set(map(str, DAYS))
    for project in WORK.glob('FreqDuet-*'):
        if not project.is_dir() or project == HERE.parent:
            continue
        for day in DAYS:
            assert not list(project.glob(f'run*/cases/*/{day}')), (project, day)
        for manifest in project.glob('source*/protocol.json'):
            assert not reserved & base.declared_days(read(manifest)), manifest
    archive = frozen.helper(frozen.ASSETS, 'gate_default')
    table = {str(h): {str(k): 0 for k in BINS} for h in BINS}
    targets = {}
    for day in DEV:
        path = OLD / 'run01/development/cases' / REGIME / str(day) / CAP
        trace = archive.raw(path / 'raw/path.pkl.gz')
        summary = read(OLD / 'run01/calibration_days' / str(day) / 'summary/results.json')
        targets[str(day)] = metrics(summary['arms'][CAP])
        assert summary['day'] == day and len(trace['queries']) == len(trace['graph_queries']) == 5200
        for q, g in zip(trace['queries'], trace['graph_queries']):
            assert all(q[k] == g[k] for k in ('trip', 'bus', 'station', 'time_s'))
            h = q['executed_request_s']
            assert h in BINS and h in q['legal'] and h == g['shadow_action_s']
            ref = base.project_native(q['action_s'], q['legal'], 10)
            table[str(ref)][str(h)] += 1
    assert all(sum(row.values()) > 0 for row in table.values())
    assert table['0']['5'] + table['0']['10'] > 0 and table['10']['0'] + table['10']['5'] > 0
    example = {'5': {'0': 1, '5': 1, '10': 1}}
    assert draw(example, 5, BINS, 0, .01) == 0 and draw(example, 5, BINS, 0, .99) == 10
    assert draw(example, 5, (5,), .5, .99) == 5
    assert draw({'5': {'5': 0}}, 5, (5,), 0, .5) == 5
    for path in (root, root / 'development'):
        frozen.prepare(path)
        bound = frozen.study.bind(frozen.ASSETS)
        bound.base.fixed.pilot.joint.optimized_runtime(path / 'runtime/import_check')
    write(root / 'preflight/summary/results.json', dict(passed=True, physical_starts=0,
          days=DAYS, development_days=DEV, counts=table, targets=targets))


def collect(root, day, arms, gate, reference=None):
    physical = frozen.helper(frozen.ASSETS, 'physical')
    args = argparse.Namespace(root=root, assets=frozen.ASSETS)
    frozen.study.HERE = HERE
    frozen.study.make_audit = lambda runtime, credit, arm, regime: make_audit(
        runtime, credit, arm, regime, day, gate['counts'], gate['tilt'])
    uniforms, generated = reference if reference is not None else (None, None)
    rows = {}
    for arm in arms:
        result, trace, _ = frozen.study.collect(args, REGIME, day, arm)
        detail, op = result['passenger_detail'], result['operating_detail']
        if uniforms is None:
            uniforms, generated = trace['uniforms'], detail['generated']
        assert result['physical_account']['passed'] and op['early_departures'] == 0
        assert detail['generated'] == generated and len(trace['uniforms']) == 5200
        assert physical.crn_pair(uniforms, trace['uniforms'])['exact']
        bins = Counter(str(q['executed_request_s']) for q in trace['queries'])
        assert sum(bins.values()) == 5200 and set(bins) <= set(map(str, BINS))
        assert sum(int(h)*n for h, n in bins.items()) == round(262 * result['endpoints'][DOSE])
        rows[arm] = dict(metrics=metrics(dict(bins=bins, metrics=dict(result['endpoints'],
            unboarded=detail['unserved'], journey_p95_s=detail['journey_p50_p90_p95_s'][2]))),
            bins=bins, wall_s=result['wall_s'], callback_p95_s=op['callback_p50_p95_max_s'][1])
    return dict(day=day, generated=generated, paired_events=5200, arms=rows)


def development(root, day):
    gate = read(root / 'preflight/summary/results.json')
    assert gate['passed'] and day in DEV
    old = OLD / 'run01/development/cases' / REGIME / str(day) / CAP
    trace = frozen.helper(frozen.ASSETS, 'gate_default').raw(old / 'raw/path.pkl.gz')
    reference = trace['uniforms'], read(old / 'summary/results.json')['passenger_detail']['generated']
    rows = {}
    for key, tilt in PLAN['tilts'].items():
        arm = RULE + '_' + key
        rows[key] = collect(root / 'development', day, (arm,),
                            dict(gate, tilt=tilt), reference)['arms'][arm]
    write(root / 'calibration_days' / str(day) / 'summary/results.json',
          dict(day=day, paired_events=5200, variants=rows))


def margins():
    return {DOSE: PLAN['dose_margin_s_per_trip'], **dict.fromkeys(SHARES, PLAN['bin_share_margin'])}


def calibrate(root):
    gate = read(root / 'preflight/summary/results.json')
    rows = {d: read(root / 'calibration_days' / str(d) / 'summary/results.json') for d in DEV}
    assert all(row['day'] == day and row['paired_events'] == 5200 for day, row in rows.items())
    limits = margins()
    errors = {variant: {k: sum(rows[d]['variants'][variant]['metrics'][k] -
              gate['targets'][str(d)][k] for d in DEV)/len(DEV) for k in limits}
              for variant in PLAN['tilts']}
    eligible = [v for v, e in errors.items() if all(abs(e[k]) <= lim for k, lim in limits.items())]
    chosen = min(eligible, key=lambda v: (abs(errors[v][DOSE]),
        max(abs(errors[v][k]) for k in SHARES), abs(PLAN['tilts'][v]))) if eligible else None
    result = dict(passed=chosen is not None, variant=chosen, development_errors=errors,
                  counts=gate['counts'], tilt=PLAN['tilts'][chosen] if chosen else None)
    out = root / 'calibration/summary'
    write(out / 'results.json', result)
    if chosen is not None:
        write(out / 'ready.json', result)


def confirmation(root, day):
    gate = read(root / 'calibration/summary/ready.json')
    assert gate['passed'] and day in DAYS
    write(root / 'pairs' / str(day) / 'summary/results.json',
          collect(root, day, PLAN['arms'], gate))


def aggregate(root):
    gate = read(root / 'calibration/summary/ready.json')
    rows = {d: read(root / 'pairs' / str(d) / 'summary/results.json') for d in DAYS}
    assert all(row['day'] == d and set(row['arms']) == set(PLAN['arms']) for d, row in rows.items())
    starts = sum(len(list(p.glob(f'cases/{REGIME}/*/*/started.json')))
                 for p in (root, root / 'development'))
    assert starts == PLAN['physical_start_cap']
    draws = np.random.default_rng(PLAN['bootstrap_seed']).integers(0, len(DAYS),
             size=(PLAN['bootstrap_draws'], len(DAYS)))
    effects = {}
    for a, b in ((CAP, RULE), (CAP, FIXED), (RULE, FIXED)):
        effect = {}
        for k in rows[DAYS[0]]['arms'][a]['metrics']:
            v = np.asarray([r['arms'][a]['metrics'][k]-r['arms'][b]['metrics'][k]
                            for r in rows.values()])
            assert np.isfinite(v).all()
            effect[k] = dict(mean=float(v.mean()), by_day=dict(zip(DAYS, v.tolist())),
                             ci95=np.quantile(v[draws].mean(1), [.025, .975]).tolist())
        effects[f'{a}_minus_{b}'] = effect
    primary = effects[f'{CAP}_minus_{RULE}']
    matching = {k: -limit <= primary[k]['ci95'][0] <= primary[k]['ci95'][1] <= limit
                for k, limit in margins().items()}
    outcomes = {base.JOURNEY: primary[base.JOURNEY]['ci95'][1] < 0,
                **{k: primary[k]['ci95'][1] <= 0 for k in base.SAFETY}}
    write(root / 'aggregate/summary/results.json', dict(complete=True, physical_starts=starts,
          training_updates=0, main_policy_changed=False, calibration=gate, matching=matching,
          outcomes=outcomes, passed=all(matching.values()) and all(outcomes.values()),
          effects=effects, by_day=rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('preflight', 'development', 'calibrate', 'confirmation', 'aggregate'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--day', type=int)
    a = p.parse_args()
    globals()[a.phase](*((a.root,) if a.day is None else (a.root, a.day)))
