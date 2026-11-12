"""Matched-dose study using the existing V408 and frozen V406 runtime."""

import argparse
from collections import Counter
import math
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
sys.path.insert(0, str(WORKSPACE / 'FreqDuet-v408-heavy-confirmation/source01'))
from confirm import declared_days, frozen
from coupled_model import project_native
from frozen_v406 import choose
import numpy as np

read, write = frozen.study.read, frozen.study.write
PLAN = read(HERE / 'protocol.json')
REGIME = 'heavy'
DAYS = PLAN['days'][REGIME]
DEV = PLAN['development_days']
ARMS = PLAN['arms']
FULL, CAP = 'v406_single', 'v406_cap_10'
DOSE = 'holding_s_per_trip'
JOURNEY = 'journey_s_per_passenger'
SAFETY = ('cv_squared', 'readiness_s_per_trip', 'public_delay_s_per_day')
OLD = WORKSPACE / 'FreqDuet-v407-service-continuation/run03'


class Capped(frozen.FrozenController):
    def select_plan(self, model):
        s = model.snapshot
        own = next(v for v in s.vehicles if v.bus == s.current_bus)
        key = own.bus, own.trip, s.routes[own.direction][own.index].station
        assert self.manager.method == 'single' and not self.manager.due and self.pending is None
        legal = tuple(h for h in own.legal if h <= 10)
        baseline = project_native(s.native_action_s, own.legal, 10)
        action, pair, edit, picked, fallback, evaluations, status, reference = choose(
            self.models, key, baseline, legal, None, (), paired=False)
        assert pair is None and edit is None and action in legal
        result = dict(action_s=action, commands={}, evaluations=evaluations, status=status)
        if reference is not None:
            if len(self.models) > 1 and action != s.native_action_s:
                self.activity['censored_overrides'] += 1
            result.update(reference=reference, selected=picked, fallback=fallback,
                          full_cost=picked['cost'])
            if action != s.native_action_s:
                result['commands'] = {key: action}
        return result


def audit(runtime, credit, arm, regime):
    if arm not in (CAP, 'fixed_cap_0'):
        return frozen.audit(runtime, credit, arm, regime)
    base = frozen.audit(runtime, credit, 'fixed_cap_10' if arm == 'fixed_cap_0' else FULL, regime)

    class DoseAudit(type(base)):
        def attach(self, runner):
            super().attach(runner)
            if arm == 'fixed_cap_0':
                self.bridge.cap = 0
            else:
                from env.bus import Bus
                self.bridge.detach()
                self.bridge = Capped(self.env, self.launches, self.credit, self.manager)
                self.bridge.attach(Bus)

    return DoseAudit('joint', credit)


def mixture(target, lo, hi, a, b):
    if not lo <= target <= hi or hi <= lo:
        return None
    p = (target - lo) / (hi - lo)
    return {a: 1 - p, b: p}


def preflight(root):
    from tests import fixture
    assert len(DAYS) == len(set(DAYS)) == PLAN['max_concurrent_days']
    assert set(DAYS).isdisjoint(DEV)
    assert len(DAYS) * len(ARMS) + len(DEV) == PLAN['physical_start_cap']
    assert len(DEV) == PLAN['development_start_cap']
    assert PLAN['regimes'][REGIME] == frozen.PLAN['regimes'][REGIME]
    for project in WORKSPACE.glob('FreqDuet-*'):
        if not project.is_dir() or project == HERE.parent:
            continue
        for day in DAYS:
            assert not list(project.glob(f'run*/cases/*/{day}')), (project, day)
        for manifest in project.glob('source*/protocol.json'):
            assert not set(map(str, DAYS)) & declared_days(read(manifest)), manifest
    for day in DEV:
        for arm in (FULL, 'fixed_cap_10', 'fixed_cap_15'):
            old = read(OLD / 'cases' / REGIME / str(day) / arm / 'summary/results.json')
            assert old['completed'] and old['physical_account']['passed']
            assert old['day'] == day and old['arm'] == arm

    # A controlled objective prefers 45 s globally and 10 s in the truncated menu.
    model, _ = fixture()
    snap = model.snapshot
    own = snap.vehicles[0]
    key = own.bus, own.trip, snap.routes[own.direction][own.index].station
    def evaluation(commands=None):
        h = (commands or {}).get(key, min(snap.native_action_s, 10))
        return dict(coverage=True, failure=False, cost=100. - h, cv2=.1,
            missing_commands=(), launches={}, public_delays={}, readiness_delays={})
    stub = SimpleNamespace(snapshot=snap, evaluate=evaluation)
    bridge = Capped.__new__(Capped)
    bridge.models, bridge.pending = [stub], None
    bridge.manager = SimpleNamespace(method='single', due=False)
    assert choose([stub], key, snap.native_action_s, own.legal, None, (), False)[0] == 45
    assert bridge.select_plan(stub)['action_s'] == 10
    assert 0. <= mixture(6, 0, 10, 'a', 'b')['b'] <= 1.
    assert mixture(11, 0, 10, 'a', 'b') is None
    frozen.prepare(root)
    frozen.prepare(root / 'development')
    bound = frozen.study.bind(frozen.ASSETS)
    bound.base.fixed.pilot.joint.optimized_runtime(root / 'runtime/import_check')
    write(root / 'preflight/summary/results.json', dict(passed=True, physical_starts=0,
        days=DAYS, development_days=DEV, frozen_source=str(frozen.HERE)))


def collect(root, day, arms, reference=None):
    args = argparse.Namespace(root=root, assets=frozen.ASSETS)
    frozen.study.HERE = HERE
    frozen.study.make_audit = audit
    physical = frozen.helper(frozen.ASSETS, 'physical')
    uniforms, generated = reference if reference is not None else (None, None)
    rows = {}
    for arm in arms:
        result, trace, _ = frozen.study.collect(args, REGIME, day, arm)
        detail, operating = result['passenger_detail'], result['operating_detail']
        if uniforms is None:
            uniforms, generated = trace['uniforms'], detail['generated']
        assert result['physical_account']['passed'] and operating['early_departures'] == 0
        assert detail['generated'] == generated and len(trace['uniforms']) == 5200
        assert physical.crn_pair(uniforms, trace['uniforms'])['exact']
        bins = Counter(q['executed_request_s'] for q in trace['queries'])
        assert sum(bins.values()) == 5200
        assert sum(h * n for h, n in bins.items()) == round(262 * result['endpoints'][DOSE])
        if arm == CAP or arm.startswith('fixed_cap_'):
            assert max(bins) <= int(arm.rsplit('_', 1)[1])
        rows[arm] = dict(metrics=dict(result['endpoints'], unboarded=detail['unserved'],
            journey_p95_s=detail['journey_p50_p90_p95_s'][2]), bins=bins,
            graph_statuses=result['graph']['statuses'], wall_s=result['wall_s'],
            callback_p95_s=operating['callback_p50_p95_max_s'][1])
    return dict(day=day, generated=generated, paired_events=5200, arms=rows)


def development(root, day):
    assert read(root / 'preflight/summary/results.json')['passed'] and day in DEV
    old = OLD / 'cases' / REGIME / str(day) / FULL
    trace = frozen.helper(frozen.ASSETS, 'gate_default').raw(old / 'raw/path.pkl.gz')
    generated = read(old / 'summary/results.json')['passenger_detail']['generated']
    row = collect(root / 'development', day, (CAP,), (trace['uniforms'], generated))
    write(root / 'calibration_days' / str(day) / 'summary/results.json', row)


def calibrate(root):
    dev = {}
    for day in DEV:
        row = read(root / 'calibration_days' / str(day) / 'summary/results.json')
        assert row['day'] == day and row['paired_events'] == 5200
        dev[day] = {'fixed_cap_0': 0., CAP: row['arms'][CAP]['metrics'][DOSE]}
        for arm in (FULL, 'fixed_cap_10', 'fixed_cap_15'):
            dev[day][arm] = read(OLD / 'cases' / REGIME / str(day) / arm /
                                 'summary/results.json')['endpoints'][DOSE]
    means = {arm: sum(row[arm] for row in dev.values()) / len(DEV) for arm in ARMS}
    weights = {candidate: mixture(means[candidate], means[low], means[high], low, high)
        for candidate, low, high in ((FULL, 'fixed_cap_10', 'fixed_cap_15'),
                                     (CAP, 'fixed_cap_0', 'fixed_cap_10'))}
    result = dict(passed=weights[FULL] is not None, weights=weights,
                  development_dose_by_day=dev, development_mean_dose=means)
    out = root / 'calibration/summary'
    write(out / 'results.json', result)
    if result['passed']:
        write(out / 'ready.json', result)


def confirmation(root, day):
    assert read(root / 'calibration/summary/ready.json')['passed'] and day in DAYS
    write(root / 'pairs' / str(day) / 'summary/results.json', collect(root, day, ARMS))


def aggregate(root):
    weights = read(root / 'calibration/summary/ready.json')['weights']
    rows = {day: read(root / 'pairs' / str(day) / 'summary/results.json') for day in DAYS}
    assert all(row['day'] == day and set(row['arms']) == set(ARMS) for day, row in rows.items())
    starts = sum(len(list(path.glob(f'cases/{REGIME}/*/*/started.json')))
                 for path in (root, root / 'development'))
    assert starts == PLAN['physical_start_cap']
    cfg = PLAN['bootstrap']
    draws = np.random.default_rng(cfg['seed']).integers(0, len(DAYS),
                                                       size=(cfg['draws'], len(DAYS)))
    contrasts = [(f'{candidate}_matched', candidate, ref) for candidate, ref in weights.items()
                 if ref is not None]
    contrasts += [(f'{candidate}_minus_{control}', candidate, {control: 1.})
                  for candidate in (FULL, CAP) for control in ARMS
                  if control != candidate and (candidate == FULL or control != FULL)]
    effects = {}
    for name, candidate, ref in contrasts:
        metrics = {}
        for metric in rows[DAYS[0]]['arms'][candidate]['metrics']:
            values = np.asarray([row['arms'][candidate]['metrics'][metric] -
                sum(w * row['arms'][arm]['metrics'][metric] for arm, w in ref.items())
                for row in rows.values()])
            assert all(map(math.isfinite, values))
            metrics[metric] = dict(mean=float(values.mean()), by_day=dict(zip(DAYS, values.tolist())),
                ci95=np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist())
        effects[name] = metrics
    gates = {}
    tol = PLAN['holding_equivalence_s_per_trip']
    for arm in (FULL, CAP):
        match = effects.get(f'{arm}_matched')
        if match is None:
            gates[arm] = dict(identified=False, passed=False)
            continue
        lo, hi = match[DOSE]['ci95']
        checks = dict(dose_matched=-tol <= lo <= hi <= tol, journey=match[JOURNEY]['ci95'][1] < 0,
                      **{k: match[k]['ci95'][1] <= 0 for k in SAFETY})
        gates[arm] = dict(**checks, identified=True, passed=all(checks.values()))
    write(root / 'aggregate/summary/results.json', dict(complete=True, physical_starts=starts,
        training_updates=0, main_policy_changed=False, weights=weights, dose_margin=tol,
        gates=gates, effects=effects, by_day=rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('preflight', 'development', 'calibrate', 'confirmation', 'aggregate'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--day', type=int)
    a = p.parse_args()
    args = (a.root, a.day) if a.day is not None else (a.root,)
    globals()[a.phase](*args)
