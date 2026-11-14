"""Compare the frozen capped policy with only its full-link clock changed."""

import argparse
from collections import Counter
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
V412 = WORK / 'FreqDuet-v412-running-speed-transfer'
V413 = WORK / 'FreqDuet-v413-public-slot-observer'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shadow_checks = load('v414_checks', V413 / 'running_shadow01/running_shadow_checks.py')
run = shadow_checks.run
old, frozen, read, write = run.old, run.old.frozen, run.read, run.write
import frozen_v406
from running_shadow import RunningByHour, detached_kernel

PLAN = read(HERE / 'protocol.json')
REGIME, CAP = run.REGIME, run.CAP
DAYS = PLAN['days'][REGIME]
BASE, HOURLY = PLAN['arms']
BINS = (0, 5, 10)
PRIMARY = ('readiness_s_per_trip', 'public_delay_s_per_day', 'journey_s_per_passenger', 'cv_squared')


class Pooled:
    def __init__(self, model):
        self.forecasts = model.forecasts

    def seconds(self, key, entry_s):
        return math.ceil(self.forecasts[key]['run_s'])


def graph_type(clock_factory, counts):
    simulate = detached_kernel()

    class RunningGraph(frozen_v406.ReferenceGraph):
        def __init__(self, *args):
            super().__init__(*args)
            self.running_clock = clock_factory(self)

        def evaluate(self, commands=None, homogeneous=False):
            submitted = frozen_v406.ReferenceCommands(commands or {}, {})
            continuation = frozen_v406._BudgetContinuation(self, submitted)
            result = simulate(self, submitted, homogeneous, continuation, self.running_clock)
            future = {k: v[self.seed_sizes[k]:] for k, v in result['arrivals'].items()}
            result['cv2'] = self.arrival_prefix.extend(future).cv_squared
            counts['evaluations'] += 1
            if isinstance(self.running_clock, RunningByHour):
                counts.update(self.running_clock.calls)
                self.running_clock.calls.clear()
            return result

    return RunningGraph


def hourly(model):
    return RunningByHour(model.completed, model.forecasts, model.snapshot.now_s)


def check_graph():
    names = shadow_checks.checks()
    fixture = load('v415_live_fixture', frozen.HERE / 'tests.py').fixture
    original, _ = fixture()
    # Reuse the exact observed current-query state for each clock.
    model = object.__new__(graph_type(Pooled, Counter()))
    model.__dict__.update(vars(original), running_clock=Pooled(original))
    for command in ({}, *({(0, 10, 0): h} for h in BINS)):
        baseline = original.evaluate(command)
        assert baseline['coverage'] and not baseline['failure']
        assert model.evaluate(command) == baseline
    hourly_cls = graph_type(hourly, Counter())
    model = object.__new__(hourly_cls)
    model.__dict__.update(vars(original), running_clock=hourly(original))
    baseline = original.evaluate()
    assert model.evaluate() == baseline
    key = (False, 2)
    model.completed = dict(original.completed)
    model.completed[key] = [replace(v, run_s=14.) for v in original.completed[key]]
    model.running_clock = hourly(model)
    changed = model.evaluate()
    assert changed['coverage'] and not changed['failure']
    assert changed['actions'][0, 10, 0] == baseline['actions'][0, 10, 0]
    assert changed['returns'][0] == baseline['returns'][0] + 4
    return [*names, 'pooled live-root reference and all three legal capped actions exactly equal',
            'hourly live-root terminal timing changes while the issued root command is identical']


def preflight(root):
    source = read(V413 / 'run01/running_shadow_readout/summary/results.json')
    assert source['complete'] and source['primary_gate']['passed'] and source['paid_confirmation_paths'] == 8
    assert BASE == CAP and REGIME == 'slow' and PLAN['road_profile_shift_mps'] == run.SHIFT[REGIME]
    assert PLAN['regimes'][REGIME] == run.old.PLAN['regimes']['heavy']
    assert PLAN['engineering_day'] == run.PARITY and run.PARITY not in DAYS
    assert len(set(DAYS)) == len(DAYS) == PLAN['max_concurrent_days'] == 8
    assert PLAN['physical_start_cap'] == 1 + len(DAYS) * len(PLAN['arms'])
    assert not PLAN['automatic_retry']
    assert not (root / 'engineering').exists() and not (root / 'cases').exists()
    reserved, examined = set(map(str, DAYS)), 0
    for project in WORK.glob('FreqDuet-*'):
        if project == HERE.parent or not project.is_dir():
            continue
        for day in DAYS:
            for pattern in ('run*/cases/*/{}', 'run*/confirmation/{}', 'run*/running_shadow_confirmation/{}'):
                assert not next(project.glob(pattern.format(day)), None), (project, day)
        for manifest in project.glob('source*/*protocol.json'):
            assert not reserved & old.base.declared_days(read(manifest)), manifest
            examined += 1
    assert not reserved & old.base.declared_days(read(V413 / 'running_shadow01/running_shadow_protocol.json'))
    assert Path(frozen_v406.__file__) == frozen.HERE / 'frozen_v406.py'
    for path in (root, root / 'engineering'):
        frozen.prepare(path)
    bound = frozen.study.bind(frozen.ASSETS)
    bound.base.fixed.pilot.joint.optimized_runtime(root / 'runtime/import_check')
    write(root / 'preflight/summary/results.json', dict(passed=True, physical_starts=0, days=DAYS,
          checks=check_graph(), manifests=examined, engineering_day=PLAN['engineering_day']))


def collect(root, day, arm, clock_factory=None):
    counts = Counter()
    gate = run.GATE['frozen_gate']

    def make_audit(runtime, credit, label, settings):
        base = old.make_audit(runtime, credit, CAP, settings, day, gate['counts'], gate['tilt'])

        class SpeedAudit(type(base)):
            def attach(self, runner):
                super().attach(runner)
                self.env.routes_set = run.shifted_profile(self.env.routes_set, PLAN['road_profile_shift_mps'])
                self.expected = self.env.routes_set.iloc[:, 5:].to_numpy(dtype=float)

            def finish(self, runner, records, holds):
                super().finish(runner, records, holds)
                run.np.testing.assert_array_equal([r.speed_history.to_numpy(dtype=float) for r in self.env.routes],
                                                  self.expected)
                assert self.env._demand_scale == 1.2 and self.contract['physical_vehicle_count'] == 12
                self.contract['applied_road_profile_shift_mps'] = PLAN['road_profile_shift_mps']

        return SpeedAudit('joint', credit)

    graph = frozen_v406.ReferenceGraph if clock_factory is None else graph_type(clock_factory, counts)
    args = SimpleNamespace(root=root, assets=frozen.ASSETS)
    with patch.object(frozen.study, 'HERE', HERE), patch.object(frozen.study, 'make_audit', make_audit), \
            patch.object(frozen_v406, 'ReferenceGraph', graph):
        result, trace, _ = frozen.study.collect(args, REGIME, day, arm)
    assert result['physical_account']['passed'] and result['operating_detail']['early_departures'] == 0
    assert len(trace['uniforms']) == len(trace['queries']) == len(trace['graph_queries']) == 5200
    assert len(trace['dispatches']) == len(trace['returns']) == 262
    assert result['physical_starts'] == 1 and not result['training_updates'] and not result['main_policy_changed']
    bins = Counter(q['executed_request_s'] for q in trace['queries'])
    assert set(bins) <= set(BINS)
    assert sum(h*n for h, n in bins.items()) == round(262 * result['endpoints']['holding_s_per_trip'])
    return result, trace, dict(counts), bins


def engineering(root):
    assert read(root / 'preflight/summary/results.json')['passed']
    day = PLAN['engineering_day']
    prior = run.V412_ROOT / 'cases' / REGIME / str(day) / CAP
    archive = frozen.helper(frozen.ASSETS, 'gate_default').raw(prior / 'raw/path.pkl.gz')
    reference = read(prior / 'summary/results.json')
    result, trace, counts, _ = collect(root / 'engineering', day, BASE, Pooled)
    compare = frozen.helper(frozen.ASSETS, 'physical').first_diff
    assert set(archive) == set(trace) and counts['evaluations'] > 0
    for name in archive:
        a, b = archive[name], trace[name]
        if name == 'graph_queries':
            a, b = ([{k: v for k, v in r.items() if k != 'total_s'} for r in rows] for rows in (a, b))
        assert (diff := compare(a, b, name)) is None, diff
    for name in ('endpoints', 'execution', 'action_chain', 'contract'):
        assert (diff := compare(reference[name], result[name], name)) is None, diff
    write(root / 'engineering/summary/results.json', dict(passed=True, physical_starts=1,
          day=day, streams=sorted(trace), paired_events=5200, evaluations=counts['evaluations'],
          exact_endpoints=True, wall_s=result['wall_s']))


def pair(root, day):
    assert day in DAYS and read(root / 'engineering/summary/results.json')['passed']
    rows, traces = {}, {}
    for arm, clock in ((BASE, None), (HOURLY, hourly)):
        result, trace, counts, bins = collect(root, day, arm, clock)
        detail, op = result['passenger_detail'], result['operating_detail']
        traces[arm] = trace
        rows[arm] = dict(metrics=dict(result['endpoints'], journey_p95_s=detail['journey_p50_p90_p95_s'][2],
                                      unboarded=detail['unserved'], incomplete=detail['incomplete']),
                         generated=detail['generated'], bins=bins, predictor=counts,
                         graph_statuses=result['graph']['statuses'], wall_s=result['wall_s'],
                         callback_p95_s=op['callback_p50_p95_max_s'][1])
    assert rows[BASE]['generated'] == rows[HOURLY]['generated']
    base, candidate = traces[BASE], traces[HOURLY]
    crn = frozen.helper(frozen.ASSETS, 'physical').crn_pair(base['uniforms'], candidate['uniforms'])
    assert crn['exact'] and crn['common_events'] == 5200 and crn['left_only'] == crn['right_only'] == 0
    assert base['public_clocks'] == candidate['public_clocks']
    dose = frozen.helper(frozen.ASSETS, 'v324').actual_dose(candidate, base)
    assert not dose['scheduled_release_changed_trips']
    write(root / 'pairs' / str(day) / 'summary/results.json', dict(day=day, paired_events=5200,
          physical_starts=2, training_updates=0, main_policy_changed=False, arms=rows, action_dose=dose))


def aggregate(root):
    rows = {day: read(root / 'pairs' / str(day) / 'summary/results.json') for day in DAYS}
    assert all(row['day'] == day and row['paired_events'] == 5200 and row['physical_starts'] == 2
               and set(row['arms']) == set(PLAN['arms']) for day, row in rows.items())
    starts = sum(1 for _ in root.glob('cases/slow/*/*/started.json'))
    starts += read(root / 'engineering/summary/results.json')['physical_starts']
    assert starts == PLAN['physical_start_cap']
    np, cfg = run.np, PLAN['bootstrap']
    draws = np.random.default_rng(cfg['seed']).integers(0, len(DAYS), (cfg['draws'], len(DAYS)))
    effects = {}
    for metric in rows[DAYS[0]]['arms'][BASE]['metrics']:
        delta = np.asarray([r['arms'][HOURLY]['metrics'][metric] - r['arms'][BASE]['metrics'][metric]
                            for r in rows.values()])
        assert np.isfinite(delta).all(), metric
        effects[metric] = dict(mean=float(delta.mean()), by_day=dict(zip(DAYS, delta.tolist())),
                              ci95=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist())
    checks = {k: effects[k]['ci95'][1] <= 0 for k in PRIMARY}
    checks[PRIMARY[0]] = effects[PRIMARY[0]]['ci95'][1] < 0
    write(root / 'aggregate/summary/results.json', dict(complete=True, passed=all(checks.values()),
          physical_starts=starts, training_updates=0, main_policy_changed=False,
          checks=checks, effects=effects, by_day=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight', 'engineering', 'pair', 'aggregate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--day', type=int)
    args = parser.parse_args()
    globals()[args.phase](args.root, *(() if args.day is None else (args.day,)))
